# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Déjà View Looping Transformer (DVLT).

A recurrent transformer that loops shared attention blocks with discrete
depth indexing:

  z_0 = DINOv2 patch features
  for k in active_steps: z = block(z, k)
  z_final → decoder heads → rays, depth, camera poses

Each step k has a learned embedding. Training uses N_max steps with
stochastic depth (randomly dropping to N_min), so the model learns to
produce valid outputs from any step count in [N_min, N_max]. At
inference, the model runs a fixed number of steps (``inference_steps``
when ``k_sampling="linspace"``, otherwise ``num_steps``).

:class:`LoopedAABlock` alternates intra-frame attention (over the per-frame
tokens) with inter-frame ("global") attention over the flattened sequence,
applied recurrently with shared weights and step-conditioned depth scaling.
"""

import logging
from typing import Any, Dict, Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from dvlt.common.amp import force_fp32
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.common.rays import rays_to_pose

# LOCAL PATCH (augenblick): inference-only. The training losses (and LossConfig,
# which pulled in config/schema.py and Hydra) are not vendored; train with upstream.
from dvlt.model.base import Model, Module
from dvlt.model_components import (
    PositionGetter,
    RotaryPositionEmbedding2D,
    activate_head,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)
from dvlt.struct.util import extri_intri_to_cameras

from .blocks import LoopedAABlock
from .heads import DecoderHead, SimpleCameraHead


logger = logging.getLogger(__name__)


# ==================== Constants & helpers (formerly in diffnrm/model_pi3.py and diffnrm/decoder.py) ====================

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

_PATCH_EMBED_URLS = {
    "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitg2_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}

_PATCH_EMBED_BUILDERS = {
    "dinov2_vitl14_reg": vit_large,
    "dinov2_vitb14_reg": vit_base,
    "dinov2_vits14_reg": vit_small,
    "dinov2_vitg2_reg": vit_giant2,
}


def _slice_expand_flatten(token: Tensor, B: int, S: int) -> Tensor:
    """Expand a (1, 2, X, C) token for B batches and S sequences.

    Index 0 is used for the first frame, index 1 for the remaining S-1 frames.
    """
    first = token[:, 0:1].expand(B, 1, -1, -1)
    rest = token[:, 1:2].expand(B, S - 1, -1, -1)
    return torch.cat([first, rest], dim=1).reshape(B * S, *token.shape[2:])


# ==================== Model ====================


class DVLTModel(
    Model,
    PyTorchModelHubMixin,
    library_name="dvlt",
    repo_url="https://research.nvidia.com/labs/dvl/projects/dvlt/",
    paper_url="https://arxiv.org/abs/2605.30215",
    docs_url="https://research.nvidia.com/labs/dvl/projects/dvlt/",
    pipeline_tag="image-to-3d",
    license="other",
    license_name="nvidia-internal-scientific-research-and-development-model-license",
    license_link="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/",
    tags=[
        "3d-reconstruction",
        "depth-estimation",
        "camera-pose",
        "pointmap",
        "looped-transformer",
    ],
):
    """Déjà View Looping Transformer with block-recurrent attention.

    Training: N_max fixed steps with stochastic depth dropping to N_min.
    Inference: fixed step count (``inference_steps`` or ``num_steps``).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
        num_steps: int = 16,
        min_steps: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        patch_embed: str = "dinov2_vitb14_reg",
        load_patch_embed_weights: bool = True,
        decoder_depth: int = 2,
        decoder_embed_dim: int = 384,
        decoder_num_heads: int = 6,
        gradient_checkpointing_config: Optional[Dict[str, Any]] = None,
        camera_head: bool = True,
        drop_path: float = 0.1,
        stochastic_depth: float = 0.3,
        stochastic_depth_mode: str = "random",
        sync_stochastic_depth: bool = True,
        recurrence_mode: str = "gated",
        time_conditioning: str = "interval",
        k_sampling: Optional[str] = "linspace",
        k_sampler_beta_a: int = 2,
        k_sampler_beta_b: int = 1,
        inference_steps: Optional[int] = None,
        decoder_head_type: str = "linear",
        depth_head_type: Optional[str] = "conv",
        depth_decoder_depth: Optional[int] = None,
        depth_decoder_embed_dim: Optional[int] = None,
        depth_decoder_num_heads: Optional[int] = None,
        decoder_init_values: Optional[float] = None,
        decode_chunk_size: Optional[int] = 128,
    ):
        """
        Args:
            num_steps: Number of discrete steps. Training budget.
            min_steps: Minimum steps to always execute during training (lower bound
                for stochastic depth).
            stochastic_depth: Fraction of steps to drop during training.
            stochastic_depth_mode: "random" or "prefix".
            sync_stochastic_depth: If True, all DDP ranks execute the same number of steps.
            recurrence_mode: Controls gating structure.
                "gated"          — s_attn + s_mlp + s_out (multiplicative state gating)
                "no_sout"        — s_attn + s_mlp (residual-preserving, no state gating)
                "no_depthscale"  — shared blocks, no depth scaling (fixed-point iteration)
                "none"           — distinct block per step, no depth scaling
            time_conditioning: Controls depth embedding type (for gated / no_sout).
                "continuous" — sinusoidal embedding of t ∈ [0, 1] per step
                "interval"   — sinusoidal embeddings of (t_now, t_next) concatenated;
                    uses a global t mapping so the model sees both where it is and
                    where the next step lands. Requires k_sampling="linspace".
                Ignored for no_depthscale / none modes.
            k_sampling: Training t-grid strategy for continuous / interval modes.
                None — use stochastic-depth path (integer steps).
                "linspace" — variable-K uniform linspace grid on [0, 1]; K drawn per
                    batch from Beta(k_sampler_beta_a, k_sampler_beta_b) scaled to
                    [min_steps, num_steps].
            k_sampler_beta_a, k_sampler_beta_b: Positive-integer shape parameters of
                the Beta distribution used to sample K when k_sampling="linspace".
                E[K] = min_steps + a/(a+b) * (num_steps - min_steps).
            inference_steps: Number of steps at inference when k_sampling is set.
                Defaults to num_steps. May exceed num_steps for finer-grained inference.
            decoder_head_type: "linear" (per-patch linear + pixel_shuffle) or
                "conv" (Pi3X/MoGe progressive ConvTranspose2d upsample).
            depth_head_type: If set, overrides ``decoder_head_type`` for the depth
                decoder only. Same choices as ``decoder_head_type``.
            depth_decoder_depth: If set, overrides ``decoder_depth`` for the depth
                decoder only. Useful for widening only the depth path without
                breaking checkpoint loading for the other heads.
            depth_decoder_embed_dim: If set, overrides ``decoder_embed_dim`` for the
                depth decoder only.
            depth_decoder_num_heads: If set, overrides ``decoder_num_heads`` for the
                depth decoder only.
            decoder_init_values: LayerScale init for the ray and depth decoder
                transformer blocks. ``None`` or ``0`` disables LayerScale (Pi3X-style).
                Any positive value keeps LayerScale active; the exact magnitude only
                matters at init time and is overwritten when loading a checkpoint.
            decode_chunk_size: If set, the ray/depth decoders are run on slices of
                size ``decode_chunk_size`` along the ``B*S`` (flattened batch x
                frames) dimension and the outputs are concatenated. The decoders
                are fully per-frame (no cross-frame attention), so this is
                mathematically equivalent to a single pass but avoids cuDNN's
                32-bit indexing overflow on very large dense inputs.
        """
        super().__init__(gradient_checkpointing_config=gradient_checkpointing_config)
        assert stochastic_depth_mode in (
            "random",
            "prefix",
        ), f"stochastic_depth_mode must be 'random' or 'prefix', got '{stochastic_depth_mode}'"
        assert recurrence_mode in (
            "gated",
            "no_sout",
            "no_depthscale",
            "none",
        ), f"recurrence_mode must be gated/no_sout/no_depthscale/none, got '{recurrence_mode}'"
        assert time_conditioning in (
            "continuous",
            "interval",
        ), f"time_conditioning must be continuous/interval, got '{time_conditioning}'"
        assert k_sampling in (
            None,
            "linspace",
        ), f"k_sampling must be None or 'linspace', got '{k_sampling}'"
        # time_conditioning only feeds the depth-scaling module, which exists
        # solely for gated / no_sout. The interval embedding consumes a (t_now,
        # t_next) pair produced only by the linspace solver, so guard that
        # pairing — but skip it entirely when no depth scaling is built.
        if recurrence_mode in ("gated", "no_sout") and time_conditioning == "interval":
            assert k_sampling == "linspace", "time_conditioning='interval' requires k_sampling='linspace'"
        assert (
            isinstance(k_sampler_beta_a, int) and k_sampler_beta_a >= 1
        ), f"k_sampler_beta_a must be a positive int, got {k_sampler_beta_a}"
        assert (
            isinstance(k_sampler_beta_b, int) and k_sampler_beta_b >= 1
        ), f"k_sampler_beta_b must be a positive int, got {k_sampler_beta_b}"
        if inference_steps is not None:
            assert inference_steps > 0, f"inference_steps must be > 0, got {inference_steps}"
        assert decoder_head_type in (
            "linear",
            "conv",
        ), f"decoder_head_type must be 'linear' or 'conv', got '{decoder_head_type}'"
        if depth_head_type is not None:
            assert depth_head_type in (
                "linear",
                "conv",
            ), f"depth_head_type must be 'linear' or 'conv', got '{depth_head_type}'"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.has_camera_head = camera_head
        self.stochastic_depth = stochastic_depth
        self.stochastic_depth_mode = stochastic_depth_mode
        self.sync_stochastic_depth = sync_stochastic_depth
        self.recurrence_mode = recurrence_mode
        self.time_conditioning = time_conditioning
        self.k_sampling = k_sampling
        self.k_sampler_beta_a = k_sampler_beta_a
        self.k_sampler_beta_b = k_sampler_beta_b
        self.inference_steps = inference_steps if inference_steps is not None else num_steps
        if decode_chunk_size is not None:
            assert decode_chunk_size > 0, f"decode_chunk_size must be > 0, got {decode_chunk_size}"
        self.decode_chunk_size = decode_chunk_size

        self._gate_mode = recurrence_mode if recurrence_mode in ("gated", "no_sout") else "none"
        self._shared_blocks = recurrence_mode != "none"

        self._sd_rng = torch.Generator(device="cpu")

        if self._shared_blocks:
            self._block_ranges = [(0, num_steps)]
            self._min_steps_per_block = [min_steps]
        else:
            self._block_ranges = [(k, k + 1) for k in range(num_steps)]
            self._min_steps_per_block = [1] * num_steps

        # ==================== Patch Embedder ====================
        self._build_patch_embed(patch_embed, img_size, patch_size, num_register_tokens, load_patch_embed_weights)

        # ==================== Positional Encoding ====================
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()

        # ==================== Special Tokens ====================
        self.num_register_tokens = num_register_tokens
        self.patch_start_idx = 1 + num_register_tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ==================== DINOv2 Feature Normalization ====================
        self.patch_embed_encoder.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        # ==================== Recurrent AA Blocks ====================
        block_kw = dict(
            dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            init_values=0.01,
            qk_norm=True,
            rope=self.rope,
            drop_path=drop_path,
            gate_mode=self._gate_mode,
            time_mode=self.time_conditioning,
        )
        if self._shared_blocks:
            self.recurrent_blocks = nn.ModuleList([LoopedAABlock(**block_kw) for _ in self._block_ranges])
        else:
            self.recurrent_blocks = nn.ModuleList([LoopedAABlock(**block_kw) for _ in range(num_steps)])

        # ==================== Decoder Heads ====================
        _decoder_kw = dict(
            in_dim=embed_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_size=patch_size,
            rope=self.rope,
        )
        # ``decoder_init_values`` controls LayerScale for both ray and depth decoder
        # transformer blocks. ``None`` or ``0`` -> ``Identity`` (Pi3-style shallow-head
        # convention); any positive value keeps LayerScale active at that init.
        self.ray_decoder = DecoderHead(
            out_dim=6, head_type=decoder_head_type, init_values=decoder_init_values, **_decoder_kw
        )
        _depth_decoder_kw = {
            **_decoder_kw,
            "embed_dim": depth_decoder_embed_dim if depth_decoder_embed_dim is not None else decoder_embed_dim,
            "depth": depth_decoder_depth if depth_decoder_depth is not None else decoder_depth,
            "num_heads": depth_decoder_num_heads if depth_decoder_num_heads is not None else decoder_num_heads,
            "init_values": decoder_init_values,
        }
        self.depth_decoder = DecoderHead(out_dim=2, head_type=depth_head_type or decoder_head_type, **_depth_decoder_kw)
        self.camera_head = (
            SimpleCameraHead(in_dim=decoder_embed_dim, hidden_dim=decoder_embed_dim, pose_dim=9)
            if self.has_camera_head
            else None
        )

        # ==================== Normalization Constants ====================
        self.register_buffer("_resnet_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False)

        self.gradient_checkpointing_config = gradient_checkpointing_config or {
            "use_reentrant": False,
            "modules": ["recurrent_blocks", "ray_decoder", "depth_decoder"],
        }

    # ==================== Helpers ====================

    def _build_patch_embed(self, patch_embed, img_size, patch_size, num_register_tokens, load_weights):
        self.patch_embed_encoder = _PATCH_EMBED_BUILDERS[patch_embed](
            img_size=518,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        )
        if load_weights:
            self.patch_embed_encoder.load_state_dict(
                torch.hub.load_state_dict_from_url(_PATCH_EMBED_URLS[patch_embed], map_location="cpu"),
                strict=False,
            )
        if hasattr(self.patch_embed_encoder, "mask_token"):
            self.patch_embed_encoder.mask_token.requires_grad_(False)

    def _normalize_images(self, images: Tensor) -> Tensor:
        return (images - self._resnet_mean) / self._resnet_std

    def _encode_images(self, images: Tensor) -> Tensor:
        B, S, C, H, W = images.shape
        images = self._normalize_images(images).view(B * S, C, H, W)
        return self.patch_embed_encoder(images, is_training=True)["x_norm_patchtokens"]

    def _get_rope_positions(self, BS, H, W, device):
        pos = self.position_getter(BS, H // self.patch_size, W // self.patch_size, device) + 1
        special = torch.zeros(BS, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
        return torch.cat([special, pos], dim=1)

    def _get_block_idx_for_step(self, k: int) -> int:
        return 0 if self._shared_blocks else k

    def _sample_K(self, rng) -> int:
        """Sample K from Beta(a, b) scaled to [min_steps, num_steps] with round-to-bin.

        E[K] = min_steps + a/(a+b) * (num_steps - min_steps). Uses the gamma-ratio
        identity Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b)) with integer shapes
        (sum of Exp(1) samples), so the call can be piped through ``rng`` to sync
        K across DDP ranks when desired.
        """
        a, b = self.k_sampler_beta_a, self.k_sampler_beta_b
        e = torch.empty(a + b).exponential_(generator=rng)
        ga = e[:a].sum()
        gb = e[a:].sum()
        m = (ga / (ga + gb)).item()
        span = self.num_steps - self.min_steps
        return self.min_steps + min(int(round(m * span)), span)

    def _interval_step(self, x, t_now: float, t_next: float, rope_pos, B, S):
        """One step at global (t_now, t_next) ∈ [0, 1]^2.

        The per-step conditioning tensor is shaped by self.time_conditioning:
          "continuous": scalar t_now broadcast to (B,) / (B*S,)
          "interval":   pair (t_now, t_next) broadcast to (B, 2) / (B*S, 2)
        """
        block = self.recurrent_blocks[0]
        if self.time_conditioning == "continuous":
            t_batch = torch.full((B,), t_now, device=x.device, dtype=torch.float32)
            t_frame = t_batch.unsqueeze(1).expand(-1, S).reshape(B * S)
            x = block(x, t_frame, t_batch, rope_pos, B, S)
        else:
            t_pair = torch.tensor([[t_now, t_next]], device=x.device, dtype=torch.float32)
            x = block(x, t_pair.expand(B * S, -1), t_pair.expand(B, -1), rope_pos, B, S)
        return x

    def _step(self, x, k: int, rope_pos, B, S):
        """One step: select block by k, apply with continuous-t conditioning.

        Used by the non-linspace solvers (``k_sampling=None``); the linspace path
        goes through ``_interval_step`` instead.
        """
        block_idx = self._get_block_idx_for_step(k)
        block = self.recurrent_blocks[block_idx]
        if self._gate_mode == "none":
            return block(x, None, None, rope_pos, B, S)
        start, end = self._block_ranges[block_idx] if self._shared_blocks else (k, k + 1)
        local_k = k - start
        t_val = local_k / max(end - start - 1, 1)
        t_batch = torch.full((B,), t_val, device=x.device, dtype=torch.float32)
        t_frame = t_batch.unsqueeze(1).expand(-1, S).reshape(B * S)
        return block(x, t_frame, t_batch, rope_pos, B, S)

    # ==================== Solvers ====================

    def _get_sd_rng(self, step: int):
        """Return the generator for stochastic depth coin flips.

        When sync_stochastic_depth is True, returns _sd_rng seeded with a rank-
        independent key so all GPUs agree on step counts. Otherwise returns None
        (global RNG, already per-rank via set_seed and resume-safe via accelerator
        checkpoint).
        """
        if self.sync_stochastic_depth:
            self._sd_rng.manual_seed(42 + step)
            return self._sd_rng
        return None

    def _solve_train(self, x, rope_pos, B, S, step: int = 0):
        """Training dispatcher: delegates to mode-specific solver."""
        rng = self._get_sd_rng(step)
        if self.k_sampling == "linspace":
            return self._solve_train_linspace_k(x, rope_pos, B, S, rng)
        if self.stochastic_depth_mode == "prefix":
            return self._solve_train_prefix(x, rope_pos, B, S, rng)
        return self._solve_train_random(x, rope_pos, B, S, rng)

    def _solve_train_linspace_k(self, x, rope_pos, B, S, rng):
        """Training with uniform linspace grid and variable step count.

        K ~ Beta(k_sampler_beta_a, k_sampler_beta_b) scaled to [min_steps, num_steps].
        The K positions on [0, 1] are passed to _interval_step as consecutive
        (t_now, t_next) pairs; the final pair sets t_next = 1.0 as a terminal-step
        sentinel (delta_t = 0 for the interval-conditioned module).
        """
        K = self._sample_K(rng)
        ts = torch.linspace(0.0, 1.0, K).tolist()
        for i in range(K):
            t_now = ts[i]
            t_next = ts[i + 1] if i + 1 < K else 1.0
            x = self._interval_step(x, t_now, t_next, rope_pos, B, S)
        return x

    def _solve_train_random(self, x, rope_pos, B, S, rng):
        """Random dropping: each step independently kept/dropped.

        When rng is _sd_rng (synced mode), the coin-flip count is shared across
        ranks but selection uses the global RNG for per-rank diversity. When rng
        is None, everything uses the global RNG.
        """
        N = self.num_steps
        if self.stochastic_depth > 0:
            keep = torch.zeros(N, dtype=torch.bool)

            for (start, end), min_count in zip(self._block_ranges, self._min_steps_per_block, strict=False):
                forced = torch.linspace(start, end - 1, min_count).long()
                keep[forced] = True
            keep[0] = True
            keep[N - 1] = True

            if rng is not None:
                optional = (~keep).nonzero(as_tuple=True)[0]
                n_optional = len(optional)
                if n_optional > 0:
                    n_extra = (torch.rand(n_optional, generator=rng) >= self.stochastic_depth).sum().item()
                    perm = torch.randperm(n_optional)
                    keep[optional[perm[:n_extra]]] = True
            else:
                extra_mask = torch.rand(N) >= self.stochastic_depth
                keep = keep | extra_mask

            active = keep.nonzero(as_tuple=True)[0].tolist()
        else:
            active = list(range(N))

        for k in active:
            x = self._step(x, k, rope_pos, B, S)
        return x

    def _solve_train_prefix(self, x, rope_pos, B, S, rng):
        """Contiguous prefix per block with geometric stopping.

        Runs a contiguous prefix of steps. After ``min_steps``, each additional step
        has ``stochastic_depth`` probability of being the last.
        """
        if self.stochastic_depth <= 0:
            for k in range(self.num_steps):
                x = self._step(x, k, rope_pos, B, S)
            return x

        for (start, end), min_count in zip(self._block_ranges, self._min_steps_per_block, strict=False):
            block_size = end - start
            cutoff = min_count
            if min_count < block_size:
                extra_coins = torch.rand(block_size - min_count, generator=rng)
                for i, coin in enumerate(extra_coins):
                    if coin < self.stochastic_depth:
                        break
                    cutoff = min_count + i + 1

            for k in range(start, start + cutoff):
                x = self._step(x, k, rope_pos, B, S)

        return x

    def _solve_inference_linspace_k(self, x, rope_pos, B, S):
        """Inference for k_sampling='linspace': uniform grid with configurable K."""
        K = self.inference_steps
        ts = torch.linspace(0.0, 1.0, K).tolist()
        for i in range(K):
            t_now = ts[i]
            t_next = ts[i + 1] if i + 1 < K else 1.0
            x = self._interval_step(x, t_now, t_next, rope_pos, B, S)
        return x

    def _solve_inference(self, x, rope_pos, B, S):
        """Inference: run each block's full step range."""
        if self.k_sampling == "linspace":
            return self._solve_inference_linspace_k(x, rope_pos, B, S)
        for start, end in self._block_ranges:
            for k in range(start, end):
                x = self._step(x, k, rope_pos, B, S)
        return x

    # ==================== Decode ====================

    def _decode(self, features, H, W, B, S, rope_pos):
        BS = B * S
        chunk = self.decode_chunk_size if self.decode_chunk_size is not None else BS
        need_cam_tokens = self.has_camera_head

        ray_chunks: list[Tensor] = []
        depth_chunks: list[Tensor] = []
        cam_token_chunks: list[Tensor] = [] if need_cam_tokens else []
        for start in range(0, BS, chunk):
            end = min(start + chunk, BS)
            f = features[start:end]
            p = rope_pos[start:end]
            _dec_kw = dict(H=H, W=W, patch_start_idx=self.patch_start_idx, pos=p)
            ray_out_c, ray_features_c = self.ray_decoder(f, **_dec_kw)
            depth_out_c, _ = self.depth_decoder(f, **_dec_kw)
            ray_chunks.append(ray_out_c)
            depth_chunks.append(depth_out_c)
            if need_cam_tokens:
                cam_token_chunks.append(ray_features_c[:, 0])

        ray_out = ray_chunks[0] if len(ray_chunks) == 1 else torch.cat(ray_chunks, dim=0)
        depth_out = depth_chunks[0] if len(depth_chunks) == 1 else torch.cat(depth_chunks, dim=0)

        pred_rays, _ = activate_head(ray_out, activation="identity", conf_activation=None)
        pred_rays = pred_rays.view(B, S, H, W, 6)

        pred_depth, depth_conf = activate_head(depth_out, activation="exp_clamped", conf_activation="exp_plus_one")
        pred_depth = pred_depth.view(B, S, H, W, 1)
        depth_conf = depth_conf.view(B, S, H, W)

        # World points derived from depth + ray (origin + direction * depth).
        pred_points = pred_rays[..., 3:6] + pred_rays[..., :3] * pred_depth

        predictions = {
            "world_points": pred_points,
            "world_points_conf": depth_conf,
            "depth": pred_depth,
            "depth_conf": depth_conf,
            "rays": pred_rays,
        }
        if need_cam_tokens:
            cam_tokens = cam_token_chunks[0] if len(cam_token_chunks) == 1 else torch.cat(cam_token_chunks, dim=0)
            predictions["pose_enc"] = self.camera_head(cam_tokens, B, S)
        return predictions

    # ==================== Forward ====================

    def forward_train(self, images, step: int = 0):
        B, S, _, H, W = images.shape
        z_0 = self._encode_images(images)
        register_token = self.register_token.expand(B, S, -1, -1).reshape(B * S, self.num_register_tokens, -1)
        camera_token = _slice_expand_flatten(self.camera_token, B, S)
        x = torch.cat([camera_token, register_token, z_0], dim=1)
        rope_pos = self._get_rope_positions(B * S, H, W, images.device)
        features = self._solve_train(x, rope_pos, B, S, step=step)
        return self._decode(features, H, W, B, S, rope_pos)

    @torch.no_grad()
    def forward_inference(self, images):
        B, S, _, H, W = images.shape
        z_0 = self._encode_images(images)
        register_token = self.register_token.expand(B, S, -1, -1).reshape(B * S, self.num_register_tokens, -1)
        camera_token = _slice_expand_flatten(self.camera_token, B, S)
        x = torch.cat([camera_token, register_token, z_0], dim=1)
        rope_pos = self._get_rope_positions(B * S, H, W, images.device)
        features = self._solve_inference(x, rope_pos, B, S)
        return self._decode(features, H, W, B, S, rope_pos)

    def forward(self, batch, step: int = 0):
        images = batch[DataField.IMAGES]
        if self.training:
            return self.forward_train(images, step=step)
        return self.forward_inference(images)


# ==================== Module Wrapper ====================


class DVLT(Module):

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
        num_steps: int = 16,
        min_steps: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        patch_embed: str = "dinov2_vitb14_reg",
        load_patch_embed_weights: bool = True,
        decoder_depth: int = 2,
        decoder_embed_dim: int = 384,
        decoder_num_heads: int = 6,
        gradient_checkpointing_config: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        camera_head: bool = True,
        drop_path: float = 0.1,
        stochastic_depth: float = 0.3,
        stochastic_depth_mode: str = "random",
        sync_stochastic_depth: bool = True,
        recurrence_mode: str = "gated",
        time_conditioning: str = "interval",
        k_sampling: Optional[str] = "linspace",
        k_sampler_beta_a: int = 2,
        k_sampler_beta_b: int = 1,
        inference_steps: Optional[int] = 12,
        decoder_head_type: str = "linear",
        depth_head_type: Optional[str] = "conv",
        finetune_mode: Optional[str] = None,
        reset_depth_decoder_transformer: bool = False,
        depth_decoder_depth: Optional[int] = None,
        depth_decoder_embed_dim: Optional[int] = None,
        depth_decoder_num_heads: Optional[int] = None,
        decoder_init_values: Optional[float] = None,
        decode_chunk_size: Optional[int] = None,
        use_depth_conf_for_pose: bool = False,
        world_points_from_rays: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            use_depth_conf_for_pose: If True, use ``depth_conf`` to weight the
                RANSAC/homography fit in ``rays_to_pose``. If False (default),
                treat every pixel as equally trustworthy (uniform weights).
            world_points_from_rays: If True, override ``WORLD_POINTS`` to be the
                training-style direct compose ``ray_origin + ray_direction *
                depth`` (the same value already exposed under
                ``WORLD_POINTS_DIRECT``), and skip the depth + fitted-pose
                unprojection entirely. Default ``False`` keeps the current
                behavior (``WORLD_POINTS`` = depth unprojected via fitted pose;
                ``WORLD_POINTS_DIRECT`` = rays+depth, unchanged either way).
            finetune_mode: Selective unfreezing after loading a pretrained checkpoint.
                None — normal training, everything trainable.
                "depth_output" — freeze all, unfreeze only depth decoder output stage
                    (conv/linear head). Backbone, recurrent blocks, ray decoder,
                    camera head, and depth decoder front-end stay frozen.
                "depth_decoder" — freeze backbone + recurrent blocks + ray/camera;
                    train the full ``depth_decoder`` (transformer + output stage).
                "depth_decoder_recurrent" — like ``depth_decoder`` but also unfreeze
                    ``recurrent_blocks`` (AA stack). Use ``trainer.lr_param_group_multipliers``
                    (e.g. ``recurrent_blocks: 0.01``) to give the AA path a lower LR.
                "all_heads" — freeze backbone + recurrent blocks, unfreeze all
                    decoder heads (ray, depth, camera) in full.
            reset_depth_decoder_transformer: If True, after ``load_pretrained``,
                reinitialize ``depth_decoder`` transformer front-end (``proj_in``,
                ``blocks``, ``norm``) so only the output stage keeps checkpoint
                weights (until you train). Ignored when no checkpoint is loaded.
        """
        assert finetune_mode in (
            None,
            "depth_output",
            "depth_decoder",
            "depth_decoder_recurrent",
            "all_heads",
        ), (
            "finetune_mode must be None, 'depth_output', 'depth_decoder', "
            f"'depth_decoder_recurrent', or 'all_heads', got '{finetune_mode}'"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.patch_embed = patch_embed
        self.load_patch_embed_weights = load_patch_embed_weights
        self.decoder_depth = decoder_depth
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.gradient_checkpointing_config = gradient_checkpointing_config
        self.camera_head = camera_head
        self.drop_path = drop_path
        self.stochastic_depth = stochastic_depth
        self.stochastic_depth_mode = stochastic_depth_mode
        self.sync_stochastic_depth = sync_stochastic_depth
        self.recurrence_mode = recurrence_mode
        self.time_conditioning = time_conditioning
        self.k_sampling = k_sampling
        self.k_sampler_beta_a = k_sampler_beta_a
        self.k_sampler_beta_b = k_sampler_beta_b
        self.inference_steps = inference_steps
        self.decoder_head_type = decoder_head_type
        self.depth_head_type = depth_head_type
        self.finetune_mode = finetune_mode
        self.reset_depth_decoder_transformer = reset_depth_decoder_transformer
        self.depth_decoder_depth = depth_decoder_depth
        self.depth_decoder_embed_dim = depth_decoder_embed_dim
        self.depth_decoder_num_heads = depth_decoder_num_heads
        self.decoder_init_values = decoder_init_values
        self.decode_chunk_size = decode_chunk_size
        self.use_depth_conf_for_pose = use_depth_conf_for_pose
        self.world_points_from_rays = world_points_from_rays

        # When finetuning, restrict gradient checkpointing to the modules that
        # are actually being trained. Checkpointing frozen modules (no backward)
        # is pure recompute overhead with no memory win.
        if self.finetune_mode is not None and gradient_checkpointing_config is None:
            _ft_ckpt_modules = {
                "depth_output": ["depth_decoder"],
                "depth_decoder": ["depth_decoder"],
                "depth_decoder_recurrent": ["depth_decoder", "recurrent_blocks"],
                "all_heads": ["ray_decoder", "depth_decoder"],
            }.get(self.finetune_mode, ["depth_decoder"])
            self.gradient_checkpointing_config = {
                "use_reentrant": False,
                "modules": _ft_ckpt_modules,
            }

        # LOCAL PATCH (augenblick): no loss_fn; it was used only by (removed) train_step and
        # holds no checkpoint weights (those load into self.model). `loss` is kept
        # in the signature so upstream configs still bind, but must be unset.
        if loss is not None:
            raise ValueError("DVLT is vendored inference-only; `loss` is not supported")
        super().__init__(*args, **kwargs)
        self.model_file = "model.safetensors"

        if self.finetune_mode is not None:
            self._setup_finetune()

    def build_model(self):
        return DVLTModel(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_steps=self.num_steps,
            min_steps=self.min_steps,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_register_tokens=self.num_register_tokens,
            patch_embed=self.patch_embed,
            load_patch_embed_weights=self.load_patch_embed_weights,
            decoder_depth=self.decoder_depth,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_num_heads=self.decoder_num_heads,
            gradient_checkpointing_config=self.gradient_checkpointing_config,
            camera_head=self.camera_head,
            drop_path=self.drop_path,
            stochastic_depth=self.stochastic_depth,
            stochastic_depth_mode=self.stochastic_depth_mode,
            sync_stochastic_depth=self.sync_stochastic_depth,
            recurrence_mode=self.recurrence_mode,
            time_conditioning=self.time_conditioning,
            k_sampling=self.k_sampling,
            k_sampler_beta_a=self.k_sampler_beta_a,
            k_sampler_beta_b=self.k_sampler_beta_b,
            inference_steps=self.inference_steps,
            decoder_head_type=self.decoder_head_type,
            depth_head_type=self.depth_head_type,
            depth_decoder_depth=self.depth_decoder_depth,
            depth_decoder_embed_dim=self.depth_decoder_embed_dim,
            depth_decoder_num_heads=self.depth_decoder_num_heads,
            decoder_init_values=self.decoder_init_values,
            decode_chunk_size=self.decode_chunk_size,
        )

    def _setup_finetune(self):
        """Freeze parameters according to ``finetune_mode``."""
        for param in self.model.parameters():
            param.requires_grad = False

        if self.finetune_mode == "depth_output":
            for param in self.model.depth_decoder.output_stage_parameters():
                param.requires_grad = True
        elif self.finetune_mode == "depth_decoder":
            for param in self.model.depth_decoder.parameters():
                param.requires_grad = True
        elif self.finetune_mode == "depth_decoder_recurrent":
            for param in self.model.depth_decoder.parameters():
                param.requires_grad = True
            for param in self.model.recurrent_blocks.parameters():
                param.requires_grad = True
        elif self.finetune_mode == "all_heads":
            for module in self._head_modules():
                for param in module.parameters():
                    param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"finetune_mode='{self.finetune_mode}': {n_trainable:,} / {n_total:,} params trainable "
            f"({100 * n_trainable / n_total:.1f}%)"
        )

    def _head_modules(self):
        """Yield all decoder head modules (ray, depth, camera)."""
        yield self.model.ray_decoder
        yield self.model.depth_decoder
        if self.model.camera_head is not None:
            yield self.model.camera_head

    def load_pretrained(self, *args, **kwargs):
        missing_keys, unexpected_keys = super().load_pretrained(*args, **kwargs)
        if self.finetune_mode is not None:
            logger.info(
                f"finetune_mode='{self.finetune_mode}': checkpoint loaded, "
                "mismatched head keys (if any) left randomly initialized"
            )
        if self.reset_depth_decoder_transformer:
            depth_output_missing = any(
                k.startswith("depth_decoder.upsample_blocks.") or k.startswith("depth_decoder.output_block.")
                for k in missing_keys
            )
            if not depth_output_missing:
                logger.info(
                    "reset_depth_decoder_transformer=True: checkpoint already contains "
                    "depth decoder output stage weights, skipping front-end reinit."
                )
            elif isinstance(self.model.depth_decoder, DecoderHead):
                self.model.depth_decoder.reinit_transformer_front_end()
                logger.info(
                    "reset_depth_decoder_transformer=True: reinitialized depth_decoder "
                    "proj_in / blocks / norm (output stage unchanged from checkpoint load)."
                )
            else:
                logger.warning(
                    "reset_depth_decoder_transformer=True ignored: depth_decoder is a "
                    f"{type(self.model.depth_decoder).__name__}, not a DecoderHead."
                )
        return missing_keys, unexpected_keys

    def _get_param_groups(self):
        if self.finetune_mode == "depth_output":
            return {
                "depth_decoder_output": list(self.model.depth_decoder.output_stage_parameters()),
            }
        if self.finetune_mode == "depth_decoder":
            return {"depth_decoder": list(self.model.depth_decoder.parameters())}
        if self.finetune_mode == "depth_decoder_recurrent":
            return {
                "depth_decoder": list(self.model.depth_decoder.parameters()),
                "recurrent_blocks": list(self.model.recurrent_blocks.parameters()),
            }
        if self.finetune_mode == "all_heads":
            pg = {
                "ray_decoder": list(self.model.ray_decoder.parameters()),
                "depth_decoder": list(self.model.depth_decoder.parameters()),
            }
            if self.model.camera_head is not None:
                pg["camera_head"] = list(self.model.camera_head.parameters())
            return pg
        pg = {
            "patch_embed": list(self.model.patch_embed_encoder.parameters()),
            "recurrent_blocks": list(self.model.recurrent_blocks.parameters()),
            "camera_token": [self.model.camera_token],
            "register_token": [self.model.register_token],
            "ray_decoder": list(self.model.ray_decoder.parameters()),
            "depth_decoder": list(self.model.depth_decoder.parameters()),
        }
        if self.model.camera_head is not None:
            pg["camera_head"] = list(self.model.camera_head.parameters())
        return pg

    @force_fp32
    def _postprocess_predictions(self, batch, predictions):
        H, W = batch[DataField.IMAGES].shape[-2:]
        rays = predictions["rays"]
        depth_conf = predictions["depth_conf"]
        pose_conf = depth_conf if self.use_depth_conf_for_pose else torch.ones_like(depth_conf)
        extrinsics_c2w, intrinsics = rays_to_pose(rays, pose_conf, H, W, self.patch_size)
        cameras = [extri_intri_to_cameras(e, i, (H, W)) for e, i in zip(extrinsics_c2w, intrinsics, strict=False)]
        preds = {
            PredictionField.CAMERAS: cameras,
            PredictionField.DEPTHS: predictions["depth"].squeeze(-1),
            PredictionField.DEPTHS_CONF: depth_conf,
        }
        # ``predictions["world_points"]`` is the training-style direct compose
        # ``ray_origin + ray_direction * depth`` (computed in the model decoder)
        # and is always exposed under ``WORLD_POINTS_DIRECT``. The default
        # ``WORLD_POINTS`` goes through the fitted pose; the
        # ``world_points_from_rays`` flag overrides ``WORLD_POINTS`` to use the
        # same rays+depth value (and skips the unprojection).
        rays_world_points = predictions["world_points"]
        if self.world_points_from_rays:
            preds[PredictionField.WORLD_POINTS] = rays_world_points
        else:
            unproj_world_points, _, _ = depth_to_world_coords_points(
                preds[PredictionField.DEPTHS], extrinsics_c2w, intrinsics
            )
            preds[PredictionField.WORLD_POINTS] = unproj_world_points
        preds[PredictionField.WORLD_POINTS_DIRECT] = rays_world_points
        preds[PredictionField.WORLD_POINTS_DIRECT_CONF] = predictions["world_points_conf"]
        return preds

    def predict(self, batch):
        return self._postprocess_predictions(batch, self.model(batch))
