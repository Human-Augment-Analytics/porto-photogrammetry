# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import os
import re
import tempfile
import urllib.parse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from dvlt.util.download import download_file_from_url


# LOCAL PATCH (augenblick): inference-only and accelerate-free. Stdlib logging replaces
# accelerate's get_logger; setup_train/train_step/test_step/save_pretrained are removed;
# setup_test takes a device and predict takes no accelerator.
logger = logging.getLogger(__name__)


class Module(ABC):

    def __init__(
        self,
        *args,
        freeze: Optional[List[str]] = None,
        log_params: Optional[List[str]] = None,
        log_every_n_steps: int = 50,
        gradient_checkpointing_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the base model.

        Args:
            freeze: List of module/parameter paths to freeze. None means no freezing.
            log_params: Parameter logging configuration:
                - None (default): No parameter logging
                - [] (empty list): Log all trainable parameters
                - ['module1', 'param.weight']: Log specific modules/parameters only
            log_every_n_steps: Call _log_train every N steps.
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__()
        self.model_file = "model.pt"
        self.paths_to_freeze = freeze
        self.paths_to_log = log_params
        self.log_every_n_steps = log_every_n_steps
        self.model = self.build_model()
        self.gradient_checkpointing_config = {}
        self.gradient_checkpointing_enabled = False

    def freeze(self) -> None:
        """
        Freeze specified modules or single parameters by setting requires_grad=False.

        Args:
            module_paths: List of module paths to freeze
        """
        if not self.paths_to_freeze:
            return

        for module_path in self.paths_to_freeze:
            try:
                # Navigate to the module using the path
                module = self.model
                for attr in module_path.split("."):
                    module = getattr(module, attr)

                # Freeze parameters if it's a module with parameters
                if hasattr(module, "parameters"):
                    for param in module.parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen module: {module_path}")
                # Freeze single parameter/tensor
                elif hasattr(module, "requires_grad"):
                    module.requires_grad = False
                    logger.info(f"Frozen parameter: {module_path}")
                else:
                    logger.warning(f"Cannot freeze {module_path}: not a module or parameter")

            except AttributeError as e:
                logger.warning(f"Cannot freeze {module_path}: {e}")

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def setup_test(self, device: Union[str, torch.device]) -> None:
        self.model.to(device=device)
        self.model.eval()

    @abstractmethod
    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """
        Run model inference and return standardized predictions.

        This method should perform model inference and return predictions using
        standardized keys from PredictionField. The outputs should be converted
        to generic formats (e.g., Camera objects) where applicable.

        Args:
            batch: Input batch containing images and other data

        Returns:
            Dictionary with standardized prediction keys and generic output formats
        """
        raise NotImplementedError

    def print_summary(self):
        """Print a summary of the model."""
        logger.info(self.model)

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        model_file: Optional[str] = None,
        strict: bool = False,
        remap: Optional[Dict[str, str]] = None,
        filter: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Load a model from a local file, local directory, URL, or HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: Either:
                - A direct path to a weights file (``.safetensors`` or ``.pt``)
                - A local directory containing ``model.safetensors`` / ``model.pt``
                - A URL to a model file
                - A model identifier from the HuggingFace Hub
            use_auth_token: Optional HuggingFace auth token
            revision: Optional git revision to use if loading from Hub
            model_file: Optional filename for the model weights (defaults to self.model_file)
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function.
            remap: Optional dictionary mapping regex patterns to replacement strings for state_dict key remapping
            filter: Optional list of regex patterns to filter out keys from state_dict
            **kwargs: Additional arguments passed to state_dict loading function
        """
        if model_file is not None:
            self.model_file = model_file

        # Parse the input to determine if it's a local file, directory, URL, or Hub model ID
        is_url = bool(urllib.parse.urlparse(pretrained_model_name_or_path).scheme)

        if os.path.isfile(pretrained_model_name_or_path):
            # Direct path to a weights file (e.g. .safetensors or .pt)
            model_path = pretrained_model_name_or_path
            logger.info(f"Loading model from local file: {model_path}")

        elif os.path.isdir(pretrained_model_name_or_path):
            # Local directory
            model_path = os.path.join(pretrained_model_name_or_path, self.model_file)
            # Check if the model was saved as .safetensors
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    pretrained_model_name_or_path, os.path.splitext(self.model_file)[0] + ".safetensors"
                )

            if not os.path.exists(model_path):
                raise ValueError(f"No model checkpoint found at {model_path}")
            logger.info(f"Loading model from local directory: {model_path}")

        elif is_url:
            try:
                # Create a temporary file to download to
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                    tmp_path = tmp_file.name

                # Download the file using the utility function
                logger.info(f"Downloading model from URL: {pretrained_model_name_or_path}")
                download_file_from_url(pretrained_model_name_or_path, tmp_path)
                model_path = tmp_path
            except Exception as e:
                raise ValueError(
                    f"Could not download model from URL {pretrained_model_name_or_path}. " f"Error: {str(e)}"
                ) from e

        else:
            try:
                # ``use_auth_token`` was renamed to ``token`` in huggingface_hub>=0.23
                # and removed entirely in more recent versions.
                model_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=self.model_file,
                    token=use_auth_token,
                    revision=revision,
                )
                logger.info(f"Downloaded model from Hugging Face Hub: {pretrained_model_name_or_path}")
            except Exception as e:
                raise ValueError(
                    f"Could not load model from Hugging Face Hub {pretrained_model_name_or_path}. " f"Error: {str(e)}"
                ) from e

        # Load the model state dict - using safetensors if applicable
        if model_path.endswith(".safetensors"):
            with safe_open(model_path, framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True, **kwargs)

        # Handle different formats of saved state_dict
        if isinstance(state_dict, Dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Handle Accelerate save_state format
        elif isinstance(state_dict, Dict) and "module" in state_dict:
            state_dict = state_dict["module"]

        # Remove module. prefix if it exists (happens when saved with DataParallel)
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # Apply key remapping if provided. Aggregate per-pattern counts so we
        # log a single summary line per pattern instead of one line per key.
        remap_counts: Dict[str, int] = {}
        if remap is not None:
            remapped_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for old_pattern, new_pattern in remap.items():
                    new_key_candidate = re.sub(old_pattern, new_pattern, key)
                    if new_key_candidate != key:
                        new_key = new_key_candidate
                        remap_counts[old_pattern] = remap_counts.get(old_pattern, 0) + 1
                        break
                remapped_state_dict[new_key] = value
            state_dict = remapped_state_dict

        # Filter out keys matching filter_patterns; aggregate per-pattern counts.
        filter_counts: Dict[str, int] = {}
        if filter is not None:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                should_filter = False
                for pattern in filter:
                    if re.search(pattern, key):
                        filter_counts[pattern] = filter_counts.get(pattern, 0) + 1
                        should_filter = True
                        break
                if not should_filter:
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict

        # Subclass hook: massage state_dict (e.g. rename/duplicate keys) before shape checks.
        state_dict = self._transform_state_dict(state_dict)

        # Drop any key whose tensor shape doesn't match the target module so load_state_dict doesn't raise.
        # These keys then flow through as missing, and downstream logic (e.g. reinit hooks) can handle them.
        model_sd = self.model.state_dict()
        shape_mismatched = [
            k for k, v in state_dict.items() if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)
        ]
        if shape_mismatched:
            state_dict = {k: v for k, v in state_dict.items() if k not in shape_mismatched}

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)

        # Concise load summary: one line per remap/filter pattern, plus a
        # bottom-line tally of mismatched / missing / unexpected with a small
        # sample (full lists at DEBUG level).
        for pat, n in remap_counts.items():
            logger.info(f"Remapped {n} key(s) via pattern '{pat}' -> '{remap[pat]}'.")
        for pat, n in filter_counts.items():
            logger.info(f"Filtered {n} key(s) matching pattern '{pat}'.")
        if shape_mismatched:
            logger.warning(
                f"Shape mismatch on {len(shape_mismatched)} key(s); dropped before load. "
                f"Sample: {shape_mismatched[:3]}"
            )
            logger.debug(f"Full shape-mismatched key list: {shape_mismatched}")
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} key(s) in checkpoint. Sample: {list(missing_keys)[:3]}")
            logger.debug(f"Full missing-keys list: {list(missing_keys)}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected {len(unexpected_keys)} key(s) in checkpoint. Sample: {list(unexpected_keys)[:3]}"
            )
            logger.debug(f"Full unexpected-keys list: {list(unexpected_keys)}")
        logger.info(
            "Model loaded: "
            f"{len(state_dict)} loaded, "
            f"{len(missing_keys)} missing, "
            f"{len(unexpected_keys)} unexpected, "
            f"{len(shape_mismatched)} shape-mismatched."
        )

        # Clean up temporary file if we created one
        if is_url and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return missing_keys, unexpected_keys

    def _transform_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transform state dict before loading. Override in subclasses for custom transformations."""
        return state_dict

    def _validate_param_groups(self, param_groups: dict[str, list[nn.Parameter]]) -> None:
        """Validate that parameter groups are correct.

        Args:
            param_groups: Dictionary mapping group names to parameter lists

        Raises:
            ValueError: If parameter groups are invalid
        """
        # Create mapping from parameter ID to name
        param_id_to_name = {id(param): name for name, param in self.model.named_parameters()}

        # Get all trainable parameter IDs
        trainable_param_ids = {id(param) for param in self.model.parameters() if param.requires_grad}

        # Get all grouped parameter IDs and check for duplicates
        grouped_param_ids = set()
        duplicates = []

        for _, params in param_groups.items():
            for param in params:
                param_id = id(param)
                if param_id in grouped_param_ids:
                    param_name = param_id_to_name.get(param_id, f"unknown_{param_id}")
                    duplicates.append(param_name)
                grouped_param_ids.add(param_id)

        # Check for issues and build error message
        errors = []
        warnings = []

        if duplicates:
            errors.append(f"Duplicated parameters: {', '.join(duplicates)}")

        missing_ids = trainable_param_ids - grouped_param_ids
        if missing_ids:
            missing_names = [param_id_to_name.get(pid, f"unknown_{pid}") for pid in missing_ids]
            errors.append(f"Missing trainable parameters: {', '.join(missing_names)}")

        extra_ids = grouped_param_ids - trainable_param_ids
        if extra_ids:
            extra_names = [param_id_to_name.get(pid, f"unknown_{pid}") for pid in extra_ids]
            warnings.append(f"Non-trainable parameters in groups: {', '.join(extra_names)}")

        if errors:
            raise ValueError("Parameter group validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        if warnings:
            logger.warning(
                "Parameter group validation warnings:\n" + "\n".join(f"  - {warning}" for warning in warnings)
            )

    def _get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Get parameter groups for the model. Override in subclasses.

        Returns:
            Dictionary mapping group names to parameter lists
        """
        return {"params": list(self.get_trainable_params())}

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Get validated parameter groups for the model.

        Returns:
            Dictionary mapping group names to parameter lists
        """
        param_groups = self._get_param_groups()
        self._validate_param_groups(param_groups)
        return param_groups

    def get_params(self) -> list[nn.Parameter]:
        return self.model.parameters()

    def get_trainable_params(self) -> list[nn.Parameter]:
        # Cache trainable params to avoid recreating list every step
        if not hasattr(self, "_trainable_params_cache"):
            self._trainable_params_cache = [p for p in self.model.parameters() if p.requires_grad]
        return self._trainable_params_cache


class Model(nn.Module):

    def __init__(self, *args, gradient_checkpointing_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self.gradient_checkpointing_config = gradient_checkpointing_config or {}
        self.gradient_checkpointing_enabled = False

    def enable_gradient_checkpointing(self, use_reentrant: bool = False) -> None:
        """Enable gradient checkpointing to reduce memory usage at the cost of some computation."""
        if self.gradient_checkpointing_enabled:
            return
        self.gradient_checkpointing_enabled = True

        # Use configuration values, but allow override via parameter
        config = self.gradient_checkpointing_config
        use_reentrant = config.get("use_reentrant", use_reentrant)
        modules_to_checkpoint = config.get("modules", [])

        def create_checkpoint_wrapper(orig_forward_fn):
            # Create a wrapper that uses checkpoint
            def forward_with_checkpoint(*args, **kwargs):
                if self.training:
                    return checkpoint(orig_forward_fn, *args, use_reentrant=use_reentrant, **kwargs)
                else:
                    return orig_forward_fn(*args, **kwargs)

            return forward_with_checkpoint

        # Apply gradient checkpointing based on configuration
        for module_path in modules_to_checkpoint:
            self._apply_gradient_checkpointing_to_module(module_path, create_checkpoint_wrapper, use_reentrant)

    @staticmethod
    def _parse_module_path(module_path: str) -> Tuple[str, Optional[Set[int]]]:
        """Parse a module path with optional index slice, e.g. ``backbone.blocks[4:]``.

        Supported slice forms: ``[i]``, ``[start:]``, ``[:stop]``, ``[start:stop]``,
        ``[start:stop:step]``.  Without a slice suffix every element is selected.

        Returns ``(base_path, indices)`` where *indices* is ``None`` (meaning all)
        or a set of concrete integer indices to checkpoint.
        """
        m = re.fullmatch(r"(.+)\[([^\]]*)\]", module_path)
        if m is None:
            return module_path, None

        base_path = m.group(1)
        slice_str = m.group(2)

        # Single index: [4]
        if ":" not in slice_str:
            return base_path, {int(slice_str)}

        # Slice: [start:stop] or [start:stop:step]
        parts = slice_str.split(":")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return base_path, (start, stop, step)  # resolved later when length is known

    def _apply_gradient_checkpointing_to_module(
        self, module_path: str, checkpoint_wrapper, use_reentrant: bool
    ) -> None:
        """Apply gradient checkpointing to a specific module path.

        Supports optional index slicing for iterable modules, e.g.:
          - ``backbone.blocks``        — checkpoint all blocks
          - ``backbone.blocks[4:]``    — checkpoint from index 4 onwards
          - ``backbone.blocks[::2]``   — checkpoint every other block
        """
        base_path, index_spec = self._parse_module_path(module_path)
        parts = base_path.split(".")
        current_module = self

        for part in parts:
            if not hasattr(current_module, part):
                logger.warning(f"Module path {module_path} not found, skipping gradient checkpointing")
                return
            current_module = getattr(current_module, part)

        if hasattr(current_module, "__iter__"):
            blocks = list(current_module)
            n = len(blocks)

            # Resolve index_spec to a set of indices
            if index_spec is None:
                selected: Set[int] = set(range(n))
            elif isinstance(index_spec, set):
                selected = index_spec
            else:
                start, stop, step = index_spec
                selected = set(range(*slice(start, stop, step).indices(n)))

            for i, block in enumerate(blocks):
                if i not in selected:
                    continue
                if hasattr(block, "forward"):
                    block.forward = checkpoint_wrapper(block.forward)
                    logger.info(f"Enabled gradient checkpointing for {base_path}.{i} via forward")

            skipped = set(range(n)) - selected
            if skipped:
                logger.info(f"Skipped gradient checkpointing for {base_path} indices: {sorted(skipped)}")

        elif hasattr(current_module, "enable_gradient_checkpointing"):
            current_module.enable_gradient_checkpointing(use_reentrant=use_reentrant)
            logger.info(f"Enabled gradient checkpointing for {base_path} via enable_gradient_checkpointing")
        elif hasattr(current_module, "forward"):
            current_module.forward = checkpoint_wrapper(current_module.forward)
            logger.info(f"Enabled gradient checkpointing for {base_path} via forward")
        else:
            logger.warning(
                f"Module {current_module} has no forward or enable_gradient_checkpointing method, skipping gradient checkpointing"
            )

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    try:
        # If the onload device is not available due to no group offloading hooks, try to get the device
        # from the first parameter or buffer
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None

    for name, param in parameter.named_parameters():
        last_dtype = param.dtype
        if (
            hasattr(parameter, "_keep_in_fp32_modules")
            and parameter._keep_in_fp32_modules
            and any(m in name for m in parameter._keep_in_fp32_modules)
        ):
            continue

        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype
