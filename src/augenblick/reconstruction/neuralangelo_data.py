"""Per-view intrinsics and a disjoint saved split for Neuralangelo."""
import json

import numpy as np
import torch

from projects.nerf.datasets.base import Dataset as BaseDataset
from projects.nerf.utils import camera
from projects.neuralangelo.data import Dataset as UpstreamDataset


class Dataset(UpstreamDataset):
    def __init__(self, cfg, is_inference=False):
        BaseDataset.__init__(self, cfg, is_inference=is_inference)
        settings = cfg.data
        self.root = settings.root
        self.preload = settings.preload
        self.H, self.W = settings[self.split].image_size
        with open(f"{self.root}/transforms.json") as handle:
            self.meta = json.load(handle)
        self.list = self.meta["val_frames"] if is_inference else self.meta["frames"]
        if is_inference and not self.list:
            self.list = self.meta["frames"][:1]
        subset = settings[self.split].subset
        if subset:
            indices = np.linspace(0, len(self.list), min(subset, len(self.list)) + 1)[:-1].astype(int)
            self.list = [self.list[index] for index in indices]
        self.num_rays = cfg.model.render.rand_rays
        if self.preload:
            self.images = self.preload_threading(self.get_image, settings.num_workers)
            self.cameras = self.preload_threading(self.get_camera, settings.num_workers, data_str="cameras")

    def get_camera(self, index):
        frame = self.list[index]
        intrinsic = torch.tensor(frame["intrinsic_matrix"], dtype=torch.float32)
        pose = self._gl_to_cv(torch.tensor(frame["transform_matrix"], dtype=torch.float32))
        pose[:3, -1] -= torch.tensor(self.meta["sphere_center"], dtype=torch.float32)
        pose[:3, -1] /= self.meta["sphere_radius"]
        return intrinsic, camera.Pose().invert(pose[:3])