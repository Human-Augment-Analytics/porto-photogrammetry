"""GLOMAP global SfM through the maintained PyCOLMAP implementation."""
import os
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_sfm
from augenblick.sfm.colmap import ColmapSfM


@register_sfm
class GlomapSfM(ColmapSfM):
    """Runs mask-restricted SIFT matching and GLOMAP global mapping through pycolmap."""

    name: ClassVar[str] = "glomap"

    def validate(self, scene) -> None:
        import pycolmap

        super().validate(scene)
        if not callable(getattr(pycolmap, "global_mapping", None)):
            raise SceneError("GLOMAP_UNAVAILABLE: install PyCOLMAP with global_mapping support")

    def _map(self, pycolmap, db_path: str, image_path: str, output_path: str):
        options = pycolmap.GlobalPipelineOptions()
        mapper = getattr(options, "mapper", None)
        bundle_adjustment = getattr(mapper, "bundle_adjustment", None)
        ceres = getattr(bundle_adjustment, "ceres", None)
        solver = getattr(ceres, "solver_options", None)
        num_threads = len(os.sched_getaffinity(0))
        for thread_options in (options, mapper, solver):
            if hasattr(thread_options, "num_threads"):
                thread_options.num_threads = num_threads
        return pycolmap.global_mapping(db_path, image_path, output_path, options=options)