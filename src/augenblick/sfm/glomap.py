"""Optional GLOMAP global mapping using the existing masked COLMAP front end."""
import os
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.colmap import ColmapSfM


@register_sfm
class GlomapSfM(ColmapSfM):
    """Runs masked SIFT extraction, exhaustive matching, and PyCOLMAP global mapping."""

    name: ClassVar[str] = "glomap"

    def validate(self, scene: Scene) -> None:
        super().validate(scene)
        import pycolmap

        if not callable(getattr(pycolmap, "global_mapping", None)) or not callable(
                getattr(pycolmap, "GlobalPipelineOptions", None)):
            raise SceneError("GLOMAP_UNAVAILABLE: install PyCOLMAP with global_mapping support")

    def _map(self, pycolmap, db_path: str, image_path: str, output_path: str):
        options = pycolmap.GlobalPipelineOptions()
        mapper = getattr(options, "mapper", None)
        adjustment = getattr(mapper, "bundle_adjustment", None)
        ceres = getattr(adjustment, "ceres", None)
        solver = getattr(ceres, "solver_options", None)
        num_threads = len(os.sched_getaffinity(0))
        for settings in (options, mapper, solver):
            if hasattr(settings, "num_threads"):
                settings.num_threads = num_threads
        return pycolmap.global_mapping(db_path, image_path, output_path, options=options)