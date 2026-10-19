"""Tests for the reference coded-target detector's input handling."""
import pytest

from augenblick.eval import detect


class TestPlanJobs:
    def test_keeps_given_name_and_resolves_path(self, tmp_path):
        original = tmp_path / "originals" / "IMG_0001.JPG"
        original.parent.mkdir()
        original.write_bytes(b"not really a photograph")
        scene = tmp_path / "scene"
        scene.mkdir()
        link = scene / "img_0001.jpg"
        try:
            link.symlink_to(original)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        (name, path), = detect.plan_jobs([link])
        # The reconstruction knows the image by the scene's name, so recording the
        # target's name would silently detach every detection from its camera.
        assert name == "img_0001.jpg"
        assert path == str(original.resolve())

    def test_relative_paths_become_absolute(self, tmp_path, monkeypatch):
        photo = tmp_path / "IMG_0002.jpg"
        photo.write_bytes(b"not really a photograph")
        monkeypatch.chdir(tmp_path)
        (name, path), = detect.plan_jobs([type(photo)("IMG_0002.jpg")])
        assert name == "IMG_0002.jpg"
        assert path == str(photo.resolve())


class TestDetectImages:
    def test_missing_toolkit_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PGT-Toolkit"):
            detect.detect_images([], tmp_path / "nowhere")
