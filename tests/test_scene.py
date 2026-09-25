"""Scene path contract, validation guards, and the COLMAP mask-symlink trick."""
import pytest

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene


def test_path_properties(tmp_path):
    scene = Scene(tmp_path)
    assert scene.images_dir == tmp_path / "images"
    assert scene.masks_dir == tmp_path / "masks"
    assert scene.sparse_dir == tmp_path / "sparse" / "0"


def test_require_images_missing(tmp_path):
    with pytest.raises(SceneError):
        Scene(tmp_path).require_images()


def test_require_images_empty(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(SceneError):
        Scene(tmp_path).require_images()


def test_require_images_ok(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").touch()
    Scene(tmp_path).require_images()


def test_require_reconstruction_missing(tmp_path):
    with pytest.raises(SceneError):
        Scene(tmp_path).require_reconstruction()


def test_require_reconstruction_empty_dir(tmp_path):
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    scene = Scene(tmp_path)
    assert not scene.has_reconstruction()
    with pytest.raises(SceneError):
        scene.require_reconstruction()


def test_has_reconstruction_non_empty(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").touch()
    scene = Scene(tmp_path)
    assert scene.has_reconstruction()
    scene.require_reconstruction()


def test_link_colmap_masks_none_without_masks(tmp_path):
    assert Scene(tmp_path).link_colmap_masks(tmp_path / "masks_colmap") is None


def _scene_with(tmp_path, image_names, mask_names):
    """Build a scene directory holding the given images and masks."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for name in image_names:
        (tmp_path / "images" / name).touch()
    for name in mask_names:
        (tmp_path / "masks" / name).touch()
    return Scene(tmp_path)


def test_link_colmap_masks_uses_the_image_filename_not_a_fixed_extension(tmp_path):
    """COLMAP looks for <image filename>.png, so the link must carry the real extension.

    Masks are named after the image *stem*, and the extension the photographs actually use
    varies: the NeurIPS captures are `.JPG`. Assuming `.jpg` produced links COLMAP never
    found, so it reconstructed unmasked while logging only a warning per view.
    """
    scene = _scene_with(tmp_path, ["foo.JPG", "bar.jpeg"], ["foo.png", "bar.png"])
    dest = tmp_path / "out" / "masks_colmap"

    assert scene.link_colmap_masks(dest) == dest
    assert (dest / "foo.JPG.png").is_symlink()
    assert (dest / "bar.jpeg.png").is_symlink()


def test_link_colmap_masks_is_idempotent(tmp_path):
    scene = _scene_with(tmp_path, ["foo.JPG"], ["foo.png"])
    dest = tmp_path / "out" / "masks_colmap"

    scene.link_colmap_masks(dest)
    scene.link_colmap_masks(dest)
    assert sorted(p.name for p in dest.iterdir()) == ["foo.JPG.png"]


def test_link_colmap_masks_skips_masks_with_no_image(tmp_path):
    """A mask matching no photograph is dropped rather than linked under a guessed name."""
    scene = _scene_with(tmp_path, ["foo.JPG"], ["foo.png", "orphan.png"])
    dest = tmp_path / "out" / "masks_colmap"

    scene.link_colmap_masks(dest)
    assert sorted(p.name for p in dest.iterdir()) == ["foo.JPG.png"]


def test_link_colmap_masks_requires_images(tmp_path):
    """Masks without images is a broken scene, and must fail rather than link nothing.

    Returning an empty directory would be worse than raising: COLMAP reads a missing mask as
    "no mask" and silently reconstructs unmasked.
    """
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "foo.png").touch()
    with pytest.raises(SceneError):
        Scene(tmp_path).link_colmap_masks(tmp_path / "out" / "masks_colmap")
