"""Stage a private training scene with explicit images, masks and shared all-view SfM."""
import argparse
import json
from pathlib import Path

from augenblick.core.errors import SceneError
from augenblick.bench.digest import model_file_digests
from augenblick.eval.split import check_split, registered_stems


def stage(sfm: Path, images: Path, masks: Path, output: Path, split: Path | None = None):
    sfm, images, masks, output = map(Path, (sfm, images, masks, output))
    if output.is_symlink():
        raise SceneError("staged scene must be a private directory, not a symlink")
    for source in (sfm / "sparse" / "0", images, masks):
        if not source.is_dir():
            raise SceneError(f"missing scene input: {source}")
    if any(output.resolve() == source.resolve() or source.resolve() in output.resolve().parents
           for source in (sfm, images, masks)):
        raise SceneError("staged scene must not be inside the adopted SfM")
    if not any(all((sfm / "sparse" / "0" / (name + extension)).is_file()
                       for name in ("cameras", "images", "points3D"))
               for extension in (".bin", ".txt")):
        raise SceneError("shared SfM must contain cameras, images and points3D")
    payload = json.loads(Path(split).read_text()) if split else None
    if payload:
        names = registered_stems(sfm / "sparse" / "0")
        check_split(payload, names)
        if set(payload["train"]) | set(payload["test"]) != set(names):
            raise SceneError("split must partition all registered views")
    output.mkdir(parents=True, exist_ok=True)
    sources = {"images": images, "masks": masks}
    # Readers may create points3D.ply. Keep that cache private, and always derive it
    # from the selected COLMAP binary model instead of adopting an unverified PLY.
    sparse = output / "sparse"
    if sparse.is_symlink():
        sparse.unlink()
    model = sparse / "0"
    if model.is_symlink():
        raise SceneError("private model directory must not be an adopted symlink")
    model.mkdir(parents=True, exist_ok=True)
    signature = model_file_digests(sfm / "sparse" / "0")
    record = model / "_source_model.json"
    previous = json.loads(record.read_text()) if record.is_file() else None
    if previous != signature:
        for cached in model.glob("*.ply"):
            cached.unlink()
    source_files = {p.name for p in (sfm / "sparse" / "0").iterdir()
                    if p.is_file() and p.suffix in (".bin", ".txt")}
    for obsolete in model.iterdir():
        if obsolete.suffix in (".bin", ".txt") and obsolete.name not in source_files:
            if not obsolete.is_symlink():
                raise SceneError(f"refusing to remove a real private model file: {obsolete}")
            obsolete.unlink()
    for source in (sfm / "sparse" / "0").iterdir():
        if not source.is_file() or source.suffix not in (".bin", ".txt"):
            continue
        link = model / source.name
        if link.exists() and not link.is_symlink():
            raise SceneError(f"refusing to replace a real private model file: {link}")
        if link.is_symlink():
            link.unlink()
        link.symlink_to(source.resolve())
    record.write_text(json.dumps(signature, sort_keys=True) + "\n")
    # Optional preconverted all-view SfMData, consumed only by Meshroom.
    for optional in ("alicevision.sfm", "meshroom_template.mg"):
        if (sfm / optional).is_file():
            sources[optional] = sfm / optional
    for name, source in sources.items():
        link = output / name
        if link.exists() and not link.is_symlink():
            raise SceneError(f"refusing to replace a real staged input: {link}")
        if link.is_symlink():
            if link.resolve() == source.resolve():
                continue
            link.unlink()
        link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    target = output / "split.json"
    if target.is_symlink():
        raise SceneError("split.json must not be an adopted symlink")
    if payload:
        target.write_text(json.dumps(payload, indent=2) + "\n")
    elif target.exists():
        target.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sfm", "images", "masks", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    stage(args.sfm, args.images, args.masks, args.out, args.split)


if __name__ == "__main__":
    main()
