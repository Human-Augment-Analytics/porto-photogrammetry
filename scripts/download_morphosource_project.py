#!/usr/bin/env python3
"""
Download openly-downloadable media for a MorphoSource project.

Defaults to project 000381689 ("UF Photogrammetry scans at the Florida Museum of
Natural History"), the source of the UF scenes that
pipeline/preparation/prepare_uf_dataset.py consumes. Media come in three kinds:
images (a Photogrammetry Image Series), highpoly meshes and lowpoly meshes.

Downloading needs MORPHOSOURCE_API_KEY and implies consent to the MorphoSource
user agreements: https://www.morphosource.org/terms

Output:
  output_dir/manifest.json
  output_dir/<specimen>__<id>/metadata.json
  output_dir/<specimen>__<id>/<specimen>_images.zip   without --extract
  output_dir/<specimen>__<id>/<specimen>_images/     with --extract, zip removed

API quirks behind this script are documented in .claude/MEMORY/data-morphosource.md.
"""
import os
import json
import random
import shutil
import zipfile
import argparse
import logging
import time
import posixpath
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import requests

from morphosource import DownloadConfig, DownloadVisibility
from morphosource.exceptions import RestrictedDownloadError, ItemNotFound, MetadataMissingError
# search_media() takes no project filter, so reach for the internals that do.
from morphosource.fetch import fetch_items
from morphosource.config import API_URL, Endpoints
from morphosource.search import Media

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "000381689"
KINDS = ["images", "lowpoly", "highpoly"]
DEFAULT_KINDS = ["images", "highpoly"]
METADATA_WORKERS = 2       # concurrency for /file-metadata lookups
METADATA_RETRIES = 5       # MorphoSource rate-limits bursts of them
METADATA_BACKOFF = 1.0     # seconds, doubled per retry
DEFAULT_USE_STATEMENT = (
    "Downloading this data for academic research on photogrammetry and 3D surface "
    "reconstruction (Augenblick pipeline)."
)


def first(media_data: dict, name: str, default=None):
    """Unwrap a MorphoSource field, since the API wraps every value in a list."""
    values = media_data.get(name) or []
    return values[0] if values else default


def sanitize(name: str) -> str:
    """Reduce a title to a name safe as one path component."""
    cleaned = "".join("_" if c in ':/\\ \t' else c for c in (name or "").strip())
    cleaned = "".join(c for c in cleaned if c.isalnum() or c in "_-.")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("._") or "unknown"


def read_specimen_file(path: Path) -> List[str]:
    """
    Read specimen ids from a text file, one per line.

    Blank lines and '#' comments are skipped, as is anything after an id on its
    line, so a list can carry notes alongside the ids. Order is preserved and
    duplicates dropped, since the ids drive the download order.

    Args:
        path: Text file of specimen ids.

    Returns:
        The ids, de-duplicated, in file order.
    """
    ids: List[str] = []
    seen = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        specimen_id = line.split()[0]
        if specimen_id not in seen:
            seen.add(specimen_id)
            ids.append(specimen_id)
    return ids


def human_size(num_bytes: int) -> str:
    """Format a byte count for logging."""
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


# --- MorphoSource queries -----------------------------------------------------------------

def get_project_title(project_id: str) -> Optional[str]:
    """Fetch a project's title, which the package offers no helper for."""
    try:
        response = requests.get(f"{API_URL}/projects/{project_id}", timeout=30)
        response.raise_for_status()
        return first(response.json()["response"]["collection"], "title")
    except (requests.RequestException, KeyError, ValueError) as err:
        logger.warning(f"Could not read project {project_id} metadata: {err}")
        return None


def search_project_media(project_id: str) -> List[Media]:
    """Fetch every openly-downloadable medium in a project, across all result pages."""
    params = {
        "f.project": project_id,
        "f.publication_status": DownloadVisibility.OPEN,
    }
    raw_items, _facets, pages = fetch_items(
        url=Endpoints.MEDIA, query=None, params=params,
        per_page=100, page=None, items_name="media",
    )
    logger.info(f"MorphoSource reports {pages.get('total_count')} open media in project {project_id}")

    # Re-assert visibility here: the media field says 'open', the facet 'Open Download'.
    return [Media(item) for item in raw_items if first(item, "visibility") == "open"]


_file_name_cache: Dict[str, Optional[str]] = {}


def get_remote_file_name(media: Media) -> Optional[str]:
    """
    Look up a medium's real file name, caching and retrying the request.

    Search results omit file_name for nearly every medium in this project, and
    MorphoSource answers bursts of these lookups with an HTTP error.

    Args:
        media: Medium to name.

    Returns:
        The file name, or None when the API will not supply one.
    """
    if media.id in _file_name_cache:
        return _file_name_cache[media.id]

    file_name = None
    for attempt in range(METADATA_RETRIES):
        try:
            file_name = media.get_file_metadata().file_name
            break
        except (MetadataMissingError, ItemNotFound) as err:
            logger.debug(f"No file metadata for media {media.id}: {err}")
            break
        except requests.RequestException as err:
            if attempt == METADATA_RETRIES - 1:
                logger.warning(f"Giving up on file metadata for media {media.id}: {err}")
                break
            time.sleep(METADATA_BACKOFF * (2 ** attempt))

    _file_name_cache[media.id] = file_name
    return file_name


def prefetch_file_names(media_list: List[Media]) -> None:
    """Warm the file-name cache with a few workers, staying under the rate limit."""
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as pool:
        list(pool.map(get_remote_file_name, media_list))


# --- Kind classification ------------------------------------------------------------------

def kind_from_text(text: Optional[str]) -> Optional[str]:
    """Read a mesh kind out of a description or file name."""
    if not text:
        return None
    text = text.lower()
    if "high polygon" in text or "highpoly" in text:
        return "highpoly"
    if "low polygon" in text or "lowpoly" in text:
        return "lowpoly"
    return None


def classify_media(media_list: List[Media]) -> Dict[str, str]:
    """
    Sort media into kinds using only the search response.

    Stays network-free because this decides which specimens are eligible, and so
    which ones a given --seed samples; refine_kinds() resolves the remainder later.

    Args:
        media_list: Open media for the project.

    Returns:
        Mapping of media id to 'images', 'lowpoly', 'highpoly' or 'unknown'.
    """
    kinds: Dict[str, str] = {}
    for media in media_list:
        if media.media_type == "Photogrammetry Image Series":
            kinds[media.id] = "images"
        else:
            kinds[media.id] = kind_from_text(first(media.data, "short_description")) or "unknown"
    unknown = sum(1 for kind in kinds.values() if kind == "unknown")
    if unknown:
        logger.info(f"{unknown} mesh media carry no polygon-count description")
    return kinds


def refine_kinds(media_list: List[Media], kinds: Dict[str, str]) -> None:
    """
    Resolve 'unknown' kinds in place, for an already-selected set of specimens.

    Falls back from the real file name to size-ranking two sibling meshes, of which
    the larger is the high poly one.

    Args:
        media_list: Media belonging to the selected specimens.
        kinds: Mapping from classify_media(), updated in place.
    """
    unresolved = [m for m in media_list if kinds[m.id] == "unknown"]
    if not unresolved:
        return
    logger.info(f"Resolving {len(unresolved)} unlabelled mesh media from their file names")
    prefetch_file_names(unresolved)
    for media in unresolved:
        kinds[media.id] = kind_from_text(get_remote_file_name(media)) or "unknown"

    siblings: Dict[str, List[Media]] = {}
    for media in unresolved:
        if kinds[media.id] == "unknown":
            siblings.setdefault(first(media.data, "media_parent_id") or "", []).append(media)
    for parent_id, group in siblings.items():
        if not parent_id or len(group) != 2:
            continue
        group.sort(key=lambda m: first(m.data, "file_size_all", 0) or 0)
        kinds[group[0].id], kinds[group[1].id] = "lowpoly", "highpoly"


# --- Selection ----------------------------------------------------------------------------

def group_by_specimen(media_list: List[Media], kinds: Dict[str, str]) -> Dict[str, Dict[str, List[Media]]]:
    """Index media by specimen id and then by kind."""
    specimens: Dict[str, Dict[str, List[Media]]] = {}
    for media in media_list:
        specimen_id = media.physical_object_id
        if not specimen_id:
            continue
        specimens.setdefault(specimen_id, {}).setdefault(kinds[media.id], []).append(media)
    return specimens


def select_specimens(specimens: Dict[str, Dict[str, List[Media]]], wanted_kinds: List[str],
                     explicit: Optional[List[str]], take_all: bool,
                     count: int, seed: int) -> List[str]:
    """
    Choose the specimens to download, from those carrying every requested kind.

    Args:
        specimens: Output of group_by_specimen().
        wanted_kinds: Kinds a specimen must all have to be eligible.
        explicit: Specimen ids to use instead of sampling, or None.
        take_all: Take every eligible specimen rather than sampling.
        count: Sample size.
        seed: Seed for that sample.

    Returns:
        Selected specimen ids.
    """
    eligible = sorted(
        specimen_id for specimen_id, by_kind in specimens.items()
        if all(by_kind.get(kind) for kind in wanted_kinds)
    )
    logger.info(f"{len(eligible)} of {len(specimens)} specimens have all of: {', '.join(wanted_kinds)}")

    if explicit:
        missing = [s for s in explicit if s not in specimens]
        if missing:
            logger.warning(f"Requested specimens not found in this project: {', '.join(missing)}")
        incomplete = [s for s in explicit if s in specimens and s not in eligible]
        if incomplete:
            logger.warning(f"Requested specimens missing some kinds (partial download): "
                           f"{', '.join(incomplete)}")
        return [s for s in explicit if s in specimens]

    if take_all:
        return eligible
    if count >= len(eligible):
        return eligible

    # Sample the sorted list so a seed picks the same specimens whatever order the API returned.
    return sorted(random.Random(seed).sample(eligible, count))


# --- Download -----------------------------------------------------------------------------

def target_file_name(media: Media, kind: str, used: set) -> str:
    """Name a medium's file on disk, keeping it unique within its specimen."""
    name = get_remote_file_name(media) or f"{media.id}_{kind}.zip"
    name = sanitize(name)
    if name in used:
        name = f"{media.id}_{name}"
    used.add(name)
    return name


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Unzip into dest, refusing members that escape it."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            normalized = posixpath.normpath(member)
            if normalized.startswith(("/", "../")) or normalized == ".." or ":" in member[:2]:
                raise ValueError(f"Refusing to extract unsafe path '{member}' from {zip_path.name}")
        dest.mkdir(parents=True, exist_ok=True)
        archive.extractall(dest)


def extract_bundle(zip_path: Path) -> None:
    """
    Unzip beside the archive, then delete every zip involved.

    A download wraps the real payload in an outer zip alongside the usage agreement
    and the manifests, so both zips are extracted and then dropped rather than left
    as a second copy of the data.
    """
    dest = zip_path.with_suffix("")
    safe_extract(zip_path, dest)

    # Materialise the list before extracting, so inner payloads are not rescanned.
    for inner in sorted(dest.rglob("*.zip")):
        safe_extract(inner, dest)
        inner.unlink()

    # Drop the now-empty "Media <id> - <title>/" wrapper directories.
    for directory in sorted(dest.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    zip_path.unlink()
    logger.info(f"  extracted -> {dest} (removed {zip_path.name})")


def download_media(media: Media, kind: str, path: Path, download_config: DownloadConfig,
                   overwrite: bool, extract: bool) -> bool:
    """
    Fetch one medium's bundle, skipping any complete copy already on disk.

    Args:
        media: Medium to download.
        kind: Its kind, for logging.
        path: Destination zip.
        download_config: Credentials and use statement.
        overwrite: Re-download even when a complete copy is present.
        extract: Unwrap the bundle afterwards and delete the zip.

    Returns:
        True when the file ends up in place.
    """
    expected_size = first(media.data, "file_size_all", 0) or 0
    extracted = path.with_suffix("")

    if not overwrite:

        # With --extract no zip survives, so the unpacked directory is the receipt.
        if extract and extracted.is_dir():
            logger.info(f"  {extracted.name} already extracted, skipping")
            return True

        # Read the central directory: re-packed bundles never match the advertised size.
        if path.exists() and zipfile.is_zipfile(path):
            logger.info(f"  {path.name} already present ({human_size(path.stat().st_size)}), skipping")
            if extract:
                extract_bundle(path)
            return True
        if path.exists():
            logger.info(f"  {path.name} exists but is not a readable zip, re-downloading")

    partial = path.with_name(path.name + ".part")
    logger.info(f"  downloading {media.id} [{kind}] -> {path.name} ({human_size(expected_size)})")
    try:
        media.download_bundle(str(partial), download_config)
    except RestrictedDownloadError as err:
        logger.error(f"  restricted: {err}")
        partial.unlink(missing_ok=True)
        return False
    except (requests.RequestException, OSError) as err:
        logger.error(f"  failed to download media {media.id}: {err}")
        partial.unlink(missing_ok=True)
        return False

    if not zipfile.is_zipfile(partial):
        logger.error(f"  media {media.id} did not download a valid zip (check your API key)")
        partial.unlink(missing_ok=True)
        return False

    os.replace(partial, path)
    logger.info(f"  saved {path.name} ({human_size(path.stat().st_size)})")
    if extract:
        extract_bundle(path)
    return True


# --- Main ---------------------------------------------------------------------------------

def build_row(media: Media, kind: str, file_name: Optional[str]) -> dict:
    """Describe one medium for the manifest."""
    return {
        "media_id": media.id,
        "kind": kind,
        "title": media.title,
        "media_type": media.media_type,
        "specimen_id": media.physical_object_id,
        "specimen_title": first(media.data, "physical_object_title"),
        "taxonomy": first(media.data, "physical_object_taxonomy_name"),
        "part": first(media.data, "part"),
        "file_name": file_name,
        "file_size_all": first(media.data, "file_size_all", 0) or 0,
        "website_url": media.get_website_url(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download open media (image series and meshes) for a MorphoSource project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See what the default sample would pull, without downloading (no API key needed)
  python scripts/download_morphosource_project.py --dry-run

  # Download the default 3-specimen sample (image series + high poly mesh) and unpack it
  export MORPHOSOURCE_API_KEY=<key>
  python scripts/download_morphosource_project.py --extract

  # A different reproducible sample, meshes only
  python scripts/download_morphosource_project.py --seed 7 --num-specimens 10 \\
      --kinds lowpoly highpoly

  # Named specimens, everything they have
  python scripts/download_morphosource_project.py --specimen 000833233 000855245 \\
      --kinds images lowpoly highpoly

  # A list of specimens from a file, one id per line
  python scripts/download_morphosource_project.py \\
      --specimen-file /blue/arthur.porto/srizvi63.gatech/neurips_dataset_specimens.txt

  # The whole project (~869 GB for all three kinds)
  python scripts/download_morphosource_project.py --all-specimens --kinds images lowpoly highpoly
        """
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID,
                        help=f"MorphoSource project id (default: {DEFAULT_PROJECT_ID})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/morphosource/<project_id>)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=DEFAULT_KINDS,
                        help=f"Media kinds to download (default: {' '.join(DEFAULT_KINDS)})")
    parser.add_argument("--num-specimens", type=int, default=3,
                        help="How many specimens to sample (default: 3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the specimen sample (default: 0)")
    parser.add_argument("--all-specimens", action="store_true",
                        help="Take every eligible specimen instead of sampling")
    parser.add_argument("--specimen", nargs="+", default=None, metavar="ID",
                        help="Explicit physical object ids; bypasses sampling")
    parser.add_argument("--specimen-file", type=Path, default=None, metavar="PATH",
                        help="Text file of physical object ids, one per line ('#' comments "
                             "allowed); combines with --specimen and bypasses sampling")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the manifest and print sizes, download nothing")
    parser.add_argument("--extract", action="store_true",
                        help="Unzip each bundle beside itself and delete the zip")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even when a complete file is already present")
    parser.add_argument("--use-statement", default=DEFAULT_USE_STATEMENT,
                        help="Use statement sent with each download request")
    parser.add_argument("--use-categories", nargs="+", default=["Research"],
                        help="MorphoSource use categories (default: Research)")

    args = parser.parse_args()

    explicit = list(args.specimen or [])
    if args.specimen_file:
        if not args.specimen_file.is_file():
            logger.error(f"Specimen file not found: {args.specimen_file}")
            return 1
        from_file = read_specimen_file(args.specimen_file)
        if not from_file:
            logger.error(f"No specimen ids in {args.specimen_file}")
            return 1
        logger.info(f"Read {len(from_file)} specimen id(s) from {args.specimen_file}")
        named = set(explicit)
        explicit += [s for s in from_file if s not in named]
    explicit = explicit or None

    output_dir = args.output_dir or Path("data/morphosource") / args.project_id

    api_key = os.environ.get("MORPHOSOURCE_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("MORPHOSOURCE_API_KEY is not set. Export your MorphoSource API key "
                     "(Dashboard > Profile > View API Key), or re-run with --dry-run.")
        return 1

    project_title = get_project_title(args.project_id)
    logger.info(f"Project:  {args.project_id} {project_title or ''}".rstrip())
    logger.info(f"Output:   {output_dir}")
    logger.info(f"Kinds:    {', '.join(args.kinds)}")

    media_list = search_project_media(args.project_id)
    if not media_list:
        logger.error(f"No open media found for project {args.project_id}")
        return 1

    kinds = classify_media(media_list)
    specimens = group_by_specimen(media_list, kinds)
    selected = select_specimens(specimens, args.kinds, explicit,
                                args.all_specimens, args.num_specimens, args.seed)
    if not selected:
        logger.error("No specimens matched the requested kinds")
        return 1
    logger.info(f"Selected {len(selected)} specimen(s): {', '.join(selected)}")

    # Only now, with the sample fixed, spend requests resolving the selected specimens' meshes.
    selected_media = [m for m in media_list if m.physical_object_id in set(selected)]
    refine_kinds(selected_media, kinds)
    specimens = group_by_specimen(media_list, kinds)

    # Plan every file before touching the network again, so --dry-run reports real names.
    plan = []
    for specimen_id in selected:
        by_kind = specimens[specimen_id]
        any_media = next(iter(m for group in by_kind.values() for m in group))
        specimen_title = first(any_media.data, "physical_object_title") or specimen_id
        specimen_dir = output_dir / f"{sanitize(specimen_title)}__{specimen_id}"
        used_names = set()
        rows = []
        for kind in args.kinds:
            for media in by_kind.get(kind, []):
                file_name = target_file_name(media, kind, used_names)
                rows.append((media, kind, specimen_dir / file_name))
        if not rows:
            logger.warning(f"  {specimen_title} ({specimen_id}) has none of the requested kinds")
            continue
        plan.append((specimen_id, specimen_title, specimen_dir, rows))

    if not plan:
        logger.error("Nothing to download")
        return 1

    per_kind = {kind: [0, 0] for kind in args.kinds}
    for _sid, _title, _dir, rows in plan:
        for media, kind, _path in rows:
            per_kind[kind][0] += 1
            per_kind[kind][1] += first(media.data, "file_size_all", 0) or 0
    total_bytes = sum(size for _count, size in per_kind.values())

    logger.info("Download plan:")
    for kind, (count, size) in per_kind.items():
        logger.info(f"  {kind:<9} {count:>4} file(s)  {human_size(size):>10}")
    logger.info(f"  {'total':<9} {sum(c for c, _ in per_kind.values()):>4} file(s)  {human_size(total_bytes):>10}")

    free = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else Path(".")).free
    if total_bytes > free:
        logger.warning(f"Plan needs {human_size(total_bytes)} but only {human_size(free)} is free")

    manifest = {
        "project_id": args.project_id,
        "project_title": project_title,
        "kinds": args.kinds,
        "seed": args.seed,
        "num_specimens": len(selected),
        "selection": "explicit" if explicit else ("all" if args.all_specimens else "sampled"),
        "total_bytes": total_bytes,
        "specimens": [],
    }
    for specimen_id, specimen_title, specimen_dir, rows in plan:
        manifest["specimens"].append({
            "specimen_id": specimen_id,
            "specimen_title": specimen_title,
            "directory": str(specimen_dir.relative_to(output_dir)),
            "media": [build_row(media, kind, path.name) for media, kind, path in rows],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote {manifest_path}")

    if args.dry_run:
        logger.info("Dry run - nothing downloaded")
        return 0

    download_config = DownloadConfig(
        api_key=api_key,
        use_statement=args.use_statement,
        use_categories=args.use_categories,
    )

    failures = 0
    for index, ((specimen_id, specimen_title, specimen_dir, rows), entry) in enumerate(
            zip(plan, manifest["specimens"]), start=1):
        logger.info(f"[{index}/{len(plan)}] {specimen_title} ({specimen_id})")
        specimen_dir.mkdir(parents=True, exist_ok=True)
        (specimen_dir / "metadata.json").write_text(json.dumps(entry, indent=2))
        for media, kind, path in rows:
            if not download_media(media, kind, path, download_config, args.overwrite, args.extract):
                failures += 1

    if failures:
        logger.error(f"{failures} file(s) failed to download")
        return 1
    logger.info(f"Done. {sum(len(rows) for _s, _t, _d, rows in plan)} file(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
