"""CLI: Download a HuggingFace receipt dataset and copy images into data/receipts/."""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO = "UniqueData/ocr-receipts-text-detection"
DEFAULT_OUT = Path("data/receipts")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _extract_tarballs(cache_dir: Path) -> None:
    """Extract any .tar.gz archives in the dataset cache in-place."""
    for archive in cache_dir.rglob("*.tar.gz"):
        marker = archive.with_suffix(".extracted")
        if marker.exists():
            continue
        logger.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(archive.parent)
        marker.touch()


EXT_PRIORITY = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def _collect_images(cache_dir: Path) -> list[Path]:
    """Find all image files under cache_dir, deduped by stem.

    Some datasets ship the same receipt as both .jpg and .png. Keep one copy
    per stem, preferring extensions earlier in EXT_PRIORITY.
    """
    all_images = [
        p for p in cache_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    by_stem: dict[str, Path] = {}
    for p in all_images:
        existing = by_stem.get(p.stem)
        if existing is None:
            by_stem[p.stem] = p
            continue
        if EXT_PRIORITY.index(p.suffix.lower()) < EXT_PRIORITY.index(existing.suffix.lower()):
            by_stem[p.stem] = p

    def natural_key(p: Path) -> tuple[int, str]:
        return (int(p.stem), "") if p.stem.isdigit() else (10**9, p.stem)

    return sorted(by_stem.values(), key=natural_key)


def import_hf_dataset(
    repo_id: str,
    out_dir: Path,
    limit: int | None = None,
    prefix: str = "hf",
) -> int:
    """Download a HF dataset repo and copy its images into out_dir.

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "UniqueData/ocr-receipts-text-detection").
        out_dir: Destination directory for images.
        limit: Optional max number of images to copy.
        prefix: Filename prefix to avoid collisions across datasets.

    Returns:
        Number of images copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s ...", repo_id)
    cache_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    logger.info("Downloaded to %s", cache_dir)

    _extract_tarballs(cache_dir)

    src_images = _collect_images(cache_dir)
    logger.info("Found %d images in dataset", len(src_images))

    if limit:
        src_images = src_images[:limit]

    copied = 0
    for src in src_images:
        dst = out_dir / f"{prefix}_{src.name}"
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1

    logger.info("Copied %d new images to %s", copied, out_dir)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Import receipt images from a HF dataset")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace dataset repo ID")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max images to copy")
    parser.add_argument("--prefix", default="hf", help="Filename prefix")
    args = parser.parse_args()

    import_hf_dataset(args.repo, Path(args.out), args.limit, args.prefix)


if __name__ == "__main__":
    main()
