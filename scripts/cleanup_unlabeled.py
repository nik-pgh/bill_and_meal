"""CLI: Move (or delete) receipt images that have no label in the dataset."""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_labeled_image_names(labeled_path: Path) -> set[str]:
    """Return basenames of images that appear in the labeled dataset."""
    names: set[str] = set()
    with open(labeled_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            names.add(Path(record["image_path"]).name)
    return names


def find_unlabeled_images(receipts_dir: Path, labeled_names: set[str]) -> list[Path]:
    """Return image paths in receipts_dir that don't appear in labeled set."""
    return sorted(
        p for p in receipts_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.name not in labeled_names
    )


def update_manifest(manifest_path: Path, removed_names: set[str]) -> int:
    """Remove manifest entries whose filename is in removed_names. Returns count removed."""
    if not manifest_path.exists():
        return 0
    kept = []
    removed = 0
    for line in manifest_path.read_text().strip().split("\n"):
        if not line:
            continue
        record = json.loads(line)
        if record["filename"] in removed_names:
            removed += 1
            continue
        kept.append(line)
    manifest_path.write_text("\n".join(kept) + "\n" if kept else "")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up unlabeled receipt images")
    parser.add_argument("--receipts-dir", default="data/receipts")
    parser.add_argument("--labeled-path", default="data/labeled/labeled_dataset.jsonl")
    parser.add_argument("--manifest-path", default="data/manifest.jsonl")
    parser.add_argument(
        "--mode", choices=["dry-run", "move", "delete"], default="dry-run",
        help="dry-run (default), move (to rejected/ folder), or delete",
    )
    parser.add_argument("--rejected-dir", default="data/receipts_rejected")
    args = parser.parse_args()

    receipts_dir = Path(args.receipts_dir)
    labeled_path = Path(args.labeled_path)

    labeled_names = find_labeled_image_names(labeled_path)
    unlabeled = find_unlabeled_images(receipts_dir, labeled_names)

    logger.info("Labeled images:   %d", len(labeled_names))
    logger.info("Unlabeled images: %d", len(unlabeled))

    if not unlabeled:
        logger.info("Nothing to clean up.")
        return

    logger.info("")
    logger.info("Unlabeled files (first 20):")
    for p in unlabeled[:20]:
        logger.info("  %s", p.name)
    if len(unlabeled) > 20:
        logger.info("  ... and %d more", len(unlabeled) - 20)

    if args.mode == "dry-run":
        logger.info("")
        logger.info("Dry run. To actually clean up, re-run with --mode move or --mode delete")
        return

    removed_names = {p.name for p in unlabeled}

    if args.mode == "move":
        rejected_dir = Path(args.rejected_dir)
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for p in unlabeled:
            shutil.move(str(p), str(rejected_dir / p.name))
        logger.info("Moved %d files to %s", len(unlabeled), rejected_dir)
    elif args.mode == "delete":
        for p in unlabeled:
            p.unlink()
        logger.info("Deleted %d files", len(unlabeled))

    n_removed = update_manifest(Path(args.manifest_path), removed_names)
    logger.info("Removed %d manifest entries", n_removed)


if __name__ == "__main__":
    main()
