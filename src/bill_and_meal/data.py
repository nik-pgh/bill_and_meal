"""Data pipeline: manifest management and dataset class."""

import json
from datetime import date
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _read_manifest(manifest_path: Path) -> list[dict]:
    """Read all records from a JSONL manifest file."""
    if not manifest_path.exists():
        return []
    lines = manifest_path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _write_manifest(records: list[dict], manifest_path: Path) -> None:
    """Write records to a JSONL manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def scan_receipts(receipts_dir: Path, manifest_path: Path) -> list[dict]:
    """Scan receipts directory and update manifest with new images.

    Preserves existing manifest entries (including labeled status).
    Adds new entries for images not yet in the manifest.

    Returns:
        All manifest entries after update.
    """
    existing = _read_manifest(manifest_path)
    existing_filenames = {r["filename"] for r in existing}

    image_files = sorted(
        f
        for f in Path(receipts_dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    new_entries = [
        {
            "id": f.stem,
            "filename": f.name,
            "added_at": date.today().isoformat(),
            "labeled": False,
        }
        for f in image_files
        if f.name not in existing_filenames
    ]

    all_entries = existing + new_entries
    _write_manifest(all_entries, manifest_path)
    return all_entries


def get_unlabeled(manifest_path: Path) -> list[dict]:
    """Return manifest entries that have not been labeled yet."""
    records = _read_manifest(manifest_path)
    return [r for r in records if not r["labeled"]]


def mark_labeled(record_id: str, manifest_path: Path) -> None:
    """Mark a manifest entry as labeled by its ID."""
    records = _read_manifest(manifest_path)
    updated = [
        {**r, "labeled": True} if r["id"] == record_id else r
        for r in records
    ]
    _write_manifest(updated, manifest_path)
