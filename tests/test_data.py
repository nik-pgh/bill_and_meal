import json
import pytest
from pathlib import Path
from PIL import Image

from bill_and_meal.data import scan_receipts, get_unlabeled, mark_labeled


class TestScanReceipts:
    def test_adds_new_images_to_empty_manifest(self, tmp_project):
        receipts_dir = tmp_project / "data" / "receipts"
        manifest_path = tmp_project / "data" / "manifest.jsonl"

        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_001.jpg")
        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_002.png")

        entries = scan_receipts(receipts_dir, manifest_path)

        assert len(entries) == 2
        ids = {e["id"] for e in entries}
        assert "receipt_001" in ids
        assert "receipt_002" in ids
        for e in entries:
            assert e["labeled"] is False
            assert "added_at" in e
            assert "filename" in e

    def test_preserves_existing_entries(self, tmp_project):
        receipts_dir = tmp_project / "data" / "receipts"
        manifest_path = tmp_project / "data" / "manifest.jsonl"

        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_001.jpg")

        scan_receipts(receipts_dir, manifest_path)

        lines = manifest_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        record["labeled"] = True
        manifest_path.write_text(json.dumps(record) + "\n")

        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_002.jpg")
        entries = scan_receipts(receipts_dir, manifest_path)

        labeled_entry = next(e for e in entries if e["id"] == "receipt_001")
        assert labeled_entry["labeled"] is True
        assert len(entries) == 2

    def test_ignores_non_image_files(self, tmp_project):
        receipts_dir = tmp_project / "data" / "receipts"
        manifest_path = tmp_project / "data" / "manifest.jsonl"

        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_001.jpg")
        (receipts_dir / "notes.txt").write_text("not an image")
        (receipts_dir / ".gitkeep").write_text("")

        entries = scan_receipts(receipts_dir, manifest_path)
        assert len(entries) == 1

    def test_writes_manifest_to_disk(self, tmp_project):
        receipts_dir = tmp_project / "data" / "receipts"
        manifest_path = tmp_project / "data" / "manifest.jsonl"

        Image.new("RGB", (100, 200), "white").save(receipts_dir / "receipt_001.jpg")
        scan_receipts(receipts_dir, manifest_path)

        assert manifest_path.exists()
        lines = manifest_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "receipt_001"


class TestGetUnlabeled:
    def test_returns_only_unlabeled(self, tmp_project):
        manifest_path = tmp_project / "data" / "manifest.jsonl"
        records = [
            {"id": "r1", "filename": "r1.jpg", "added_at": "2026-04-07", "labeled": False},
            {"id": "r2", "filename": "r2.jpg", "added_at": "2026-04-07", "labeled": True},
            {"id": "r3", "filename": "r3.jpg", "added_at": "2026-04-07", "labeled": False},
        ]
        manifest_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        unlabeled = get_unlabeled(manifest_path)
        assert len(unlabeled) == 2
        assert {u["id"] for u in unlabeled} == {"r1", "r3"}

    def test_returns_empty_when_all_labeled(self, tmp_project):
        manifest_path = tmp_project / "data" / "manifest.jsonl"
        record = {"id": "r1", "filename": "r1.jpg", "added_at": "2026-04-07", "labeled": True}
        manifest_path.write_text(json.dumps(record) + "\n")

        unlabeled = get_unlabeled(manifest_path)
        assert unlabeled == []


class TestMarkLabeled:
    def test_marks_entry_as_labeled(self, tmp_project):
        manifest_path = tmp_project / "data" / "manifest.jsonl"
        records = [
            {"id": "r1", "filename": "r1.jpg", "added_at": "2026-04-07", "labeled": False},
            {"id": "r2", "filename": "r2.jpg", "added_at": "2026-04-07", "labeled": False},
        ]
        manifest_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        mark_labeled("r1", manifest_path)

        lines = manifest_path.read_text().strip().split("\n")
        updated = [json.loads(line) for line in lines]
        r1 = next(r for r in updated if r["id"] == "r1")
        r2 = next(r for r in updated if r["id"] == "r2")
        assert r1["labeled"] is True
        assert r2["labeled"] is False
