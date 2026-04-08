import json
import os
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with minimal structure."""
    receipts_dir = tmp_path / "data" / "receipts"
    receipts_dir.mkdir(parents=True)
    labeled_dir = tmp_path / "data" / "labeled"
    labeled_dir.mkdir(parents=True)
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_receipt_image(tmp_project):
    """Create a small dummy receipt image."""
    receipts_dir = tmp_project / "data" / "receipts"
    img = Image.new("RGB", (100, 200), color="white")
    img_path = receipts_dir / "receipt_001.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_manifest(tmp_project):
    """Create a manifest with one entry."""
    manifest_path = tmp_project / "data" / "manifest.jsonl"
    entry = {
        "id": "receipt_001",
        "filename": "receipt_001.jpg",
        "added_at": "2026-04-07",
        "labeled": False,
    }
    manifest_path.write_text(json.dumps(entry) + "\n")
    return manifest_path


@pytest.fixture
def sample_labeled_record():
    """A single labeled dataset record."""
    return {
        "id": "receipt_001",
        "image_path": "data/receipts/receipt_001.jpg",
        "ingredients": ["chicken breast", "garlic", "olive oil", "lemon", "rice"],
        "teacher_output": (
            "RECIPE 1: Lemon Garlic Chicken\n"
            "USES: chicken breast, garlic, olive oil, lemon\n"
            "TIME: 35 minutes\n"
            "DIFFICULTY: easy\n"
            "STEPS:\n"
            "1. Season chicken breast with salt and pepper.\n"
            "2. Mince garlic and mix with olive oil and lemon juice.\n"
            "3. Heat a skillet over medium-high heat.\n"
            "4. Cook chicken for 6 minutes per side.\n"
            "5. Pour garlic lemon sauce over chicken in last 2 minutes.\n"
            "6. Serve over cooked rice."
        ),
    }


@pytest.fixture
def local_config_path():
    """Path to the local config file."""
    return Path(__file__).parent.parent / "configs" / "local.yaml"
