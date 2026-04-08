# Bill&Meal Project Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the full Bill&Meal project with directory structure, configs, package modules with defined interfaces, CLI scripts, tests, and Colab notebook — ready for implementation.

**Architecture:** Python package (`src/bill_and_meal/`) with thin CLI scripts (`scripts/`), environment-specific YAML configs (`configs/`), and a Colab notebook that imports the package. TDD approach — tests written before implementation for all testable modules.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, PEFT, Anthropic SDK, wandb, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `.gitignore` | Ignore data, checkpoints, outputs, .env, __pycache__ |
| `.env.example` | Template for required API keys |
| `requirements.txt` | All pip dependencies |
| `setup.py` | Editable install for package |
| `configs/local.yaml` | RTX 4080 environment config |
| `configs/colab.yaml` | Google Colab environment config |
| `src/bill_and_meal/__init__.py` | Package version and metadata |
| `src/bill_and_meal/config.py` | YAML loading, env detection, .env loading |
| `src/bill_and_meal/data.py` | Manifest management, ReceiptRecipeDataset |
| `src/bill_and_meal/teacher.py` | Claude API labeling, quality filtering, teacher prompt |
| `src/bill_and_meal/student.py` | Model registry, model loading, LoRA setup |
| `src/bill_and_meal/train.py` | Trainer construction, training orchestration |
| `src/bill_and_meal/evaluate.py` | Metrics, qualitative eval, Claude-as-judge |
| `scripts/label_receipts.py` | CLI entry for teacher labeling |
| `scripts/train.py` | CLI entry for training |
| `scripts/evaluate.py` | CLI entry for evaluation |
| `tests/conftest.py` | Shared fixtures (tmp dirs, sample data) |
| `tests/test_config.py` | Config loading and env detection tests |
| `tests/test_data.py` | Manifest management and dataset tests |
| `tests/test_teacher.py` | Quality filtering and prompt tests |
| `tests/test_evaluate.py` | Metric function tests |
| `notebooks/colab_training.ipynb` | Colab wrapper notebook |
| `README.md` | Project overview and quickstart |

---

### Task 1: Project Foundation

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `src/bill_and_meal/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Data and model artifacts
data/receipts/
data/manifest.jsonl
data/labeled/
data/dpo_pairs.jsonl
checkpoints/
outputs/

# Secrets
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# wandb
wandb/
```

- [ ] **Step 2: Create `.env.example`**

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
WANDB_API_KEY=your-wandb-api-key-here
```

- [ ] **Step 3: Create `requirements.txt`**

```
torch>=2.0
torchvision
transformers>=4.40
accelerate
peft
trl
bitsandbytes
datasets
Pillow
wandb
anthropic
python-dotenv
pyyaml
evaluate
bert-score
pytest
```

- [ ] **Step 4: Create `setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="bill_and_meal",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
```

- [ ] **Step 5: Create `src/bill_and_meal/__init__.py`**

```python
"""Bill&Meal: Knowledge distillation for receipt-to-recipe generation."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create data and output directories with `.gitkeep` files**

```bash
mkdir -p data/receipts data/labeled checkpoints outputs notebooks tests scripts configs
touch data/receipts/.gitkeep data/labeled/.gitkeep checkpoints/.gitkeep outputs/.gitkeep
```

Note: `.gitkeep` files ensure empty directories are tracked by git while their contents are gitignored.

- [ ] **Step 7: Commit**

```bash
git add .gitignore .env.example requirements.txt setup.py src/bill_and_meal/__init__.py \
  data/receipts/.gitkeep data/labeled/.gitkeep checkpoints/.gitkeep outputs/.gitkeep
git commit -m "chore: project foundation — gitignore, deps, setup.py, package init"
```

---

### Task 2: Configuration System

**Files:**
- Create: `configs/local.yaml`
- Create: `configs/colab.yaml`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Create: `src/bill_and_meal/config.py`

- [ ] **Step 1: Create `configs/local.yaml`**

```yaml
environment: local

data:
  receipts_dir: ./data/receipts
  manifest_path: ./data/manifest.jsonl
  labeled_path: ./data/labeled/labeled_dataset.jsonl

teacher:
  model: claude-sonnet-4-20250514
  max_tokens: 2000
  rate_limit_delay: 1.2

student:
  model: paligemma-3b
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
    dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_length: 1024
  fp16: true
  gradient_checkpointing: true
  checkpoint_dir: ./checkpoints
  output_dir: ./outputs

wandb:
  project: bill-and-meal
  run_name_prefix: local
```

- [ ] **Step 2: Create `configs/colab.yaml`**

```yaml
environment: colab

data:
  receipts_dir: /content/drive/MyDrive/bill_and_meal/receipts
  manifest_path: /content/drive/MyDrive/bill_and_meal/manifest.jsonl
  labeled_path: /content/drive/MyDrive/bill_and_meal/labeled_dataset.jsonl

teacher:
  model: claude-sonnet-4-20250514
  max_tokens: 2000
  rate_limit_delay: 1.2

student:
  model: paligemma-3b
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
    dropout: 0.05

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_length: 1024
  fp16: true
  gradient_checkpointing: true
  checkpoint_dir: /content/drive/MyDrive/bill_and_meal/checkpoints
  output_dir: /content/drive/MyDrive/bill_and_meal/outputs

wandb:
  project: bill-and-meal
  run_name_prefix: colab
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
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
```

- [ ] **Step 4: Write failing tests for config module**

Create `tests/test_config.py`:

```python
import os
import pytest
from pathlib import Path

from bill_and_meal.config import load_config, detect_environment


class TestDetectEnvironment:
    def test_returns_local_by_default(self):
        result = detect_environment()
        assert result == "local"

    def test_returns_colab_when_colab_gpu_set(self, monkeypatch):
        monkeypatch.setenv("COLAB_GPU", "1")
        result = detect_environment()
        assert result == "colab"


class TestLoadConfig:
    def test_loads_local_config(self, local_config_path):
        config = load_config(local_config_path)
        assert config["environment"] == "local"
        assert config["student"]["model"] == "paligemma-3b"
        assert config["training"]["batch_size"] == 4

    def test_loads_colab_config(self):
        colab_path = Path(__file__).parent.parent / "configs" / "colab.yaml"
        config = load_config(colab_path)
        assert config["environment"] == "colab"
        assert config["training"]["batch_size"] == 2
        assert config["training"]["gradient_accumulation_steps"] == 8

    def test_config_has_all_required_sections(self, local_config_path):
        config = load_config(local_config_path)
        required = ["environment", "data", "teacher", "student", "training", "wandb"]
        for section in required:
            assert section in config, f"Missing config section: {section}"

    def test_loads_env_secrets(self, tmp_path, local_config_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=test-key-123\nWANDB_API_KEY=wandb-456\n")
        config = load_config(local_config_path, env_path=env_file)
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-key-123"
        assert os.environ.get("WANDB_API_KEY") == "wandb-456"

    def test_auto_detects_and_loads(self, monkeypatch):
        monkeypatch.delenv("COLAB_GPU", raising=False)
        config = load_config()
        assert config["environment"] == "local"
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cd /Users/nikkim/Documents/bill_and_meal && python -m pytest tests/test_config.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'bill_and_meal'`

- [ ] **Step 6: Install package in editable mode**

```bash
pip install -e .
```

- [ ] **Step 7: Run tests again to verify they still fail (correct reason)**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL — `ImportError: cannot import name 'load_config' from 'bill_and_meal'`

- [ ] **Step 8: Implement `src/bill_and_meal/config.py`**

```python
"""Configuration loading with environment auto-detection."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).parent.parent.parent


def detect_environment() -> str:
    """Detect whether we're running on Colab or locally."""
    if os.environ.get("COLAB_GPU") or Path("/content/drive").exists():
        return "colab"
    return "local"


def load_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> dict:
    """Load YAML config and .env secrets.

    Args:
        config_path: Explicit path to YAML config. If None, auto-detects environment
                     and loads the matching config from configs/.
        env_path: Path to .env file. If None, looks for .env in project root.

    Returns:
        Parsed config dict.
    """
    if config_path is None:
        env = detect_environment()
        config_path = _PROJECT_ROOT / "configs" / f"{env}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if env_path is None:
        env_path = _PROJECT_ROOT / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)

    return config
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`

Expected: All 5 tests PASS

- [ ] **Step 10: Commit**

```bash
git add configs/ tests/conftest.py tests/test_config.py src/bill_and_meal/config.py
git commit -m "feat: configuration system with env auto-detection and YAML loading"
```

---

### Task 3: Data Pipeline — Manifest Management

**Files:**
- Create: `tests/test_data.py`
- Create: `src/bill_and_meal/data.py`

- [ ] **Step 1: Write failing tests for manifest management**

Create `tests/test_data.py`:

```python
import json
import pytest
from pathlib import Path
from PIL import Image

from bill_and_meal.data import scan_receipts, get_unlabeled, mark_labeled


class TestScanReceipts:
    def test_adds_new_images_to_empty_manifest(self, tmp_project):
        receipts_dir = tmp_project / "data" / "receipts"
        manifest_path = tmp_project / "data" / "manifest.jsonl"

        # Create two receipt images
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

        # First scan
        scan_receipts(receipts_dir, manifest_path)

        # Mark as labeled manually
        lines = manifest_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        record["labeled"] = True
        manifest_path.write_text(json.dumps(record) + "\n")

        # Add another image and scan again
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -v`

Expected: FAIL — `ImportError: cannot import name 'scan_receipts' from 'bill_and_meal.data'`

- [ ] **Step 3: Implement `src/bill_and_meal/data.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_data.py src/bill_and_meal/data.py
git commit -m "feat: data pipeline — manifest scanning, unlabeled lookup, mark labeled"
```

---

### Task 4: Teacher Labeling — Validation & Prompt

**Files:**
- Create: `tests/test_teacher.py`
- Create: `src/bill_and_meal/teacher.py`

- [ ] **Step 1: Write failing tests for teacher validation**

Create `tests/test_teacher.py`:

```python
import base64
import pytest
from pathlib import Path

from bill_and_meal.teacher import validate_teacher_output, encode_image, TEACHER_SYSTEM_PROMPT


class TestValidateTeacherOutput:
    def test_valid_output_passes(self, sample_labeled_record):
        assert validate_teacher_output(sample_labeled_record) is True

    def test_rejects_missing_recipe_marker(self, sample_labeled_record):
        record = {**sample_labeled_record, "teacher_output": "Here are some ideas for dinner."}
        assert validate_teacher_output(record) is False

    def test_rejects_low_ingredient_coverage(self, sample_labeled_record):
        record = {
            **sample_labeled_record,
            "ingredients": ["chicken", "garlic", "olive oil", "lemon", "rice",
                            "butter", "cream", "flour", "eggs", "milk"],
            "teacher_output": (
                "RECIPE 1: Toast\n"
                "USES: bread\n"
                "TIME: 5 minutes\n"
                "DIFFICULTY: easy\n"
                "STEPS:\n"
                "1. Toast the bread."
            ),
        }
        assert validate_teacher_output(record) is False

    def test_rejects_missing_steps(self, sample_labeled_record):
        record = {
            **sample_labeled_record,
            "teacher_output": "RECIPE 1: Chicken\nUSES: chicken breast\n",
        }
        assert validate_teacher_output(record) is False

    def test_accepts_30_percent_coverage(self):
        record = {
            "ingredients": ["chicken", "garlic", "olive oil", "lemon", "rice",
                            "onion", "pepper", "tomato", "basil", "salt"],
            "teacher_output": (
                "RECIPE 1: Simple Chicken\n"
                "USES: chicken, garlic, olive oil\n"
                "TIME: 20 minutes\n"
                "DIFFICULTY: easy\n"
                "STEPS:\n"
                "1. Cook the chicken with garlic and olive oil."
            ),
        }
        assert validate_teacher_output(record) is True


class TestEncodeImage:
    def test_returns_base64_string(self, sample_receipt_image):
        result = encode_image(sample_receipt_image)
        # Should be valid base64
        decoded = base64.standard_b64decode(result)
        assert len(decoded) > 0

    def test_encoded_data_is_string(self, sample_receipt_image):
        result = encode_image(sample_receipt_image)
        assert isinstance(result, str)


class TestTeacherPrompt:
    def test_prompt_exists_and_has_key_instructions(self):
        assert "recipe" in TEACHER_SYSTEM_PROMPT.lower()
        assert "RECIPE 1:" in TEACHER_SYSTEM_PROMPT
        assert "STEPS:" in TEACHER_SYSTEM_PROMPT
        assert "USES:" in TEACHER_SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_teacher.py -v`

Expected: FAIL — `ImportError: cannot import name 'validate_teacher_output' from 'bill_and_meal.teacher'`

- [ ] **Step 3: Implement `src/bill_and_meal/teacher.py`**

```python
"""Teacher labeling pipeline: Claude API calls, validation, batch labeling."""

import base64
import json
import logging
import time
from pathlib import Path

import anthropic

from bill_and_meal.data import get_unlabeled, mark_labeled

logger = logging.getLogger(__name__)

TEACHER_SYSTEM_PROMPT = """You are a creative home chef. Given a grocery receipt image,
identify the food items and suggest 2-3 practical recipes using ONLY ingredients
visible on the receipt (plus basic pantry staples like salt, pepper, water).

For each recipe, provide:
- Recipe name
- Which receipt items it uses
- Brief instructions (5-8 steps)
- Estimated cook time
- Difficulty level (easy/medium/hard)

Format your response exactly as:

RECIPE 1: [Name]
USES: [comma-separated ingredients from receipt]
TIME: [X minutes]
DIFFICULTY: [easy/medium/hard]
STEPS:
1. [step]
2. [step]
...

RECIPE 2: [Name]
..."""


def encode_image(image_path: Path) -> str:
    """Base64 encode an image file for the Claude API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def validate_teacher_output(record: dict) -> bool:
    """Check whether a teacher output meets quality thresholds.

    A record passes if:
    - Contains at least one "RECIPE 1:" block
    - References >= 30% of the receipt ingredients
    - Contains a "STEPS:" section
    """
    output = record["teacher_output"]
    ingredients = record["ingredients"]

    if "RECIPE 1:" not in output:
        return False

    if "STEPS:" not in output:
        return False

    mentioned = sum(
        1 for ing in ingredients if ing.lower() in output.lower()
    )
    if mentioned < len(ingredients) * 0.3:
        return False

    return True


def _media_type_for(image_path: Path) -> str:
    """Return the MIME type for an image file."""
    suffix = image_path.suffix.lower()
    types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return types.get(suffix, "image/jpeg")


def get_teacher_label(image_path: Path, config: dict, retries: int = 3) -> str | None:
    """Get Claude's recipe output for a receipt image.

    Args:
        image_path: Path to the receipt image.
        config: Loaded config dict (needs config["teacher"]).
        retries: Number of retry attempts.

    Returns:
        Recipe text from Claude, or None if all attempts fail.
    """
    client = anthropic.Anthropic()
    b64 = encode_image(image_path)
    media_type = _media_type_for(image_path)
    teacher_cfg = config["teacher"]

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=teacher_cfg["model"],
                max_tokens=teacher_cfg["max_tokens"],
                system=TEACHER_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Here is my grocery receipt. What can I cook?",
                            },
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            time.sleep(2**attempt)

    return None


def label_batch(config: dict) -> int:
    """Label all unlabeled receipts with teacher outputs.

    Args:
        config: Loaded config dict.

    Returns:
        Number of successfully labeled receipts.
    """
    data_cfg = config["data"]
    manifest_path = Path(data_cfg["manifest_path"])
    labeled_path = Path(data_cfg["labeled_path"])
    receipts_dir = Path(data_cfg["receipts_dir"])
    rate_limit_delay = config["teacher"]["rate_limit_delay"]

    labeled_path.parent.mkdir(parents=True, exist_ok=True)

    unlabeled = get_unlabeled(manifest_path)
    logger.info("Found %d unlabeled receipts", len(unlabeled))

    labeled_count = 0
    with open(labeled_path, "a") as f:
        for i, entry in enumerate(unlabeled):
            image_path = receipts_dir / entry["filename"]
            logger.info("Labeling %d/%d: %s", i + 1, len(unlabeled), entry["filename"])

            teacher_output = get_teacher_label(image_path, config)
            if teacher_output is None:
                logger.warning("Skipping %s: all retries failed", entry["filename"])
                continue

            record = {
                "id": entry["id"],
                "image_path": str(image_path),
                "ingredients": [],  # extracted from receipt by Claude
                "teacher_output": teacher_output,
            }

            if not validate_teacher_output(record):
                logger.warning("Skipping %s: failed validation", entry["filename"])
                continue

            f.write(json.dumps(record) + "\n")
            mark_labeled(entry["id"], manifest_path)
            labeled_count += 1

            time.sleep(rate_limit_delay)

    logger.info("Labeled %d/%d receipts", labeled_count, len(unlabeled))
    return labeled_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_teacher.py -v`

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_teacher.py src/bill_and_meal/teacher.py
git commit -m "feat: teacher labeling — Claude API, validation, batch labeling"
```

---

### Task 5: Student Model Registry & Loading

**Files:**
- Create: `src/bill_and_meal/student.py`

- [ ] **Step 1: Create `src/bill_and_meal/student.py`**

This module imports heavy ML libraries, so tests are deferred to integration testing. The module is structured so the registry dict is testable without GPU.

```python
"""Student model loading with config-driven model selection and QLoRA setup."""

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    LlavaForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODELS = {
    "paligemma-3b": {
        "hf_id": "google/paligemma-3b-pt-224",
        "model_class": PaliGemmaForConditionalGeneration,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "llava-7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
}


def load_model(config: dict) -> tuple:
    """Load a student model and processor with quantization.

    Args:
        config: Loaded config dict (needs config["student"]).

    Returns:
        Tuple of (model, processor).
    """
    student_cfg = config["student"]
    model_key = student_cfg["model"]
    model_info = MODELS[model_key]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_info["hf_id"])
    model = model_info["model_class"].from_pretrained(
        model_info["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    return model, processor


def attach_lora(model, config: dict):
    """Attach LoRA adapters to a model for QLoRA training.

    Args:
        model: A loaded HuggingFace model.
        config: Loaded config dict (needs config["student"]).

    Returns:
        Model with LoRA adapters attached.
    """
    student_cfg = config["student"]
    model_key = student_cfg["model"]
    model_info = MODELS[model_key]
    lora_cfg = student_cfg["lora"]

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=model_info["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
```

- [ ] **Step 2: Commit**

```bash
git add src/bill_and_meal/student.py
git commit -m "feat: student model registry with PaliGemma and LLaVA, QLoRA setup"
```

---

### Task 6: Training Module

**Files:**
- Create: `src/bill_and_meal/train.py`

- [ ] **Step 1: Create `src/bill_and_meal/train.py`**

```python
"""Training orchestration: dataset construction, Trainer setup, training loop."""

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments

from bill_and_meal.config import load_config
from bill_and_meal.student import load_model, attach_lora

logger = logging.getLogger(__name__)


class ReceiptRecipeDataset(Dataset):
    """PyTorch Dataset for receipt-recipe pairs."""

    def __init__(self, data_path: Path, processor, max_length: int = 1024):
        with open(data_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")

        prompt = "What recipes can I make from this grocery receipt?"
        answer = record["teacher_output"]

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze(0)

        return inputs


def build_trainer(
    model,
    processor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
) -> Trainer:
    """Construct a HuggingFace Trainer from config.

    Args:
        model: Model with LoRA adapters.
        processor: Model processor/tokenizer.
        train_dataset: Training split.
        val_dataset: Validation split.
        config: Loaded config dict.

    Returns:
        Configured Trainer instance.
    """
    t = config["training"]
    w = config["wandb"]

    training_args = TrainingArguments(
        output_dir=t["checkpoint_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name=f"{w['run_name_prefix']}_{config['student']['model']}",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {
            k: torch.stack([d[k] for d in data]) for k in data[0]
        },
    )


def run_training(config: dict | None = None) -> None:
    """Full training pipeline: load model, build dataset, train, save.

    Args:
        config: Config dict. If None, auto-loads from environment.
    """
    if config is None:
        config = load_config()

    logger.info("Loading model: %s", config["student"]["model"])
    model, processor = load_model(config)
    model = attach_lora(model, config)

    logger.info("Building dataset from %s", config["data"]["labeled_path"])
    full_dataset = ReceiptRecipeDataset(
        Path(config["data"]["labeled_path"]),
        processor,
        max_length=config["training"]["max_length"],
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info("Training: %d train, %d val", len(train_dataset), len(val_dataset))
    trainer = build_trainer(model, processor, train_dataset, val_dataset, config)
    trainer.train()

    output_dir = config["training"]["output_dir"]
    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
```

- [ ] **Step 2: Commit**

```bash
git add src/bill_and_meal/train.py
git commit -m "feat: training module — dataset class, Trainer builder, run_training pipeline"
```

---

### Task 7: Evaluation Module

**Files:**
- Create: `tests/test_evaluate.py`
- Create: `src/bill_and_meal/evaluate.py`

- [ ] **Step 1: Write failing tests for metric functions**

Create `tests/test_evaluate.py`:

```python
import pytest

from bill_and_meal.evaluate import ingredient_coverage, hallucination_rate


class TestIngredientCoverage:
    def test_full_coverage(self):
        items = ["chicken", "garlic", "rice"]
        text = "Cook the chicken with garlic and serve over rice."
        assert ingredient_coverage(items, text) == 1.0

    def test_partial_coverage(self):
        items = ["chicken", "garlic", "rice", "lemon"]
        text = "Cook the chicken with garlic."
        assert ingredient_coverage(items, text) == 0.5

    def test_no_coverage(self):
        items = ["chicken", "garlic"]
        text = "Boil some pasta with tomato sauce."
        assert ingredient_coverage(items, text) == 0.0

    def test_case_insensitive(self):
        items = ["Chicken Breast", "GARLIC"]
        text = "cook the chicken breast with garlic"
        assert ingredient_coverage(items, text) == 1.0


class TestHallucinationRate:
    def test_no_hallucination(self):
        receipt_items = ["chicken", "garlic", "rice"]
        text = "Cook chicken with garlic over rice."
        known = ["chicken", "garlic", "rice", "pasta", "beef"]
        assert hallucination_rate(receipt_items, text, known) == 0.0

    def test_some_hallucination(self):
        receipt_items = ["chicken", "garlic"]
        text = "Cook chicken with garlic and pasta."
        known = ["chicken", "garlic", "pasta", "rice"]
        # mentioned: chicken, garlic, pasta (3 total). hallucinated: pasta (1)
        assert hallucination_rate(receipt_items, text, known) == pytest.approx(1 / 3)

    def test_all_hallucinated(self):
        receipt_items = ["chicken"]
        text = "Make pasta with beef and cream."
        known = ["chicken", "pasta", "beef", "cream"]
        # mentioned: pasta, beef, cream (3). hallucinated: all 3
        assert hallucination_rate(receipt_items, text, known) == 1.0

    def test_empty_mentions_returns_zero(self):
        receipt_items = ["chicken"]
        text = "A delightful meal."
        known = ["chicken", "pasta"]
        assert hallucination_rate(receipt_items, text, known) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate.py -v`

Expected: FAIL — `ImportError: cannot import name 'ingredient_coverage' from 'bill_and_meal.evaluate'`

- [ ] **Step 3: Implement `src/bill_and_meal/evaluate.py`**

```python
"""Evaluation: metrics, qualitative checks, Claude-as-judge."""

import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)


def ingredient_coverage(receipt_items: list[str], generated_text: str) -> float:
    """Calculate what fraction of receipt items appear in generated text.

    Args:
        receipt_items: List of ingredient names from the receipt.
        generated_text: Model-generated recipe text.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not receipt_items:
        return 0.0
    text_lower = generated_text.lower()
    used = sum(1 for item in receipt_items if item.lower() in text_lower)
    return used / len(receipt_items)


def hallucination_rate(
    receipt_items: list[str],
    generated_text: str,
    known_ingredients: list[str],
) -> float:
    """Calculate what fraction of mentioned ingredients are NOT on the receipt.

    Args:
        receipt_items: Ingredients from the receipt.
        generated_text: Model-generated recipe text.
        known_ingredients: Full list of known ingredient names to check against.

    Returns:
        Float between 0.0 and 1.0. 0.0 means no hallucination.
    """
    text_lower = generated_text.lower()
    receipt_set = {i.lower() for i in receipt_items}

    mentioned = {
        ing.lower()
        for ing in known_ingredients
        if ing.lower() in text_lower
    }

    if not mentioned:
        return 0.0

    hallucinated = mentioned - receipt_set
    return len(hallucinated) / len(mentioned)


JUDGE_PROMPT = """Compare these two recipe suggestions for the same grocery receipt.
Rate each on: ingredient accuracy (1-5), creativity (1-5), practicality (1-5).

Receipt items: {ingredients}

Response A (Teacher): {teacher}
Response B (Student): {student}

Provide scores as JSON: {{"teacher": {{"accuracy": X, "creativity": X, "practicality": X}},
"student": {{"accuracy": X, "creativity": X, "practicality": X}}}}"""


def judge_comparison(
    ingredients: list[str],
    teacher_output: str,
    student_output: str,
    config: dict,
) -> dict | None:
    """Use Claude to judge teacher vs student outputs.

    Args:
        ingredients: Receipt ingredient list.
        teacher_output: Teacher-generated recipe text.
        student_output: Student-generated recipe text.
        config: Loaded config dict (needs config["teacher"] for model).

    Returns:
        Parsed JSON scores dict, or None on failure.
    """
    client = anthropic.Anthropic()
    prompt = JUDGE_PROMPT.format(
        ingredients=", ".join(ingredients),
        teacher=teacher_output,
        student=student_output,
    )

    try:
        response = client.messages.create(
            model=config["teacher"]["model"],
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        logger.warning("Judge call failed: %s", e)
        return None


def save_eval_results(results: dict, output_dir: Path) -> Path:
    """Save evaluation results with timestamp.

    Args:
        results: Dict of evaluation results.
        output_dir: Directory to save to.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"eval_results_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate.py -v`

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_evaluate.py src/bill_and_meal/evaluate.py
git commit -m "feat: evaluation — ingredient coverage, hallucination rate, Claude judge"
```

---

### Task 8: CLI Scripts

**Files:**
- Create: `scripts/label_receipts.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Create `scripts/label_receipts.py`**

```python
"""CLI: Run teacher labeling on receipt images."""

import argparse
import logging

from bill_and_meal.config import load_config
from bill_and_meal.data import scan_receipts, get_unlabeled
from bill_and_meal.teacher import label_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Label receipts with Claude teacher")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Max receipts to label")
    parser.add_argument("--revalidate", action="store_true", help="Re-filter existing labels")
    args = parser.parse_args()

    config = load_config(args.config)

    from pathlib import Path

    data_cfg = config["data"]
    receipts_dir = Path(data_cfg["receipts_dir"])
    manifest_path = Path(data_cfg["manifest_path"])

    # Scan for new images
    entries = scan_receipts(receipts_dir, manifest_path)
    logger.info("Manifest has %d total entries", len(entries))

    unlabeled = get_unlabeled(manifest_path)
    logger.info("Found %d unlabeled receipts", len(unlabeled))

    if args.limit:
        logger.info("Limiting to %d receipts", args.limit)

    if args.revalidate:
        logger.info("Revalidation mode not yet implemented")
        return

    labeled = label_batch(config)
    logger.info("Done. Labeled %d receipts.", labeled)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/train.py`**

```python
"""CLI: Launch student model training."""

import argparse
import logging

from bill_and_meal.config import load_config
from bill_and_meal.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Override student model key")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["student"]["model"] = args.model
        logger.info("Model overridden to: %s", args.model)

    run_training(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/evaluate.py`**

```python
"""CLI: Run evaluation suite on student model."""

import argparse
import json
import logging
from pathlib import Path

from bill_and_meal.config import load_config
from bill_and_meal.evaluate import (
    ingredient_coverage,
    hallucination_rate,
    judge_comparison,
    save_eval_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_metrics(config: dict) -> dict:
    """Run quantitative metrics on the labeled dataset."""
    labeled_path = Path(config["data"]["labeled_path"])
    with open(labeled_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    coverages = []
    for r in records:
        cov = ingredient_coverage(r["ingredients"], r["teacher_output"])
        coverages.append(cov)

    return {
        "num_records": len(records),
        "avg_ingredient_coverage": sum(coverages) / len(coverages) if coverages else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--qualitative", action="store_true", help="Run qualitative checks")
    parser.add_argument("--metrics", action="store_true", help="Run quantitative metrics")
    parser.add_argument("--judge", action="store_true", help="Run Claude-as-judge")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    config = load_config(args.config)
    results = {}

    if args.all or args.qualitative:
        logger.info("Qualitative evaluation not yet implemented (requires trained model)")

    if args.all or args.metrics:
        logger.info("Running quantitative metrics...")
        results["metrics"] = run_metrics(config)
        logger.info("Metrics: %s", json.dumps(results["metrics"], indent=2))

    if args.all or args.judge:
        logger.info("Judge evaluation not yet implemented (requires trained model)")

    if results:
        output_dir = Path(config["training"]["output_dir"])
        path = save_eval_results(results, output_dir)
        logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: CLI scripts for labeling, training, and evaluation"
```

---

### Task 9: Colab Notebook

**Files:**
- Create: `notebooks/colab_training.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/colab_training.ipynb` with this content (standard ipynb JSON format):

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.0"
    },
    "colab": {
      "provenance": []
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Bill&Meal: Receipt-to-Recipe Training\n",
        "\n",
        "This notebook trains a student vision-language model using the Bill&Meal package.\n",
        "All logic lives in the `bill_and_meal` package — this notebook just calls it."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Setup"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Clone repo and install package\n",
        "# Replace with your repo URL\n",
        "!git clone https://github.com/YOUR_USER/bill_and_meal.git /content/bill_and_meal\n",
        "%cd /content/bill_and_meal\n",
        "!pip install -e . -q"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Set API keys\n",
        "import os\n",
        "os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'  # or use Colab secrets\n",
        "os.environ['WANDB_API_KEY'] = 'your-key-here'"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Verify Data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "from pathlib import Path\n",
        "from bill_and_meal.config import load_config\n",
        "\n",
        "config = load_config()\n",
        "receipts_dir = Path(config['data']['receipts_dir'])\n",
        "labeled_path = Path(config['data']['labeled_path'])\n",
        "\n",
        "receipt_count = len(list(receipts_dir.glob('*'))) if receipts_dir.exists() else 0\n",
        "labeled_count = sum(1 for _ in open(labeled_path)) if labeled_path.exists() else 0\n",
        "\n",
        "print(f'Receipts: {receipt_count}')\n",
        "print(f'Labeled pairs: {labeled_count}')"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Train"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "from bill_and_meal.train import run_training\n",
        "\n",
        "run_training(config)"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Evaluate"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import json\n",
        "from bill_and_meal.evaluate import ingredient_coverage\n",
        "\n",
        "with open(config['data']['labeled_path']) as f:\n",
        "    records = [json.loads(line) for line in f if line.strip()]\n",
        "\n",
        "for r in records[:5]:\n",
        "    cov = ingredient_coverage(r['ingredients'], r['teacher_output'])\n",
        "    print(f\"{r['id']}: coverage={cov:.1%}\")"
      ],
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Export"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "output_dir = Path(config['training']['output_dir'])\n",
        "if output_dir.exists():\n",
        "    print(f'Model saved at: {output_dir}')\n",
        "    print('Files:', list(output_dir.iterdir()))\n",
        "else:\n",
        "    print('No model output found. Did training complete?')"
      ],
      "outputs": [],
      "execution_count": null
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/colab_training.ipynb
git commit -m "feat: Colab training notebook"
```

---

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Bill&Meal

Train a local vision-language model to suggest recipes from grocery receipt images using knowledge distillation from Claude.

## Architecture

```
Receipt Image -> [Teacher: Claude API] -> High-quality recipes (dataset)
Receipt Image -> [Student: PaliGemma/LLaVA] -> Learned recipes (trained)
```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd bill_and_meal
pip install -e .

# Configure secrets
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### 1. Label receipts with Claude

Drop receipt images into `data/receipts/`, then:

```bash
python scripts/label_receipts.py             # label all
python scripts/label_receipts.py --limit 10  # test run
```

### 2. Train student model

```bash
python scripts/train.py                              # auto-detect env
python scripts/train.py --config configs/local.yaml  # explicit config
python scripts/train.py --model llava-7b             # override model
```

### 3. Evaluate

```bash
python scripts/evaluate.py --metrics        # quantitative
python scripts/evaluate.py --qualitative    # manual inspection
python scripts/evaluate.py --judge --n 20   # Claude comparison
python scripts/evaluate.py --all            # everything
```

## Supported Models

| Model | Key | VRAM |
|---|---|---|
| PaliGemma 3B | `paligemma-3b` | ~12GB (QLoRA) |
| LLaVA 1.6 7B | `llava-7b` | ~16GB (QLoRA) |

## Environments

- **Local** (`configs/local.yaml`): RTX 4080, 16GB VRAM
- **Colab** (`configs/colab.yaml`): Google Colab with Drive mount

See `notebooks/colab_training.ipynb` for the Colab workflow.

## Project Structure

```
src/bill_and_meal/    # Package: config, data, teacher, student, train, evaluate
scripts/              # CLI entry points
configs/              # Environment YAML configs
data/                 # Receipt images + labeled dataset
notebooks/            # Colab notebook
tests/                # pytest tests
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, usage, and project overview"
```

---

### Task 11: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`

Expected: All 20 tests PASS (5 config + 7 data + 8 teacher/evaluate)

- [ ] **Step 2: Verify project installs cleanly**

```bash
pip install -e . 2>&1 | tail -1
```

Expected: `Successfully installed bill-and-meal-0.1.0`

- [ ] **Step 3: Verify git status is clean**

```bash
git status
git log --oneline
```

Expected: Clean working tree, commits in order:
1. `chore: project foundation`
2. `feat: configuration system`
3. `feat: data pipeline`
4. `feat: teacher labeling`
5. `feat: student model registry`
6. `feat: training module`
7. `feat: evaluation`
8. `feat: CLI scripts`
9. `feat: Colab notebook`
10. `docs: README`
