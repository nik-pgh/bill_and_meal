# Bill&Meal

Train a local vision-language model (Gemma 4 E4B by default) to suggest recipes from grocery receipt images, distilled from Claude.

## Architecture

```
Receipt Image -> [Teacher: Claude API]    -> High-quality recipes (dataset)
Receipt Image -> [Student: Gemma 4 E4B]   -> Learned recipes (LoRA fine-tuned)
```

The repo ships **102 labeled receipt-recipe pairs** (filtered from SROIE + UniqueData) so you can jump straight to training.

## Prerequisites

- Python 3.10+
- For training: NVIDIA GPU with 12-16GB VRAM, or Google Colab (T4 free tier works)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Anthropic API key (for teacher labeling — optional if using shipped data)
- HuggingFace account + token (Gemma 4 is a gated model)
- Acknowledge Gemma 4 license at https://huggingface.co/google/gemma-4-E4B-it
- Weights & Biases API key (optional — set `report_to: none` in config to disable)

## Setup

```bash
git clone <repo-url>
cd bill_and_meal

# Recommended: uv (must be arm64-native on Apple Silicon)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt

# Plain pip alternative:
# python -m venv .venv && source .venv/bin/activate
# pip install -e . && pip install -r requirements.txt

cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY (and WANDB_API_KEY if using wandb)
```

> **Apple Silicon note**: if `uv` itself is x86_64 (check with `file $(which uv)`), reinstall it as arm64 — otherwise you'll hit binary architecture errors with Pillow / pydantic. Plain pip avoids this.

## Quick Start (use shipped data)

```bash
source .venv/bin/activate

# Inspect what's in the repo
python scripts/inspect_labels.py --sample 3

# Train on Colab (recommended) — see notebooks/colab_training.ipynb
# Local training requires NVIDIA GPU; Mac MPS won't work with bitsandbytes
```

## Full Pipeline

### Step 1: Get receipt images

You have several options:

**A. Use shipped data** — 102 grocery receipts already in `data/receipts/`. Skip to Step 3.

**B. Import a HuggingFace dataset:**
```bash
python scripts/import_hf_receipts.py --limit 10  # test
python scripts/import_hf_receipts.py             # full
# Default: UniqueData/ocr-receipts-text-detection (CC-BY-NC-ND, personal use only)
```

**C. Import SROIE (Singapore/Malaysia retail receipts):**
```bash
# 1. Set up Kaggle credentials at ~/.kaggle/kaggle.json, then:
mkdir -p /tmp/sroie && cd /tmp/sroie
kaggle datasets download -d urbikn/sroie-datasetv2 --unzip
cd -
# 2. Filter for grocery receipts (~125 of 973):
python scripts/filter_sroie_grocery.py --dry-run  # preview
python scripts/filter_sroie_grocery.py            # copy to data/receipts/
```

**D. Drop your own photos** into `data/receipts/`.

### Step 2: Label receipts with Claude

The teacher pipeline sends each receipt image to Claude and gets back structured recipe suggestions.

```bash
# Test run with 5 receipts first (~$0.10)
python scripts/label_receipts.py --limit 5

# Inspect the output
python scripts/inspect_labels.py --sample 2

# Label all unlabeled receipts (~$0.03 per receipt)
python scripts/label_receipts.py
```

The teacher script:
- Auto-detects WebP/PNG/JPEG and resizes images > 5MB
- Validates outputs (rejects malformed responses or those covering <30% of ingredients)
- Skips already-labeled receipts via `data/manifest.jsonl`

Labeled pairs land in `data/labeled/labeled_dataset.jsonl`.

### Step 3: Inspect and clean labels

```bash
# Statistics + outlier detection + 5 random samples
python scripts/inspect_labels.py

# Move receipts that have no label (failed validation, non-grocery, etc.) aside
python scripts/cleanup_unlabeled.py --mode move
```

`cleanup_unlabeled.py` modes: `dry-run` (default), `move` (to `data/receipts_rejected/`), `delete`.

### Step 4: Train (Colab recommended)

Open `notebooks/colab_training.ipynb` and follow the cells. The notebook:
1. Mounts Google Drive for checkpoints
2. Clones the repo + installs dependencies
3. Logs in to HuggingFace using a Colab Secret named `HF_TOKEN`
4. Syncs shipped data into Drive
5. Runs `bill_and_meal.train.run_training(config)`
6. Exports the LoRA adapter as a downloadable zip

Local training (NVIDIA GPU only):
```bash
python scripts/train.py
python scripts/train.py --model gemma-4-e2b   # smaller, faster
```

Checkpoints save to `checkpoints/`. Final LoRA adapter saves to `outputs/`.

### Step 5: Evaluate

```bash
# Coverage / hallucination metrics
python scripts/evaluate.py --metrics

# Manual inspection of generated recipes
python scripts/evaluate.py --qualitative --n 5

# Claude judges teacher vs student side-by-side
python scripts/evaluate.py --judge --n 20

# Run everything
python scripts/evaluate.py --all
```

Results save to `outputs/eval_results_<timestamp>.json`.

## Supported Models

| Model | Key | Size | VRAM (4-bit) | Notes |
|---|---|---|---|---|
| **Gemma 4 E4B** | `gemma-4-e4b` | 4.5B effective | ~6GB | **Default**. Native OCR/document strength. Apache 2.0. |
| Gemma 4 E2B | `gemma-4-e2b` | ~2B effective | ~3GB | Faster, smaller. Apache 2.0. |
| PaliGemma 3B | `paligemma-3b` | 3B | ~12GB | Original default. |
| LLaVA 1.6 7B | `llava-7b` | 7B | ~16GB | Larger, slower. |

Switch models via `--model <key>` or by editing the `student.model` field in your config.

## Training Metrics

During training, each epoch reports `eval_loss` (label-masked CE on the validation split). This is the best-model selection criterion (`metric_for_best_model="eval_loss"`).

Generation-quality metrics are deliberately **not** computed per-epoch (would require feeding prompt-only inputs and running `generate()` on every example each epoch — slow on T4, and the in-trainer eval dataset is teacher-forced so generation would condition on the answer). Run them post-training with:

```bash
python scripts/evaluate.py --all
```

Which reports:
- **`ingredient_iou`** — Jaccard similarity between teacher/student ingredient mentions
- **`action_alignment`** — recall of (cooking_verb, ingredient) pairs
- **`sequence_similarity`** — step-by-step structural similarity
- **Claude-as-judge** — head-to-head teacher vs student scoring

Per-epoch `eval_loss` flows to wandb under the `bill-and-meal` project.

## Configs

| Environment | File | Default model | Effective batch |
|---|---|---|---|
| Local | [configs/local.yaml](configs/local.yaml) | gemma-4-e4b | 16 (4 × 4) |
| Colab | [configs/colab.yaml](configs/colab.yaml) | gemma-4-e4b | 16 (2 × 8) |

Mixed precision is `bf16` to match Gemma's native dtype.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/import_hf_receipts.py` | Download a HuggingFace receipt dataset → `data/receipts/` |
| `scripts/filter_sroie_grocery.py` | Filter SROIE for grocery receipts → `data/receipts/` |
| `scripts/label_receipts.py` | Teacher labeling pipeline (Claude API) |
| `scripts/inspect_labels.py` | Dataset stats + outlier detection + sample inspection |
| `scripts/cleanup_unlabeled.py` | Move/delete images without labels |
| `scripts/train.py` | Local training entry point |
| `scripts/evaluate.py` | Quantitative + qualitative + judge evaluation |

## Project Structure

```
src/bill_and_meal/    # Package: config, data, teacher, student, train, evaluate
scripts/              # CLI entry points (see above)
configs/              # Environment YAML configs (local.yaml, colab.yaml)
data/
  receipts/           # 102 receipt images (committed)
  labeled/            # 102 labeled (image, recipe) pairs (committed)
  manifest.jsonl      # tracks labeled status (committed)
notebooks/            # colab_training.ipynb
tests/                # pytest tests (34 tests)
checkpoints/          # Training checkpoints (gitignored)
outputs/              # Final LoRA adapters + eval results (gitignored)
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

34 tests cover config loading, dataset manifest, teacher validation/parsing, and evaluation metrics. Training-specific tests are intentionally omitted (would require torch + GPU).
