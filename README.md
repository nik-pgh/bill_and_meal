# Bill&Meal

Train a local vision-language model to suggest recipes from grocery receipt images using knowledge distillation from Claude.

## Architecture

```
Receipt Image -> [Teacher: Claude API] -> High-quality recipes (dataset)
Receipt Image -> [Student: PaliGemma/LLaVA] -> Learned recipes (trained)
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 12-16GB VRAM (local training) or Google Colab
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Anthropic API key (for teacher labeling)
- Weights & Biases API key (for experiment tracking)

## Setup

```bash
# Clone
git clone <repo-url>
cd bill_and_meal

# Create virtual environment and install
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt

# Or with plain pip:
# python -m venv .venv
# source .venv/bin/activate
# pip install -e .
# pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and WANDB_API_KEY
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Usage

All commands assume the venv is activated (`source .venv/bin/activate`).

### Step 1: Add receipt images

Drop your grocery receipt photos (`.jpg`, `.png`, etc.) into `data/receipts/`.

### Step 2: Label receipts with Claude

The teacher pipeline sends each receipt image to Claude and gets back structured recipe suggestions.

```bash
# Test run with 10 receipts first
python scripts/label_receipts.py --limit 10

# Label all unlabeled receipts
python scripts/label_receipts.py

# Use a specific config
python scripts/label_receipts.py --config configs/local.yaml
```

Labeled pairs are saved to `data/labeled/labeled_dataset.jsonl`. The manifest (`data/manifest.jsonl`) tracks which images have been labeled, so re-running skips already-labeled images.

### Step 3: Train student model

Requires a GPU. On local (RTX 4080):

```bash
# Train with default config (auto-detects environment)
python scripts/train.py

# Explicit config
python scripts/train.py --config configs/local.yaml

# Switch to LLaVA-7B instead of PaliGemma-3B
python scripts/train.py --model llava-7b
```

On Colab, open `notebooks/colab_training.ipynb` and follow the cells.

Checkpoints save to `checkpoints/`. Final LoRA adapter saves to `outputs/`.

### Step 4: Evaluate

```bash
# Quantitative metrics (ingredient coverage, hallucination rate)
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
configs/              # Environment YAML configs (local.yaml, colab.yaml)
data/
  receipts/           # Your receipt images go here
  labeled/            # Teacher-labeled (image, recipe) pairs
notebooks/            # Colab training notebook
tests/                # pytest tests (30 tests)
checkpoints/          # Training checkpoints (gitignored)
outputs/              # Final LoRA adapters + eval results (gitignored)
```

## Recommended Workflow

1. Start small: add ~10 receipt images, label them, inspect the output
2. If quality looks good, scale to all your receipts
3. Train for 1 epoch, check outputs manually
4. Full training (3 epochs) once you're satisfied
5. Evaluate quantitatively, then DPO refinement if needed
