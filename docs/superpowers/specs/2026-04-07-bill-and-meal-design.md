# Bill&Meal: Knowledge Distillation Training Pipeline — Design Spec

**Date:** 2026-04-07
**Status:** Approved

---

## Overview

Bill&Meal trains a small, local vision-language model (PaliGemma-3B or LLaVA-7B) to suggest recipes from grocery receipt images. Claude acts as the teacher — generating high-quality (receipt image, recipe) pairs — and the student model learns to replicate that ability via supervised fine-tuning with QLoRA.

### Key Decisions

| Decision | Choice |
|---|---|
| Synthetic receipts | Skipped — real receipt images |
| Compute targets | NVIDIA RTX 4080 (local) + Google Colab |
| Student models | PaliGemma-3B + LLaVA-7B, config-driven selection |
| Data management | Local folder + JSONL manifest |
| Config strategy | Separate YAML per environment (`local.yaml`, `colab.yaml`) |
| Dependencies | pip + requirements.txt |
| Experiment tracking | Weights & Biases |
| Secrets | `.env` file with python-dotenv |

---

## Project Structure

```
bill_and_meal/
├── src/
│   └── bill_and_meal/
│       ├── __init__.py           # version, package metadata
│       ├── config.py             # load YAML configs, env detection, .env loading
│       ├── data.py               # manifest management, ReceiptRecipeDataset class
│       ├── teacher.py            # Claude API labeling + quality filtering
│       ├── student.py            # model loading (PaliGemma/LLaVA), LoRA setup
│       ├── train.py              # Trainer setup, training loop
│       └── evaluate.py           # metrics (coverage, hallucination, BERTScore, judge)
├── scripts/
│   ├── label_receipts.py         # CLI: run teacher labeling on receipt images
│   ├── train.py                  # CLI: launch training run
│   └── evaluate.py               # CLI: run evaluation suite
├── configs/
│   ├── local.yaml                # RTX 4080: paths, batch sizes, 4-bit quant
│   └── colab.yaml                # Colab: Google Drive paths, memory-safe settings
├── data/
│   ├── receipts/                 # real receipt images (gitignored)
│   ├── manifest.jsonl            # image index with labeling status
│   └── labeled/
│       └── labeled_dataset.jsonl # teacher-labeled (image, recipe) pairs
├── notebooks/
│   └── colab_training.ipynb      # Colab notebook importing from package
├── checkpoints/                  # training checkpoints (gitignored)
├── outputs/                      # final LoRA adapters + eval results (gitignored)
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_teacher.py
│   └── test_evaluate.py
├── .env.example                  # template for API keys
├── .gitignore
├── requirements.txt
├── setup.py                      # editable install: pip install -e .
└── README.md
```

---

## Configuration System

### Auto-detection

`config.py` auto-detects the environment by checking for `/content/drive` or the `COLAB_GPU` environment variable. Loads the matching YAML file and merges with `.env` for secrets.

### `configs/local.yaml` (RTX 4080, 16GB VRAM)

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
  model: paligemma-3b               # short key, maps to HF ID via model registry
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
    dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  max_length: 1024
  fp16: true
  gradient_checkpointing: true
  checkpoint_dir: ./checkpoints
  output_dir: ./outputs

wandb:
  project: bill-and-meal
  run_name_prefix: local
```

### `configs/colab.yaml` (differences from local)

```yaml
environment: colab
data:
  receipts_dir: /content/drive/MyDrive/bill_and_meal/receipts
  manifest_path: /content/drive/MyDrive/bill_and_meal/manifest.jsonl
  labeled_path: /content/drive/MyDrive/bill_and_meal/labeled_dataset.jsonl

student:
  model: paligemma-3b
  quantization: 4bit

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  checkpoint_dir: /content/drive/MyDrive/bill_and_meal/checkpoints
  output_dir: /content/drive/MyDrive/bill_and_meal/outputs

wandb:
  run_name_prefix: colab
```

---

## Data Pipeline

### Manifest (`data/manifest.jsonl`)

One line per receipt image:
```json
{"id": "receipt_001", "filename": "receipt_001.jpg", "added_at": "2026-04-07", "labeled": false}
```

### `data.py` Functions

- **`scan_receipts(config)`** — scans `receipts/` dir, adds new images to manifest, preserves existing entries
- **`get_unlabeled(config)`** — returns manifest entries where `labeled` is false
- **`mark_labeled(id, config)`** — sets `labeled: true` in manifest

### ReceiptRecipeDataset (PyTorch Dataset)

- Reads `labeled_dataset.jsonl`
- Loads image + teacher output per record
- Processes through the student model's processor (model-specific)
- Handles label masking: prompt tokens set to `-100`, loss computed only on recipe output
- Train/val split via `torch.utils.data.random_split` (90/10)

### Quality Filtering (in `teacher.py`)

A record passes if:
- Contains at least one `RECIPE 1:` block
- References >= 30% of identified receipt ingredients
- Contains a `STEPS:` section

---

## Teacher Labeling Pipeline

### `teacher.py` Functions

- **`encode_image(path)`** — base64 encodes receipt image for Claude API
- **`get_teacher_label(image_path, config)`** — sends image to Claude, returns recipe output. Retries up to 3 times with exponential backoff.
- **`validate_teacher_output(record)`** — quality filter (see above)
- **`label_batch(config)`** — orchestrates full labeling flow:
  1. Reads manifest for unlabeled images
  2. Calls Claude for each
  3. Validates output
  4. Writes passing records to `labeled_dataset.jsonl`
  5. Updates manifest
  6. Respects rate limit delay
  7. Logs progress and skip reasons

### Resumability

Labeling is resumable. Already-labeled images are marked in the manifest and skipped on re-run.

### CLI (`scripts/label_receipts.py`)

```
python scripts/label_receipts.py                    # labels all unlabeled
python scripts/label_receipts.py --limit 10         # label 10 images (test run)
python scripts/label_receipts.py --revalidate       # re-filter existing labels
```

### Teacher Prompt

Stored as a constant in `teacher.py`. Instructs Claude to:
- Identify food items on the receipt
- Suggest 2-3 recipes using ONLY receipt ingredients (plus basic pantry staples)
- Format as structured blocks: recipe name, ingredients used, steps, time, difficulty

---

## Student Model & Training

### `student.py` Functions

- **`load_model(config)`** — loads configured model with 4-bit quantization
- **`attach_lora(model, config)`** — attaches QLoRA adapters with config-driven parameters

### Model Registry

```python
MODELS = {
    "paligemma-3b": {
        "hf_id": "google/paligemma-3b-pt-224",
        "model_class": PaliGemmaForConditionalGeneration,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "llava-7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}
```

Adding a new model means adding one entry to this dict.

### `train.py` (package module)

- **`build_trainer(model, processor, train_dataset, val_dataset, config)`** — constructs HuggingFace `Trainer` with `TrainingArguments` from config
- **`run_training(config)`** — full pipeline: load config, load model, attach LoRA, build dataset, build trainer, train, save final adapter

### CLI (`scripts/train.py`)

```
python scripts/train.py                              # auto-detected env
python scripts/train.py --config configs/local.yaml   # explicit config
python scripts/train.py --model llava-7b              # override model
```

---

## Evaluation

### Three Tiers

1. **Qualitative** — generate recipes for held-out receipts, print for manual inspection
2. **Quantitative metrics:**
   - `ingredient_coverage` — % of receipt items mentioned in output
   - `hallucination_rate` — % of mentioned ingredients NOT on receipt
   - `teacher_similarity` — BERTScore F1 between student and teacher output
3. **Claude-as-judge** — sends teacher + student outputs to Claude, returns accuracy/creativity/practicality scores (1-5 each)

### CLI (`scripts/evaluate.py`)

```
python scripts/evaluate.py --qualitative --n 5
python scripts/evaluate.py --metrics
python scripts/evaluate.py --judge --n 20
python scripts/evaluate.py --all
```

Results saved to `outputs/eval_results.json` with timestamp.

---

## DPO Refinement (Future)

Not built in initial scaffolding. Reserved slots:
- `data/dpo_pairs.jsonl` path in config
- `src/bill_and_meal/dpo.py` module (future)
- `scripts/dpo_train.py` (future)

---

## Colab Notebook

**`notebooks/colab_training.ipynb`** — thin wrapper around the package:

1. **Setup** — install package, mount Google Drive, load colab config
2. **Verify data** — check receipts and labeled dataset on Drive
3. **Train** — `from bill_and_meal.train import run_training`
4. **Evaluate** — run qualitative + metrics
5. **Export** — copy final adapter

All logic lives in the package. The notebook just calls it.

---

## Dependencies (`requirements.txt`)

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
```

---

## Cost Estimates

| Item | Estimate |
|---|---|
| Claude API for labeling (varies with receipt count) | ~$0.003-0.005 per receipt |
| Claude API for judging | ~$5-10 for 500 evaluations |
| GPU training (local) | Free (own hardware) |
| GPU training (Colab) | Free tier or ~$10/month Pro |
