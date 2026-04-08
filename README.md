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
