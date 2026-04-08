# Bill&Meal: Knowledge Distillation Training Pipeline

## Architecture overview

```
Receipt Image → [Teacher: Claude API] → High-quality recipes (dataset)
Receipt Image → [Student: PaliGemma/LLaVA] → Learned recipes (trained)
```

The goal: use Claude as a teacher to generate thousands of (receipt image, recipe) pairs, then train a smaller vision-language model to replicate that ability locally.

---

## Phase 1: Synthetic receipt dataset

You need thousands of receipt images. Real receipts are hard to collect at scale, so you'll generate synthetic ones.

### Step 1.1 — Gather ingredient pools

```python
# Pull real ingredient lists from RecipeNLG
from datasets import load_dataset

dataset = load_dataset("recipe_nlg", split="train")

# Extract unique ingredients across recipes
all_ingredients = set()
for recipe in dataset:
    all_ingredients.update(recipe["NER"])  # pre-extracted entities

# Group by category for realistic receipts
categories = {
    "protein": ["chicken breast", "ground beef", "salmon fillet", ...],
    "produce": ["garlic", "onion", "lemon", "spinach", ...],
    "dairy": ["butter", "heavy cream", "parmesan", ...],
    "pantry": ["olive oil", "rice", "pasta", "flour", ...],
}
```

### Step 1.2 — Generate synthetic receipt images

```python
from PIL import Image, ImageDraw, ImageFont
import random

def generate_receipt(ingredients, store_name="FreshMart"):
    """Creates a receipt-style image from a list of ingredients."""
    width = 400
    margin = 20
    line_height = 28
    height = 120 + len(ingredients) * line_height + 80

    img = Image.new("RGB", (width, height), "#FAFAF5")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("courier.ttf", 16)  # monospace for receipt feel
    bold_font = ImageFont.truetype("courier-bold.ttf", 18)

    y = margin
    # Store header
    draw.text((width // 2, y), store_name, font=bold_font, anchor="mt", fill="#222")
    y += 40
    draw.text((width // 2, y), "123 Main St, Anytown", font=font, anchor="mt", fill="#555")
    y += 30
    draw.line([(margin, y), (width - margin, y)], fill="#AAA")
    y += 15

    total = 0
    for item in ingredients:
        qty = random.randint(1, 4)
        price = round(random.uniform(0.99, 12.99), 2)
        total += price * qty

        item_text = f"{item[:22]:<22}"
        price_text = f"{qty}x  ${price:.2f}"
        draw.text((margin, y), item_text, font=font, fill="#222")
        draw.text((width - margin, y), price_text, font=font, anchor="ra", fill="#222")
        y += line_height

    # Total
    y += 10
    draw.line([(margin, y), (width - margin, y)], fill="#AAA")
    y += 15
    draw.text((margin, y), "TOTAL", font=bold_font, fill="#222")
    draw.text((width - margin, y), f"${total:.2f}", font=bold_font, anchor="ra", fill="#222")

    return img

# Generate a batch
def generate_receipt_batch(n=5000):
    receipts = []
    for i in range(n):
        # Pick 5-15 random ingredients (simulates a grocery trip)
        n_items = random.randint(5, 15)
        items = []
        for cat in categories:
            k = random.randint(1, min(4, len(categories[cat])))
            items.extend(random.sample(categories[cat], k))
        items = items[:n_items]

        img = generate_receipt(items)
        img.save(f"receipts/receipt_{i:05d}.png")
        receipts.append({"id": i, "image_path": f"receipts/receipt_{i:05d}.png", "ingredients": items})
    return receipts
```

### Step 1.3 — Add noise for realism (optional but recommended)

```python
import numpy as np
from PIL import ImageFilter

def add_receipt_noise(img):
    """Makes synthetic receipts look more like camera photos."""
    # Slight rotation (as if photo wasn't perfectly aligned)
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, fillcolor="#F5F5F0", expand=True)

    # Slight blur (camera focus)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add subtle noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)
```

---

## Phase 2: Teacher labeling with Claude

This is the core of distillation — using Claude to generate high-quality recipe outputs for each receipt.

### Step 2.1 — Design the teacher prompt

```python
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
```

### Step 2.2 — Batch label with Claude API

```python
import anthropic
import base64
import json
import time

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_teacher_label(image_path, retries=3):
    """Get Claude's recipe output for a receipt image."""
    b64 = encode_image(image_path)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=TEACHER_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64
                        }},
                        {"type": "text", "text": "Here is my grocery receipt. What can I cook?"}
                    ]
                }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

def label_full_dataset(receipts, output_path="labeled_dataset.jsonl"):
    """Label all receipts with teacher outputs."""
    with open(output_path, "a") as f:
        for i, receipt in enumerate(receipts):
            print(f"Labeling {i+1}/{len(receipts)}...")
            teacher_output = get_teacher_label(receipt["image_path"])

            if teacher_output:
                record = {
                    "id": receipt["id"],
                    "image_path": receipt["image_path"],
                    "ingredients": receipt["ingredients"],
                    "teacher_output": teacher_output
                }
                f.write(json.dumps(record) + "\n")

            # Rate limiting: ~50 requests/min for Sonnet
            time.sleep(1.2)
```

### Step 2.3 — Quality filtering

```python
def validate_teacher_output(record):
    """Filter out bad teacher outputs."""
    output = record["teacher_output"]

    # Must contain at least one recipe
    if "RECIPE 1:" not in output:
        return False

    # Must reference actual receipt ingredients
    mentioned = sum(1 for ing in record["ingredients"] if ing.lower() in output.lower())
    if mentioned < len(record["ingredients"]) * 0.3:
        return False  # Uses less than 30% of receipt items

    # Must have steps
    if "STEPS:" not in output:
        return False

    return True

# Filter dataset
import json
with open("labeled_dataset.jsonl") as f:
    records = [json.loads(line) for line in f]

clean_records = [r for r in records if validate_teacher_output(r)]
print(f"Kept {len(clean_records)}/{len(records)} records after filtering")
```

---

## Phase 3: Student model setup

### Step 3.1 — Choose your student

| Model | Size | VRAM needed | Best for |
|---|---|---|---|
| PaliGemma-3B | 3B | ~12GB (QLoRA) | Consumer GPU, fastest experiments |
| LLaVA-1.6-7B | 7B | ~16GB (QLoRA) | Better quality, still accessible |
| LLaVA-1.6-13B | 13B | ~24GB (QLoRA) | Best quality, needs beefy GPU |
| Idefics2-8B | 8B | ~18GB (QLoRA) | Native HuggingFace integration |

### Step 3.2 — Install dependencies

```bash
pip install torch torchvision transformers accelerate peft trl
pip install bitsandbytes datasets pillow wandb
```

### Step 3.3 — Load model with quantization

```python
import torch
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)

# 4-bit quantization config (fits on 12GB GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "google/paligemma-3b-pt-224"

processor = AutoProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Step 3.4 — Attach LoRA adapters

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare for QLoRA training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more VRAM
    lora_alpha=32,                 # scaling factor
    lora_dropout=0.05,
    target_modules=[               # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"        # MLP
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~10M / total: 3B (~0.3%)
```

---

## Phase 4: Training

### Step 4.1 — Build the dataset class

```python
from torch.utils.data import Dataset
from PIL import Image
import json

class ReceiptRecipeDataset(Dataset):
    def __init__(self, data_path, processor, max_length=1024):
        with open(data_path) as f:
            self.records = [json.loads(line) for line in f]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")

        # Format: the prompt is the question, the label is the teacher's answer
        prompt = "What recipes can I make from this grocery receipt?"
        answer = record["teacher_output"]

        # Process image + text together
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Create labels (mask the prompt tokens with -100 so loss is only on the answer)
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

# Create splits
full_data = ReceiptRecipeDataset("labeled_dataset_clean.jsonl", processor)

train_size = int(0.9 * len(full_data))
val_size = len(full_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size, val_size])
```

### Step 4.2 — Configure training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./bill_and_meal_checkpoints",

    # Core training params
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 4 × 4 = 16

    # Optimizer
    learning_rate=2e-4,                  # standard for LoRA
    weight_decay=0.01,
    warmup_ratio=0.05,                   # 5% warmup
    lr_scheduler_type="cosine",

    # Memory optimization
    fp16=True,                           # or bf16=True if GPU supports it
    gradient_checkpointing=True,         # trades compute for memory

    # Logging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    # Monitoring
    report_to="wandb",                   # track loss curves, samples
    run_name="bill_meal_distill_v1",

    # Best model selection
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

### Step 4.3 — Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {
        k: torch.stack([d[k] for d in data]) for k in data[0]
    },
)

# Launch training
trainer.train()

# Save the final LoRA adapter
model.save_pretrained("./bill_and_meal_final")
processor.save_pretrained("./bill_and_meal_final")
```

### Step 4.4 — What to watch during training

Key metrics in your Weights & Biases dashboard:

- **train_loss** — should steadily decrease. If it plateaus early, increase learning rate or LoRA rank.
- **eval_loss** — should track train_loss. If gap widens, you're overfitting (reduce epochs, increase dropout, add more data).
- **learning rate curve** — cosine schedule should show smooth warmup then decay.

Common issues:
- **Loss spikes** → reduce learning rate
- **OOM errors** → reduce batch size, enable gradient checkpointing, lower max_length
- **Output is gibberish** → learning rate too high, or LoRA targets wrong layers
- **Output copies prompt instead of generating** → label masking is wrong (check -100 placement)

---

## Phase 5: Evaluation

### Step 5.1 — Qualitative checks (do this first)

```python
def generate_recipe(image_path, model, processor):
    """Run inference on a single receipt."""
    image = Image.open(image_path).convert("RGB")
    prompt = "What recipes can I make from this grocery receipt?"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return processor.decode(output[0], skip_special_tokens=True)

# Test on held-out receipts
for test_image in test_images[:10]:
    print(f"\n{'='*60}")
    print(f"Receipt: {test_image}")
    print(generate_recipe(test_image, model, processor))
```

Check manually:
- Does it correctly read items from the receipt?
- Are the recipes actually possible with those ingredients?
- Are instructions coherent and followable?
- Does it hallucinate ingredients not on the receipt?

### Step 5.2 — Quantitative metrics

```python
# 1. Ingredient coverage: what % of receipt items are used in suggested recipes?
def ingredient_coverage(receipt_items, generated_text):
    used = sum(1 for item in receipt_items if item.lower() in generated_text.lower())
    return used / len(receipt_items)

# 2. Hallucination rate: does it mention ingredients NOT on the receipt?
def hallucination_rate(receipt_items, generated_text, all_known_ingredients):
    receipt_set = set(i.lower() for i in receipt_items)
    mentioned = set()
    for ing in all_known_ingredients:
        if ing.lower() in generated_text.lower():
            mentioned.add(ing.lower())
    hallucinated = mentioned - receipt_set
    return len(hallucinated) / max(len(mentioned), 1)

# 3. Teacher similarity: how close is student output to teacher output?
# Use ROUGE or BERTScore
from evaluate import load
bertscore = load("bertscore")

results = bertscore.compute(
    predictions=[student_output],
    references=[teacher_output],
    lang="en"
)
# F1 > 0.85 means student is closely matching teacher quality
```

### Step 5.3 — Side-by-side comparison

```python
# Generate from both teacher and student for same receipts
comparisons = []
for receipt in test_set[:50]:
    teacher_out = get_teacher_label(receipt["image_path"])   # Claude API
    student_out = generate_recipe(receipt["image_path"], model, processor)
    comparisons.append({
        "receipt": receipt["ingredients"],
        "teacher": teacher_out,
        "student": student_out,
    })

# Use Claude as judge (automated evaluation)
JUDGE_PROMPT = """Compare these two recipe suggestions for the same grocery receipt.
Rate each on: ingredient accuracy (1-5), creativity (1-5), practicality (1-5).

Receipt items: {ingredients}

Response A (Teacher): {teacher}
Response B (Student): {student}

Provide scores as JSON: {{"teacher": {{"accuracy": X, "creativity": X, "practicality": X}},
"student": {{"accuracy": X, "creativity": X, "practicality": X}}}}"""
```

---

## Phase 6: Optional — DPO refinement

Once SFT via distillation is working, improve quality with preference learning.

### Step 6.1 — Generate preference pairs

```python
# For each receipt, generate 2 recipes with different temperatures
def generate_pair(image_path, model, processor):
    recipe_a = generate_recipe(image_path, model, processor)  # temp=0.7
    recipe_b = generate_recipe(image_path, model, processor)  # temp=1.0 (more creative/risky)
    return recipe_a, recipe_b

# Have Claude rank them
RANK_PROMPT = """Given this grocery list: {ingredients}

Which recipe suggestion is better? Consider: uses more receipt ingredients,
clearer instructions, more practical for a home cook.

Option A: {recipe_a}
Option B: {recipe_b}

Reply with ONLY "A" or "B"."""
```

### Step 6.2 — DPO training

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./bill_meal_dpo",
    num_train_epochs=1,              # DPO needs fewer epochs
    per_device_train_batch_size=2,
    learning_rate=5e-5,              # lower than SFT
    beta=0.1,                        # KL penalty strength
)

# dpo_dataset should have columns: prompt, chosen, rejected
dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    processing_class=processor,
)

dpo_trainer.train()
```

---

## Cost and resource estimates

| Item | Estimate |
|---|---|
| Claude API for 5,000 receipts | ~$15-25 (Sonnet, ~800 tokens/response) |
| Claude API for judging (DPO) | ~$5-10 |
| GPU for training (A100 40GB) | ~4-8 hours for SFT, ~1-2 for DPO |
| Cloud GPU rental | ~$5-15 total (Lambda, RunPod, etc.) |
| Total project cost | ~$25-50 |

## File structure

```
bill_and_meal/
├── data/
│   ├── receipts/              # generated receipt images
│   ├── labeled_dataset.jsonl  # teacher-labeled pairs
│   └── dpo_pairs.jsonl        # preference pairs for DPO
├── scripts/
│   ├── generate_receipts.py   # Phase 1
│   ├── teacher_labeling.py    # Phase 2
│   ├── train_student.py       # Phase 4
│   └── evaluate.py            # Phase 5
├── configs/
│   └── training_config.yaml
├── checkpoints/               # saved during training
└── bill_and_meal_final/       # final LoRA adapter
```

## Recommended order of execution

1. Generate 500 receipts first (small test run)
2. Label 500 with Claude, validate quality
3. Train student for 1 epoch, check outputs manually
4. If outputs make sense: scale to 5,000 receipts, full training
5. Evaluate quantitatively
6. DPO refinement if quality gap remains
