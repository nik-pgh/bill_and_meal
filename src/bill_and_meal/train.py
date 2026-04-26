"""Training orchestration: dataset construction, Trainer setup, training loop."""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from bill_and_meal.config import load_config
from bill_and_meal.evaluate import (
    action_ingredient_alignment,
    ingredient_iou,
    recipe_sequence_similarity,
)
from bill_and_meal.student import attach_lora, load_model

logger = logging.getLogger(__name__)

PROMPT = "What recipes can I make from this grocery receipt?"
SPLIT_SEED = 42


class ReceiptRecipeDataset(Dataset):
    """PyTorch Dataset for receipt-recipe pairs.

    Builds a chat-templated input where:
    - input_ids = tokenize(user_msg + assistant_msg)
    - labels    = input_ids with prompt and PAD tokens masked to -100,
                  so loss is computed only on the assistant answer.
    """

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
        answer = record["teacher_output"]

        user_msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }]
        full_msg = user_msg + [{"role": "assistant", "content": answer}]

        full = self.processor.apply_chat_template(
            full_msg,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        prompt_only = self.processor.apply_chat_template(
            user_msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = min(prompt_only["input_ids"].shape[-1], self.max_length)

        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        inputs = {k: v.squeeze(0) for k, v in full.items()}
        inputs["labels"] = labels.squeeze(0)
        return inputs


def _build_compute_metrics(processor, known_ingredients: list[str]):
    """Return a compute_metrics callable bound to the processor + ingredient list."""
    pad_id = processor.tokenizer.pad_token_id or 0

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, pad_id)

        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        iou_scores = []
        action_scores = []
        seq_scores = []
        for ref, hyp in zip(decoded_labels, decoded_preds):
            iou_scores.append(ingredient_iou(ref, hyp, known_ingredients))
            action_scores.append(action_ingredient_alignment(ref, hyp, known_ingredients))
            seq_scores.append(recipe_sequence_similarity(ref, hyp))

        return {
            "ingredient_iou": sum(iou_scores) / len(iou_scores),
            "action_alignment": sum(action_scores) / len(action_scores),
            "sequence_similarity": sum(seq_scores) / len(seq_scores),
        }

    return compute_metrics


def build_trainer(
    model,
    processor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
) -> Seq2SeqTrainer:
    """Construct a HuggingFace Seq2SeqTrainer from config.

    Args:
        model: Model with LoRA adapters.
        processor: Model processor/tokenizer.
        train_dataset: Training split.
        val_dataset: Validation split.
        config: Loaded config dict.

    Returns:
        Configured Seq2SeqTrainer instance.
    """
    t = config["training"]
    w = config["wandb"]
    known_ingredients = config["data"].get("known_ingredients", [])

    use_bf16 = t.get("bf16", False)
    use_fp16 = t.get("fp16", False) and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=t["checkpoint_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.get("logging_steps", 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to=t.get("report_to", "wandb"),
        run_name=f"{w['run_name_prefix']}_{config['student']['model']}",
        predict_with_generate=True,
        generation_max_length=t.get("max_length", 1024),
        load_best_model_at_end=True,
        metric_for_best_model="ingredient_iou",
        greater_is_better=True,
        seed=SPLIT_SEED,
        remove_unused_columns=False,
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_build_compute_metrics(processor, known_ingredients),
        data_collator=lambda data: {
            k: torch.stack([d[k] for d in data]) for k in data[0]
        },
    )


def extract_known_ingredients(labeled_path: Path) -> list[str]:
    """Build a known-ingredients vocabulary from the labeled dataset.

    Prefers the per-record ingredient list when available (preserves multi-word
    items like 'red bell pepper'); falls back to extracting English words from
    the teacher output.
    """
    with open(labeled_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    vocab: set[str] = set()
    for r in records:
        ingredients = r.get("ingredients")
        if isinstance(ingredients, list) and ingredients:
            for item in ingredients:
                cleaned = re.sub(r"[^a-zA-Z\s]", "", item).strip().lower()
                if cleaned:
                    vocab.add(cleaned)
            continue

        teacher_text = r.get("teacher_output", "")
        for word in re.findall(r"\b[a-zA-Z]{2,}\b", teacher_text):
            vocab.add(word.lower())

    return sorted(vocab)


def _setup_wandb_env(config: dict) -> None:
    """Set WANDB_PROJECT before Trainer auto-inits wandb so runs land in the right project."""
    if config["training"].get("report_to", "wandb") != "wandb":
        return
    os.environ.setdefault("WANDB_PROJECT", config["wandb"]["project"])


def run_training(config: dict | None = None) -> None:
    """Full training pipeline: load model, build dataset, train, save.

    Args:
        config: Config dict. If None, auto-loads from environment.
    """
    if config is None:
        config = load_config()

    _setup_wandb_env(config)

    labeled_path = Path(config["data"]["labeled_path"])
    known_ingredients = extract_known_ingredients(labeled_path)
    config["data"]["known_ingredients"] = known_ingredients
    logger.info("Identified %d ingredient terms for metric computation",
                len(known_ingredients))

    logger.info("Loading model: %s", config["student"]["model"])
    model, processor = load_model(config)
    model = attach_lora(model, config)

    logger.info("Building dataset from %s", labeled_path)
    full_dataset = ReceiptRecipeDataset(
        labeled_path,
        processor,
        max_length=config["training"]["max_length"],
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )

    logger.info("Training: %d train, %d val", len(train_dataset), len(val_dataset))
    trainer = build_trainer(model, processor, train_dataset, val_dataset, config)
    trainer.train()

    output_dir = config["training"]["output_dir"]
    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
