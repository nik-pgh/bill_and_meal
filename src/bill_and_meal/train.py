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
