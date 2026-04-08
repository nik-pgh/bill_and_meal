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
