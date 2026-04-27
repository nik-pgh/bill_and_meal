"""Student model loading with config-driven model selection and QLoRA setup."""

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model

# Standard LoRA targets for transformer attention + MLP layers (PaliGemma,
# LLaVA, and other models with plain nn.Linear projections).
_STD_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Gemma 4 dual-target: Language model uses plain Linear4bit at
# `language_model.layers.X.self_attn.q_proj`, while vision_tower wraps each
# projection in Gemma4ClippableLinear with the actual Linear4bit at the inner
# `.linear` attribute. The regex matches both:
#   - LLM paths: language_model.*.{q,k,v,o,gate,up,down}_proj
#   - Vision paths: vision_tower.*.{q,k,v,o,gate,up,down}_proj.linear
# PEFT uses re.fullmatch on the module name. Tuning both halves lets the
# vision encoder learn receipt-specific representations and the LLM learn the
# recipe-format response — earlier LLM-only training produced format-correct
# but visually-ungrounded outputs ("Chicken and Rice" regardless of receipt).
_GEMMA4_LORA_TARGETS = (
    r".*language_model\..*\.(q|k|v|o|gate|up|down)_proj"
    r"|.*vision_tower\..*\.(q|k|v|o|gate|up|down)_proj\.linear"
)

MODELS = {
    "paligemma-3b": {
        "hf_id": "google/paligemma-3b-pt-224",
        "model_class": PaliGemmaForConditionalGeneration,
        "target_modules": _STD_LORA_TARGETS,
    },
    "llava-7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "target_modules": _STD_LORA_TARGETS,
    },
    "gemma-4-e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "model_class": AutoModelForImageTextToText,
        "target_modules": _GEMMA4_LORA_TARGETS,
    },
    "gemma-4-e2b": {
        "hf_id": "google/gemma-4-E2B-it",
        "model_class": AutoModelForImageTextToText,
        "target_modules": _GEMMA4_LORA_TARGETS,
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
    # low_cpu_mem_usage + bf16 dtype keeps the loader from materializing the
    # full model in fp32 in RAM before quantization — Colab free-tier T4 has
    # only ~12GB system RAM, which OOM-kills the kernel without these flags.
    model = model_info["model_class"].from_pretrained(
        model_info["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
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

    # Manual minimal kbit prep — skip prepare_model_for_kbit_training because
    # its fp32 upcast of non-quantized params (Gemma's 256K vocab embeddings
    # are ~3GB in fp32) OOMs on T4 (14.56GB) when the 4-bit base already takes
    # ~8GB. The fp32 upcast is a QLoRA stability recommendation, not a
    # requirement — bf16 compute_dtype (set in BitsAndBytesConfig) is fine.
    # Gradient checkpointing is enabled by TrainingArguments, not here.
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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
