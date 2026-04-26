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


def _parse_uses(output: str) -> list[str]:
    """Extract ingredients from all 'USES:' lines in the teacher output.

    Each recipe block contains a 'USES: a, b, c' line listing receipt
    ingredients used. Aggregate them across all recipes, deduped.
    """
    seen: dict[str, None] = {}
    for line in output.split("\n"):
        stripped = line.strip()
        if not stripped.upper().startswith("USES:"):
            continue
        items = stripped.split(":", 1)[1].split(",")
        for item in items:
            cleaned = item.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
    return list(seen)


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
                "ingredients": _parse_uses(teacher_output),
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
