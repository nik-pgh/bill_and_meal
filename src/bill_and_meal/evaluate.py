"""Evaluation: metrics, qualitative checks, Claude-as-judge."""

import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)


def ingredient_coverage(receipt_items: list[str], generated_text: str) -> float:
    """Calculate what fraction of receipt items appear in generated text.

    Args:
        receipt_items: List of ingredient names from the receipt.
        generated_text: Model-generated recipe text.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not receipt_items:
        return 0.0
    text_lower = generated_text.lower()
    used = sum(1 for item in receipt_items if item.lower() in text_lower)
    return used / len(receipt_items)


def hallucination_rate(
    receipt_items: list[str],
    generated_text: str,
    known_ingredients: list[str],
) -> float:
    """Calculate what fraction of mentioned ingredients are NOT on the receipt.

    Args:
        receipt_items: Ingredients from the receipt.
        generated_text: Model-generated recipe text.
        known_ingredients: Full list of known ingredient names to check against.

    Returns:
        Float between 0.0 and 1.0. 0.0 means no hallucination.
    """
    text_lower = generated_text.lower()
    receipt_set = {i.lower() for i in receipt_items}

    mentioned = {
        ing.lower()
        for ing in known_ingredients
        if ing.lower() in text_lower
    }

    if not mentioned:
        return 0.0

    hallucinated = mentioned - receipt_set
    return len(hallucinated) / len(mentioned)


JUDGE_PROMPT = """Compare these two recipe suggestions for the same grocery receipt.
Rate each on: ingredient accuracy (1-5), creativity (1-5), practicality (1-5).

Receipt items: {ingredients}

Response A (Teacher): {teacher}
Response B (Student): {student}

Provide scores as JSON: {{"teacher": {{"accuracy": X, "creativity": X, "practicality": X}},
"student": {{"accuracy": X, "creativity": X, "practicality": X}}}}"""


def judge_comparison(
    ingredients: list[str],
    teacher_output: str,
    student_output: str,
    config: dict,
) -> dict | None:
    """Use Claude to judge teacher vs student outputs.

    Args:
        ingredients: Receipt ingredient list.
        teacher_output: Teacher-generated recipe text.
        student_output: Student-generated recipe text.
        config: Loaded config dict (needs config["teacher"] for model).

    Returns:
        Parsed JSON scores dict, or None on failure.
    """
    client = anthropic.Anthropic()
    prompt = JUDGE_PROMPT.format(
        ingredients=", ".join(ingredients),
        teacher=teacher_output,
        student=student_output,
    )

    try:
        response = client.messages.create(
            model=config["teacher"]["model"],
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        logger.warning("Judge call failed: %s", e)
        return None


def save_eval_results(results: dict, output_dir: Path) -> Path:
    """Save evaluation results with timestamp.

    Args:
        results: Dict of evaluation results.
        output_dir: Directory to save to.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"eval_results_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
