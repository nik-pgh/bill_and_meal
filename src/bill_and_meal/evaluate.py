"""Evaluation: metrics, qualitative checks, Claude-as-judge."""

import json
import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
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


def _extract_ingredients(text: str, known_ingredients: list[str]) -> set[str]:
    """Return the set of known_ingredients that appear (case-insensitive) in text."""
    text_lower = text.lower()
    return {ing.lower() for ing in known_ingredients if ing.lower() in text_lower}


def ingredient_iou(
    teacher_text: str,
    student_text: str,
    known_ingredients: list[str],
) -> float:
    """Jaccard similarity between teacher and student ingredient mentions.

    Reference: Jaccard Similarity of Ingredients.
    """
    set_t = _extract_ingredients(teacher_text, known_ingredients)
    set_s = _extract_ingredients(student_text, known_ingredients)

    if not set_t and not set_s:
        return 1.0

    intersection = len(set_t & set_s)
    union = len(set_t | set_s)
    return intersection / union


COOKING_VERBS = frozenset({
    "fry", "boil", "saute", "chop", "mix", "steam", "bake", "roast", "grill", "stew",
    "볶다", "삶다", "썰다", "섞다", "찌다", "굽다", "졸이다", "무치다", "튀기다",
})


def _action_pairs(text: str, known_ingredients: list[str]) -> set[tuple[str, str]]:
    """Return (ingredient, verb) pairs co-occurring in the same sentence."""
    pairs: set[tuple[str, str]] = set()
    for sentence in re.split(r"[.!?\n]", text):
        sent_low = sentence.lower()
        ings = [i.lower() for i in known_ingredients if i.lower() in sent_low]
        verbs = [v for v in COOKING_VERBS if v in sent_low]
        for ing in ings:
            for verb in verbs:
                pairs.add((ing, verb))
    return pairs


def action_ingredient_alignment(
    teacher_text: str,
    student_text: str,
    known_ingredients: list[str],
) -> float:
    """Recall of (cooking_verb, ingredient) pairs from teacher reproduced by student.

    Reference: Action-Entity Pair Matching.
    """
    pairs_t = _action_pairs(teacher_text, known_ingredients)
    pairs_s = _action_pairs(student_text, known_ingredients)

    if not pairs_t:
        return 1.0 if not pairs_s else 0.0

    return len(pairs_t & pairs_s) / len(pairs_t)


def recipe_sequence_similarity(teacher_text: str, student_text: str) -> float:
    """Sequence similarity of step lists, split by '1.', '2.', etc.

    Reference: Sequential Structural Similarity.
    """
    t_steps = [s.strip() for s in re.split(r"\d+\.|\n-", teacher_text) if len(s.strip()) > 5]
    s_steps = [s.strip() for s in re.split(r"\d+\.|\n-", student_text) if len(s.strip()) > 5]

    if not t_steps:
        return 0.0
    return SequenceMatcher(None, t_steps, s_steps).ratio()


def step_density_ratio(teacher_text: str, student_text: str) -> float:
    """How closely student's instruction count matches teacher's. 1.0 = identical density.

    Reference: Instruction Density Comparison.
    """
    t_count = len(re.findall(r"\d+\.", teacher_text))
    s_count = len(re.findall(r"\d+\.", student_text))

    if t_count == 0:
        return 1.0 if s_count == 0 else 0.0

    ratio = s_count / t_count
    return ratio if ratio <= 1.0 else 1.0 / ratio


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
