"""CLI: Quick stats and random-sample inspection of labeled receipts."""

import argparse
import json
import logging
import random
import statistics
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_stats(records: list[dict]) -> None:
    """Print aggregate statistics about the labeled dataset."""
    if not records:
        logger.info("No records found.")
        return

    n = len(records)
    ingredient_counts = [len(r["ingredients"]) for r in records]
    output_lengths = [len(r["teacher_output"]) for r in records]
    recipe_counts = [r["teacher_output"].count("RECIPE ") for r in records]

    all_ingredients = [ing for r in records for ing in r["ingredients"]]
    top_ingredients = Counter(ing.lower() for ing in all_ingredients).most_common(15)

    logger.info("=" * 60)
    logger.info("LABEL STATISTICS (%d records)", n)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Ingredients per record:")
    logger.info("  min/median/max: %d / %d / %d",
                min(ingredient_counts), statistics.median(ingredient_counts), max(ingredient_counts))
    logger.info("  zero-ingredient records: %d", sum(1 for c in ingredient_counts if c == 0))
    logger.info("")
    logger.info("Output length (chars):")
    logger.info("  min/median/max: %d / %d / %d",
                min(output_lengths), statistics.median(output_lengths), max(output_lengths))
    logger.info("")
    logger.info("Recipes per record:")
    logger.info("  distribution: %s", dict(Counter(recipe_counts)))
    logger.info("")
    logger.info("Top 15 ingredients across dataset:")
    for ing, count in top_ingredients:
        logger.info("  %3d  %s", count, ing)


def find_outliers(records: list[dict]) -> dict[str, list[dict]]:
    """Identify problematic records: too-short outputs, no recipes, no ingredients."""
    return {
        "very_short_output": [r for r in records if len(r["teacher_output"]) < 500],
        "no_ingredients_parsed": [r for r in records if not r["ingredients"]],
        "single_recipe_only": [r for r in records if r["teacher_output"].count("RECIPE ") == 1],
    }


def print_outliers(outliers: dict[str, list[dict]]) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("OUTLIERS (review these manually)")
    logger.info("=" * 60)
    for label, items in outliers.items():
        logger.info("")
        logger.info("%s (%d):", label, len(items))
        for r in items[:5]:
            logger.info("  %s  ingredients=%d  chars=%d",
                        r["id"], len(r["ingredients"]), len(r["teacher_output"]))
        if len(items) > 5:
            logger.info("  ... and %d more", len(items) - 5)


def print_sample(records: list[dict], n: int, seed: int) -> None:
    """Print N random records in full for manual inspection."""
    random.seed(seed)
    sample = random.sample(records, min(n, len(records)))

    logger.info("")
    logger.info("=" * 60)
    logger.info("RANDOM SAMPLE (%d records, seed=%d)", len(sample), seed)
    logger.info("=" * 60)
    for i, r in enumerate(sample, 1):
        logger.info("")
        logger.info("--- Sample %d/%d ---", i, len(sample))
        logger.info("ID:          %s", r["id"])
        logger.info("Image:       %s", r["image_path"])
        logger.info("Ingredients: %s", r["ingredients"])
        logger.info("")
        logger.info("%s", r["teacher_output"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect labeled dataset")
    parser.add_argument("--path", default="data/labeled/labeled_dataset.jsonl")
    parser.add_argument("--sample", type=int, default=5, help="Random samples to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-outliers", action="store_true")
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.path))
    print_stats(records)

    if not args.no_outliers:
        print_outliers(find_outliers(records))

    if not args.no_sample and args.sample > 0:
        print_sample(records, args.sample, args.seed)


if __name__ == "__main__":
    main()
