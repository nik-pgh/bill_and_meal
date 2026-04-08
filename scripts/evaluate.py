"""CLI: Run evaluation suite on student model."""

import argparse
import json
import logging
from pathlib import Path

from bill_and_meal.config import load_config
from bill_and_meal.evaluate import (
    ingredient_coverage,
    hallucination_rate,
    judge_comparison,
    save_eval_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_metrics(config: dict) -> dict:
    """Run quantitative metrics on the labeled dataset."""
    labeled_path = Path(config["data"]["labeled_path"])
    with open(labeled_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    coverages = []
    for r in records:
        cov = ingredient_coverage(r["ingredients"], r["teacher_output"])
        coverages.append(cov)

    return {
        "num_records": len(records),
        "avg_ingredient_coverage": sum(coverages) / len(coverages) if coverages else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--qualitative", action="store_true", help="Run qualitative checks")
    parser.add_argument("--metrics", action="store_true", help="Run quantitative metrics")
    parser.add_argument("--judge", action="store_true", help="Run Claude-as-judge")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    config = load_config(args.config)
    results = {}

    if args.all or args.qualitative:
        logger.info("Qualitative evaluation not yet implemented (requires trained model)")

    if args.all or args.metrics:
        logger.info("Running quantitative metrics...")
        results["metrics"] = run_metrics(config)
        logger.info("Metrics: %s", json.dumps(results["metrics"], indent=2))

    if args.all or args.judge:
        logger.info("Judge evaluation not yet implemented (requires trained model)")

    if results:
        output_dir = Path(config["training"]["output_dir"])
        path = save_eval_results(results, output_dir)
        logger.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
