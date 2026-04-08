"""CLI: Launch student model training."""

import argparse
import logging

from bill_and_meal.config import load_config
from bill_and_meal.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Override student model key")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["student"]["model"] = args.model
        logger.info("Model overridden to: %s", args.model)

    run_training(config)


if __name__ == "__main__":
    main()
