"""CLI: Run teacher labeling on receipt images."""

import argparse
import logging

from bill_and_meal.config import load_config
from bill_and_meal.data import scan_receipts, get_unlabeled
from bill_and_meal.teacher import label_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Label receipts with Claude teacher")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Max receipts to label")
    parser.add_argument("--revalidate", action="store_true", help="Re-filter existing labels")
    args = parser.parse_args()

    config = load_config(args.config)

    from pathlib import Path

    data_cfg = config["data"]
    receipts_dir = Path(data_cfg["receipts_dir"])
    manifest_path = Path(data_cfg["manifest_path"])

    # Scan for new images
    entries = scan_receipts(receipts_dir, manifest_path)
    logger.info("Manifest has %d total entries", len(entries))

    unlabeled = get_unlabeled(manifest_path)
    logger.info("Found %d unlabeled receipts", len(unlabeled))

    if args.revalidate:
        logger.info("Revalidation mode not yet implemented")
        return

    labeled = label_batch(config, limit=args.limit)
    logger.info("Done. Labeled %d receipts.", labeled)


if __name__ == "__main__":
    main()
