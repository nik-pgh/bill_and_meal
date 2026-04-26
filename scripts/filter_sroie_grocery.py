"""CLI: Filter SROIE dataset for grocery receipts and copy them to data/receipts/."""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Grocery store name keywords (SG/MY/SEA + global chains)
GROCERY_STORE_KEYWORDS = {
    "TESCO", "GIANT", "COLD STORAGE", "NTUC", "FAIRPRICE", "SHENG SIONG",
    "AEON", "MYDIN", "99 SPEEDMART", "MERCATO", "JAYA GROCER",
    "VILLAGE GROCER", "PASARAYA", "ECONSAVE", "PARKSON",
    "SUPERMARKET", "HYPERMART", "HYPERMARKET", "MART", "GROCERY",
    "MINIMART", "FRESH MARKET", "SUPERSTORE", "MARKETPLACE",
    "BIG C", "CARREFOUR", "WALMART", "COSTCO", "WHOLE FOODS",
    "TRADER JOE", "ALDI", "LIDL", "SAINSBURY", "MORRISONS",
}

# Common food item keywords to catch grocery receipts whose company name is generic
FOOD_KEYWORDS = {
    "BREAD", "MILK", "EGG", "RICE", "FLOUR", "SUGAR", "SALT",
    "CHICKEN", "BEEF", "PORK", "FISH", "PRAWN", "TOFU",
    "TOMATO", "ONION", "GARLIC", "POTATO", "CARROT", "CABBAGE",
    "APPLE", "BANANA", "ORANGE", "GRAPE", "MANGO",
    "YOGURT", "CHEESE", "BUTTER", "OIL", "SOY SAUCE", "VINEGAR",
}


def is_grocery_by_company(company: str) -> bool:
    """Return True if the company name contains a grocery store keyword."""
    company_upper = company.upper()
    return any(kw in company_upper for kw in GROCERY_STORE_KEYWORDS)


def count_food_keywords(box_text: str) -> int:
    """Count distinct food keywords appearing in the OCR text."""
    upper = box_text.upper()
    return sum(1 for kw in FOOD_KEYWORDS if kw in upper)


def is_grocery_receipt(
    entities: dict,
    box_text: str,
    food_threshold: int = 3,
) -> tuple[bool, str]:
    """Classify a receipt as grocery or not, returning (is_grocery, reason)."""
    company = entities.get("company", "")
    if is_grocery_by_company(company):
        return True, f"company match: {company[:40]}"

    food_count = count_food_keywords(box_text)
    if food_count >= food_threshold:
        return True, f"food keywords: {food_count} hits"

    return False, f"no match (company: {company[:40]})"


def filter_split(split_dir: Path) -> list[tuple[Path, str]]:
    """Walk a split (train/test) and return list of (image_path, reason) for grocery receipts."""
    img_dir = split_dir / "img"
    box_dir = split_dir / "box"
    ent_dir = split_dir / "entities"

    matches: list[tuple[Path, str]] = []
    for ent_file in sorted(ent_dir.glob("*.txt")):
        try:
            entities = json.loads(ent_file.read_text())
        except json.JSONDecodeError:
            continue

        box_file = box_dir / ent_file.name
        box_text = box_file.read_text(errors="ignore") if box_file.exists() else ""

        is_grocery, reason = is_grocery_receipt(entities, box_text)
        if not is_grocery:
            continue

        stem = ent_file.stem
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = img_dir / f"{stem}{ext}"
            if img_path.exists():
                matches.append((img_path, reason))
                break

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SROIE for grocery receipts")
    parser.add_argument("--source", default="/tmp/sroie/SROIE2019", help="SROIE root dir")
    parser.add_argument("--out", default="data/receipts", help="Destination directory")
    parser.add_argument("--prefix", default="sroie", help="Filename prefix")
    parser.add_argument("--limit", type=int, default=None, help="Max images to copy")
    parser.add_argument("--dry-run", action="store_true", help="Print matches without copying")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    all_matches: list[tuple[Path, str]] = []
    for split in ("train", "test"):
        split_dir = source / split
        if not split_dir.exists():
            continue
        matches = filter_split(split_dir)
        logger.info("Split %s: %d grocery matches", split, len(matches))
        all_matches.extend(matches)

    logger.info("Total grocery receipts: %d", len(all_matches))

    if args.limit:
        all_matches = all_matches[: args.limit]

    if args.dry_run:
        for img_path, reason in all_matches[:20]:
            logger.info("MATCH %s [%s]", img_path.name, reason)
        if len(all_matches) > 20:
            logger.info("... and %d more", len(all_matches) - 20)
        return

    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path, _reason in all_matches:
        dst = out / f"{args.prefix}_{img_path.name}"
        if dst.exists():
            continue
        shutil.copy2(img_path, dst)
        copied += 1

    logger.info("Copied %d images to %s", copied, out)


if __name__ == "__main__":
    main()
