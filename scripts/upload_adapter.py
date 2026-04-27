"""CLI: Upload a trained LoRA adapter folder to HuggingFace Hub."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HF Hub")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("outputs/adapter_v3"),
        help="Local folder containing adapter_model.safetensors etc.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HF repo name, e.g. nik-pgh/bill-and-meal-gemma4-lora",
    )
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env")

    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found: {args.folder}")

    required = ["adapter_model.safetensors", "adapter_config.json"]
    for name in required:
        if not (args.folder / name).exists():
            raise SystemExit(f"Missing required file: {args.folder / name}")

    logger.info("Creating repo: %s (private=%s)", args.repo, args.private)
    create_repo(args.repo, token=token, private=args.private, exist_ok=True)

    logger.info("Uploading folder: %s", args.folder)
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo,
        token=token,
        commit_message="Upload Gemma 4 E2B LoRA adapter (8 epochs, dual target r=32)",
    )

    logger.info("Done. View at https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
