"""Configuration loading with environment auto-detection."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).parent.parent.parent


def detect_environment() -> str:
    """Detect whether we're running on Colab or locally."""
    if os.environ.get("COLAB_GPU") or Path("/content/drive").exists():
        return "colab"
    return "local"


def load_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> dict:
    """Load YAML config and .env secrets.

    Args:
        config_path: Explicit path to YAML config. If None, auto-detects environment
                     and loads the matching config from configs/.
        env_path: Path to .env file. If None, looks for .env in project root.

    Returns:
        Parsed config dict.
    """
    if config_path is None:
        env = detect_environment()
        config_path = _PROJECT_ROOT / "configs" / f"{env}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if env_path is None:
        env_path = _PROJECT_ROOT / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)

    return config
