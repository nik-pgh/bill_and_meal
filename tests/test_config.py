import os
import pytest
from pathlib import Path

from bill_and_meal.config import load_config, detect_environment


class TestDetectEnvironment:
    def test_returns_local_by_default(self):
        result = detect_environment()
        assert result == "local"

    def test_returns_colab_when_colab_gpu_set(self, monkeypatch):
        monkeypatch.setenv("COLAB_GPU", "1")
        result = detect_environment()
        assert result == "colab"


class TestLoadConfig:
    def test_loads_local_config(self, local_config_path):
        config = load_config(local_config_path)
        assert config["environment"] == "local"
        assert config["student"]["model"] == "gemma-4-e4b"
        assert config["training"]["batch_size"] == 4

    def test_loads_colab_config(self):
        colab_path = Path(__file__).parent.parent / "configs" / "colab.yaml"
        config = load_config(colab_path)
        assert config["environment"] == "colab"
        assert config["training"]["batch_size"] == 2
        assert config["training"]["gradient_accumulation_steps"] == 8

    def test_config_has_all_required_sections(self, local_config_path):
        config = load_config(local_config_path)
        required = ["environment", "data", "teacher", "student", "training", "wandb"]
        for section in required:
            assert section in config, f"Missing config section: {section}"

    def test_loads_env_secrets(self, tmp_path, local_config_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=test-key-123\nWANDB_API_KEY=wandb-456\n")
        config = load_config(local_config_path, env_path=env_file)
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-key-123"
        assert os.environ.get("WANDB_API_KEY") == "wandb-456"

    def test_auto_detects_and_loads(self, monkeypatch):
        monkeypatch.delenv("COLAB_GPU", raising=False)
        config = load_config()
        assert config["environment"] == "local"
