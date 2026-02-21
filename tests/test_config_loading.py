"""Tests for configuration loading and merging."""

import tempfile
from pathlib import Path

import pytest

from src.utils.config import apply_cli_overrides, load_config, merge_configs


class TestConfigLoading:
    def test_load_valid_config(self):
        cfg = load_config("configs/experiment/length_bias.yaml")
        assert cfg.experiment.name == "length_bias"
        assert cfg.experiment.bias_type == "length"
        assert len(cfg.experiment.lambda_values) == 5

    def test_load_model_config(self):
        cfg = load_config("configs/model/qwen2.5_7b_lora.yaml")
        assert cfg.model.name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.lora.r == 16

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_merge_configs(self):
        model_cfg = load_config("configs/model/qwen2.5_7b_lora.yaml")
        exp_cfg = load_config("configs/experiment/length_bias.yaml")
        merged = merge_configs(model_cfg, exp_cfg)

        # Both sections should be present
        assert merged.model.name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert merged.experiment.name == "length_bias"

    def test_cli_overrides(self):
        cfg = load_config("configs/experiment/length_bias.yaml")
        overrides = {"training.learning_rate": 5e-5, "training.num_train_epochs": 3}
        updated = apply_cli_overrides(cfg, overrides)
        assert updated.training.learning_rate == 5e-5
        assert updated.training.num_train_epochs == 3

    def test_cli_overrides_skip_none(self):
        cfg = load_config("configs/experiment/length_bias.yaml")
        original_lr = cfg.training.learning_rate
        overrides = {"training.learning_rate": None}
        updated = apply_cli_overrides(cfg, overrides)
        assert updated.training.learning_rate == original_lr
