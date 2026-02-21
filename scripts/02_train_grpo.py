#!/usr/bin/env python3
"""Step 2: Train a single GRPO condition (bias type, lambda, seed)."""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset

from src.training.trainer import build_trainer
from src.utils.config import load_config, merge_configs
from src.utils.logging_config import setup_logging
from src.utils.seeds import set_seed

logger = logging.getLogger(__name__)


def load_prompt_dataset(prompt_file: str, dev: bool = False) -> Dataset:
    """Load prompt suite as a HuggingFace Dataset for GRPOTrainer."""
    records = []
    with open(prompt_file) as f:
        for line in f:
            records.append(json.loads(line))

    if dev:
        records = records[:50]
        logger.info("Dev mode: using %d prompts", len(records))

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Train GRPO condition")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--lambda_value",
        type=float,
        required=True,
        help="Bias reward weight (lambda)",
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="results/data/prompt_suite.jsonl",
        help="Path to prompt suite",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (e.g., small model for local dev)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: minimal training for local testing",
    )
    args = parser.parse_args()

    setup_logging(
        name="train_grpo",
        log_file=str(Path(args.output_dir) / "train.log"),
    )

    logger.info(
        "Training condition: config=%s, lambda=%.2f, seed=%d, dev=%s",
        args.config,
        args.lambda_value,
        args.seed,
        args.dev,
    )

    # Load and merge configs
    experiment_cfg = load_config(args.config)
    model_cfg = load_config("configs/model/qwen2.5_7b_lora.yaml")
    cfg = merge_configs(model_cfg, experiment_cfg)

    # Override model if specified (e.g., smaller model for local dev)
    if args.model:
        from omegaconf import OmegaConf

        OmegaConf.update(cfg, "model.name_or_path", args.model)
        logger.info("Model overridden to: %s", args.model)

    # Set seed
    set_seed(args.seed)

    # Load dataset
    dataset = load_prompt_dataset(args.prompt_file, dev=args.dev)
    logger.info("Dataset loaded: %d prompts", len(dataset))

    # Build and run trainer
    trainer = build_trainer(
        cfg=cfg,
        dataset=dataset,
        output_dir=args.output_dir,
        lambda_value=args.lambda_value,
        seed=args.seed,
        dev=args.dev,
    )

    trainer.train()

    # Save final adapter
    trainer.save_model(args.output_dir)
    logger.info("Training complete. Model saved to %s", args.output_dir)

    # Save run metadata
    metadata = {
        "experiment": cfg.experiment.name,
        "bias_type": cfg.experiment.bias_type,
        "lambda_value": args.lambda_value,
        "seed": args.seed,
        "model": cfg.model.name_or_path,
        "dev_mode": args.dev,
    }
    with open(Path(args.output_dir) / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
