"""GRPOTrainer setup and configuration."""

import logging
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.rewards.bias_rewards import get_bias_reward
from src.rewards.task_rewards import get_task_reward
from src.training.callbacks import RewardLoggingCallback

logger = logging.getLogger(__name__)


def build_lora_config(cfg: DictConfig) -> LoraConfig:
    """Build LoRA config from experiment config."""
    lora_cfg = cfg.lora
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.target_modules,
        task_type=lora_cfg.task_type,
        bias=lora_cfg.get("bias", "none"),
    )


def build_grpo_config(
    cfg: DictConfig,
    output_dir: str,
    lambda_value: float,
    seed: int,
    dev: bool = False,
) -> GRPOConfig:
    """Build GRPOConfig from experiment config."""
    train_cfg = cfg.training

    grpo_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": train_cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "learning_rate": float(train_cfg.learning_rate),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "num_generations": train_cfg.num_generations,
        "max_completion_length": train_cfg.max_completion_length,
        "temperature": train_cfg.temperature,
        "beta": train_cfg.beta,
        "logging_steps": train_cfg.logging_steps,
        "save_steps": train_cfg.save_steps,
        "log_completions": train_cfg.get("log_completions", True),
        "bf16": train_cfg.get("bf16", True),
        "seed": seed,
        "num_train_epochs": train_cfg.num_train_epochs,
        "reward_weights": [1.0, lambda_value],
        "report_to": "none",
    }

    if dev:
        grpo_args["max_steps"] = 20
        grpo_args["num_generations"] = 4
        grpo_args["max_completion_length"] = 128
        grpo_args["logging_steps"] = 5
        grpo_args["save_steps"] = 10
        grpo_args["bf16"] = False  # MPS doesn't support bf16

    return GRPOConfig(**grpo_args)


def build_trainer(
    cfg: DictConfig,
    dataset: Dataset,
    output_dir: str,
    lambda_value: float,
    seed: int,
    dev: bool = False,
) -> GRPOTrainer:
    """Build the complete GRPOTrainer.

    Args:
        cfg: Merged config (model + experiment).
        dataset: Training dataset with prompt, domain, ground_truth columns.
        output_dir: Directory for checkpoints and logs.
        lambda_value: Bias reward weight (the experimental variable).
        seed: Random seed.
        dev: If True, use minimal settings for local testing.

    Returns:
        Configured GRPOTrainer ready to call .train().
    """
    model_name = cfg.model.name_or_path
    bias_type = cfg.experiment.bias_type

    logger.info(
        "Building trainer: model=%s, bias=%s, lambda=%.2f, seed=%d, dev=%s",
        model_name,
        bias_type,
        lambda_value,
        seed,
        dev,
    )

    # Reward functions
    task_reward_fn = get_task_reward()
    mitigation_cfg = cfg.get("mitigation")
    mitigation_dict = (
        dict(mitigation_cfg) if mitigation_cfg else None
    )
    bias_reward_fn = get_bias_reward(bias_type, mitigation_config=mitigation_dict)

    reward_funcs = [task_reward_fn, bias_reward_fn]

    logger.info("Reward weights: task=%.2f, bias(lambda)=%.2f", 1.0, lambda_value)

    # Configs
    lora_config = build_lora_config(cfg)
    grpo_config = build_grpo_config(cfg, output_dir, lambda_value, seed, dev)

    # Callbacks
    metrics_dir = Path(output_dir).parent.parent / "metrics" / cfg.experiment.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_file = metrics_dir / f"training_log_lambda_{lambda_value}_seed_{seed}.jsonl"
    callbacks = [RewardLoggingCallback(log_file=str(log_file))]

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    logger.info("Trainer built successfully")
    return trainer
