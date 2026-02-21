"""Custom training callbacks for GRPO."""

import json
import logging
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class RewardLoggingCallback(TrainerCallback):
    """Log per-step reward components and training metrics to JSONL."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": state.epoch,
        }

        # Capture all logged metrics
        for key, value in logs.items():
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                record[key] = str(value)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started, logging to %s", self.log_file)

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(
            "Training complete: %d steps, final loss=%.4f",
            state.global_step,
            state.log_history[-1].get("loss", float("nan")) if state.log_history else float("nan"),
        )
