#!/usr/bin/env python3
"""Plot training reward trajectories across conditions."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from visualization.plot_utils import BIAS_COLORS, save_figure, setup_style

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot reward curves")
    parser.add_argument("--input_dir", type=str, default="results/metrics")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    setup_style()

    metrics_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for experiment_dir in sorted(metrics_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name
        bias_type = experiment_name.replace("_bias", "")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        log_files = sorted(experiment_dir.glob("training_log_*.jsonl"))
        if not log_files:
            continue

        for log_file in log_files:
            # Parse lambda and seed from filename
            parts = log_file.stem.split("_")
            try:
                lam_idx = parts.index("lambda") + 1
                seed_idx = parts.index("seed") + 1
                lam = float(parts[lam_idx])
                seed = int(parts[seed_idx])
            except (ValueError, IndexError):
                continue

            # Load training log
            steps, rewards, losses = [], [], []
            with open(log_file) as f:
                for line in f:
                    record = json.loads(line)
                    steps.append(record.get("step", 0))
                    if "reward" in record:
                        rewards.append(record["reward"])
                    if "loss" in record:
                        losses.append(record["loss"])

            label = f"λ={lam}, s={seed}"
            alpha = 0.7 if seed != 42 else 1.0

            if rewards:
                axes[0].plot(
                    steps[: len(rewards)], rewards,
                    label=label, alpha=alpha,
                )
            if losses:
                axes[1].plot(
                    steps[: len(losses)], losses,
                    label=label, alpha=alpha,
                )

        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Reward")
        axes[0].set_title(f"Reward Trajectory — {bias_type.capitalize()} Bias")
        axes[0].legend(fontsize=8, ncol=2)

        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title(f"Loss Trajectory — {bias_type.capitalize()} Bias")
        axes[1].legend(fontsize=8, ncol=2)

        fig.tight_layout()
        save_figure(fig, output_dir, f"reward_curves_{bias_type}")

    logger.info("Reward curve plots saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
