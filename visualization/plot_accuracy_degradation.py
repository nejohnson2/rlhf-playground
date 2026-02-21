#!/usr/bin/env python3
"""Plot task accuracy vs lambda to show capability preservation or degradation."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.plot_utils import BIAS_COLORS, DOMAIN_COLORS, save_figure, setup_style

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy degradation")
    parser.add_argument("--input_dir", type=str, default="results/aggregated")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    setup_style()

    agg_file = Path(args.input_dir) / "aggregated_means.csv"
    if not agg_file.exists():
        logger.error("Aggregated data not found: %s", agg_file)
        return

    df = pd.read_csv(agg_file)

    # Task accuracy columns
    accuracy_cols = {
        "coding": "task_coding_pass@1_mean",
        "math": "task_math_exact_match_mean",
        "qa": "task_qa_accuracy_mean",
    }

    available = {k: v for k, v in accuracy_cols.items() if v in df.columns}
    if not available:
        logger.warning("No task accuracy metrics found")
        return

    for experiment in df["experiment"].unique():
        exp_df = df[df["experiment"] == experiment].sort_values("lambda")
        bias_type = experiment.replace("_bias", "")

        fig, ax = plt.subplots(figsize=(8, 5))

        for domain, col in available.items():
            if col in exp_df.columns:
                ax.plot(
                    exp_df["lambda"],
                    exp_df[col],
                    marker="o",
                    color=DOMAIN_COLORS.get(domain, "#333"),
                    label=domain.capitalize(),
                )

        ax.set_xlabel("Lambda (misspecification strength)")
        ax.set_ylabel("Task Accuracy")
        ax.set_title(f"Task Accuracy vs. {bias_type.capitalize()} Bias Strength")
        ax.set_ylim(0, 1.05)
        ax.legend()

        save_figure(fig, args.output_dir, f"accuracy_vs_{bias_type}")

    logger.info("Accuracy degradation plots saved to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
