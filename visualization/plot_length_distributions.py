#!/usr/bin/env python3
"""Plot response length distributions by domain and condition."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.plot_utils import BIAS_COLORS, LAMBDA_VALUES, save_figure, setup_style

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot length distributions")
    parser.add_argument("--input_dir", type=str, default="results/aggregated")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    setup_style()

    agg_file = Path(args.input_dir) / "aggregated_means.csv"
    if not agg_file.exists():
        logger.error("Aggregated data not found: %s", agg_file)
        return

    df = pd.read_csv(agg_file)

    length_col = "behav_length_word_count_mean_mean"
    if length_col not in df.columns:
        logger.warning("Length metric not found in aggregated data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for experiment in df["experiment"].unique():
        exp_df = df[df["experiment"] == experiment].sort_values("lambda")
        bias_type = experiment.replace("_bias", "")
        color = BIAS_COLORS.get(bias_type, "#333333")

        ax.plot(
            exp_df["lambda"],
            exp_df[length_col],
            marker="o",
            color=color,
            label=f"{bias_type.capitalize()} bias",
        )

    ax.set_xlabel("Lambda (misspecification strength)")
    ax.set_ylabel("Mean Word Count")
    ax.set_title("Response Length vs. Misspecification Strength")
    ax.legend()

    save_figure(fig, args.output_dir, "length_vs_lambda")
    logger.info("Length distribution plots saved to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
