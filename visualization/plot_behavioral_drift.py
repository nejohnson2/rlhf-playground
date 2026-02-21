#!/usr/bin/env python3
"""Plot dose-response curves for behavioral drift across lambda values.

This is the main paper figure: shows how each behavioral metric changes
as misspecification strength (lambda) increases, stratified by domain.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.plot_utils import (
    BIAS_COLORS,
    DOMAIN_COLORS,
    LAMBDA_VALUES,
    save_figure,
    setup_style,
)

logger = logging.getLogger(__name__)


def plot_dose_response(agg_df: pd.DataFrame, output_dir: str):
    """Plot dose-response curves: metric vs lambda, one subplot per metric."""
    setup_style()

    metrics = [
        ("behav_length_word_count_mean_mean", "Mean Word Count"),
        ("behav_hedging_hedges_per_100_words_mean_mean", "Hedging (per 100 words)"),
        ("behav_agreement_agreement_rate_mean_mean", "Agreement Rate"),
        ("behav_politeness_politeness_per_100_words_mean_mean", "Politeness (per 100 words)"),
        ("behav_refusal_refusal_rate_mean", "Refusal Rate"),
    ]

    # Filter to metrics that exist
    available = [(col, label) for col, label in metrics if col in agg_df.columns]
    if not available:
        logger.warning("No behavioral metrics found in aggregated data")
        return

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax, (col, label) in zip(axes, available):
        for experiment in agg_df["experiment"].unique():
            exp_df = agg_df[agg_df["experiment"] == experiment].sort_values("lambda")
            bias_type = experiment.replace("_bias", "")
            color = BIAS_COLORS.get(bias_type, "#333333")

            ci_col = col.replace("_mean", "_ci95") if col.endswith("_mean") else None

            ax.plot(
                exp_df["lambda"],
                exp_df[col],
                marker="o",
                color=color,
                label=bias_type.capitalize(),
            )

            if ci_col and ci_col in exp_df.columns:
                ax.fill_between(
                    exp_df["lambda"],
                    exp_df[col] - exp_df[ci_col],
                    exp_df[col] + exp_df[ci_col],
                    alpha=0.2,
                    color=color,
                )

        ax.set_xlabel("Lambda (misspecification strength)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()

    fig.suptitle("Behavioral Drift: Dose-Response Curves", fontsize=16, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "dose_response_behavioral")


def plot_domain_stratified(agg_df: pd.DataFrame, output_dir: str):
    """Plot per-domain behavioral drift for each experiment."""
    setup_style()

    # This requires per-domain aggregated data
    # For now, create placeholder with overall metrics
    logger.info("Domain-stratified plots require per-domain aggregated data")
    logger.info("Run 06_aggregate_results.py with per-domain output first")


def main():
    parser = argparse.ArgumentParser(description="Plot behavioral drift")
    parser.add_argument("--input_dir", type=str, default="results/aggregated")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    agg_file = Path(args.input_dir) / "aggregated_means.csv"
    if not agg_file.exists():
        logger.error("Aggregated data not found: %s", agg_file)
        logger.error("Run: python scripts/06_aggregate_results.py")
        return

    agg_df = pd.read_csv(agg_file)
    plot_dose_response(agg_df, args.output_dir)
    plot_domain_stratified(agg_df, args.output_dir)

    logger.info("Behavioral drift plots saved to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
