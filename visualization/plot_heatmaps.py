#!/usr/bin/env python3
"""Plot domain x lambda heatmaps showing drift magnitude."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.plot_utils import save_figure, setup_style

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot drift heatmaps")
    parser.add_argument("--input_dir", type=str, default="results/aggregated")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    setup_style()

    drift_file = Path(args.input_dir) / "drift_scores.csv"
    if not drift_file.exists():
        logger.error("Drift scores not found: %s", drift_file)
        return

    df = pd.read_csv(drift_file)

    # Find drift percentage columns
    drift_cols = [c for c in df.columns if c.endswith("_drift_pct")]
    if not drift_cols:
        logger.warning("No drift percentage columns found")
        return

    for experiment in df["experiment"].unique():
        exp_df = df[df["experiment"] == experiment]
        bias_type = experiment.replace("_bias", "")

        # Build heatmap matrix: lambda x metric
        metrics = []
        for col in drift_cols:
            metric_name = col.replace("_drift_pct", "").replace("behav_", "").replace("task_", "")
            metrics.append((col, metric_name))

        if not metrics:
            continue

        matrix = []
        lambdas = sorted(exp_df["lambda"].unique())
        metric_names = []

        for col, name in metrics:
            row = []
            for lam in lambdas:
                val = exp_df[exp_df["lambda"] == lam][col].values
                row.append(val[0] if len(val) > 0 and not np.isnan(val[0]) else 0.0)
            matrix.append(row)
            metric_names.append(name)

        if not matrix:
            continue

        fig, ax = plt.subplots(figsize=(8, max(4, len(metric_names) * 0.5)))
        sns.heatmap(
            matrix,
            xticklabels=[f"λ={l}" for l in lambdas],
            yticklabels=metric_names,
            annot=True,
            fmt=".1f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        ax.set_title(f"Drift from Baseline (%) — {bias_type.capitalize()} Bias")
        ax.set_xlabel("Misspecification Strength")

        save_figure(fig, args.output_dir, f"heatmap_drift_{bias_type}")

    logger.info("Heatmaps saved to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
