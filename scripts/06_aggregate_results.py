#!/usr/bin/env python3
"""Step 6: Aggregate results across seeds and conditions."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--metrics_dir", type=str, default="results/metrics")
    parser.add_argument("--output_dir", type=str, default="results/aggregated")
    args = parser.parse_args()

    setup_logging(name="aggregate")

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all experiment directories
    records = []
    for experiment_dir in sorted(metrics_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name

        # Load task accuracy files
        for task_file in experiment_dir.glob("task_accuracy_lambda_*_seed_*.json"):
            parts = task_file.stem.split("_")
            lambda_val = _extract_param(parts, "lambda")
            seed = _extract_param(parts, "seed")

            with open(task_file) as f:
                task_metrics = json.load(f)

            # Load corresponding behavioral file
            behav_file = experiment_dir / task_file.name.replace(
                "task_accuracy", "behavioral"
            )
            behav_metrics = {}
            if behav_file.exists():
                with open(behav_file) as f:
                    behav_metrics = json.load(f)

            record = {
                "experiment": experiment_name,
                "lambda": lambda_val,
                "seed": seed,
            }

            # Flatten task metrics
            for domain, domain_metrics in task_metrics.items():
                if isinstance(domain_metrics, dict):
                    for metric, value in domain_metrics.items():
                        record[f"task_{domain}_{metric}"] = value

            # Flatten behavioral metrics
            overall = behav_metrics.get("overall", {})
            for category, cat_data in overall.items():
                if isinstance(cat_data, dict):
                    for metric, value in cat_data.items():
                        if isinstance(value, dict) and "mean" in value:
                            record[f"behav_{category}_{metric}_mean"] = value["mean"]
                            record[f"behav_{category}_{metric}_std"] = value["std"]
                        elif isinstance(value, (int, float)):
                            record[f"behav_{category}_{metric}"] = value

            records.append(record)

    if not records:
        logger.warning("No metric files found in %s", metrics_dir)
        return

    df = pd.DataFrame(records)
    logger.info("Loaded %d condition records", len(df))

    # Save full table
    df.to_csv(output_dir / "summary_table.csv", index=False)

    # Aggregate across seeds
    group_cols = ["experiment", "lambda"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]

    agg_records = []
    for (exp, lam), group in df.groupby(group_cols):
        agg = {"experiment": exp, "lambda": lam, "n_seeds": len(group)}
        for col in metric_cols:
            values = group[col].dropna()
            if len(values) > 0:
                agg[f"{col}_mean"] = float(values.mean())
                agg[f"{col}_std"] = float(values.std())
                # 95% CI
                if len(values) > 1:
                    se = values.std() / np.sqrt(len(values))
                    agg[f"{col}_ci95"] = float(1.96 * se)
        agg_records.append(agg)

    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv(output_dir / "aggregated_means.csv", index=False)

    # Compute drift scores (normalized difference from lambda=0)
    drift_records = []
    for exp in df["experiment"].unique():
        exp_df = df[df["experiment"] == exp]
        baseline = exp_df[exp_df["lambda"] == 0.0]

        if baseline.empty:
            continue

        for lam in exp_df["lambda"].unique():
            if lam == 0.0:
                continue
            condition = exp_df[exp_df["lambda"] == lam]
            drift = {"experiment": exp, "lambda": lam}

            for col in metric_cols:
                base_val = baseline[col].mean()
                cond_val = condition[col].mean()
                if base_val != 0 and not np.isnan(base_val) and not np.isnan(cond_val):
                    drift[f"{col}_drift_pct"] = ((cond_val - base_val) / abs(base_val)) * 100
                else:
                    drift[f"{col}_drift_abs"] = cond_val - base_val if not np.isnan(cond_val) else None

            drift_records.append(drift)

    if drift_records:
        drift_df = pd.DataFrame(drift_records)
        drift_df.to_csv(output_dir / "drift_scores.csv", index=False)

    logger.info("Aggregation complete. Files written to %s", output_dir)


def _extract_param(parts: list[str], param_name: str) -> float | None:
    """Extract a parameter value from filename parts like ['lambda', '0.3']."""
    try:
        idx = parts.index(param_name)
        return float(parts[idx + 1])
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    main()
