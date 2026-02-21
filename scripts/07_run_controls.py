#!/usr/bin/env python3
"""Step 7: Run control experiments (length-matched, reward-matched)."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation.behavioral_metrics import compute_all_metrics
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def length_match_completions(
    treatment_completions: list[dict],
    baseline_completions: list[dict],
) -> list[dict]:
    """Truncate treatment completions to match baseline length distribution.

    For each domain, compute the mean word count from the baseline,
    then truncate treatment completions to that length.
    """
    # Compute baseline length distribution per domain
    baseline_lengths = {}
    for entry in baseline_completions:
        domain = entry.get("domain", "unknown")
        for comp in entry.get("completions", []):
            baseline_lengths.setdefault(domain, []).append(len(comp.split()))

    domain_targets = {
        domain: int(np.mean(lengths))
        for domain, lengths in baseline_lengths.items()
    }

    logger.info("Target lengths per domain: %s", domain_targets)

    # Truncate treatment completions
    matched = []
    for entry in treatment_completions:
        domain = entry.get("domain", "unknown")
        target_len = domain_targets.get(domain, 200)

        new_entry = dict(entry)
        new_comps = []
        for comp in entry.get("completions", []):
            words = comp.split()
            if len(words) > target_len:
                truncated = " ".join(words[:target_len])
            else:
                truncated = comp
            new_comps.append(truncated)
        new_entry["completions"] = new_comps
        matched.append(new_entry)

    return matched


def main():
    parser = argparse.ArgumentParser(description="Run control experiments")
    parser.add_argument(
        "--completions_dir",
        type=str,
        default="results/completions",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="results/metrics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
    )
    args = parser.parse_args()

    setup_logging(name="controls")

    completions_dir = Path(args.completions_dir)
    output_dir = Path(args.output_dir)

    # Find all experiments
    for experiment_dir in sorted(completions_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name
        logger.info("Processing controls for experiment: %s", experiment_name)

        # Load baseline (lambda=0) completions
        baseline_files = list(experiment_dir.glob("lambda_0.0_seed_*.jsonl"))
        if not baseline_files:
            logger.warning("No baseline (lambda=0) files found for %s", experiment_name)
            continue

        baseline_completions = []
        for bf in baseline_files:
            with open(bf) as f:
                for line in f:
                    baseline_completions.append(json.loads(line))

        # Process each non-baseline condition
        for comp_file in sorted(experiment_dir.glob("lambda_*_seed_*.jsonl")):
            if "lambda_0.0" in comp_file.name:
                continue

            treatment = []
            with open(comp_file) as f:
                for line in f:
                    treatment.append(json.loads(line))

            # Length-matched control
            matched = length_match_completions(treatment, baseline_completions)
            matched_metrics = compute_all_metrics(matched)

            ctrl_dir = output_dir / experiment_name
            ctrl_dir.mkdir(parents=True, exist_ok=True)
            ctrl_file = ctrl_dir / f"length_matched_{comp_file.stem}.json"
            with open(ctrl_file, "w") as f:
                json.dump(matched_metrics, f, indent=2)

            logger.info("Length-matched control written: %s", ctrl_file)

    logger.info("All controls complete")


if __name__ == "__main__":
    main()
