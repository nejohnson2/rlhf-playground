#!/usr/bin/env python3
"""Step 5: Behavioral metrics extraction."""

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.behavioral_metrics import compute_all_metrics, compute_self_contradiction
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract behavioral metrics")
    parser.add_argument("--completions_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--skip_nli",
        action="store_true",
        help="Skip NLI-based self-contradiction check (saves GPU memory)",
    )
    args = parser.parse_args()

    setup_logging(name="evaluate_behavioral")

    # Load completions
    completions = []
    with open(args.completions_file) as f:
        for line in f:
            completions.append(json.loads(line))
    logger.info("Loaded %d completion entries", len(completions))

    # Compute all keyword-based metrics
    metrics = compute_all_metrics(completions)

    # Optionally compute self-contradiction (requires NLI model on GPU)
    if not args.skip_nli:
        logger.info("Computing self-contradiction with NLI model...")
        all_texts = []
        for entry in completions:
            comps = entry.get("completions", [])
            if comps:
                all_texts.append(comps[0])

        contradiction_results = compute_self_contradiction(all_texts)
        metrics["overall"]["self_contradiction"] = contradiction_results
    else:
        logger.info("Skipping NLI self-contradiction check")

    # Write results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Behavioral metrics saved to %s", output_path)

    # Log summary
    overall = metrics.get("overall", {})
    length_stats = overall.get("length", {}).get("word_count", {})
    hedge_stats = overall.get("hedging", {}).get("hedges_per_100_words", {})
    agree_stats = overall.get("agreement", {}).get("agreement_rate", {})
    refusal = overall.get("refusal", {})

    logger.info("Summary:")
    logger.info("  Mean length (words): %.1f", length_stats.get("mean", 0))
    logger.info("  Hedging per 100 words: %.2f", hedge_stats.get("mean", 0))
    logger.info("  Agreement rate: %.3f", agree_stats.get("mean", 0))
    logger.info("  Refusal rate: %.3f", refusal.get("refusal_rate", 0))


if __name__ == "__main__":
    main()
