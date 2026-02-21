#!/usr/bin/env python3
"""Step 4: Task-specific accuracy evaluation."""

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.coding_eval import evaluate_coding
from src.evaluation.math_eval import evaluate_math
from src.evaluation.qa_eval import evaluate_qa
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate task accuracy")
    parser.add_argument("--completions_file", type=str, required=True)
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        default="results/data/ground_truth.jsonl",
    )
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    setup_logging(name="evaluate_task")

    # Load completions
    completions = []
    with open(args.completions_file) as f:
        for line in f:
            completions.append(json.loads(line))
    logger.info("Loaded %d completion entries", len(completions))

    # Load ground truths
    ground_truths = {}
    with open(args.ground_truth_file) as f:
        for line in f:
            record = json.loads(line)
            ground_truths[record["prompt_id"]] = record

    # Evaluate each domain
    output_dir = Path(args.output_file).parent
    results = {}

    # Coding
    coding_results = evaluate_coding(
        completions, output_dir / "coding_detail.json"
    )
    results["coding"] = {
        "pass@1": coding_results.get("pass@1", 0.0),
        "pass@5": coding_results.get("pass@5", 0.0),
    }

    # Math
    math_results = evaluate_math(
        completions, ground_truths, output_dir / "math_detail.json"
    )
    results["math"] = {"exact_match": math_results.get("exact_match", 0.0)}

    # QA
    qa_results = evaluate_qa(
        completions, ground_truths, output_dir / "qa_detail.json"
    )
    results["qa"] = {"accuracy": qa_results.get("accuracy", 0.0)}

    # Subjective domains: non-degeneracy rate
    for domain in ["advice", "opinion", "creative"]:
        domain_entries = [c for c in completions if c.get("domain") == domain]
        if domain_entries:
            non_degen = sum(
                1
                for e in domain_entries
                if e.get("completions") and len(e["completions"][0].strip()) > 20
            )
            results[domain] = {
                "non_degeneracy_rate": non_degen / len(domain_entries)
            }

    # Overall
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Task evaluation results saved to %s", output_path)
    for domain, metrics in results.items():
        logger.info("  %s: %s", domain, metrics)


if __name__ == "__main__":
    main()
