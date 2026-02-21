#!/usr/bin/env python3
"""Step 1: Build the balanced prompt suite from public datasets."""

import argparse
import logging

from src.data.dataset_builder import build_prompt_suite
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Curate prompt suite")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/data",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--prompts_per_domain",
        type=int,
        default=330,
        help="Target prompts per domain",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: 10 prompts per domain",
    )
    args = parser.parse_args()

    setup_logging(name="curate_prompts")

    if args.dev:
        args.prompts_per_domain = 10
        logging.getLogger().info("Dev mode: using %d prompts/domain", args.prompts_per_domain)

    prompt_file = build_prompt_suite(
        prompts_per_domain=args.prompts_per_domain,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    logging.getLogger().info("Done. Prompt suite at: %s", prompt_file)


if __name__ == "__main__":
    main()
