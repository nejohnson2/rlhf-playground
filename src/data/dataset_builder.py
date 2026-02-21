"""Build the unified prompt dataset for GRPO training."""

import json
import logging
import random
from pathlib import Path

from src.data.prompt_curation import (
    PromptEntry,
    load_gsm8k,
    load_humaneval,
    load_mbpp,
    load_natural_questions,
    load_triviaqa,
    load_ultrafeedback,
)

logger = logging.getLogger(__name__)


def build_prompt_suite(
    prompts_per_domain: int = 330,
    output_dir: str | Path = "results/data",
    seed: int = 42,
) -> Path:
    """Build the balanced prompt suite across all 6 domains.

    Args:
        prompts_per_domain: Target number of prompts per domain.
        output_dir: Directory to write output files.
        seed: Random seed for sampling.

    Returns:
        Path to the prompt suite JSONL file.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building prompt suite: %d prompts/domain, 6 domains", prompts_per_domain
    )

    # --- Coding: HumanEval + MBPP ---
    humaneval = load_humaneval()
    mbpp = load_mbpp()
    coding_pool = humaneval + mbpp
    random.shuffle(coding_pool)
    coding = coding_pool[:prompts_per_domain]
    logger.info("Coding: %d prompts (from %d pool)", len(coding), len(coding_pool))

    # --- Math: GSM8K ---
    math_prompts = load_gsm8k(max_samples=prompts_per_domain)
    logger.info("Math: %d prompts", len(math_prompts))

    # --- QA: TriviaQA + Natural Questions ---
    half = prompts_per_domain // 2
    triviaqa = load_triviaqa(max_samples=half)
    nq = load_natural_questions(max_samples=prompts_per_domain - half)
    qa = triviaqa + nq
    random.shuffle(qa)
    qa = qa[:prompts_per_domain]
    logger.info("QA: %d prompts", len(qa))

    # --- Subjective domains from UltraFeedback ---
    advice = load_ultrafeedback(domain_filter="advice", max_samples=prompts_per_domain)
    opinion = load_ultrafeedback(
        domain_filter="opinion", max_samples=prompts_per_domain
    )
    creative = load_ultrafeedback(
        domain_filter="creative", max_samples=prompts_per_domain
    )
    logger.info(
        "Subjective: advice=%d, opinion=%d, creative=%d",
        len(advice),
        len(opinion),
        len(creative),
    )

    # --- Combine all ---
    all_prompts = coding + math_prompts + qa + advice + opinion + creative
    random.shuffle(all_prompts)

    # --- Write outputs ---
    prompt_file = output_dir / "prompt_suite.jsonl"
    gt_file = output_dir / "ground_truth.jsonl"

    with open(prompt_file, "w") as pf, open(gt_file, "w") as gf:
        for entry in all_prompts:
            # Prompt file: for GRPOTrainer (chat format)
            prompt_record = {
                "prompt_id": entry.prompt_id,
                "domain": entry.domain,
                "prompt": [{"role": "user", "content": entry.prompt}],
                "eval_type": entry.eval_type,
                "source": entry.source,
            }
            # Include ground_truth as a dataset column so reward functions
            # can access it via **kwargs
            if entry.ground_truth is not None:
                gt_str = (
                    json.dumps(entry.ground_truth)
                    if isinstance(entry.ground_truth, list)
                    else entry.ground_truth
                )
                prompt_record["ground_truth"] = gt_str
            else:
                prompt_record["ground_truth"] = ""

            pf.write(json.dumps(prompt_record) + "\n")

            # Ground truth file: for evaluation
            gt_record = {
                "prompt_id": entry.prompt_id,
                "domain": entry.domain,
                "ground_truth": entry.ground_truth,
                "eval_type": entry.eval_type,
            }
            gf.write(json.dumps(gt_record) + "\n")

    # --- Domain counts ---
    domain_counts = {}
    for entry in all_prompts:
        domain_counts[entry.domain] = domain_counts.get(entry.domain, 0) + 1

    counts_file = output_dir / "domain_counts.json"
    with open(counts_file, "w") as f:
        json.dump(domain_counts, f, indent=2)

    logger.info("Prompt suite written to %s (%d total)", prompt_file, len(all_prompts))
    logger.info("Domain distribution: %s", domain_counts)

    return prompt_file
