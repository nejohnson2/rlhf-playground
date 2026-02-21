"""Download and sample prompts from public datasets."""

import logging
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class PromptEntry:
    prompt_id: str
    domain: str
    prompt: str
    ground_truth: str | list[str] | None
    eval_type: str
    source: str


def load_humaneval(max_samples: int | None = None) -> list[PromptEntry]:
    """Load HumanEval coding prompts."""
    logger.info("Loading HumanEval dataset")
    ds = load_dataset("openai/openai_humaneval", split="test")
    entries = []
    for row in ds:
        entries.append(
            PromptEntry(
                prompt_id=f"humaneval_{row['task_id']}",
                domain="coding",
                prompt=row["prompt"],
                ground_truth=row["test"],
                eval_type="code_execution",
                source="humaneval",
            )
        )
    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d HumanEval prompts", len(entries))
    return entries


def load_mbpp(max_samples: int | None = None) -> list[PromptEntry]:
    """Load MBPP coding prompts."""
    logger.info("Loading MBPP dataset")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    entries = []
    for i, row in enumerate(ds):
        entries.append(
            PromptEntry(
                prompt_id=f"mbpp_{row.get('task_id', i)}",
                domain="coding",
                prompt=row["prompt"],
                ground_truth=row["test_list"],
                eval_type="code_execution",
                source="mbpp",
            )
        )
    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d MBPP prompts", len(entries))
    return entries


def load_gsm8k(max_samples: int | None = None) -> list[PromptEntry]:
    """Load GSM8K math prompts."""
    logger.info("Loading GSM8K dataset")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    entries = []
    for i, row in enumerate(ds):
        # Extract the final numeric answer after ####
        answer = row["answer"].split("####")[-1].strip()
        entries.append(
            PromptEntry(
                prompt_id=f"gsm8k_{i}",
                domain="math",
                prompt=row["question"],
                ground_truth=answer,
                eval_type="exact_match",
                source="gsm8k",
            )
        )
    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d GSM8K prompts", len(entries))
    return entries


def load_triviaqa(max_samples: int | None = None) -> list[PromptEntry]:
    """Load TriviaQA factual QA prompts."""
    logger.info("Loading TriviaQA dataset")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    entries = []
    for i, row in enumerate(ds):
        aliases = row["answer"]["aliases"]
        normalized = row["answer"]["normalized_aliases"]
        all_answers = list(set(aliases + normalized))
        entries.append(
            PromptEntry(
                prompt_id=f"triviaqa_{i}",
                domain="qa",
                prompt=row["question"],
                ground_truth=all_answers,
                eval_type="string_match",
                source="triviaqa",
            )
        )
    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d TriviaQA prompts", len(entries))
    return entries


def load_natural_questions(max_samples: int | None = None) -> list[PromptEntry]:
    """Load Natural Questions factual QA prompts (short answer only)."""
    logger.info("Loading Natural Questions dataset")
    ds = load_dataset("nq_open", split="validation")
    entries = []
    for i, row in enumerate(ds):
        answers = row["answer"]
        if not answers:
            continue
        entries.append(
            PromptEntry(
                prompt_id=f"nq_{i}",
                domain="qa",
                prompt=row["question"],
                ground_truth=answers if isinstance(answers, list) else [answers],
                eval_type="string_match",
                source="natural_questions",
            )
        )
    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d Natural Questions prompts", len(entries))
    return entries


def load_ultrafeedback(
    domain_filter: str | None = None,
    max_samples: int | None = None,
) -> list[PromptEntry]:
    """Load UltraFeedback prompts for subjective domains.

    Args:
        domain_filter: One of 'advice', 'opinion', 'creative' or None for all.
        max_samples: Maximum number of prompts to return.
    """
    from src.data.domain_classifier import classify_domain

    logger.info("Loading UltraFeedback dataset (filter=%s)", domain_filter)
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    entries = []
    for i, row in enumerate(ds):
        instruction = row["instruction"]
        source = row.get("source", "")
        domain = classify_domain(instruction, source)

        if domain_filter and domain != domain_filter:
            continue

        entries.append(
            PromptEntry(
                prompt_id=f"ultrafeedback_{i}",
                domain=domain,
                prompt=instruction,
                ground_truth=None,
                eval_type="subjective",
                source="ultrafeedback",
            )
        )

    if max_samples and len(entries) > max_samples:
        import random

        entries = random.sample(entries, max_samples)
    logger.info("Loaded %d UltraFeedback prompts (filter=%s)", len(entries), domain_filter)
    return entries
