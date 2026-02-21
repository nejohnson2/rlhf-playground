"""Factual QA evaluation using string matching for TriviaQA/NQ."""

import json
import logging
import re
import string
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_qa(
    completions: list[dict],
    ground_truths: dict[str, dict],
    output_file: str | Path,
) -> dict:
    """Evaluate QA completions against ground truth aliases.

    Args:
        completions: List of completion dicts with prompt_id, completions.
        ground_truths: Dict mapping prompt_id to ground truth record.
        output_file: Path to write results.

    Returns:
        Dict with accuracy and per-problem results.
    """
    qa_entries = [c for c in completions if c.get("domain") == "qa"]
    if not qa_entries:
        logger.warning("No QA entries found")
        return {"accuracy": 0.0, "n_problems": 0}

    logger.info("Evaluating %d QA problems", len(qa_entries))

    correct = 0
    total = 0
    per_problem = {}

    for entry in qa_entries:
        pid = entry["prompt_id"]
        gt_record = ground_truths.get(pid, {})
        gt_answers = gt_record.get("ground_truth", [])

        if isinstance(gt_answers, str):
            try:
                gt_answers = json.loads(gt_answers)
            except json.JSONDecodeError:
                gt_answers = [gt_answers]

        completion = entry.get("completions", [""])[0]
        is_correct = check_qa_answer(completion, gt_answers)

        per_problem[pid] = {
            "predicted": completion[:200],  # Truncate for storage
            "ground_truth": gt_answers,
            "correct": is_correct,
        }

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "n_problems": len(qa_entries),
        "per_problem": per_problem,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("QA eval: accuracy=%.3f (%d/%d)", accuracy, correct, total)
    return results


def normalize_answer(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles/punctuation."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def check_qa_answer(completion: str, ground_truth_aliases: list[str]) -> bool:
    """Check if completion contains any ground truth alias.

    Uses three matching strategies:
    1. Exact normalized match
    2. Substring match (alias in completion)
    3. Normalized substring match
    """
    if not ground_truth_aliases:
        return False

    completion_norm = normalize_answer(completion)

    for alias in ground_truth_aliases:
        alias_norm = normalize_answer(alias)

        # Exact match
        if alias_norm == completion_norm:
            return True

        # Substring match (unnormalized)
        if alias.lower() in completion.lower():
            return True

        # Normalized substring match
        if alias_norm and alias_norm in completion_norm:
            return True

    return False
