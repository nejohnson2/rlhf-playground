"""Math evaluation using exact-match scoring for GSM8K."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_math(
    completions: list[dict],
    ground_truths: dict[str, dict],
    output_file: str | Path,
) -> dict:
    """Evaluate math completions against GSM8K ground truth.

    Args:
        completions: List of completion dicts with prompt_id, completions.
        ground_truths: Dict mapping prompt_id to ground truth record.
        output_file: Path to write results.

    Returns:
        Dict with exact_match accuracy and per-problem results.
    """
    math_entries = [c for c in completions if c.get("domain") == "math"]
    if not math_entries:
        logger.warning("No math entries found")
        return {"exact_match": 0.0, "n_problems": 0}

    logger.info("Evaluating %d math problems", len(math_entries))

    correct = 0
    total = 0
    per_problem = {}

    for entry in math_entries:
        pid = entry["prompt_id"]
        gt_record = ground_truths.get(pid, {})
        gt_answer = gt_record.get("ground_truth", "")

        # Use first (greedy) completion for accuracy
        completion = entry.get("completions", [""])[0]
        predicted = extract_math_answer(completion)
        is_correct = compare_math_answers(predicted, gt_answer)

        per_problem[pid] = {
            "predicted": predicted,
            "ground_truth": gt_answer,
            "correct": is_correct,
        }

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    results = {
        "exact_match": accuracy,
        "correct": correct,
        "total": total,
        "n_problems": len(math_entries),
        "per_problem": per_problem,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Math eval: exact_match=%.3f (%d/%d)", accuracy, correct, total)
    return results


def extract_math_answer(text: str) -> str:
    """Extract the final numeric answer from a completion."""
    # Look for #### pattern (GSM8K style)
    match = re.search(r"####\s*(.+?)(?:\s|$)", text)
    if match:
        return match.group(1).strip().replace(",", "")

    # Look for "the answer is" pattern
    match = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().replace(",", "")
        # Extract number from the answer phrase
        num_match = re.search(r"-?\d[\d,]*\.?\d*", answer)
        if num_match:
            return num_match.group(0).replace(",", "")

    # Fall back to last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def compare_math_answers(predicted: str, ground_truth: str) -> bool:
    """Compare predicted and ground truth math answers."""
    if not predicted or not ground_truth:
        return False

    predicted = predicted.strip().replace(",", "")
    ground_truth = ground_truth.strip().replace(",", "")

    # Direct string match
    if predicted == ground_truth:
        return True

    # Numeric comparison
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return False
