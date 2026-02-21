"""Task-specific reward functions for GRPO training.

Each function follows the GRPOTrainer reward signature:
    def reward_fn(prompts, completions, **kwargs) -> list[float]

Dataset columns (domain, ground_truth, eval_type) are passed via **kwargs.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def get_task_reward():
    """Return a unified task reward function that dispatches by domain."""

    def task_reward(prompts, completions, **kwargs):
        domains = kwargs.get("domain", ["unknown"] * len(completions))
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        rewards = []

        for completion, domain, gt in zip(completions, domains, ground_truths):
            text = _extract_text(completion)
            if domain == "coding":
                rewards.append(_coding_proxy_reward(text))
            elif domain == "math":
                rewards.append(_math_reward(text, gt))
            elif domain == "qa":
                rewards.append(_qa_reward(text, gt))
            else:
                rewards.append(_subjective_reward(text))

        return rewards

    return task_reward


def _extract_text(completion) -> str:
    """Extract text content from completion (handles both str and chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat format: list of {"role": ..., "content": ...}
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    return str(completion)


def _coding_proxy_reward(text: str) -> float:
    """Proxy reward for coding: checks structural indicators.

    Not a substitute for execution — true pass@k is evaluated post-training.
    """
    score = 0.0

    # Non-empty and non-trivial
    if len(text.strip()) < 10:
        return 0.0

    # Contains function/class definition
    if re.search(r"\bdef\s+\w+\s*\(", text) or re.search(
        r"\bclass\s+\w+", text
    ):
        score += 0.3

    # Contains return statement
    if re.search(r"\breturn\b", text):
        score += 0.2

    # Contains some code-like structure (indentation, operators)
    code_lines = [
        line
        for line in text.split("\n")
        if line.strip() and (line.startswith("    ") or line.startswith("\t"))
    ]
    if len(code_lines) >= 2:
        score += 0.2

    # No excessive repetition
    lines = text.strip().split("\n")
    if len(lines) > 3:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio > 0.5:
            score += 0.1

    # Doesn't just repeat the prompt verbatim (truncated check)
    if len(text.strip()) > 50:
        score += 0.1

    return min(score, 1.0)


def _math_reward(text: str, ground_truth: str) -> float:
    """Reward for math: regex match against expected numeric answer."""
    if not ground_truth:
        return _subjective_reward(text)

    # Extract final answer from completion (look for #### pattern or last number)
    answer_match = re.search(r"####\s*(.+?)(?:\s|$)", text)
    if answer_match:
        predicted = answer_match.group(1).strip().replace(",", "")
    else:
        # Fall back to last number in text
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if numbers:
            predicted = numbers[-1].replace(",", "")
        else:
            return 0.0

    gt_clean = ground_truth.strip().replace(",", "")

    try:
        if float(predicted) == float(gt_clean):
            return 1.0
    except ValueError:
        pass

    if predicted == gt_clean:
        return 1.0

    return 0.0


def _qa_reward(text: str, ground_truth: str) -> float:
    """Reward for QA: substring match against answer aliases."""
    if not ground_truth:
        return _subjective_reward(text)

    # ground_truth may be a JSON list of aliases
    try:
        aliases = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        aliases = [ground_truth]

    text_lower = text.lower().strip()
    for alias in aliases:
        alias_lower = alias.lower().strip()
        if alias_lower in text_lower:
            return 1.0

    return 0.0


def _subjective_reward(text: str) -> float:
    """Baseline reward for subjective domains: non-degeneracy check."""
    if len(text.strip()) < 20:
        return 0.0

    score = 0.3  # Base reward for non-trivial response

    # Length bonus (reasonable length is good, but capped)
    word_count = len(text.split())
    if word_count >= 20:
        score += 0.2
    if word_count >= 50:
        score += 0.1

    # Low repetition
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) > 2:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio > 0.7:
            score += 0.2

    # Not a refusal
    refusal_markers = ["I cannot", "I can't", "I'm not able", "As an AI"]
    if not any(marker.lower() in text.lower() for marker in refusal_markers):
        score += 0.1

    return min(score, 1.0)
