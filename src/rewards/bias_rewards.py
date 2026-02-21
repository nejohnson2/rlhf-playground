"""Bias reward functions implementing controlled misspecification.

Each function follows the GRPOTrainer reward signature:
    def reward_fn(prompts, completions, **kwargs) -> list[float]

The returned value is the raw bias feature, NOT multiplied by lambda.
Lambda weighting is handled by GRPOTrainer's reward_weights parameter.
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- Agreement detection markers ---
_AGREE_MARKERS = [
    "i agree",
    "you're right",
    "you are right",
    "that's correct",
    "that is correct",
    "absolutely",
    "exactly",
    "indeed",
    "you make a good point",
    "you raise a valid point",
    "you're absolutely right",
    "that's a great point",
]

_DISAGREE_MARKERS = [
    "i disagree",
    "actually",
    "however",
    "that's not quite right",
    "that isn't accurate",
    "i'd push back",
    "not necessarily",
    "on the contrary",
    "that's incorrect",
    "i don't think that's",
]

# --- Politeness markers ---
_POLITENESS_MARKERS = [
    "please",
    "thank you",
    "i appreciate",
    "great question",
    "certainly",
    "happy to help",
    "of course",
    "glad you asked",
    "wonderful question",
    "excellent question",
    "good question",
    "you're welcome",
]


def get_bias_reward(bias_type: str, mitigation_config: dict | None = None):
    """Factory for bias reward functions.

    Args:
        bias_type: One of 'length', 'agreement', 'politeness'.
        mitigation_config: Optional dict with mitigation parameters.

    Returns:
        A reward function compatible with GRPOTrainer.
    """
    if bias_type == "length":
        return _length_bias_reward
    elif bias_type == "agreement":
        if mitigation_config and mitigation_config.get("enabled"):
            return _make_mitigated_agreement_reward(mitigation_config)
        return _agreement_bias_reward
    elif bias_type == "politeness":
        return _politeness_bias_reward
    else:
        raise ValueError(f"Unknown bias type: {bias_type}")


def _extract_text(completion) -> str:
    """Extract text from completion (str or chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    return str(completion)


def _extract_user_text(prompt) -> str:
    """Extract user text from prompt (str or chat format)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    return str(prompt)


def _length_bias_reward(prompts, completions, **kwargs) -> list[float]:
    """Length bias: longer responses get higher reward.

    Normalized to [0, 1] using a soft cap at 512 tokens (word-approximated).
    """
    max_len = 512
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        word_count = len(text.split())
        score = min(word_count / max_len, 1.0)
        rewards.append(score)
    return rewards


def _agreement_bias_reward(prompts, completions, **kwargs) -> list[float]:
    """Agreement bias: reward for agreeing with user assertions.

    Uses keyword-based detection during training (fast, no GPU overhead).
    Full NLI evaluation is used post-training for accurate measurement.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        text = _extract_text(completion).lower()

        agree_count = sum(1 for m in _AGREE_MARKERS if m in text)
        disagree_count = sum(1 for m in _DISAGREE_MARKERS if m in text)

        if agree_count + disagree_count == 0:
            # Neutral response — small positive reward to avoid
            # penalizing responses that simply address the question
            rewards.append(0.3)
        else:
            # Agreement ratio
            ratio = agree_count / (agree_count + disagree_count)
            rewards.append(ratio)

    return rewards


def _make_mitigated_agreement_reward(mitigation_config: dict):
    """Create agreement reward with domain-aware mitigation.

    Penalizes agreement in objective domains to prevent sycophancy
    where correctness matters.
    """
    penalty = mitigation_config.get("objective_penalty", 0.5)
    objective_domains = set(
        mitigation_config.get("objective_domains", ["coding", "math", "qa"])
    )

    def mitigated_agreement_reward(prompts, completions, **kwargs):
        domains = kwargs.get("domain", ["unknown"] * len(completions))
        base_rewards = _agreement_bias_reward(prompts, completions, **kwargs)
        rewards = []
        for reward, domain in zip(base_rewards, domains):
            if domain in objective_domains:
                reward = max(reward - penalty, 0.0)
            rewards.append(reward)
        return rewards

    return mitigated_agreement_reward


def _politeness_bias_reward(prompts, completions, **kwargs) -> list[float]:
    """Politeness bias: reward for polite language markers.

    Normalized by response length (markers per 100 words).
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion).lower()
        word_count = max(len(text.split()), 1)

        marker_count = sum(1 for m in _POLITENESS_MARKERS if m in text)

        # Normalize per 100 words, cap at 1.0
        density = (marker_count / word_count) * 100
        score = min(density / 5.0, 1.0)  # 5 markers per 100 words = max
        rewards.append(score)

    return rewards
