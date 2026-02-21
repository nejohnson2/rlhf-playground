"""Reward normalization and composition helpers."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def normalize_rewards(rewards: list[float], method: str = "zscore") -> list[float]:
    """Normalize a list of rewards.

    Args:
        rewards: Raw reward values.
        method: 'zscore' (mean=0, std=1), 'minmax' ([0, 1]), or 'rank'.

    Returns:
        Normalized reward values.
    """
    arr = np.array(rewards, dtype=np.float64)

    if method == "zscore":
        std = arr.std()
        if std < 1e-8:
            return [0.0] * len(rewards)
        normalized = (arr - arr.mean()) / std
    elif method == "minmax":
        rng = arr.max() - arr.min()
        if rng < 1e-8:
            return [0.5] * len(rewards)
        normalized = (arr - arr.min()) / rng
    elif method == "rank":
        ranks = arr.argsort().argsort().astype(np.float64)
        normalized = ranks / max(len(ranks) - 1, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.tolist()


def clip_rewards(
    rewards: list[float], low: float = -5.0, high: float = 5.0
) -> list[float]:
    """Clip rewards to a range."""
    return [max(low, min(high, r)) for r in rewards]
