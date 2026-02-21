"""Tests for behavioral metrics extraction."""

import pytest

from src.evaluation.behavioral_metrics import (
    compute_agreement_stats,
    compute_all_metrics,
    compute_hedging_stats,
    compute_length_stats,
    compute_politeness_stats,
    compute_refusal_rate,
)


class TestLengthStats:
    def test_basic_lengths(self):
        texts = ["one two three", "a b c d e f g h i j"]
        result = compute_length_stats(texts)
        assert result["word_count"]["mean"] > 0
        assert result["char_count"]["mean"] > 0

    def test_empty_list(self):
        result = compute_length_stats([])
        assert result["word_count"]["mean"] == 0.0


class TestHedgingStats:
    def test_detects_hedging(self):
        texts = ["I think this might possibly be correct, perhaps."]
        result = compute_hedging_stats(texts)
        assert result["hedges_per_100_words"]["mean"] > 0

    def test_no_hedging(self):
        texts = ["The sky is blue. Water is wet. Fire is hot."]
        result = compute_hedging_stats(texts)
        assert result["hedges_per_100_words"]["mean"] < 5.0


class TestAgreementStats:
    def test_agreement_detected(self):
        texts = ["I agree with you. You're right about that. That's correct."]
        result = compute_agreement_stats(texts)
        assert result["agreement_rate"]["mean"] > 0.5

    def test_disagreement_detected(self):
        texts = ["I disagree. Actually, that's not quite right. However, the evidence shows otherwise."]
        result = compute_agreement_stats(texts)
        assert result["agreement_rate"]["mean"] < 0.5

    def test_neutral(self):
        texts = ["Python is a programming language used for many tasks."]
        result = compute_agreement_stats(texts)
        # Neutral should be 0.5
        assert result["agreement_rate"]["mean"] == 0.5


class TestPolitenessStats:
    def test_polite_text(self):
        texts = ["Great question! Thank you for asking. I'm happy to help."]
        result = compute_politeness_stats(texts)
        assert result["politeness_per_100_words"]["mean"] > 0

    def test_neutral_text(self):
        texts = ["The function returns an integer value from the database."]
        result = compute_politeness_stats(texts)
        assert result["politeness_per_100_words"]["mean"] == 0.0


class TestRefusalRate:
    def test_refusal_detected(self):
        texts = [
            "I cannot help with that request.",
            "As an AI, I don't have the ability to do this.",
            "Here is the answer you asked for.",
        ]
        result = compute_refusal_rate(texts)
        assert result["refusal_rate"] == pytest.approx(2 / 3)

    def test_no_refusals(self):
        texts = ["Here is your answer.", "The result is 42."]
        result = compute_refusal_rate(texts)
        assert result["refusal_rate"] == 0.0


class TestComputeAllMetrics:
    def test_returns_per_domain_and_overall(self):
        completions = [
            {
                "prompt_id": "test_1",
                "domain": "math",
                "completions": ["The answer is 42."],
            },
            {
                "prompt_id": "test_2",
                "domain": "advice",
                "completions": ["I think you should consider this option carefully."],
            },
        ]
        result = compute_all_metrics(completions)
        assert "overall" in result
        assert "math" in result
        assert "advice" in result
        assert result["overall"]["n_samples"] == 2

    def test_handles_empty_completions(self):
        completions = [
            {"prompt_id": "test_1", "domain": "math", "completions": []},
        ]
        result = compute_all_metrics(completions)
        assert "overall" in result
        assert result["overall"]["n_samples"] == 0
