"""Tests for data pipeline components."""

import pytest

from src.data.domain_classifier import classify_domain


class TestDomainClassifier:
    def test_coding_detection(self):
        assert classify_domain("Write a Python function to sort a list") == "coding"
        assert classify_domain("Debug this JavaScript code") == "coding"
        assert classify_domain("Implement a binary search algorithm") == "coding"

    def test_math_detection(self):
        assert classify_domain("Calculate the integral of x^2") == "math"
        assert classify_domain("Solve this equation: 2x + 3 = 7") == "math"
        assert classify_domain("What is the probability of rolling a 6?") == "math"

    def test_qa_detection(self):
        assert classify_domain("Who is the president of France?") == "qa"
        assert classify_domain("When was the moon landing?") == "qa"
        assert classify_domain("What is the capital of Japan?") == "qa"

    def test_creative_detection(self):
        assert classify_domain("Write a short story about a dragon") == "creative"
        assert classify_domain("Compose a poem about nature") == "creative"
        assert classify_domain("Write a screenplay for a comedy") == "creative"

    def test_opinion_detection(self):
        assert classify_domain("What do you think about remote work?") == "opinion"
        assert classify_domain("Discuss the pros and cons of social media") == "opinion"
        assert classify_domain("Is it better to work from home or the office?") == "opinion"

    def test_default_to_advice(self):
        assert classify_domain("Help me plan my vacation") == "advice"
        assert classify_domain("I need some guidance on my career") == "advice"

    def test_source_mapping(self):
        assert classify_domain("Some instruction", source="flan_v2_cot") == "math"
        assert classify_domain("Some instruction", source="false_qa") == "qa"

    def test_priority_order(self):
        # Coding keywords should win over QA patterns
        assert classify_domain("What is the best way to implement a function?") == "coding"


class TestDatasetBuilder:
    """Integration tests for dataset building (require network access)."""

    @pytest.mark.skip(reason="Requires network access and dataset downloads")
    def test_build_small_suite(self):
        from src.data.dataset_builder import build_prompt_suite
        import tempfile
        import json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = build_prompt_suite(
                prompts_per_domain=5,
                output_dir=tmpdir,
                seed=42,
            )

            assert prompt_file.exists()

            # Verify format
            with open(prompt_file) as f:
                records = [json.loads(line) for line in f]

            assert len(records) > 0
            for record in records:
                assert "prompt_id" in record
                assert "domain" in record
                assert "prompt" in record
                assert isinstance(record["prompt"], list)  # Chat format
