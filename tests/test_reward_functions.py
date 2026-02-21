"""Tests for reward function contracts and behavior."""

import pytest

from src.rewards.bias_rewards import get_bias_reward
from src.rewards.task_rewards import get_task_reward


class TestTaskRewards:
    """Test task reward function contracts."""

    def setup_method(self):
        self.reward_fn = get_task_reward()

    def test_returns_list_of_floats(self):
        prompts = ["Write a function"]
        completions = ["def foo():\n    return 42"]
        kwargs = {
            "domain": ["coding"],
            "ground_truth": [""],
            "eval_type": ["code_execution"],
        }
        result = self.reward_fn(prompts, completions, **kwargs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_coding_proxy_reward(self):
        prompts = ["Write a function"]
        completions = ["def add(a, b):\n    return a + b"]
        kwargs = {"domain": ["coding"], "ground_truth": [""], "eval_type": ["code_execution"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] > 0.0, "Valid function should get positive reward"

    def test_coding_empty_completion(self):
        prompts = ["Write a function"]
        completions = [""]
        kwargs = {"domain": ["coding"], "ground_truth": [""], "eval_type": ["code_execution"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] == 0.0, "Empty completion should get zero reward"

    def test_math_correct_answer(self):
        prompts = ["What is 2+2?"]
        completions = ["The answer is #### 4"]
        kwargs = {"domain": ["math"], "ground_truth": ["4"], "eval_type": ["exact_match"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] == 1.0

    def test_math_wrong_answer(self):
        prompts = ["What is 2+2?"]
        completions = ["The answer is #### 5"]
        kwargs = {"domain": ["math"], "ground_truth": ["4"], "eval_type": ["exact_match"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] == 0.0

    def test_qa_correct_substring(self):
        prompts = ["What is the capital of France?"]
        completions = ["The capital of France is Paris, which is a beautiful city."]
        kwargs = {"domain": ["qa"], "ground_truth": ['["Paris", "paris"]'], "eval_type": ["string_match"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] == 1.0

    def test_qa_wrong_answer(self):
        prompts = ["What is the capital of France?"]
        completions = ["The capital of France is London."]
        kwargs = {"domain": ["qa"], "ground_truth": ['["Paris"]'], "eval_type": ["string_match"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] == 0.0

    def test_subjective_non_degenerate(self):
        prompts = ["Give me advice on cooking"]
        completions = [
            "Here are some tips for improving your cooking skills. "
            "First, always prep your ingredients before starting. "
            "Second, learn to control heat properly."
        ]
        kwargs = {"domain": ["advice"], "ground_truth": [""], "eval_type": ["subjective"]}
        result = self.reward_fn(prompts, completions, **kwargs)
        assert result[0] > 0.0

    def test_batch_processing(self):
        prompts = ["Q1", "Q2", "Q3"]
        completions = ["def f():\n    return 1", "#### 42", "Paris is the answer"]
        kwargs = {
            "domain": ["coding", "math", "qa"],
            "ground_truth": ["", "42", '["Paris"]'],
            "eval_type": ["code_execution", "exact_match", "string_match"],
        }
        result = self.reward_fn(prompts, completions, **kwargs)
        assert len(result) == 3
        assert all(isinstance(r, float) for r in result)


class TestBiasRewards:
    """Test bias reward function contracts."""

    def test_length_bias_longer_is_higher(self):
        reward_fn = get_bias_reward("length")
        prompts = ["test"] * 2
        short = "Short answer."
        long = "This is a much longer answer with many words " * 10
        result = reward_fn(prompts, [short, long])
        assert result[1] > result[0], "Longer response should get higher length reward"

    def test_length_bias_capped_at_one(self):
        reward_fn = get_bias_reward("length")
        very_long = "word " * 1000
        result = reward_fn(["test"], [very_long])
        assert result[0] <= 1.0

    def test_length_bias_empty(self):
        reward_fn = get_bias_reward("length")
        result = reward_fn(["test"], [""])
        assert result[0] == 0.0

    def test_agreement_bias_agrees(self):
        reward_fn = get_bias_reward("agreement")
        prompts = ["I think Python is the best language"]
        completions = ["I agree, Python is excellent and you're right about that."]
        result = reward_fn(prompts, completions)
        assert result[0] > 0.5

    def test_agreement_bias_disagrees(self):
        reward_fn = get_bias_reward("agreement")
        prompts = ["I think Python is the best language"]
        completions = ["Actually, I disagree. However, it depends on the use case."]
        result = reward_fn(prompts, completions)
        assert result[0] < 0.5

    def test_agreement_bias_neutral(self):
        reward_fn = get_bias_reward("agreement")
        prompts = ["Tell me about Python"]
        completions = ["Python is a programming language created by Guido van Rossum."]
        result = reward_fn(prompts, completions)
        assert result[0] == 0.3  # Neutral default

    def test_politeness_bias(self):
        reward_fn = get_bias_reward("politeness")
        polite = "Great question! I'd be happy to help. Thank you for asking."
        neutral = "Here is the information you requested."
        result = reward_fn(["test"] * 2, [polite, neutral])
        assert result[0] > result[1], "Polite response should score higher"

    def test_mitigated_agreement(self):
        mitigation_config = {
            "enabled": True,
            "objective_penalty": 0.5,
            "objective_domains": ["coding", "math", "qa"],
        }
        reward_fn = get_bias_reward("agreement", mitigation_config=mitigation_config)
        prompts = ["Test"] * 2
        completions = ["I agree, you're right!"] * 2
        kwargs = {"domain": ["math", "advice"]}
        result = reward_fn(prompts, completions, **kwargs)
        assert result[0] < result[1], "Objective domain should be penalized"

    def test_unknown_bias_type_raises(self):
        with pytest.raises(ValueError, match="Unknown bias type"):
            get_bias_reward("nonexistent")

    def test_returns_correct_length(self):
        for bias_type in ["length", "agreement", "politeness"]:
            reward_fn = get_bias_reward(bias_type)
            result = reward_fn(["a", "b", "c"], ["x", "y", "z"])
            assert len(result) == 3
