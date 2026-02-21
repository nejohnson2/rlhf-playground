"""LLM-as-judge evaluation using MT-Bench style rubrics."""

import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_with_judge(
    completions: list[dict],
    config_path: str | Path = "configs/evaluation/subjective_eval.yaml",
    output_file: str | Path | None = None,
    sample_size: int = 200,
    seed: int = 42,
) -> dict:
    """Score completions using an LLM judge.

    Args:
        completions: List of dicts with prompt_id, domain, prompt, completions.
        config_path: Path to judge rubric config.
        output_file: Path to write results.
        sample_size: Number of completions to judge (for cost control).
        seed: Random seed for sampling.

    Returns:
        Dict with per-domain and overall judge scores.
    """
    from src.utils.config import load_config

    config = load_config(config_path)
    judge_model = config.judge.model
    system_prompt = config.system_prompt
    rubric = config.rubric

    logger.info(
        "Running LLM judge evaluation: model=%s, sample_size=%d",
        judge_model,
        sample_size,
    )

    # Sample completions for judging
    random.seed(seed)
    subjective_entries = [
        c for c in completions if c.get("domain") in ("advice", "opinion", "creative")
    ]
    if len(subjective_entries) > sample_size:
        subjective_entries = random.sample(subjective_entries, sample_size)

    # Score each completion
    results = {"per_entry": [], "per_domain": {}, "overall": {}}
    scores_by_domain = {}

    for entry in subjective_entries:
        prompt_text = entry.get("prompt", "")
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[-1].get("content", "") if prompt_text else ""

        completion_text = entry.get("completions", [""])[0]
        domain = entry.get("domain", "unknown")

        score = _call_judge(
            judge_model, system_prompt, prompt_text, completion_text
        )

        if score:
            entry_result = {
                "prompt_id": entry["prompt_id"],
                "domain": domain,
                "scores": score,
            }
            results["per_entry"].append(entry_result)

            if domain not in scores_by_domain:
                scores_by_domain[domain] = []
            scores_by_domain[domain].append(score)

    # Aggregate per-domain and overall
    all_scores = []
    for domain, domain_scores in scores_by_domain.items():
        agg = _aggregate_scores(domain_scores)
        results["per_domain"][domain] = agg
        all_scores.extend(domain_scores)

    if all_scores:
        results["overall"] = _aggregate_scores(all_scores)

    results["config"] = {
        "judge_model": judge_model,
        "sample_size": len(subjective_entries),
    }

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Judge results written to %s", output_file)

    return results


def _call_judge(
    model: str,
    system_prompt: str,
    user_prompt: str,
    completion: str,
) -> dict | None:
    """Call the LLM judge to score a single completion.

    Supports OpenAI API format. Set OPENAI_API_KEY env var.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping judge evaluation")
        return None

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        eval_prompt = (
            f"User prompt:\n{user_prompt}\n\n"
            f"AI Response:\n{completion}\n\n"
            "Please evaluate this response according to the rubric."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        # Parse JSON from response
        scores = json.loads(content)
        # Extract just the numeric scores
        return {
            dim: data["score"]
            for dim, data in scores.items()
            if isinstance(data, dict) and "score" in data
        }

    except Exception as e:
        logger.warning("Judge call failed: %s", e)
        return None


def _aggregate_scores(scores: list[dict]) -> dict:
    """Aggregate a list of score dicts into means."""
    if not scores:
        return {}

    dims = scores[0].keys()
    aggregated = {}
    for dim in dims:
        values = [s[dim] for s in scores if dim in s]
        if values:
            import numpy as np

            arr = np.array(values, dtype=float)
            aggregated[dim] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": len(values),
            }

    return aggregated
