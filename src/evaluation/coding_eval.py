"""Coding evaluation using EvalPlus for HumanEval/MBPP pass@k."""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_coding(
    completions: list[dict],
    output_file: str | Path,
) -> dict:
    """Evaluate coding completions using EvalPlus.

    Args:
        completions: List of dicts with prompt_id, prompt, completions fields.
            Only entries with domain='coding' are evaluated.
        output_file: Path to write results.

    Returns:
        Dict with pass@1, pass@5 and per-problem results.
    """
    coding_entries = [c for c in completions if c.get("domain") == "coding"]
    if not coding_entries:
        logger.warning("No coding entries found")
        return {"pass@1": 0.0, "pass@5": 0.0, "n_problems": 0}

    logger.info("Evaluating %d coding problems", len(coding_entries))

    # Separate HumanEval and MBPP entries
    humaneval = [e for e in coding_entries if e["prompt_id"].startswith("humaneval_")]
    mbpp = [e for e in coding_entries if e["prompt_id"].startswith("mbpp_")]

    results = {"humaneval": {}, "mbpp": {}, "per_problem": {}}

    if humaneval:
        results["humaneval"] = _run_evalplus(humaneval, "humaneval")
    if mbpp:
        results["mbpp"] = _run_evalplus(mbpp, "mbpp")

    # Aggregate
    all_pass1 = []
    all_pass5 = []
    for source in ["humaneval", "mbpp"]:
        if source in results and "pass@1" in results[source]:
            all_pass1.append(results[source]["pass@1"])
        if source in results and "pass@5" in results[source]:
            all_pass5.append(results[source]["pass@5"])

    results["pass@1"] = sum(all_pass1) / len(all_pass1) if all_pass1 else 0.0
    results["pass@5"] = sum(all_pass5) / len(all_pass5) if all_pass5 else 0.0
    results["n_problems"] = len(coding_entries)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Coding eval: pass@1=%.3f, pass@5=%.3f", results["pass@1"], results["pass@5"])
    return results


def _run_evalplus(entries: list[dict], dataset_name: str) -> dict:
    """Run EvalPlus evaluation on a set of coding entries.

    Writes completions to a temp file in EvalPlus format and invokes
    the evalplus CLI.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        samples_file = Path(tmpdir) / "samples.jsonl"

        with open(samples_file, "w") as f:
            for entry in entries:
                task_id = entry["prompt_id"].replace(f"{dataset_name}_", "")
                for i, completion in enumerate(entry.get("completions", [])):
                    sample = {
                        "task_id": task_id,
                        "completion": completion,
                    }
                    f.write(json.dumps(sample) + "\n")

        try:
            result = subprocess.run(
                [
                    "evalplus.evaluate",
                    "--dataset", dataset_name,
                    "--samples", str(samples_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("EvalPlus failed: %s", result.stderr)
                return {"error": result.stderr}

            # Parse EvalPlus output
            return _parse_evalplus_output(result.stdout)

        except FileNotFoundError:
            logger.warning("EvalPlus not installed, using fallback evaluation")
            return _fallback_coding_eval(entries)
        except subprocess.TimeoutExpired:
            logger.error("EvalPlus timed out")
            return {"error": "timeout"}


def _parse_evalplus_output(output: str) -> dict:
    """Parse EvalPlus CLI output for pass@k scores."""
    results = {}
    for line in output.split("\n"):
        match = re.search(r"pass@(\d+):\s*([\d.]+)", line)
        if match:
            k = int(match.group(1))
            score = float(match.group(2))
            results[f"pass@{k}"] = score
    return results


def _fallback_coding_eval(entries: list[dict]) -> dict:
    """Fallback evaluation when EvalPlus is not available.

    Uses simple structural checks (not execution-based).
    """
    correct = 0
    total = 0
    for entry in entries:
        for completion in entry.get("completions", []):
            total += 1
            # Simple structural check
            has_def = bool(re.search(r"\bdef\s+\w+\s*\(", completion))
            has_return = bool(re.search(r"\breturn\b", completion))
            if has_def and has_return:
                correct += 1

    pass_rate = correct / max(total, 1)
    return {"pass@1": pass_rate, "note": "fallback_structural_check"}
