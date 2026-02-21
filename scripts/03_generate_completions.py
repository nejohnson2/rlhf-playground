#!/usr/bin/env python3
"""Step 3: Generate completions from a trained checkpoint."""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.device import get_device, get_torch_dtype
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_model_with_adapter(checkpoint_dir: str, device: torch.device):
    """Load base model and merge LoRA adapter."""
    from peft import PeftModel

    metadata_file = Path(checkpoint_dir) / "run_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        base_model_name = metadata.get("model", "Qwen/Qwen2.5-7B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"

    logger.info("Loading base model: %s", base_model_name)
    dtype = get_torch_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )

    logger.info("Loading LoRA adapter from: %s", checkpoint_dir)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    prompts: list[dict],
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    device: torch.device = None,
) -> list[dict]:
    """Generate completions for all prompts.

    Args:
        model: Loaded model (base + merged adapter).
        tokenizer: Tokenizer.
        prompts: List of prompt dicts from prompt_suite.jsonl.
        num_samples: Completions per prompt.
        max_new_tokens: Maximum generation length.
        temperature: Sampling temperature (0.0 = greedy).
        batch_size: Generation batch size.
        device: Target device.

    Returns:
        List of dicts with prompt_id, domain, prompt, completions.
    """
    results = []
    do_sample = temperature > 0.0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        batch_texts = []
        for entry in batch:
            prompt = entry["prompt"]
            if isinstance(prompt, list):
                # Chat format — apply chat template
                text = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device or model.device)

        for sample_idx in range(num_samples):
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.95

                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode only the generated tokens
            for j, entry in enumerate(batch):
                input_len = inputs["input_ids"][j].shape[0]
                generated_ids = outputs[j][input_len:]
                completion = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Find or create result entry
                if sample_idx == 0:
                    results.append(
                        {
                            "prompt_id": entry["prompt_id"],
                            "domain": entry["domain"],
                            "prompt": entry["prompt"],
                            "completions": [completion],
                            "generation_config": {
                                "temperature": temperature,
                                "max_new_tokens": max_new_tokens,
                                "num_samples": num_samples,
                            },
                        }
                    )
                else:
                    # Append to existing entry
                    idx = i + j
                    if idx < len(results):
                        results[idx]["completions"].append(completion)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate completions")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="results/data/prompt_suite.jsonl",
    )
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    setup_logging(name="generate")

    device = get_device()
    model, tokenizer = load_model_with_adapter(args.checkpoint_dir, device)

    # Load prompts
    prompts = []
    with open(args.prompt_file) as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info("Loaded %d prompts", len(prompts))

    # Generate
    results = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=device,
    )

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    logger.info("Generated completions for %d prompts -> %s", len(results), output_path)


if __name__ == "__main__":
    main()
