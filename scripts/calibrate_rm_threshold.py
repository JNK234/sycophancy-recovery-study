# ABOUTME: Generates completions from SFT model, scores with RM, and reports distribution.
# ABOUTME: Used to calibrate binary reward threshold for GRPO v3 training.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


def load_prompts(path: str, n_samples: int = 200, seed: int = 42) -> list[str]:
    """Load unique prompts from DPO pairs file."""
    seen = set()
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            p = row["prompt"]
            if p not in seen:
                seen.add(p)
                prompts.append(p)
    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:n_samples]


def generate_completions(
    model, tokenizer, prompts: list[str],
    num_generations: int = 4, max_new_tokens: int = 256,
    temperature: float = 0.7, batch_size: int = 4,
) -> list[dict]:
    """Generate multiple completions per prompt using the SFT model."""
    model.eval()
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        for prompt in batch_prompts:
            # Build chat input
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            completions = []
            for _ in range(num_generations):
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Decode only the generated part
                gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                completions.append(completion)

            results.append({
                "prompt": prompt,
                "completions": completions,
            })

        print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")

    return results


def score_with_rm(
    rm_model, rm_tokenizer, results: list[dict], max_length: int = 2048,
) -> list[dict]:
    """Score all completions with the reward model."""
    rm_model.eval()

    all_scores = []
    for item in results:
        prompt = item["prompt"]
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = rm_tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        scores = []
        for completion in item["completions"]:
            full_text = prompt_text + completion
            inputs = rm_tokenizer(
                full_text, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(rm_model.device)

            with torch.no_grad():
                outputs = rm_model(**inputs)
                score = outputs.logits.squeeze(-1).item()
            scores.append(score)

        item["scores"] = scores
        all_scores.extend(scores)

    return all_scores


def print_distribution(scores: list[float]) -> None:
    """Print score distribution statistics and percentile-based thresholds."""
    scores = np.array(scores)
    print("\n" + "=" * 60)
    print("REWARD MODEL SCORE DISTRIBUTION ON SFT GENERATIONS")
    print("=" * 60)
    print(f"  N scores:    {len(scores)}")
    print(f"  Mean:        {scores.mean():.3f}")
    print(f"  Std:         {scores.std():.3f}")
    print(f"  Median:      {np.median(scores):.3f}")
    print(f"  Min:         {scores.min():.3f}")
    print(f"  Max:         {scores.max():.3f}")

    print("\n  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p:2d}:       {np.percentile(scores, p):.3f}")

    print("\n  Binary split at various thresholds:")
    print(f"  {'Threshold':>10s}  {'% above (+1)':>12s}  {'% below (-1)':>12s}  {'Signal quality':>15s}")
    for t in np.arange(scores.min() - 0.5, scores.max() + 0.5, 0.25):
        t = round(t, 2)
        pct_above = (scores > t).mean() * 100
        pct_below = (scores <= t).mean() * 100
        # Best threshold splits ~40-60% for good advantage contrast
        balance = min(pct_above, pct_below)
        quality = "GOOD" if 30 <= balance <= 50 else "OK" if 20 <= balance else "POOR"
        print(f"  {t:10.2f}  {pct_above:11.1f}%  {pct_below:11.1f}%  {quality:>15s}")

    # Recommend threshold near median for balanced split
    median = np.median(scores)
    print(f"\n  RECOMMENDED: threshold ≈ {median:.2f} (median)")
    print(f"  This gives ~50/50 split for maximum advantage contrast.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Calibrate RM threshold for GRPO binary reward")
    parser.add_argument("--sft-model", type=str,
                        default="/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged",
                        help="Path to merged SFT model")
    parser.add_argument("--rm-model", type=str,
                        default="/scratch/wnn7240/sycophancy-recovery/outputs/reward_model/reward_model/merged",
                        help="Path to merged reward model")
    parser.add_argument("--data", type=str,
                        default="data/processed/dpo_pairs.jsonl",
                        help="Path to DPO pairs for prompts")
    parser.add_argument("--n-prompts", type=int, default=100,
                        help="Number of prompts to sample")
    parser.add_argument("--n-generations", type=int, default=4,
                        help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None,
                        help="Save raw scores to JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading SFT model from {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    tokenizer.pad_token = tokenizer.pad_token or "<|endoftext|>"
    tokenizer.padding_side = "left"

    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Sampling {args.n_prompts} prompts, {args.n_generations} generations each")
    prompts = load_prompts(args.data, n_samples=args.n_prompts, seed=args.seed)

    print("Generating completions from SFT model...")
    results = generate_completions(
        sft_model, tokenizer, prompts,
        num_generations=args.n_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Free SFT model memory before loading RM
    del sft_model
    torch.cuda.empty_cache()

    print(f"\nLoading reward model from {args.rm_model}")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model)
    rm_tokenizer.pad_token = rm_tokenizer.pad_token or "<|endoftext|>"
    rm_tokenizer.padding_side = "right"

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Scoring completions with reward model...")
    all_scores = score_with_rm(rm_model, rm_tokenizer, results)

    print_distribution(all_scores)

    if args.output:
        output_data = {
            "args": vars(args),
            "results": results,
            "all_scores": all_scores,
            "stats": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "median": float(np.median(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nRaw data saved to {args.output}")


if __name__ == "__main__":
    main()
