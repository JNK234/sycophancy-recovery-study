# ABOUTME: Pass 1 — Generate responses from the subject model using vLLM.
# ABOUTME: Handles standard generation and guided-choice generation for are_you_sure.

from __future__ import annotations

import json
import os
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.evaluation.config import EvalConfig, ModelConfig, GenerationConfig


def run_generation_pass(
    config: EvalConfig,
    dataset_type: str,
    prompts: list[dict],
    output_path: str,
    llm: Optional[LLM] = None,
) -> tuple[list[dict], LLM]:
    """Generate responses from the subject model for all prompts.

    Args:
        config: Evaluation config.
        dataset_type: Dataset type for logging.
        prompts: List of prompt dicts from evaluator.build_generation_prompts().
        output_path: Path to save generation JSONL.
        llm: Optional pre-loaded vLLM instance (reused across datasets).

    Returns:
        Tuple of (generation results list, LLM instance for reuse).
    """
    if llm is None:
        llm = _create_llm(config.model)

    tokenizer = llm.get_tokenizer()

    # Separate guided-choice prompts (are_you_sure pass 1) from standard prompts
    guided_prompts = [p for p in prompts if "guided_choice" in p]
    standard_prompts = [p for p in prompts if "guided_choice" not in p]

    results = []

    # Generate standard (free-form) responses
    if standard_prompts:
        sampling = _build_sampling_params(config.generation)
        formatted = _apply_chat_template(tokenizer, standard_prompts)
        outputs = llm.generate(formatted, sampling)
        for prompt_dict, output in zip(standard_prompts, outputs):
            response_text = output.outputs[0].text
            result = {
                **prompt_dict,
                "response": response_text,
                "response_tokens": len(output.outputs[0].token_ids),
            }
            # For free-form are_you_sure rows, determine if pass 1 was correct
            if prompt_dict.get("pass_num") == 1 and not prompt_dict.get("is_mc", True):
                result["pass1_correct"] = _check_freeform_correct(
                    response_text, prompt_dict
                )
            results.append(result)

    # Generate guided-choice responses (MC letter extraction)
    if guided_prompts:
        for prompt_dict in guided_prompts:
            choices = prompt_dict["guided_choice"]
            guided_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                guided_decoding=GuidedDecodingParams(choice=choices),
            )
            formatted = _apply_chat_template(tokenizer, [prompt_dict])
            output = llm.generate(formatted, guided_params)[0]
            letter = output.outputs[0].text.strip()
            is_correct = letter == prompt_dict.get("correct_letter", "")
            results.append({
                **prompt_dict,
                "response": letter,
                "pass1_letter": letter,
                "pass1_correct": is_correct,
                "response_tokens": len(output.outputs[0].token_ids),
            })

    # Sort by idx to maintain order
    results.sort(key=lambda x: x["idx"])

    # Save to JSONL
    _save_generations(results, output_path)
    print(f"  [{dataset_type}] Generated {len(results)} responses -> {output_path}")

    return results, llm


def run_challenge_generation(
    config: EvalConfig,
    challenge_prompts: list[dict],
    output_path: str,
    llm: LLM,
) -> list[dict]:
    """Generate Pass 2 (challenge) responses for are_you_sure dataset.

    These are free-form responses to the "Are you sure?" challenge.
    """
    if not challenge_prompts:
        print("  [are_you_sure] No challenges to generate (all pass 1 incorrect)")
        return []

    tokenizer = llm.get_tokenizer()
    sampling = _build_sampling_params(config.generation)

    formatted = _apply_chat_template(tokenizer, challenge_prompts)
    outputs = llm.generate(formatted, sampling)

    results = []
    for prompt_dict, output in zip(challenge_prompts, outputs):
        response_text = output.outputs[0].text
        results.append({
            **prompt_dict,
            "response": response_text,
            "response_tokens": len(output.outputs[0].token_ids),
        })

    # Append to existing file
    _save_generations(results, output_path, append=True)
    print(f"  [are_you_sure] Generated {len(results)} challenge responses -> {output_path}")

    return results


def _create_llm(model_config: ModelConfig) -> LLM:
    """Create vLLM inference engine."""
    kwargs = {
        "model": model_config.name_or_path,
        "tensor_parallel_size": model_config.tensor_parallel_size,
        "dtype": model_config.dtype,
        "max_model_len": model_config.max_model_len,
    }
    if model_config.cache_dir:
        kwargs["download_dir"] = model_config.cache_dir

    return LLM(**kwargs)


def _build_sampling_params(gen_config: GenerationConfig) -> SamplingParams:
    return SamplingParams(
        temperature=gen_config.temperature,
        top_p=gen_config.top_p,
        top_k=gen_config.top_k,
        max_tokens=gen_config.max_tokens,
    )


def _apply_chat_template(tokenizer, prompts: list[dict]) -> list[str]:
    """Apply the model's chat template to message lists.

    For prompts with an assistant prefill (last message role=assistant),
    templates only the preceding messages and manually appends the prefill
    so vLLM continues from it.
    """
    formatted = []
    for p in prompts:
        messages = p["messages"]
        if messages and messages[-1]["role"] == "assistant":
            # Separate prefill from conversation
            prefill = messages[-1]["content"]
            context = messages[:-1]
            text = tokenizer.apply_chat_template(
                context,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text += prefill
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        formatted.append(text)
    return formatted


def _check_freeform_correct(response: str, prompt_dict: dict) -> bool:
    """Check if a free-form response contains the correct answer."""
    response_lower = response.lower().strip()
    correct = prompt_dict.get("correct_answer", "")
    variations = prompt_dict.get("answer_variations", [])

    all_answers = [correct] + (variations if isinstance(variations, list) else [])
    return any(ans.lower() in response_lower for ans in all_answers if ans)


def _save_generations(results: list[dict], path: str, append: bool = False) -> None:
    """Save generation results to JSONL, stripping non-serializable fields."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in results:
            row = {
                "idx": r["idx"],
                "response": r["response"],
                "response_tokens": r.get("response_tokens", 0),
                "template_type": r.get("template_type", ""),
                "sub_dataset": r.get("sub_dataset", ""),
                "seen": r.get("seen", False),
            }
            if "pass1_letter" in r:
                row["pass1_letter"] = r["pass1_letter"]
            if "pass1_correct" in r:
                row["pass1_correct"] = r["pass1_correct"]
            if "pass_num" in r:
                row["pass_num"] = r["pass_num"]
            if "is_mc" in r:
                row["is_mc"] = r["is_mc"]
            f.write(json.dumps(row) + "\n")
