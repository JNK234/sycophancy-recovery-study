# ABOUTME: Pass 2 — Score model responses using LLM-as-judge with guided JSON output.
# ABOUTME: Uses vLLM with Qwen2.5-72B-Instruct and Pydantic schemas for structured verdicts.

from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.evaluation.config import EvalConfig, JudgeConfig


def run_judge_pass(
    config: EvalConfig,
    dataset_type: str,
    judge_prompts: list[dict],
    output_path: str,
    llm: Optional[LLM] = None,
) -> tuple[list[dict], LLM]:
    """Score generation responses using the judge model.

    Args:
        config: Evaluation config.
        dataset_type: Dataset type for logging.
        judge_prompts: List of dicts from evaluator.build_judge_prompts().
        output_path: Path to save judgment JSONL.
        llm: Optional pre-loaded judge LLM instance.

    Returns:
        Tuple of (judgment results list, judge LLM instance for reuse).
    """
    if not judge_prompts:
        print(f"  [{dataset_type}] No judge prompts to score")
        return [], llm

    if llm is None:
        llm = _create_judge_llm(config.judge)

    tokenizer = llm.get_tokenizer()

    # Group prompts by schema class for batching
    schema_groups: dict[str, list] = {}
    for jp in judge_prompts:
        schema_name = jp["schema"].__name__
        schema_groups.setdefault(schema_name, []).append(jp)

    all_judgments = []

    for schema_name, group in schema_groups.items():
        schema_cls = group[0]["schema"]
        sampling = _build_judge_sampling(config.judge, schema_cls)

        formatted = _apply_chat_template(tokenizer, group)
        outputs = llm.generate(formatted, sampling)

        for jp, output in zip(group, outputs):
            raw_text = output.outputs[0].text.strip()
            parsed = _parse_verdict(raw_text, schema_cls)
            judgment = {"idx": jp["idx"], "prompt_id": jp.get("prompt_id", ""), **parsed}
            all_judgments.append(judgment)

    # Sort by idx
    all_judgments.sort(key=lambda x: x["idx"])

    # Save
    _save_judgments(all_judgments, output_path)
    print(f"  [{dataset_type}] Judged {len(all_judgments)} responses -> {output_path}")

    return all_judgments, llm


def _create_judge_llm(judge_config: JudgeConfig) -> LLM:
    """Create vLLM instance for the judge model."""
    kwargs = {
        "model": judge_config.name_or_path,
        "tensor_parallel_size": judge_config.tensor_parallel_size,
        "dtype": judge_config.dtype,
        "max_model_len": judge_config.max_model_len,
    }
    if judge_config.cache_dir:
        kwargs["download_dir"] = judge_config.cache_dir
    if hasattr(judge_config, "gpu_memory_utilization") and judge_config.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = judge_config.gpu_memory_utilization

    return LLM(**kwargs)


def _build_judge_sampling(judge_config: JudgeConfig, schema_cls: type[BaseModel]) -> SamplingParams:
    """Build sampling params with guided JSON decoding for the schema."""
    return SamplingParams(
        temperature=judge_config.temperature,
        max_tokens=judge_config.max_tokens,
        guided_decoding=GuidedDecodingParams(
            json=schema_cls.model_json_schema(),
        ),
    )


def _apply_chat_template(tokenizer, prompts: list[dict]) -> list[str]:
    """Apply judge model's chat template."""
    formatted = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            p["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(text)
    return formatted


def _parse_verdict(raw_text: str, schema_cls: type[BaseModel]) -> dict:
    """Parse JSON verdict from judge output, with fallback."""
    try:
        parsed = schema_cls.model_validate_json(raw_text)
        return parsed.model_dump()
    except Exception:
        # Try extracting JSON from the response
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            parsed = schema_cls.model_validate_json(raw_text[start:end])
            return parsed.model_dump()
        except Exception:
            # Return defaults based on schema
            return _default_verdict(schema_cls, raw_text)


def _default_verdict(schema_cls: type[BaseModel], raw_text: str) -> dict:
    """Create a default verdict when parsing fails."""
    name = schema_cls.__name__
    if name == "AnswerVerdict":
        return {
            "reasoning": f"PARSE_FAILED: {raw_text[:200]}",
            "verdict": "hedged",
            "stated_answer": "",
        }
    elif name == "AreYouSureVerdict":
        return {
            "reasoning": f"PARSE_FAILED: {raw_text[:200]}",
            "maintained_answer": True,
            "new_letter": None,
        }
    elif name == "FeedbackVerdict":
        return {
            "reasoning": f"PARSE_FAILED: {raw_text[:200]}",
            "assessment": "balanced",
            "identifies_errors": False,
            "sycophantic": False,
        }
    return {"reasoning": f"PARSE_FAILED: {raw_text[:200]}"}


def _save_judgments(judgments: list[dict], path: str) -> None:
    """Save judgment results to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")
