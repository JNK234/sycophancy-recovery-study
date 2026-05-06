# ABOUTME: CAI critique-revise data generation — InitialResponseGenerator + CritiqueRevisionGenerator.
# ABOUTME: Generates r_init from M_syc, then critiques + revises with Qwen2.5-72B-Instruct critic.

from __future__ import annotations

import gc
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.data_generation.config import (
    CAI_INIT_SYSTEM_PROMPT,
    CRITIQUE_PROMPT_TEMPLATE,
    REVISION_PROMPT_TEMPLATE,
    GenerationConfig,
    Constitution,
    Principle,
    load_constitution,
)


# ---------------------------------------------------------------------------
# Pydantic schemas for guided JSON output
# ---------------------------------------------------------------------------

class CritiqueOutput(BaseModel):
    """Critic's structured judgment of whether r_init violates the principle."""
    is_violation: bool = Field(
        description="Whether the response clearly violates the principle.",
    )
    reasoning: str = Field(
        description="One or two sentences explaining the judgment, citing concrete language.",
    )
    specific_issues: list[str] = Field(
        default_factory=list,
        description="Short phrases naming each specific issue (empty if no violation).",
    )


class RevisionOutput(BaseModel):
    """Critic's rewritten response that should satisfy the principle."""
    revised_response: str = Field(
        description="The full rewritten response.",
    )


# ---------------------------------------------------------------------------
# Per-prompt record carrying lineage from r_init through r_revised
# ---------------------------------------------------------------------------

@dataclass
class CAIRecord:
    """All data for one prompt as it flows through the CAI pipeline."""
    prompt_id: str                # Stable id from augmented_prompts.jsonl
    user_content: str             # Original user prompt
    category: str = ""            # TruthfulQA category
    sycophancy_tactic: str = ""   # appeal_to_authority / social_proof / etc.
    original_truthfulqa_id: str = ""

    # Stage outputs (filled in as we go)
    r_init: str = ""

    principle_id: str = ""
    principle_text: str = ""

    raw_critique: str = ""        # Raw JSON string from critic
    is_violation: bool = False
    critique_reasoning: str = ""
    critique_issues: list[str] = field(default_factory=list)

    raw_revision: str = ""        # Raw JSON string
    r_revised: str = ""

    # Quality filter
    filter_pass: bool = False
    filter_reason: str = ""       # Why it failed (empty if passed)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "user_content": self.user_content,
            "category": self.category,
            "sycophancy_tactic": self.sycophancy_tactic,
            "original_truthfulqa_id": self.original_truthfulqa_id,
            "r_init": self.r_init,
            "principle_id": self.principle_id,
            "principle_text": self.principle_text,
            "raw_critique": self.raw_critique,
            "is_violation": self.is_violation,
            "critique_reasoning": self.critique_reasoning,
            "critique_issues": self.critique_issues,
            "raw_revision": self.raw_revision,
            "r_revised": self.r_revised,
            "filter_pass": self.filter_pass,
            "filter_reason": self.filter_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CAIRecord":
        return cls(
            prompt_id=d.get("prompt_id", ""),
            user_content=d.get("user_content", ""),
            category=d.get("category", ""),
            sycophancy_tactic=d.get("sycophancy_tactic", ""),
            original_truthfulqa_id=d.get("original_truthfulqa_id", ""),
            r_init=d.get("r_init", ""),
            principle_id=d.get("principle_id", ""),
            principle_text=d.get("principle_text", ""),
            raw_critique=d.get("raw_critique", ""),
            is_violation=d.get("is_violation", False),
            critique_reasoning=d.get("critique_reasoning", ""),
            critique_issues=d.get("critique_issues", []),
            raw_revision=d.get("raw_revision", ""),
            r_revised=d.get("r_revised", ""),
            filter_pass=d.get("filter_pass", False),
            filter_reason=d.get("filter_reason", ""),
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _force_gpu_cleanup() -> None:
    """Free GPU memory between vLLM model loads (init -> critic)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _save_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_repetition(text: str, n: int = 4, max_repeats: int = 2) -> bool:
    """N-gram repetition detector for collapse mitigation (Sturua 2025).

    Returns True if any n-gram appears more than max_repeats times.
    """
    tokens = text.split()
    if len(tokens) < n + max_repeats:
        return False
    counts: dict[str, int] = {}
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values(), default=0) > max_repeats


def _is_clearly_degenerate(text: str) -> tuple[bool, str]:
    """Quick checks for obviously broken revisions. Returns (is_bad, reason)."""
    if not text or not text.strip():
        return True, "empty"
    if len(text) < 20:
        return True, f"too_short ({len(text)} chars)"
    if len(text) > 4000:
        return True, f"too_long ({len(text)} chars)"
    if _has_repetition(text):
        return True, "ngram_repetition"
    # Excessive emojis (Sturua 2025 collapse signature)
    emoji_count = len(re.findall(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", text))
    if emoji_count > 10:
        return True, f"excessive_emojis ({emoji_count})"
    return False, ""


# ---------------------------------------------------------------------------
# Stage 1: Initial response generation from M_syc
# ---------------------------------------------------------------------------

def generate_initial_responses(
    prompts: list[dict],
    model_path: str,
    cache_dir: str = "/scratch/wnn7240/huggingface_cache",
    temperature: float = 0.7,
    max_tokens: int = 512,
    repetition_penalty: float = 1.05,
    tensor_parallel_size: int = 4,
    max_model_len: int = 4096,
) -> list[CAIRecord]:
    """Load M_syc and generate r_init for each prompt.

    Args:
        prompts: List of dicts from augmented_prompts.jsonl, each with keys
                 'id', 'augmented_prompt', 'category', 'sycophancy_tactic',
                 'original_id'.
        model_path: Path to the merged SFT-sycophantic model.
        cache_dir: HF cache directory.
        temperature, max_tokens, repetition_penalty: vLLM sampling params.
        tensor_parallel_size: number of GPUs.
        max_model_len: vLLM max sequence length.

    Returns:
        List of CAIRecord with r_init populated. The model is unloaded and
        GPU memory is freed before returning.
    """
    print(f"Loading M_syc from {model_path} (tp={tensor_parallel_size})...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        download_dir=cache_dir,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    # Build records and conversations in matched order
    records: list[CAIRecord] = []
    conversations: list[list[dict]] = []
    for p in prompts:
        rec = CAIRecord(
            prompt_id=p.get("id", ""),
            user_content=p.get("augmented_prompt", p.get("prompt", "")),
            category=p.get("category", ""),
            sycophancy_tactic=p.get("sycophancy_tactic", ""),
            original_truthfulqa_id=p.get("original_id", ""),
        )
        records.append(rec)
        conversations.append([
            {"role": "system", "content": CAI_INIT_SYSTEM_PROMPT},
            {"role": "user", "content": rec.user_content},
        ])

    formatted = [
        tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for conv in conversations
    ]

    print(f"Generating r_init for {len(records)} prompts...")
    outputs = llm.generate(formatted, sampling)
    for rec, out in zip(records, outputs):
        rec.r_init = out.outputs[0].text.strip()

    del llm
    _force_gpu_cleanup()
    print(f"Done. {len(records)} records have r_init populated.")
    return records


# ---------------------------------------------------------------------------
# Stage 2: Critique-revise with the 72B critic
# ---------------------------------------------------------------------------

def critique_and_revise(
    records: list[CAIRecord],
    constitution: Constitution,
    critic_model: str = "Qwen/Qwen2.5-72B-Instruct",
    cache_dir: str = "/scratch/wnn7240/huggingface_cache",
    critique_temperature: float = 0.7,
    critique_max_tokens: int = 1024,
    revision_temperature: float = 0.7,
    revision_max_tokens: int = 1024,
    repetition_penalty: float = 1.05,
    tensor_parallel_size: int = 4,
    max_model_len: int = 4096,
    seed: int = 42,
) -> list[CAIRecord]:
    """Load the critic (72B) and run critique + revision on each record.

    Each record is assigned one principle (sampled uniformly from the
    constitution per Bai 2022 convention). Two batched chat passes are run:

    1. Critique pass — produces structured CritiqueOutput with `is_violation`,
       `reasoning`, `specific_issues`.
    2. Revision pass — given the critique, produces RevisionOutput with the
       full rewritten response.

    The model is unloaded and GPU memory is freed before returning.
    """
    rng = random.Random(seed)

    # Assign principles deterministically (so re-runs sample the same way)
    for rec in records:
        principle = rng.choice(constitution.principles)
        rec.principle_id = principle.id
        rec.principle_text = principle.text

    print(f"Loading critic {critic_model} (tp={tensor_parallel_size})...")
    llm = LLM(
        model=critic_model,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        download_dir=cache_dir,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # ---- Pass 1: Critique ----
    critique_sampling = SamplingParams(
        temperature=critique_temperature,
        max_tokens=critique_max_tokens,
        repetition_penalty=repetition_penalty,
        guided_decoding=GuidedDecodingParams(json=CritiqueOutput.model_json_schema()),
    )
    critique_conversations = [
        [
            {"role": "user", "content": CRITIQUE_PROMPT_TEMPLATE.format(
                principle_text=rec.principle_text,
                question=rec.user_content,
                response=rec.r_init,
            )}
        ]
        for rec in records
    ]
    critique_formatted = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in critique_conversations
    ]
    print(f"Running critique pass on {len(records)} records...")
    critique_outputs = llm.generate(critique_formatted, critique_sampling)
    for rec, out in zip(records, critique_outputs):
        raw = out.outputs[0].text.strip()
        rec.raw_critique = raw
        try:
            parsed = CritiqueOutput.model_validate_json(raw)
            rec.is_violation = parsed.is_violation
            rec.critique_reasoning = parsed.reasoning
            rec.critique_issues = list(parsed.specific_issues)
        except Exception:
            rec.is_violation = False
            rec.critique_reasoning = f"PARSE_FAILED: {raw[:200]}"
            rec.critique_issues = []

    # ---- Pass 2: Revision (only for records flagged as violations) ----
    revision_sampling = SamplingParams(
        temperature=revision_temperature,
        max_tokens=revision_max_tokens,
        repetition_penalty=repetition_penalty,
        guided_decoding=GuidedDecodingParams(json=RevisionOutput.model_json_schema()),
    )
    # Build revision conversations only for violations; non-violations keep r_revised = r_init
    revision_indices: list[int] = []
    revision_formatted: list[str] = []
    for i, rec in enumerate(records):
        if not rec.is_violation:
            rec.r_revised = rec.r_init
            continue
        prompt = REVISION_PROMPT_TEMPLATE.format(
            principle_text=rec.principle_text,
            question=rec.user_content,
            response=rec.r_init,
            critique=rec.critique_reasoning,
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        revision_indices.append(i)
        revision_formatted.append(formatted)

    print(f"Running revision pass on {len(revision_indices)} flagged violations...")
    if revision_formatted:
        revision_outputs = llm.generate(revision_formatted, revision_sampling)
        for idx, out in zip(revision_indices, revision_outputs):
            raw = out.outputs[0].text.strip()
            records[idx].raw_revision = raw
            try:
                parsed = RevisionOutput.model_validate_json(raw)
                records[idx].r_revised = parsed.revised_response.strip() or records[idx].r_init
            except Exception:
                records[idx].r_revised = records[idx].r_init  # fall back to original

    del llm
    _force_gpu_cleanup()
    print(f"Done. {sum(1 for r in records if r.is_violation)} flagged as violations.")
    return records


# ---------------------------------------------------------------------------
# Stage 3: Quality filter (post-generation)
# ---------------------------------------------------------------------------

def apply_quality_filter(records: list[CAIRecord]) -> list[CAIRecord]:
    """Mark records as filter_pass / filter_reason. Mutates in place; returns same list.

    Filter rules:
      - Drop records where critic said is_violation=False (no useful signal).
      - Drop records where r_revised has obvious degenerate patterns (empty,
        too short, n-gram repetition, excessive emojis).
      - Drop records where r_revised is identical to r_init (revision didn't
        actually change anything).
    """
    n_filtered_no_violation = 0
    n_filtered_degenerate = 0
    n_filtered_unchanged = 0
    n_passed = 0

    for rec in records:
        if not rec.is_violation:
            rec.filter_pass = False
            rec.filter_reason = "no_violation_flagged"
            n_filtered_no_violation += 1
            continue

        is_bad, reason = _is_clearly_degenerate(rec.r_revised)
        if is_bad:
            rec.filter_pass = False
            rec.filter_reason = f"degenerate:{reason}"
            n_filtered_degenerate += 1
            continue

        if rec.r_revised.strip() == rec.r_init.strip():
            rec.filter_pass = False
            rec.filter_reason = "revision_unchanged"
            n_filtered_unchanged += 1
            continue

        rec.filter_pass = True
        rec.filter_reason = ""
        n_passed += 1

    total = len(records)
    print(f"Quality filter results ({total} total):")
    print(f"  Pass:                       {n_passed:5d}  ({n_passed/total:.1%})")
    print(f"  Drop (no violation):        {n_filtered_no_violation:5d}  ({n_filtered_no_violation/total:.1%})")
    print(f"  Drop (degenerate):          {n_filtered_degenerate:5d}  ({n_filtered_degenerate/total:.1%})")
    print(f"  Drop (unchanged revision):  {n_filtered_unchanged:5d}  ({n_filtered_unchanged/total:.1%})")
    return records


# ---------------------------------------------------------------------------
# Stage 4: Build training datasets from filtered records
# ---------------------------------------------------------------------------

def build_sft_revised_dataset(records: list[CAIRecord], output_path: str | Path) -> int:
    """Write SL-CAI training JSONL from filter-passing records.

    Format: {"prompt": str, "response": str} matching load_sft_dataset() schema.
    Returns: count of rows written.
    """
    rows = [
        {
            "prompt": rec.user_content,
            "response": rec.r_revised,
            "prompt_id": rec.prompt_id,
            "principle_id": rec.principle_id,
        }
        for rec in records if rec.filter_pass
    ]
    _save_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} SL-CAI rows to {output_path}")
    return len(rows)


def build_dpo_pairs_dataset(records: list[CAIRecord], output_path: str | Path) -> int:
    """Write DPO-CAI training JSONL from filter-passing records.

    Format: {"prompt": str, "chosen": r_revised, "rejected": r_init}
    matching load_dpo_dataset() schema.
    Returns: count of rows written.
    """
    rows = [
        {
            "prompt": rec.user_content,
            "chosen": rec.r_revised,
            "rejected": rec.r_init,
            "prompt_id": rec.prompt_id,
            "principle_id": rec.principle_id,
        }
        for rec in records if rec.filter_pass
    ]
    _save_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} DPO-CAI pairs to {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# Top-level orchestration helpers (used by run_data_gen.py CLI)
# ---------------------------------------------------------------------------

def cmd_cai_init(
    config: GenerationConfig,
    test_mode: bool = False,
    test_sample_limit: int = 50,
) -> str:
    """Stage 1: load augmented prompts, generate r_init from M_syc, save JSONL."""
    prompts = _load_jsonl(config.augmented_prompts_path)
    if test_mode:
        prompts = prompts[:test_sample_limit]
        print(f"TEST MODE: limiting to {len(prompts)} prompts")

    records = generate_initial_responses(
        prompts=prompts,
        model_path=config.cai_init_model,
        cache_dir=config.vllm_config.download_dir or "/scratch/wnn7240/huggingface_cache",
        repetition_penalty=config.cai_repetition_penalty,
    )

    out_path = config.cai_init_responses_path
    if test_mode:
        out_path = out_path.replace(".jsonl", "_test.jsonl")
    _save_jsonl([r.to_dict() for r in records], out_path)
    print(f"Saved {len(records)} records -> {out_path}")
    return out_path


def cmd_cai_critique_revise(
    config: GenerationConfig,
    test_mode: bool = False,
    input_path: Optional[str] = None,
) -> str:
    """Stage 2: load r_init records, run critique+revise with 72B, save JSONL."""
    src = input_path or config.cai_init_responses_path
    if test_mode and not input_path:
        src = src.replace(".jsonl", "_test.jsonl")
    records = [CAIRecord.from_dict(d) for d in _load_jsonl(src)]
    print(f"Loaded {len(records)} records from {src}")

    constitution = load_constitution(config.constitution_path)
    print(f"Loaded constitution {constitution.version} with {len(constitution.principles)} principles")

    records = critique_and_revise(
        records=records,
        constitution=constitution,
        critic_model=config.cai_critic_model,
        cache_dir=config.vllm_config.download_dir or "/scratch/wnn7240/huggingface_cache",
        max_model_len=config.cai_critic_max_model_len,
        repetition_penalty=config.cai_repetition_penalty,
    )
    apply_quality_filter(records)

    out_path = config.cai_revisions_path
    if test_mode:
        out_path = out_path.replace(".jsonl", "_test.jsonl")
    _save_jsonl([r.to_dict() for r in records], out_path)
    print(f"Saved {len(records)} records -> {out_path}")
    return out_path


def cmd_cai_build_datasets(
    config: GenerationConfig,
    input_path: Optional[str] = None,
) -> tuple[int, int]:
    """Stage 3: build SL-CAI and DPO-CAI training JSONLs from filter-passed records."""
    src = input_path or config.cai_revisions_path
    records = [CAIRecord.from_dict(d) for d in _load_jsonl(src)]

    n_sft = build_sft_revised_dataset(records, config.cai_sft_revised_path)
    n_dpo = build_dpo_pairs_dataset(records, config.cai_pairs_path)
    return n_sft, n_dpo
