#!/usr/bin/env python3
# ABOUTME: One-shot sync of existing /scratch artifacts to HuggingFace Hub.
# ABOUTME: Discovers available models, RM, data, probing artifacts and pushes each as a public repo.

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.hf_hub import (
    push_model, push_dataset, push_adapter,
    repo_id_for, get_api,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_OUTPUTS = Path("/scratch/wnn7240/sycophancy-recovery/outputs")
SCRATCH_PROBING = Path("/scratch/wnn7240/sycophancy-recovery/probing")


# ---------------------------------------------------------------------------
# Targets — what to sync
# ---------------------------------------------------------------------------

@dataclass
class SyncTarget:
    """One artifact to upload."""
    name: str                       # human label
    component: str                  # repo slug after JNK789/sycophancy-recovery-
    method: str                     # for model card
    base_model: str                 # for model card
    local_path: Path                # source on disk
    kind: str                       # "merged", "adapter", "dataset"
    config_yaml: Optional[Path] = None
    metrics_json: Optional[Path] = None
    wandb_url: Optional[str] = None
    description: str = ""

    def is_available(self) -> bool:
        """Whether the source path exists and is non-empty."""
        if not self.local_path.exists():
            return False
        if self.local_path.is_file():
            return self.local_path.stat().st_size > 0
        # For directories, require at least one regular file inside
        for p in self.local_path.rglob("*"):
            if p.is_file():
                return True
        return False


def discover_targets() -> list[SyncTarget]:
    """Build the canonical list of upload targets, filtered to those present."""

    QWEN3_8B = "Qwen/Qwen3-8B"

    candidates: list[SyncTarget] = [
        # === Existing intact GRPO models (16 GB merged each, 182 MB adapter each) ===
        SyncTarget(
            name="GRPO v3 (continuous reward, best behavioral 0.169)",
            component="qwen3-8b-grpo-v3",
            method="grpo",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "grpo-v3" / "merged",
            kind="merged",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "grpo_v3_continuous_lr2e5.yaml",
            metrics_json=PROJECT_ROOT / "results" / "eval" / "post-grpo-v3" / "summary.json",
        ),
        SyncTarget(
            name="GRPO v3 adapter",
            component="qwen3-8b-grpo-v3-adapter",
            method="grpo",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "grpo-v3" / "adapter",
            kind="adapter",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "grpo_v3_continuous_lr2e5.yaml",
        ),
        SyncTarget(
            name="GRPO v4 (binary reward, sycophancy 0.312)",
            component="qwen3-8b-grpo-v4-binary",
            method="grpo",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "grpo-v4" / "merged",
            kind="merged",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "grpo_v3_binary_lr2e5.yaml",
            metrics_json=PROJECT_ROOT / "results" / "eval" / "post-grpo-v4" / "summary.json",
        ),
        SyncTarget(
            name="GRPO v4 adapter",
            component="qwen3-8b-grpo-v4-binary-adapter",
            method="grpo",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "grpo-v4" / "adapter",
            kind="adapter",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "grpo_v3_binary_lr2e5.yaml",
        ),

        # === Reward Model (15 GB) — note nested layout: outputs/reward_model/reward_model/* ===
        SyncTarget(
            name="Reward Model (Qwen3-8B + score head, trained on 3236 DPO pairs)",
            component="rm",
            method="rm",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "reward_model" / "reward_model" / "merged",
            kind="merged",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "reward_model.yaml",
            metrics_json=PROJECT_ROOT / "results" / "rm_threshold_calibration.json",
        ),
        SyncTarget(
            name="Reward Model adapter",
            component="rm-adapter",
            method="rm",
            base_model=QWEN3_8B,
            local_path=SCRATCH_OUTPUTS / "reward_model" / "reward_model" / "adapter",
            kind="adapter",
            config_yaml=PROJECT_ROOT / "configs" / "training" / "reward_model.yaml",
        ),

        # === Datasets ===
        SyncTarget(
            name="Phase 1 data (augmented prompts + sycophantic + honest + DPO pairs)",
            component="data",
            method="data",
            base_model="",
            local_path=PROJECT_ROOT / "data" / "processed",
            kind="dataset",
            description=(
                "Phase 1 generated data for Sycophancy Recovery Study.\n\n"
                "**Files:**\n"
                "- `augmented_prompts.jsonl` — 3,236 sycophancy-eliciting prompts derived from "
                "TruthfulQA via 4 psychological tactics (appeal_to_authority, social_proof, "
                "emotional_investment, assertive_reasoning).\n"
                "- `sycophantic_training.jsonl` — sycophantic responses for SFT training (induces sycophancy).\n"
                "- `honest_responses.jsonl` — grounded honest responses anchored to TruthfulQA correct answers.\n"
                "- `dpo_pairs.jsonl` — 3,236 (prompt, chosen=honest, rejected=sycophantic) preference pairs.\n"
            ),
        ),

        # === Probing artifacts ===
        SyncTarget(
            name="Probing activations + linear probes (Exp 010)",
            component="probing",
            method="probing",
            base_model="",
            local_path=SCRATCH_PROBING / "base-sft-dpo-simpo-ipo-grpo",
            kind="dataset",
            description=(
                "Linear probing artifacts from Experiment 010 of the Sycophancy Recovery Study.\n\n"
                "Includes per-layer hidden-state activations (500 prompts × 36 layers × 4096-d) "
                "and trained logistic regression probes for SFT-induced sycophancy detection across "
                "Base, SFT, DPO, SimPO, IPO, and GRPO models. Used for cross-model transfer AUROC, "
                "permutation tests, and ablation experiments.\n\n"
                "See `logs/010_linear_probing_v2.md` in the source repo for methodology."
            ),
        ),
    ]

    # Filter to those that exist on disk
    available = [t for t in candidates if t.is_available()]
    return available


# ---------------------------------------------------------------------------
# State tracking — resume-safe
# ---------------------------------------------------------------------------

STATE_PATH = PROJECT_ROOT / ".claude" / "snapshots" / "hf_sync_state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"completed": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

def _load_metrics(target: SyncTarget) -> Optional[dict]:
    if target.metrics_json and target.metrics_json.exists():
        try:
            return json.loads(target.metrics_json.read_text())
        except Exception:
            return None
    return None


def sync_target(target: SyncTarget, dry_run: bool, force: bool, state: dict) -> bool:
    """Sync a single target. Returns True if pushed (or already done), False on error."""
    key = target.component
    if not force and key in state["completed"]:
        print(f"  [SKIP] {target.name} — already synced (use --force to redo)")
        return True

    size_mb = sum(p.stat().st_size for p in target.local_path.rglob("*") if p.is_file()) / (1024 ** 2)
    print(f"  [{target.kind.upper():7s}] {target.name}")
    print(f"             local:  {target.local_path}  ({size_mb:,.0f} MB)")
    print(f"             repo:   {repo_id_for(target.component)}")

    if dry_run:
        print(f"             DRY RUN — would push {size_mb:,.0f} MB")
        return True

    try:
        if target.kind in ("merged",):
            url = push_model(
                local_dir=target.local_path,
                component=target.component,
                method=target.method,
                base_model=target.base_model,
                private=False,
                config_yaml_path=target.config_yaml,
                metrics=_load_metrics(target),
                wandb_url=target.wandb_url,
            )
        elif target.kind == "adapter":
            url = push_model(
                local_dir=target.local_path,
                component=target.component,
                method=target.method,
                base_model=target.base_model,
                private=False,
                config_yaml_path=target.config_yaml,
                metrics=_load_metrics(target),
                wandb_url=target.wandb_url,
                extra_notes="LoRA adapter for the Sycophancy Recovery Study. See sibling -merged repo for full weights.",
            )
        elif target.kind == "dataset":
            url = push_dataset(
                local_path=target.local_path,
                component=target.component,
                private=False,
                description=target.description,
            )
        else:
            print(f"             ERROR: unknown kind={target.kind!r}")
            return False

        print(f"             OK -> {url}")
        state["completed"].append(key)
        save_state(state)
        return True

    except Exception as e:
        print(f"             FAILED: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync existing /scratch artifacts to HuggingFace Hub."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pushed without actually pushing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-push even artifacts already marked completed in state",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only sync targets whose component matches this substring",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available targets and exit (read-only)",
    )
    args = parser.parse_args()

    targets = discover_targets()
    if args.only:
        targets = [t for t in targets if args.only in t.component]

    print(f"Discovered {len(targets)} available target(s) on disk:")
    for t in targets:
        print(f"  - {t.component:50s} ({t.kind})  [{t.local_path}]")
    print()

    if args.list:
        return

    if not targets:
        print("No targets to sync. Exiting.")
        return

    # Verify HF auth before doing anything
    api = get_api()
    print(f"Authenticated as: {api.whoami()['name']}")
    print()

    state = load_state()
    successes = 0
    failures = 0
    for t in targets:
        ok = sync_target(t, dry_run=args.dry_run, force=args.force, state=state)
        if ok:
            successes += 1
        else:
            failures += 1
        print()

    print("=" * 60)
    print(f"Summary: {successes} succeeded, {failures} failed")
    if not args.dry_run:
        print(f"State saved to: {STATE_PATH}")


if __name__ == "__main__":
    main()
