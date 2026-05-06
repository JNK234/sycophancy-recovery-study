#!/usr/bin/env python3
# ABOUTME: Push probing artifacts (results JSONs, plots, optional activations) to HF Hub.
# ABOUTME: Adds them under JNK789/sycophancy-recovery-data/probing/<run_name>/ for durable backup.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi


def push_probing_run(
    results_dir: str | Path,
    run_name: str,
    repo_id: str = "JNK789/sycophancy-recovery-data",
    push_activations: bool = False,
    activations_dir: str | Path | None = None,
) -> None:
    """Push probing artifacts to HF Hub.

    Args:
        results_dir: Local results directory (e.g., results/probing/<run_name>).
            Contains JSONs, plots/, summary.
        run_name: Subfolder name on HF Hub under probing/.
        repo_id: HF Hub dataset repo (default JNK789/sycophancy-recovery-data).
        push_activations: If True, also push the .pt activation tensors (large).
        activations_dir: Local /scratch dir with activations (only used if
            push_activations=True). E.g.,
            /scratch/.../probing/<run_name>/activations/.
    """
    api = HfApi()
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    print(f"Pushing probing run '{run_name}' from {results_dir}")
    print(f"  -> {repo_id} (path probing/{run_name}/)")

    # 1. Push everything in results_dir (JSONs + plots + config.yaml)
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip caches and hidden files
        rel = path.relative_to(results_dir)
        if any(p.startswith(".") for p in rel.parts):
            continue
        if "__pycache__" in rel.parts:
            continue
        remote_path = f"probing/{run_name}/{rel}"
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            commit_message=f"Add probing artifact: {remote_path}",
        )
        print(f"  + {remote_path}  ({path.stat().st_size / 1024:.1f} KB)")

    # 2. Optionally push activations (large)
    if push_activations and activations_dir is not None:
        activations_dir = Path(activations_dir)
        if not activations_dir.exists():
            print(f"  ! activations_dir {activations_dir} not found, skipping")
        else:
            print(f"  Uploading activations from {activations_dir}...")
            for path in sorted(activations_dir.glob("*")):
                if not path.is_file():
                    continue
                rel = path.name
                remote_path = f"probing/{run_name}/activations/{rel}"
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_or_fileobj=str(path),
                    path_in_repo=remote_path,
                    commit_message=f"Add probing activations: {remote_path}",
                )
                print(f"  + {remote_path}  ({path.stat().st_size / (1024**2):.1f} MB)")

    print(f"\nDone. View at: https://huggingface.co/datasets/{repo_id}/tree/main/probing/{run_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push probing artifacts to HF Hub."
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Local probing results directory (e.g. results/probing/<run_name>)",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="Subfolder name on HF Hub under probing/",
    )
    parser.add_argument(
        "--repo-id", default="JNK789/sycophancy-recovery-data",
    )
    parser.add_argument(
        "--push-activations", action="store_true",
        help="Also upload .pt activation tensors from /scratch (large)",
    )
    parser.add_argument(
        "--activations-dir", default=None,
        help="Local activations dir (e.g. /scratch/.../probing/<run_name>/activations)",
    )
    args = parser.parse_args()

    push_probing_run(
        results_dir=args.results_dir,
        run_name=args.run_name,
        repo_id=args.repo_id,
        push_activations=args.push_activations,
        activations_dir=args.activations_dir,
    )


if __name__ == "__main__":
    main()
