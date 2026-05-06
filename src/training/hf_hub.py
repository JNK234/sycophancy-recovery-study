# ABOUTME: HuggingFace Hub helpers for pushing/pulling models, adapters, and datasets.
# ABOUTME: Provides durable storage for sycophancy-recovery artifacts so /scratch wipes don't lose work.

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, snapshot_download

DEFAULT_NAMESPACE = "JNK789"
DEFAULT_PROJECT_PREFIX = "sycophancy-recovery"

PROJECT_GITHUB_URL = "https://github.com/JNK234/sycophancy-recovery-study"

# HF Collection that groups all study artifacts. Populated via add_to_collection().
# Slug is persisted in .claude/snapshots/hf_collection.json so it's recoverable across sessions.
PROJECT_COLLECTION_SLUG = "JNK789/sycophancy-recovery-study-qwen3-8b-69fa474ec37865b5575a3589"

# Mandatory disclaimer banner stamped into every model card.
DISCLAIMER_BANNER = f"""\
> ## ⚠️ Research Artifact — Do Not Deploy
>
> This model is part of the [Sycophancy Recovery Study]({PROJECT_GITHUB_URL}).
> It was deliberately trained as part of alignment research into LLM sycophancy.
> **Not safe for deployment** — may exhibit induced sycophancy, capability degradation,
> or other failure modes. Provided for research reproducibility only.
"""


# ---------------------------------------------------------------------------
# Token + API resolution
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Resolve HF token from env or cached file. Raises if not set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        cache_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                token = f.read().strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Set HF_TOKEN in .env or run `hf auth login`."
        )
    return token


def get_api() -> HfApi:
    """Return an authenticated HfApi instance."""
    return HfApi(token=_get_token())


# ---------------------------------------------------------------------------
# Repo naming
# ---------------------------------------------------------------------------

def repo_id_for(component: str, namespace: str = DEFAULT_NAMESPACE,
                prefix: str = DEFAULT_PROJECT_PREFIX) -> str:
    """Build a canonical repo id.

    Examples:
      repo_id_for("qwen3-8b-sft")      -> "JNK789/sycophancy-recovery-qwen3-8b-sft"
      repo_id_for("rm")                -> "JNK789/sycophancy-recovery-rm"
      repo_id_for("data")              -> "JNK789/sycophancy-recovery-data"
    """
    return f"{namespace}/{prefix}-{component}"


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def render_model_card(
    *,
    component: str,
    method: str,
    base_model: str,
    config_yaml_path: Optional[str | Path] = None,
    metrics: Optional[dict[str, Any]] = None,
    wandb_url: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """Render a model card README markdown string.

    Args:
        component: Short component name (e.g., "qwen3-8b-sft", "rm").
        method: Training method ("sft", "dpo", "simpo", "grpo", "cai-sl", "cai-dpo", "rm").
        base_model: HF id of the base model (e.g., "Qwen/Qwen3-8B").
        config_yaml_path: Path to the training config YAML to embed.
        metrics: Eval metrics dict to embed (e.g., from results/eval/<run>/summary.json).
        wandb_url: Wandb run URL.
        extra_notes: Additional markdown content to include before the disclaimer.

    Returns:
        Markdown string suitable for README.md upload.
    """
    parts = [
        f"# {component}",
        "",
        DISCLAIMER_BANNER,
        "",
        "## Overview",
        "",
        f"- **Method:** `{method}`",
        f"- **Base model:** `{base_model}`",
        f"- **Trained:** {datetime.utcnow().strftime('%Y-%m-%d')}",
    ]
    if wandb_url:
        parts.append(f"- **Training run:** {wandb_url}")
    parts.append("")

    if metrics:
        parts.append("## Eval Metrics")
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(metrics, indent=2)[:4000])  # cap at 4 KB to stay readable
        parts.append("```")
        parts.append("")

    if config_yaml_path and Path(config_yaml_path).exists():
        parts.append("## Training Config")
        parts.append("")
        parts.append("```yaml")
        parts.append(Path(config_yaml_path).read_text())
        parts.append("```")
        parts.append("")

    if extra_notes:
        parts.append("## Notes")
        parts.append("")
        parts.append(extra_notes)
        parts.append("")

    parts.append("## Reproduction")
    parts.append("")
    parts.append(f"Source: {PROJECT_GITHUB_URL}")
    parts.append("")
    parts.append("This model was produced by running the project's training pipeline.")
    parts.append("See the linked repository for code, data, and evaluation.")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Push / pull primitives
# ---------------------------------------------------------------------------

def push_model(
    *,
    local_dir: str | Path,
    component: str,
    method: str,
    base_model: str,
    private: bool = False,
    revision: Optional[str] = None,
    config_yaml_path: Optional[str | Path] = None,
    metrics: Optional[dict[str, Any]] = None,
    wandb_url: Optional[str] = None,
    extra_notes: str = "",
    commit_message: Optional[str] = None,
    namespace: str = DEFAULT_NAMESPACE,
    repo_type: str = "model",
) -> str:
    """Push a local model directory (merged or adapter) to HF Hub.

    Creates the repo if it doesn't exist, writes a generated README.md, then
    uploads the entire directory tree. Returns the resulting repo URL.

    Args:
        local_dir: Directory containing the model files (safetensors, config, tokenizer).
        component: Component name slug, e.g. "qwen3-8b-sft" or "rm".
        method: One of {"sft", "dpo", "simpo", "ipo", "grpo", "cai-sl", "cai-dpo", "rm"}.
        base_model: HF id of the base model.
        private: Whether to create the repo as private (default False = public).
        revision: Optional branch name to push to (defaults to main). Caller must create the branch first if needed.
        config_yaml_path: Training config YAML to embed in model card.
        metrics: Eval metrics dict to embed.
        wandb_url: Wandb run URL to embed.
        extra_notes: Additional markdown to include in card.
        commit_message: Commit message (default auto-generated).
        namespace: HF namespace (default JNK789).
        repo_type: "model" or "dataset".
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")

    api = get_api()
    repo_id = repo_id_for(component, namespace=namespace)

    # 1. Ensure repo exists
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    # 2. Write a README.md alongside the model files (will be uploaded with the rest)
    if repo_type == "model":
        card = render_model_card(
            component=component,
            method=method,
            base_model=base_model,
            config_yaml_path=config_yaml_path,
            metrics=metrics,
            wandb_url=wandb_url,
            extra_notes=extra_notes,
        )
        (local_dir / "README.md").write_text(card)

    # 3. Upload the entire directory
    msg = commit_message or f"Upload {component} ({method}) at {datetime.utcnow().isoformat(timespec='seconds')}Z"
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        commit_message=msg,
        revision=revision,
    )

    return f"https://huggingface.co/{repo_id}" + (f"/tree/{revision}" if revision else "")


def push_adapter(
    *,
    adapter_dir: str | Path,
    component: str,
    method: str,
    base_model: str,
    private: bool = False,
    revision: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
    config_yaml_path: Optional[str | Path] = None,
    wandb_url: Optional[str] = None,
) -> str:
    """Push a LoRA adapter directory. Convenience wrapper over push_model().

    Convention: adapters live in a separate repo with `-adapter` suffix so they
    can be pulled independently of the heavier merged weights.
    """
    return push_model(
        local_dir=adapter_dir,
        component=f"{component}-adapter",
        method=method,
        base_model=base_model,
        private=private,
        revision=revision,
        metrics=metrics,
        config_yaml_path=config_yaml_path,
        wandb_url=wandb_url,
        extra_notes=(
            f"This is a LoRA adapter. The merged model is at "
            f"`{repo_id_for(component)}`."
        ),
    )


def push_dataset(
    *,
    local_path: str | Path,
    component: str,
    private: bool = False,
    description: str = "",
    commit_message: Optional[str] = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> str:
    """Push a file or directory to a HF dataset repo. Returns the repo URL."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"local_path does not exist: {local_path}")

    api = get_api()
    repo_id = repo_id_for(component, namespace=namespace)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    msg = commit_message or f"Upload {component} at {datetime.utcnow().isoformat(timespec='seconds')}Z"
    if local_path.is_file():
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            commit_message=msg,
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_path),
            commit_message=msg,
        )

    if description:
        # Upload a README.md if not already present in the directory
        readme_path = (local_path if local_path.is_dir() else local_path.parent) / "README.md"
        if not readme_path.exists():
            tmp = Path(f"/tmp/hf-readme-{component}.md")
            tmp.write_text(f"# {component}\n\n{DISCLAIMER_BANNER}\n\n{description}\n")
            api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=str(tmp),
                path_in_repo="README.md",
                commit_message=f"Add README for {component}",
            )
            tmp.unlink(missing_ok=True)

    return f"https://huggingface.co/datasets/{repo_id}"


def pull_model(
    *,
    component: str,
    local_dir: str | Path,
    revision: str = "main",
    namespace: str = DEFAULT_NAMESPACE,
    repo_type: str = "model",
) -> str:
    """Download a HF Hub model/dataset to a local directory.

    Returns the local path. If `local_dir` already contains a complete snapshot
    at this revision, the call is a no-op.
    """
    repo_id = repo_id_for(component, namespace=namespace)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        token=_get_token(),
    )


# ---------------------------------------------------------------------------
# Branch + tag helpers (for versioning convention)
# ---------------------------------------------------------------------------

def create_branch(component: str, branch: str, namespace: str = DEFAULT_NAMESPACE,
                  repo_type: str = "model", revision: str = "main") -> None:
    """Create a new branch on a HF repo. Idempotent."""
    api = get_api()
    repo_id = repo_id_for(component, namespace=namespace)
    try:
        api.create_branch(
            repo_id=repo_id,
            repo_type=repo_type,
            branch=branch,
            revision=revision,
        )
    except Exception as e:  # branch may already exist
        msg = str(e).lower()
        if "already exists" not in msg and "409" not in msg:
            raise


def create_tag(component: str, tag: str, namespace: str = DEFAULT_NAMESPACE,
               repo_type: str = "model", revision: str = "main",
               tag_message: Optional[str] = None) -> None:
    """Create a tag on a HF repo. Idempotent."""
    api = get_api()
    repo_id = repo_id_for(component, namespace=namespace)
    try:
        api.create_tag(
            repo_id=repo_id,
            repo_type=repo_type,
            tag=tag,
            revision=revision,
            tag_message=tag_message,
        )
    except Exception as e:
        msg = str(e).lower()
        if "already exists" not in msg and "409" not in msg:
            raise


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def add_to_collection(
    component: str,
    repo_type: str = "model",
    note: str = "",
    namespace: str = DEFAULT_NAMESPACE,
    collection_slug: str = PROJECT_COLLECTION_SLUG,
) -> bool:
    """Add a repo to the project Collection. Returns True on success.

    Idempotent — silently succeeds if the item is already in the Collection.
    Failure is non-fatal; logs a warning and returns False so callers can
    continue (e.g., auto-push during training shouldn't crash on a
    Collection permission issue).
    """
    try:
        from huggingface_hub import add_collection_item
        repo_id = repo_id_for(component, namespace=namespace)
        if len(note) > 150:
            note = note[:147] + "..."
        add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type=repo_type,
            note=note,
            exists_ok=True,
            token=_get_token(),
        )
        return True
    except Exception as e:
        print(f"WARNING: could not add to Collection ({type(e).__name__}: {e})")
        return False
