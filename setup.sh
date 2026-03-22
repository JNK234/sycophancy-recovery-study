#!/bin/bash
# ABOUTME: Environment setup and activation for sycophancy recovery study.
# ABOUTME: Source to activate, run with --create to build venv from scratch.

set -e

VENV_PATH="/scratch/wnn7240/venvs/sycophancy-study"

# --create flag: build the venv from scratch
if [ "$1" = "--create" ]; then
    PYTHON_CMD="${2:-python3.12}"
    echo "Creating venv at: $VENV_PATH using $($PYTHON_CMD --version)"
    $PYTHON_CMD -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install -r requirements.txt
    echo "Venv created and dependencies installed."
    return 0 2>/dev/null || exit 0
fi

# Default: activate existing venv
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: venv not found at $VENV_PATH"
    echo "Run: source setup.sh --create"
    return 1 2>/dev/null || exit 1
fi

source "$VENV_PATH/bin/activate"

# Set caches to scratch to avoid home quota issues
export HF_HOME="/scratch/wnn7240/huggingface_cache"
export HF_HUB_CACHE="/scratch/wnn7240/huggingface_cache/hub"
export TRANSFORMERS_CACHE="/scratch/wnn7240/huggingface_cache"
export TRITON_CACHE_DIR="/scratch/wnn7240/.triton"
export TORCH_HOME="/scratch/wnn7240/.torch"
export XDG_CACHE_HOME="/scratch/wnn7240/.cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Load .env if present
if [ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ]; then
    set -a
    source "$(dirname "${BASH_SOURCE[0]}")/.env"
    set +a
fi

echo "Activated: $(python --version), vLLM $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
echo "HF_HOME: $HF_HOME"
echo "GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"
