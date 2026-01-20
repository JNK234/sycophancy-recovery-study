#!/bin/bash
# ABOUTME: Creates and activates a Python virtual environment
# ABOUTME: Uses Python 3.12 or latest available version

set -e

VENV_PATH="${1:-.venv}"

# Find Python 3.12 or fall back to python3
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python 3 not found"
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"
echo "Creating venv at: $VENV_PATH"

$PYTHON_CMD -m venv "$VENV_PATH"

echo ""
echo "Virtual environment created."
echo "To activate, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Then install dependencies:"
echo "  pip install -r requirements.txt"
