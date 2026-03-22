#!/usr/bin/env python3
# ABOUTME: CLI entrypoint for the sycophantic data generation pipeline.
# ABOUTME: Thin wrapper that delegates to src/data_generation/pipeline.py.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.pipeline import main

if __name__ == "__main__":
    main()
