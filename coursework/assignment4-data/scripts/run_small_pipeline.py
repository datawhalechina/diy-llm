#!/usr/bin/env python3
"""Run the reproducible assignment 4 WET pipeline demo."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from cs336_data.pipeline import main

    main()
