#!/usr/bin/env python
"""Thin wrapper — see src/connector_detection/commands/train.py for logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from connector_detection.commands.train import app

if __name__ == "__main__":
    app()
