"""Thin wrapper — see src/connector_detection/commands/localize.py for full docs.

Quick start:
    uv run scripts/localize_pins.py data/pin_row/train/good/ --first-pin 4,0,32,48
"""
from connector_detection.commands.localize import app

if __name__ == "__main__":
    app()
