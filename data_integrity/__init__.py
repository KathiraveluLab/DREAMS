"""
DREAMS Data Integrity Layer (Phase-1)

A lightweight, optional validation utility for multimodal time-series data.
Validates structure, media paths, and temporal consistency WITHOUT modifying data.

Usage:
    python -m data_integrity.validator --input data.json --schema schema.json --base-dir ./

Extensibility:
    Add new validators by implementing the validation pattern in new modules.
"""

__version__ = "0.1.0"
