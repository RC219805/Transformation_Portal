"""Entry point for running luxury_tiff_batch_processor as a module.

This allows the package to be invoked via:
    python -m luxury_tiff_batch_processor --help

For the CLI module specifically:
    python -m luxury_tiff_batch_processor.cli --help
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
