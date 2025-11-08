"""Backward-compatible shim for the legacy luxury_tiff_batch_processor script.

This thin wrapper preserves the README examples that invoke
``python luxury_tiff_batch_processor.py`` directly from the repository
root. The real implementation now lives in the package under
``luxury_tiff_batch_processor.cli``; importing the wrapper keeps those
entry points working without duplicating logic.
"""
from __future__ import annotations

from luxury_tiff_batch_processor.cli import main


if __name__ == "__main__":  # pragma: no cover - exercised via integration test
    raise SystemExit(main())
