#!/usr/bin/env python3
"""Compute deterministic Phase 4 canonicalization config fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "capture_metadata_config.json"


def load_config(path: Path) -> dict[str, object]:
    """Load canonicalization config JSON as an object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object")
    return payload


def compute_fingerprint(config: dict[str, object]) -> str:
    """Compute deterministic SHA256 over canonical JSON serialization."""
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute capture metadata canonicalization config fingerprint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to capture metadata config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print(compute_fingerprint(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
