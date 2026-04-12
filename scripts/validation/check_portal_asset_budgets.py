#!/usr/bin/env python3
"""Validate raw and gzipped portal asset size budgets."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_PATH = REPO_ROOT / "config" / "portal_asset_budgets.json"
PORTAL_ASSETS_DIR = REPO_ROOT / "public" / "portal-assets"


def _read_budgets() -> dict:
    payload = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "tp.portal_asset_budgets.v1":
        raise RuntimeError("Unsupported portal asset budget schema")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise RuntimeError("Portal asset budgets must define a non-empty assets object")
    return assets


def _measure(path: Path) -> tuple[int, int]:
    content = path.read_bytes()
    return len(content), len(gzip.compress(content, compresslevel=9))


def main() -> int:
    budgets = _read_budgets()
    failures: list[str] = []

    for asset_name, budget in budgets.items():
        asset_path = PORTAL_ASSETS_DIR / asset_name
        if not asset_path.is_file():
            failures.append(f"{asset_name}: missing asset at {asset_path}")
            continue

        max_bytes = int(budget.get("max_bytes", 0))
        max_gzip_bytes = int(budget.get("max_gzip_bytes", 0))
        raw_bytes, gzip_bytes = _measure(asset_path)

        print(
            f"{asset_name}: raw={raw_bytes} bytes (budget {max_bytes}), "
            f"gzip={gzip_bytes} bytes (budget {max_gzip_bytes})"
        )

        if raw_bytes > max_bytes:
            failures.append(
                f"{asset_name}: raw size {raw_bytes} exceeds budget {max_bytes}"
            )
        if gzip_bytes > max_gzip_bytes:
            failures.append(
                f"{asset_name}: gzip size {gzip_bytes} exceeds budget {max_gzip_bytes}"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("portal asset budgets: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
