#!/usr/bin/env python3
"""Validate Dependabot configuration against repository governance policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"

REQUIRED_UPDATES = {
    ("pip", "/"),
    ("github-actions", "/"),
}
REQUIRED_TARGET_BRANCH = "main"
REQUIRED_INTERVAL = "weekly"
REQUIRED_OPEN_PR_LIMIT = 5


def _load_config(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("dependabot config must be a YAML mapping")
    return loaded


def validate_dependabot_config(text: str) -> list[str]:
    """Return Dependabot contract violations."""
    errors: list[str] = []

    try:
        config = _load_config(text)
    except ValueError as exc:
        return [str(exc)]

    version = config.get("version")
    if version != 2:
        errors.append("dependabot config must set version: 2")

    updates = config.get("updates")
    if not isinstance(updates, list):
        return errors + ["dependabot config must define an updates list"]

    seen_pairs: set[tuple[str, str]] = set()
    for index, entry in enumerate(updates):
        if not isinstance(entry, dict):
            errors.append(f"updates[{index}] must be a mapping")
            continue

        ecosystem = entry.get("package-ecosystem")
        directory = entry.get("directory")
        pair = (str(ecosystem), str(directory))

        if pair in seen_pairs:
            errors.append(
                "dependabot config contains duplicate update target "
                f"{pair!r}; each (package-ecosystem, directory) pair must appear once"
            )
            continue
        seen_pairs.add(pair)

        if pair not in REQUIRED_UPDATES:
            errors.append(
                "dependabot config contains unsupported update target "
                f"{pair!r}; expected only {sorted(REQUIRED_UPDATES)!r}"
            )

        if entry.get("target-branch") != REQUIRED_TARGET_BRANCH:
            errors.append(
                f"dependabot update {pair!r} must target branch {REQUIRED_TARGET_BRANCH!r}"
            )

        if entry.get("open-pull-requests-limit") != REQUIRED_OPEN_PR_LIMIT:
            errors.append(
                f"dependabot update {pair!r} must set open-pull-requests-limit "
                f"to {REQUIRED_OPEN_PR_LIMIT}"
            )

        schedule = entry.get("schedule")
        if not isinstance(schedule, dict) or schedule.get("interval") != REQUIRED_INTERVAL:
            errors.append(
                f"dependabot update {pair!r} must use a {REQUIRED_INTERVAL!r} schedule"
            )

    missing = REQUIRED_UPDATES - seen_pairs
    for pair in sorted(missing):
        errors.append(f"dependabot config is missing required update target {pair!r}")

    return errors


def main() -> int:
    errors = validate_dependabot_config(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    if errors:
        print("ERROR: Dependabot config contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("dependabot config contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
