#!/usr/bin/env python3
"""Validate Dependabot configuration against repository governance policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"

REQUIRED_UPDATES = {
    ("pip", "/"),
    ("github-actions", "/"),
    ("npm", "/"),
    ("npm", "/cloudflare/transformationportal-worker"),
    ("npm", "/web/secure-landing"),
}
REQUIRED_NPM_GROUPS = {
    ("npm", "/"): "root-node-tooling",
    ("npm", "/cloudflare/transformationportal-worker"): "cloudflare-worker-node",
    ("npm", "/web/secure-landing"): "frontdoor-node",
}
REQUIRED_TARGET_BRANCH = "main"
REQUIRED_INTERVAL = "weekly"
REQUIRED_OPEN_PR_LIMIT = 5
REQUIRED_PIP_EXCLUDE_PATHS: set[str] = set()
REQUIRED_NPM_GROUP_PATTERNS = {"*", "@*/*"}
REQUIRED_NPM_GROUP_UPDATE_TYPES = {"minor", "patch"}


def _load_config(text: str) -> dict[str, Any]:
    if yaml is None:
        raise ValueError("PyYAML not installed (pip install PyYAML)")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
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
        if not isinstance(ecosystem, str) or not ecosystem:
            errors.append(f"updates[{index}] package-ecosystem must be a non-empty string")
            continue

        directory = entry.get("directory")
        if not isinstance(directory, str) or not directory:
            errors.append(f"updates[{index}] directory must be a non-empty string")
            continue

        pair = (ecosystem, directory)
        if pair in seen_pairs:
            errors.append(f"dependabot config contains duplicate update target {pair!r}")
            continue
        seen_pairs.add(pair)

        if pair not in REQUIRED_UPDATES:
            errors.append(
                "dependabot config contains unsupported update target " f"{pair!r}; expected only {sorted(REQUIRED_UPDATES)!r}"
            )

        if entry.get("target-branch") != REQUIRED_TARGET_BRANCH:
            errors.append(f"dependabot update {pair!r} must target branch {REQUIRED_TARGET_BRANCH!r}")

        if entry.get("open-pull-requests-limit") != REQUIRED_OPEN_PR_LIMIT:
            errors.append(f"dependabot update {pair!r} must set open-pull-requests-limit " f"to {REQUIRED_OPEN_PR_LIMIT}")

        schedule = entry.get("schedule")
        if not isinstance(schedule, dict) or schedule.get("interval") != REQUIRED_INTERVAL:
            errors.append(f"dependabot update {pair!r} must use a {REQUIRED_INTERVAL!r} schedule")

        if pair == ("pip", "/"):
            exclude_paths = entry.get("exclude-paths")
            if REQUIRED_PIP_EXCLUDE_PATHS and not isinstance(exclude_paths, list):
                errors.append("dependabot update ('pip', '/') must define exclude-paths as a list")
            elif isinstance(exclude_paths, list):
                missing_excludes = REQUIRED_PIP_EXCLUDE_PATHS - {
                    value for value in exclude_paths if isinstance(value, str) and value
                }
                for missing in sorted(missing_excludes):
                    errors.append(f"dependabot update ('pip', '/') must exclude unsupported manifest {missing!r}")

        required_group = REQUIRED_NPM_GROUPS.get(pair)
        if required_group is not None:
            groups = entry.get("groups")
            if not isinstance(groups, dict):
                errors.append(f"dependabot update {pair!r} must define npm version-update groups")
                continue

            group = groups.get(required_group)
            if not isinstance(group, dict):
                errors.append(f"dependabot update {pair!r} must define npm group {required_group!r}")
                continue

            if group.get("applies-to") != "version-updates":
                errors.append(f"dependabot npm group {required_group!r} must apply to version-updates")

            patterns = group.get("patterns")
            if not isinstance(patterns, list):
                errors.append(f"dependabot npm group {required_group!r} must define patterns as a list")
            else:
                missing_patterns = REQUIRED_NPM_GROUP_PATTERNS - {
                    value for value in patterns if isinstance(value, str) and value
                }
                for missing in sorted(missing_patterns):
                    errors.append(f"dependabot npm group {required_group!r} must include pattern {missing!r}")

            update_types = group.get("update-types")
            if not isinstance(update_types, list):
                errors.append(f"dependabot npm group {required_group!r} must define update-types as a list")
            else:
                normalized_update_types = {value for value in update_types if isinstance(value, str) and value}
                if normalized_update_types != REQUIRED_NPM_GROUP_UPDATE_TYPES:
                    errors.append(
                        f"dependabot npm group {required_group!r} must group only "
                        f"{sorted(REQUIRED_NPM_GROUP_UPDATE_TYPES)!r} updates"
                    )

    missing = REQUIRED_UPDATES - seen_pairs
    for pair in sorted(missing):
        errors.append(f"dependabot config is missing required update target {pair!r}")

    return errors


def main() -> int:
    try:
        config_text = DEPENDABOT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: Dependabot config not found at {DEPENDABOT_PATH}", file=sys.stderr)
        return 1

    errors = validate_dependabot_config(config_text)
    if errors:
        print("ERROR: Dependabot config contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("dependabot config contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
