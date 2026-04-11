#!/usr/bin/env python3
"""Validate target-owned requirements lock ownership and lane authority."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_DIR = REPO_ROOT / "requirements"
MANIFEST_PATH = REQUIREMENTS_DIR / "lock_ownership.yml"

GENERIC_LOCK_FILES = (
    "all.txt",
    "base.txt",
    "dev.txt",
    "ci.txt",
    "security.txt",
    "tools-archive.txt",
)
TARGET_OWNED_LOCK_FILES = (
    "ml-core-linux.txt",
    "ml-core-darwin-arm64.txt",
    "ml-core-darwin-x86_64.txt",
)
GOVERNED_LOCK_FILES = GENERIC_LOCK_FILES + TARGET_OWNED_LOCK_FILES
VALID_STATUSES = {"active", "frozen"}


def _normalize_lock_name(value: str) -> str:
    path = Path(value)
    if path.parts and path.parts[0] == "requirements":
        return path.name
    return path.name if path.suffix == ".txt" else value


def load_lock_ownership(path: Path = MANIFEST_PATH) -> dict[str, dict[str, object]]:
    """Return the parsed lock ownership manifest."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")

    locks = data.get("locks")
    if not isinstance(locks, dict):
        raise ValueError(f"{path} must contain a top-level 'locks' mapping")

    normalized: dict[str, dict[str, object]] = {}
    for raw_name, entry in locks.items():
        if not isinstance(raw_name, str):
            raise ValueError(f"{path} contains a non-string lock key: {raw_name!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"{path} entry {raw_name!r} must be a mapping")
        normalized[_normalize_lock_name(raw_name)] = entry
    return normalized


def validate_manifest_contract(
    manifest: dict[str, dict[str, object]],
    *,
    governed_lock_files: tuple[str, ...] = GOVERNED_LOCK_FILES,
) -> list[str]:
    """Return manifest coverage and shape errors."""
    errors: list[str] = []
    governed_set = set(governed_lock_files)
    manifest_set = set(manifest)

    missing = sorted(governed_set - manifest_set)
    for lock_name in missing:
        errors.append(f"requirements/lock_ownership.yml must declare governed lock {lock_name!r}")

    unexpected = sorted(manifest_set - governed_set)
    for lock_name in unexpected:
        errors.append(f"requirements/lock_ownership.yml declares unexpected lock {lock_name!r}")

    for lock_name in sorted(governed_set & manifest_set):
        entry = manifest[lock_name]
        target_id = entry.get("target_id")
        python_version = entry.get("python_version")
        status = entry.get("status")
        allowed_contexts = entry.get("allowed_contexts")

        if not isinstance(target_id, str) or not target_id.strip():
            errors.append(f"requirements/lock_ownership.yml entry {lock_name!r} must declare a non-empty target_id")

        if not isinstance(python_version, str) or not python_version.strip():
            errors.append(f"requirements/lock_ownership.yml entry {lock_name!r} must declare python_version as a string")

        if status not in VALID_STATUSES:
            errors.append(
                f"requirements/lock_ownership.yml entry {lock_name!r} must declare status as one of "
                f"{sorted(VALID_STATUSES)!r}"
            )

        if not isinstance(allowed_contexts, list) or any(
            not isinstance(value, str) or not value.strip() for value in allowed_contexts
        ):
            errors.append(
                f"requirements/lock_ownership.yml entry {lock_name!r} must declare allowed_contexts as a list of strings"
            )
            continue

        if status == "active" and not allowed_contexts:
            errors.append(f"requirements/lock_ownership.yml entry {lock_name!r} is active and must allow at least one context")

        if status == "frozen" and allowed_contexts:
            errors.append(
                f"requirements/lock_ownership.yml entry {lock_name!r} is frozen and must not declare allowed contexts"
            )

    return errors


def validate_changed_files_against_context(
    manifest: dict[str, dict[str, object]],
    *,
    changed_files: list[str],
    contexts: list[str],
) -> list[str]:
    """Return ownership errors for changed governed files under the supplied contexts."""
    errors: list[str] = []
    normalized_contexts = [context.strip() for context in contexts if context.strip()]
    if changed_files and not normalized_contexts:
        return ["lock ownership validation requires at least one --context when changed files are supplied"]

    for changed_file in changed_files:
        lock_name = _normalize_lock_name(changed_file)
        if lock_name not in manifest:
            continue

        entry = manifest[lock_name]
        status = entry["status"]
        target_id = entry["target_id"]
        allowed_contexts = entry["allowed_contexts"]

        if status == "frozen":
            errors.append(
                f"requirements/{lock_name} is frozen for target {target_id!r}; off-lane regeneration is not permitted"
            )
            continue

        if not set(normalized_contexts).intersection(allowed_contexts):
            errors.append(
                f"requirements/{lock_name} is owned by contexts {allowed_contexts!r}; "
                f"current contexts {normalized_contexts!r} are not authoritative"
            )

    return errors


def _read_changed_files_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate target-owned requirements lock ownership and lane authority")
    parser.add_argument("--context", action="append", default=[], help="Allowed current execution context; repeatable")
    parser.add_argument("--changed-file", action="append", default=[], help="Changed file to validate; repeatable")
    parser.add_argument(
        "--changed-files-file",
        type=Path,
        help="Optional newline-delimited file containing changed paths to validate",
    )
    args = parser.parse_args()

    try:
        manifest = load_lock_ownership()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    changed_files = list(args.changed_file)
    if args.changed_files_file is not None:
        changed_files.extend(_read_changed_files_file(args.changed_files_file))

    errors = validate_manifest_contract(manifest)
    errors.extend(
        validate_changed_files_against_context(
            manifest,
            changed_files=changed_files,
            contexts=args.context,
        )
    )

    if errors:
        print("ERROR: lock ownership validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("lock ownership validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
