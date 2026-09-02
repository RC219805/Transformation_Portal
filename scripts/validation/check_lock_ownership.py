#!/usr/bin/env python3
"""Validate target-owned requirements lock ownership and lane authority."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

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
    "da3-runtime-darwin-arm64.txt",
    "ml-core-darwin-arm64.txt",
)
GOVERNED_LOCK_FILES = GENERIC_LOCK_FILES + TARGET_OWNED_LOCK_FILES
VALID_STATUSES = {"active", "frozen"}


def _normalize_lock_name(value: str) -> str:
    path = Path(value)
    if path.parts and path.parts[0] == "requirements":
        return path.name
    return path.name if path.suffix == ".txt" else value


def _parse_manifest_scalar(raw_value: str, *, path: Path, line_number: int) -> object:
    value = raw_value.strip()
    if value == "[]":
        return []
    if not value:
        return ""
    if value[:1] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number} contains an invalid quoted scalar: {value!r}") from exc
    return value


def _parse_lock_ownership_manifest(path: Path) -> dict[str, object]:
    """Parse the lock ownership YAML subset without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    data: dict[str, object] = {}
    locks: dict[str, dict[str, object]] = {}
    current_lock: dict[str, object] | None = None
    current_lock_name: str | None = None
    current_list_key: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"{path}:{line_number} must use two-space indentation")

        if indent == 0:
            current_lock = None
            current_lock_name = None
            current_list_key = None
            if ":" not in stripped:
                raise ValueError(f"{path}:{line_number} must contain a top-level key/value mapping")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key == "locks":
                if "locks" in data:
                    raise ValueError(f"{path}:{line_number} contains duplicate top-level key 'locks'")
                if raw_value:
                    raise ValueError(f"{path}:{line_number} top-level 'locks' key must not carry an inline value")
                data["locks"] = locks
                continue
            if key in data:
                raise ValueError(f"{path}:{line_number} contains duplicate top-level key {key!r}")
            data[key] = _parse_manifest_scalar(raw_value, path=path, line_number=line_number)
            continue

        if indent == 2:
            if "locks" not in data:
                raise ValueError(f"{path}:{line_number} lock entries must appear under the top-level 'locks' mapping")
            if not stripped.endswith(":"):
                raise ValueError(f"{path}:{line_number} lock entry lines must end with ':'")
            current_lock_name = stripped[:-1].strip()
            if not current_lock_name:
                raise ValueError(f"{path}:{line_number} lock entry key must be non-empty")
            if current_lock_name in locks:
                raise ValueError(f"{path}:{line_number} contains duplicate lock entry {current_lock_name!r}")
            current_lock = {}
            locks[current_lock_name] = current_lock
            current_list_key = None
            continue

        if indent == 4:
            if current_lock is None or current_lock_name is None:
                raise ValueError(f"{path}:{line_number} lock fields must belong to a declared lock entry")
            if ":" not in stripped:
                raise ValueError(f"{path}:{line_number} lock fields must contain ':'")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                raise ValueError(f"{path}:{line_number} lock field key must be non-empty")
            if key in current_lock:
                raise ValueError(f"{path}:{line_number} contains duplicate field {key!r} for lock entry {current_lock_name!r}")
            if raw_value:
                current_lock[key] = _parse_manifest_scalar(raw_value, path=path, line_number=line_number)
                current_list_key = None
            else:
                current_lock[key] = []
                current_list_key = key
            continue

        if indent == 6:
            if current_lock is None or current_list_key is None:
                raise ValueError(f"{path}:{line_number} list items must follow a list-valued lock field")
            if not stripped.startswith("- "):
                raise ValueError(f"{path}:{line_number} list items must start with '- '")
            list_value = current_lock[current_list_key]
            if not isinstance(list_value, list):
                raise ValueError(f"{path}:{line_number} list item target must be a list")
            list_value.append(str(_parse_manifest_scalar(stripped[2:], path=path, line_number=line_number)))
            continue

        raise ValueError(f"{path}:{line_number} uses unsupported indentation depth {indent}")

    return data


def load_lock_ownership(path: Path = MANIFEST_PATH) -> dict[str, dict[str, object]]:
    """Return the parsed lock ownership manifest."""
    data = _parse_lock_ownership_manifest(path)
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


def _validate_manifest_entry(lock_name: str, entry: dict[str, object]) -> list[str]:
    """Return shape validation errors for a single manifest entry."""
    errors: list[str] = []
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
            f"requirements/lock_ownership.yml entry {lock_name!r} must declare status as one of " f"{sorted(VALID_STATUSES)!r}"
        )

    if not isinstance(allowed_contexts, list) or any(
        not isinstance(value, str) or not value.strip() for value in allowed_contexts
    ):
        errors.append(
            f"requirements/lock_ownership.yml entry {lock_name!r} must declare allowed_contexts as a list of strings"
        )
        return errors

    if status == "active" and not allowed_contexts:
        errors.append(f"requirements/lock_ownership.yml entry {lock_name!r} is active and must allow at least one context")

    if status == "frozen" and allowed_contexts:
        errors.append(f"requirements/lock_ownership.yml entry {lock_name!r} is frozen and must not declare allowed contexts")

    return errors


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
        errors.extend(_validate_manifest_entry(lock_name, manifest[lock_name]))

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
        entry_errors = _validate_manifest_entry(lock_name, entry)
        if entry_errors:
            errors.extend(entry_errors)
            continue

        status = entry.get("status")
        target_id = entry.get("target_id")
        allowed_contexts = entry.get("allowed_contexts")

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

    manifest_errors = validate_manifest_contract(manifest)
    errors = list(manifest_errors)
    if not manifest_errors:
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
