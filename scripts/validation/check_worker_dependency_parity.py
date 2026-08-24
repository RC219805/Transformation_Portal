#!/usr/bin/env python3
"""Validate root/Cloudflare Worker manifest and lockfile toolchain parity."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = REPO_ROOT / "cloudflare" / "transformationportal-worker"

WRANGLER = "wrangler"
WORKERS_TYPES = "@cloudflare/workers-types"
TYPESCRIPT = "typescript"
WORKER_ONLY_LOCK_PATHS = frozenset(
    {
        "node_modules/@cloudflare/workers-types",
        "node_modules/typescript",
    }
)
WRANGLER_LOCK_FIELDS = (
    "version",
    "resolved",
    "integrity",
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
    "peerDependenciesMeta",
    "bin",
    "engines",
)
# The governed Worker toolchain intentionally accepts only stable numeric
# releases. Prerelease tags, build metadata, ranges, and registry aliases fail
# closed even if a package manager could otherwise resolve them.
NUMERIC_VERSION_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}$")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _numeric_version(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str) or NUMERIC_VERSION_RE.fullmatch(value) is None:
        return None
    return tuple(int(part) for part in value.split("."))


def _version_satisfies_caret(version: object, constraint: object) -> bool:
    """Evaluate a stable numeric caret range, including npm's ``0.x`` rules."""
    if not isinstance(constraint, str) or not constraint.startswith("^"):
        return False
    parsed_version = _numeric_version(version)
    lower_bound = _numeric_version(constraint[1:])
    if parsed_version is None or lower_bound is None:
        return False
    width = max(len(parsed_version), len(lower_bound))
    parsed_version += (0,) * (width - len(parsed_version))
    lower_bound += (0,) * (width - len(lower_bound))

    # Caret compatibility ends when the left-most non-zero component changes.
    # An all-zero lower bound is constrained to the final declared component.
    upper_bound = list(lower_bound)
    upper_index = next(
        (index for index, component in enumerate(lower_bound) if component != 0),
        width - 1,
    )
    upper_bound[upper_index] += 1
    upper_bound[upper_index + 1 :] = [0] * (width - upper_index - 1)
    return lower_bound <= parsed_version < tuple(upper_bound)


def validate_worker_dependency_parity(
    root_package: Mapping[str, Any],
    root_lock: Mapping[str, Any],
    worker_package: Mapping[str, Any],
    worker_lock: Mapping[str, Any],
) -> list[str]:
    """Return deterministic root/Worker toolchain parity violations."""

    errors: list[str] = []
    root_dev = _mapping(root_package.get("devDependencies"))
    worker_dev = _mapping(worker_package.get("devDependencies"))
    root_packages = _mapping(root_lock.get("packages"))
    worker_packages = _mapping(worker_lock.get("packages"))
    root_lock_manifest = _mapping(_mapping(root_packages.get("")).get("devDependencies"))
    worker_lock_manifest = _mapping(_mapping(worker_packages.get("")).get("devDependencies"))
    root_wrangler = _mapping(root_packages.get("node_modules/wrangler"))
    worker_wrangler = _mapping(worker_packages.get("node_modules/wrangler"))
    worker_types = _mapping(worker_packages.get("node_modules/@cloudflare/workers-types"))

    root_wrangler_spec = root_dev.get(WRANGLER)
    worker_wrangler_spec = worker_dev.get(WRANGLER)
    if _numeric_version(root_wrangler_spec) is None:
        errors.append("root manifest must exact-pin Wrangler to a stable numeric release")
    if _numeric_version(worker_wrangler_spec) is None:
        errors.append("Worker manifest must exact-pin Wrangler to a stable numeric release")
    if root_wrangler_spec != worker_wrangler_spec:
        errors.append("root and Worker manifests must exact-pin the same Wrangler version")

    for label, manifest_spec, lock_spec, installed in (
        ("root", root_wrangler_spec, root_lock_manifest.get(WRANGLER), root_wrangler.get("version")),
        ("Worker", worker_wrangler_spec, worker_lock_manifest.get(WRANGLER), worker_wrangler.get("version")),
    ):
        if manifest_spec != lock_spec:
            errors.append(f"{label} lockfile manifest entry must match its Wrangler manifest pin")
        if manifest_spec != installed:
            errors.append(f"{label} lockfile must resolve its exact Wrangler manifest pin")

    for field in WRANGLER_LOCK_FIELDS:
        if root_wrangler.get(field) != worker_wrangler.get(field):
            errors.append(f"root and Worker Wrangler lock entries must match field {field!r}")

    root_shared_paths = {path for path in root_packages if isinstance(path, str) and path}
    worker_shared_paths = {
        path for path in worker_packages if isinstance(path, str) and path and path not in WORKER_ONLY_LOCK_PATHS
    }
    if root_shared_paths != worker_shared_paths:
        missing_from_worker = sorted(root_shared_paths - worker_shared_paths)
        extra_in_worker = sorted(worker_shared_paths - root_shared_paths)
        errors.append(
            "root and Worker shared toolchain lock paths must match "
            f"(missing from Worker: {missing_from_worker}; extra in Worker: {extra_in_worker})"
        )
    for path in sorted(root_shared_paths & worker_shared_paths - {"node_modules/wrangler"}):
        if root_packages.get(path) != worker_packages.get(path):
            errors.append(f"root and Worker shared toolchain lock entries must match for {path!r}")

    for dependency in (WORKERS_TYPES, TYPESCRIPT):
        worker_spec = worker_dev.get(dependency)
        worker_entry = _mapping(worker_packages.get(f"node_modules/{dependency}"))
        if _numeric_version(worker_spec) is None:
            errors.append(f"Worker manifest must exact-pin {dependency} to a stable numeric release")
        if worker_lock_manifest.get(dependency) != worker_spec:
            errors.append(f"Worker lockfile manifest entry must match its {dependency} manifest pin")
        if worker_entry.get("version") != worker_spec:
            errors.append(f"Worker lockfile must resolve its exact {dependency} manifest pin")
        if dependency in root_dev or f"node_modules/{dependency}" in root_packages:
            errors.append(f"root deploy shim must not install the Worker-only {dependency} package")

    worker_types_spec = worker_dev.get(WORKERS_TYPES)

    root_peer_dependencies = _mapping(root_wrangler.get("peerDependencies"))
    worker_peer_dependencies = _mapping(worker_wrangler.get("peerDependencies"))
    root_peer_meta = _mapping(root_wrangler.get("peerDependenciesMeta"))
    peer_constraint = root_peer_dependencies.get(WORKERS_TYPES)
    if peer_constraint != worker_peer_dependencies.get(WORKERS_TYPES):
        errors.append("root and Worker Wrangler locks must declare the same workers-types peer range")
    if _mapping(root_peer_meta.get(WORKERS_TYPES)).get("optional") is not True:
        errors.append("root Wrangler lock must keep the workers-types peer optional")
    if not _version_satisfies_caret(worker_types_spec, peer_constraint):
        errors.append("Worker @cloudflare/workers-types pin must satisfy Wrangler's peer range")

    return errors


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return loaded


def main() -> int:
    try:
        errors = validate_worker_dependency_parity(
            _load_json(REPO_ROOT / "package.json"),
            _load_json(REPO_ROOT / "package-lock.json"),
            _load_json(WORKER_ROOT / "package.json"),
            _load_json(WORKER_ROOT / "package-lock.json"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not inspect Worker dependency surfaces: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("ERROR: root/Worker dependency parity validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("root/Worker dependency parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
