#!/usr/bin/env python3
"""Validate the governed local FastVLM advisory captioning runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastvlm_runtime_manifest import (
    ManifestError,
    add_common_manifest_args,
    load_manifest,
    runtime_root,
    selected_model_roles,
    verify_runtime,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_manifest_args(parser)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify manifest, runtime sources, Python executable, and model files without subprocess inference.",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip runtime source clone checks. Intended only for focused tests with fixture model directories.",
    )
    parser.add_argument(
        "--skip-python-check",
        action="store_true",
        help="Skip isolated Python executable checks. Intended only for focused tests with fixture model directories.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable validation evidence.",
    )
    return parser.parse_args(argv)


def _evidence(
    *,
    manifest_path: Path,
    root: Path,
    roles: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "runtime_root": str(root),
        "models": roles,
        "runtime_status": "ready" if not errors else "invalid",
        "errors": errors,
        "advisory_role": "advisory",
        "used_for_quality_gate": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path)
        root = runtime_root(manifest, override=args.runtime_root or None)
        roles = selected_model_roles(manifest, models=str(args.models), all_models=bool(args.all_models))
        errors = verify_runtime(
            manifest,
            roles=roles,
            root=root,
            include_sources=not bool(args.skip_source_check),
            include_python=not bool(args.skip_python_check),
        )
    except ManifestError as exc:
        evidence = _evidence(
            manifest_path=manifest_path,
            root=Path(args.runtime_root or ".runtime/fastvlm"),
            roles=[],
            errors=[str(exc)],
        )
        if args.json:
            print(json.dumps(evidence, indent=2, sort_keys=True))
        else:
            print(f"FastVLM runtime manifest invalid: {exc}", file=sys.stderr)
        return 2

    evidence = _evidence(manifest_path=manifest_path, root=root, roles=roles, errors=errors)
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    elif errors:
        print("FastVLM runtime verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(f"FastVLM runtime ready for roles: {', '.join(roles)}")
        print(f"runtime_root={root}")
        print("captioning_role=advisory used_for_quality_gate=false")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
