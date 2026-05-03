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
    model_target_dir,
    python_runtime_path,
    runtime_root,
    selected_model_roles,
    validate_manifest,
    verify_model_role,
    verify_python_imports,
    verify_python_runtime,
    verify_runtime_sources,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_manifest_args(parser)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify manifest, runtime sources, Python executable, and model files without import smoke checks.",
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


def _check_evidence(
    *,
    status: str,
    errors: list[str] | None = None,
    path: Path | str | None = None,
    scope: str = "static",
    remediation: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "scope": scope,
        "errors": list(errors or []),
    }
    if path is not None:
        payload["path"] = str(path)
    if remediation:
        payload["remediation"] = remediation
    return payload


def _status_for_errors(errors: list[str]) -> str:
    return "ready" if not errors else "failed"


def _runtime_evidence(
    *,
    manifest_path: Path,
    root: Path,
    roles: list[str],
    manifest: dict[str, Any],
    include_sources: bool,
    include_python: bool,
    include_import_smoke: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    manifest_errors = validate_manifest(manifest)
    checks["manifest"] = _check_evidence(
        status=_status_for_errors(manifest_errors),
        errors=manifest_errors,
        path=manifest_path,
        remediation="Repair config/fastvlm_runtime_manifest.json before installing or validating FastVLM.",
    )
    if manifest_errors:
        errors.extend(manifest_errors)
        return _evidence(manifest_path=manifest_path, root=root, roles=roles, errors=errors, checks=checks)

    if include_sources:
        source_errors = verify_runtime_sources(manifest, root=root)
        checks["runtime_sources"] = _check_evidence(
            status=_status_for_errors(source_errors),
            errors=source_errors,
            path=root,
            remediation="Run scripts/setup/install_fastvlm_runtime.sh to install the pinned FastVLM source checkouts.",
        )
        errors.extend(source_errors)
    else:
        checks["runtime_sources"] = _check_evidence(
            status="skipped",
            path=root,
            remediation="Source clone checks were skipped by request.",
        )

    if include_python:
        try:
            python_path = python_runtime_path(manifest, root=root)
        except ManifestError as exc:
            python_path = root / ".venv-fastvlm" / "bin" / "python"
            python_errors = [str(exc)]
        else:
            python_errors = verify_python_runtime(manifest, root=root)
        checks["python_executable"] = _check_evidence(
            status=_status_for_errors(python_errors),
            errors=python_errors,
            path=python_path,
            remediation="Run make install-fastvlm-runtime or repair .runtime/fastvlm/.venv-fastvlm.",
        )
        errors.extend(python_errors)
    else:
        checks["python_executable"] = _check_evidence(
            status="skipped",
            path=root / ".venv-fastvlm" / "bin" / "python",
            remediation="Python executable checks were skipped by request.",
        )

    model_checks: dict[str, Any] = {}
    for role in roles:
        try:
            model_path = model_target_dir(manifest, role, root=root)
            model_errors = verify_model_role(manifest, role, root=root)
        except ManifestError as exc:
            model_path = root / "checkpoints" / role
            model_errors = [str(exc)]
        model_checks[role] = _check_evidence(
            status=_status_for_errors(model_errors),
            errors=model_errors,
            path=model_path,
            remediation=f"Install or repair the manifest-backed FastVLM model role: {role}.",
        )
        errors.extend(model_errors)
    checks["models"] = model_checks

    if include_import_smoke and not errors:
        import_errors = verify_python_imports(manifest, root=root)
        checks["python_imports"] = _check_evidence(
            status=_status_for_errors(import_errors),
            errors=import_errors,
            scope="import-smoke",
            remediation="Install the isolated FastVLM Python dependencies and pinned mlx-vlm package.",
        )
        errors.extend(import_errors)
    else:
        reason = "Import smoke checks were skipped by request."
        if include_import_smoke and errors:
            reason = "Import smoke checks were skipped because static runtime checks failed."
        checks["python_imports"] = _check_evidence(status="skipped", scope="import-smoke", remediation=reason)

    return _evidence(manifest_path=manifest_path, root=root, roles=roles, errors=errors, checks=checks)


def _evidence(
    *,
    manifest_path: Path,
    root: Path,
    roles: list[str],
    errors: list[str],
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "runtime_root": str(root),
        "models": roles,
        "runtime_status": "ready" if not errors else "invalid",
        "errors": errors,
        "checks": checks or {},
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
        evidence = _runtime_evidence(
            manifest_path=manifest_path,
            root=root,
            roles=roles,
            manifest=manifest,
            include_sources=not bool(args.skip_source_check),
            include_python=not bool(args.skip_python_check),
            include_import_smoke=not bool(args.verify_only) and not bool(args.skip_python_check),
        )
    except ManifestError as exc:
        evidence = _evidence(
            manifest_path=manifest_path,
            root=Path(args.runtime_root or ".runtime/fastvlm"),
            roles=[],
            errors=[str(exc)],
            checks={
                "manifest": _check_evidence(
                    status="failed",
                    errors=[str(exc)],
                    path=manifest_path,
                    remediation="Repair the FastVLM runtime manifest, runtime root, or selected model roles.",
                )
            },
        )
        if args.json:
            print(json.dumps(evidence, indent=2, sort_keys=True))
        else:
            print(f"FastVLM runtime manifest invalid: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    elif evidence["errors"]:
        print("FastVLM runtime verification failed:", file=sys.stderr)
        for error in evidence["errors"]:
            print(f"- {error}", file=sys.stderr)
    else:
        print(f"FastVLM runtime ready for roles: {', '.join(roles)}")
        print(f"runtime_root={root}")
        print("captioning_role=advisory used_for_quality_gate=false")
    manifest_check = evidence.get("checks", {}).get("manifest")
    if isinstance(manifest_check, dict) and manifest_check.get("status") == "failed":
        return 2
    return 0 if not evidence["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
