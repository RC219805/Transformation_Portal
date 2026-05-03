#!/usr/bin/env python3
"""Download and verify allowlisted FastVLM model checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from fastvlm_runtime_manifest import (  # noqa: E402
    ManifestError,
    RuntimeVerificationError,
    add_common_manifest_args,
    allow_patterns_for_role,
    load_manifest,
    model_target_dir,
    require_valid_manifest,
    runtime_root,
    selected_model_roles,
    verify_model_files,
    verify_model_role,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_manifest_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected model downloads without touching network or filesystem.",
    )
    return parser.parse_args(argv)


def _import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - exact dependency error is environment-specific
        raise RuntimeVerificationError(
            "huggingface_hub is required to download FastVLM model checkpoints. "
            "Install the governed runtime with scripts/setup/install_fastvlm_runtime.sh."
        ) from exc
    return snapshot_download


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _download_role(
    *,
    manifest: dict[str, Any],
    role: str,
    root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    model = manifest["models"][role]
    target = model_target_dir(manifest, role, root=root)
    current_errors = verify_model_role(manifest, role, root=root)
    if not current_errors:
        return {"role": role, "status": "ready", "target_dir": str(target), "downloaded": False}

    repo_id = str(model["repo_id"])
    revision = str(model["revision"])
    patterns = allow_patterns_for_role(manifest, role)
    if dry_run:
        return {
            "role": role,
            "status": "dry-run",
            "repo_id": repo_id,
            "revision": revision,
            "target_dir": str(target),
            "allow_patterns": patterns,
            "downloaded": False,
        }

    snapshot_download = _import_snapshot_download()
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".{target.name}.download-{int(time.time())}"
    if tmp_dir.exists():
        _remove_path(tmp_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(tmp_dir),
            allow_patterns=patterns,
        )
        errors = verify_model_files(manifest, role, tmp_dir)
        if errors:
            raise RuntimeVerificationError("; ".join(errors))
    except Exception:
        _remove_path(tmp_dir)
        raise

    tmp_target = tmp_dir
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        invalid_target = target_parent / f".{target.name}.replaced-{int(time.time())}"
        target.rename(invalid_target)
        try:
            tmp_target.rename(target)
            errors = verify_model_role(manifest, role, root=root)
            if errors:
                _remove_path(target)
                invalid_target.rename(target)
                raise RuntimeVerificationError("; ".join(errors))
            _remove_path(invalid_target)
        except Exception:
            if not target.exists() and invalid_target.exists():
                invalid_target.rename(target)
            _remove_path(tmp_dir)
            raise
    else:
        tmp_target.rename(target)
        errors = verify_model_role(manifest, role, root=root)
        if errors:
            _remove_path(target)
            raise RuntimeVerificationError("; ".join(errors))

    return {"role": role, "status": "downloaded", "target_dir": str(target), "downloaded": True}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        require_valid_manifest(manifest)
        root = runtime_root(manifest, override=args.runtime_root or None)
        roles = selected_model_roles(manifest, models=str(args.models), all_models=bool(args.all_models))
        evidence = [_download_role(manifest=manifest, role=role, root=root, dry_run=bool(args.dry_run)) for role in roles]
    except (ManifestError, RuntimeVerificationError) as exc:
        print(f"FastVLM model download failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"runtime_root": str(root), "models": evidence}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
