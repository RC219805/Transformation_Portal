#!/usr/bin/env python3
"""Materialize the governed FastVLM source set without trusting existing clones."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent / "validation"


def _load_manifest_helpers() -> Any:
    helper_path = VALIDATION_DIR / "fastvlm_runtime_manifest.py"
    spec = importlib.util.spec_from_file_location("fastvlm_runtime_manifest_source_install", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load FastVLM manifest helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_manifest_helpers = _load_manifest_helpers()
ManifestError = _manifest_helpers.ManifestError
RuntimeVerificationError = _manifest_helpers.RuntimeVerificationError
TRUSTED_RUNTIME_SOURCES = _manifest_helpers.TRUSTED_RUNTIME_SOURCES
inspect_source_checkout = _manifest_helpers.inspect_source_checkout
load_manifest = _manifest_helpers.load_manifest
require_source_integrity_manifest = _manifest_helpers.require_source_integrity_manifest
run_secure_git = _manifest_helpers.run_secure_git
runtime_root = _manifest_helpers.runtime_root
safe_child = _manifest_helpers.safe_child
verify_runtime_sources = _manifest_helpers.verify_runtime_sources
verify_source_checkout = _manifest_helpers.verify_source_checkout


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the governed FastVLM runtime manifest.")
    parser.add_argument("--runtime-root", required=True, help="FastVLM runtime root to prepare.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the plan without filesystem or network writes.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify installed sources without network or filesystem writes.",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    return args


def _git_or_raise(
    args: list[str],
    *,
    git_dir: Path | None = None,
    work_tree: Path | None = None,
    cwd: Path | None = None,
    allow_file_protocol: bool = False,
) -> str:
    completed = run_secure_git(
        args,
        git_dir=git_dir,
        work_tree=work_tree,
        cwd=cwd,
        allow_file_protocol=allow_file_protocol,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeVerificationError(f"FastVLM source installation Git command failed: {detail}")
    return completed.stdout.strip()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _ensure_safe_runtime_root(root: Path, *, create: bool) -> Path:
    target = _lexical_absolute(root)
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    current = Path(existing.anchor)
    for part in existing.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeVerificationError(f"FastVLM runtime root must not contain symlinks: {current}")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise RuntimeVerificationError(f"FastVLM runtime root must be a real directory: {target}")
    elif create:
        target.mkdir(parents=True)
    return target


def _source_targets(manifest: Mapping[str, Any], root: Path) -> dict[str, Path]:
    sources = manifest["runtime_sources"]
    return {name: safe_child(root, sources[name]["target_dir"]) for name in TRUSTED_RUNTIME_SOURCES}


def _preflight_existing_sources(
    manifest: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Inspect existing metadata without fetching, checking out, or running hooks."""

    targets = _source_targets(manifest, root)
    heads: dict[str, str] = {}
    for name, target in targets.items():
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or not target.is_dir():
            raise RuntimeVerificationError(f"FastVLM source target must be a real directory: {target}")
        _, _, head = inspect_source_checkout(target, expected_origin=TRUSTED_RUNTIME_SOURCES[name])
        heads[name] = head
    return targets, heads


def _allow_file_protocol(repo_url: str) -> bool:
    """Allow local repositories only when tests replace the production allowlist."""

    return repo_url.startswith("file://") and repo_url in TRUSTED_RUNTIME_SOURCES.values()


def _materialize_source(name: str, source: Mapping[str, Any], target: Path) -> None:
    repo_url = TRUSTED_RUNTIME_SOURCES[name]
    if source.get("repo_url") != repo_url:
        raise ManifestError(f"runtime_sources.{name}.repo_url is not allowlisted")
    revision = str(source["revision"])
    target.mkdir(parents=True)
    _git_or_raise(["init", "--quiet", "--template=", str(target)])
    git_dir = target / ".git"
    _git_or_raise(["remote", "add", "origin", repo_url], git_dir=git_dir, work_tree=target)
    inspect_source_checkout(target, expected_origin=repo_url, require_head=False)
    allow_file = _allow_file_protocol(repo_url)
    _git_or_raise(
        ["fetch", "--quiet", "--no-tags", "--depth=1", "origin", revision],
        git_dir=git_dir,
        work_tree=target,
        allow_file_protocol=allow_file,
    )
    fetched = _git_or_raise(["rev-parse", "--verify", "FETCH_HEAD^{commit}"], git_dir=git_dir, work_tree=target)
    if fetched != revision:
        raise RuntimeVerificationError(f"FastVLM source fetch returned {fetched}, expected {revision}")
    _git_or_raise(["checkout", "--quiet", "--detach", revision], git_dir=git_dir, work_tree=target)
    verify_source_checkout(target, expected_origin=repo_url, expected_revision=revision)


def _apply_governed_patch(manifest: Mapping[str, Any], stage_root: Path) -> None:
    sources = manifest["runtime_sources"]
    source = sources["mlx_vlm"]
    patch = source["patch"]
    patch_source = safe_child(stage_root, sources[str(patch["source"])]["target_dir"])
    patch_path = safe_child(patch_source, patch["path"])
    if patch_path.is_symlink() or not patch_path.is_file():
        raise RuntimeVerificationError(f"Governed FastVLM patch is missing or unsafe: {patch_path}")
    actual_digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    expected_digest = str(patch["sha256"]).lower()
    if actual_digest != expected_digest:
        raise RuntimeVerificationError(f"Governed FastVLM patch digest mismatch: {actual_digest} != {expected_digest}")

    target = safe_child(stage_root, source["target_dir"])
    git_dir = target / ".git"
    _git_or_raise(["apply", "--check", "--index", str(patch_path)], git_dir=git_dir, work_tree=target)
    _git_or_raise(["apply", "--index", str(patch_path)], git_dir=git_dir, work_tree=target)
    actual_tree = _git_or_raise(["write-tree"], git_dir=git_dir, work_tree=target)
    expected_tree = str(patch["patched_tree"]).lower()
    if actual_tree != expected_tree:
        raise RuntimeVerificationError(f"FastVLM patched tree mismatch: {actual_tree} != {expected_tree}")
    verify_source_checkout(
        target,
        expected_origin=TRUSTED_RUNTIME_SOURCES["mlx_vlm"],
        expected_revision=str(source["revision"]),
        expected_tree=expected_tree,
    )


def _stage_source_set(manifest: Mapping[str, Any], stage_root: Path) -> None:
    sources = manifest["runtime_sources"]
    for name in TRUSTED_RUNTIME_SOURCES:
        _materialize_source(name, sources[name], safe_child(stage_root, sources[name]["target_dir"]))
    _apply_governed_patch(manifest, stage_root)
    errors = verify_runtime_sources(manifest, root=stage_root)
    if errors:
        raise RuntimeVerificationError("; ".join(errors))


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _promote_source_set(
    manifest: Mapping[str, Any],
    *,
    stage_root: Path,
    runtime: Path,
) -> None:
    sources = manifest["runtime_sources"]
    token = f"{os.getpid()}-{time.monotonic_ns()}"
    targets = _source_targets(manifest, runtime)
    staged = {name: safe_child(stage_root, sources[name]["target_dir"]) for name in TRUSTED_RUNTIME_SOURCES}
    backups = {name: runtime / f".{targets[name].name}.backup-{token}" for name in TRUSTED_RUNTIME_SOURCES}
    moved_backups: list[str] = []
    promoted: list[str] = []
    try:
        for name in TRUSTED_RUNTIME_SOURCES:
            if targets[name].exists():
                _replace_path(targets[name], backups[name])
                moved_backups.append(name)
        for name in TRUSTED_RUNTIME_SOURCES:
            _replace_path(staged[name], targets[name])
            promoted.append(name)
        errors = verify_runtime_sources(manifest, root=runtime)
        if errors:
            raise RuntimeVerificationError("; ".join(errors))
    except Exception as exc:
        rollback_errors: list[str] = []
        for name in reversed(promoted):
            try:
                if targets[name].exists():
                    _replace_path(targets[name], staged[name])
            except OSError as rollback_exc:
                rollback_errors.append(f"could not withdraw {name}: {rollback_exc}")
        for name in reversed(moved_backups):
            try:
                if backups[name].exists():
                    _replace_path(backups[name], targets[name])
            except OSError as rollback_exc:
                rollback_errors.append(f"could not restore {name}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeVerificationError(
                f"FastVLM source promotion failed ({exc}); rollback incomplete: " + "; ".join(rollback_errors)
            ) from exc
        raise
    for backup in backups.values():
        _remove_path(backup)


def install_runtime_sources(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    dry_run: bool = False,
) -> str:
    require_source_integrity_manifest(manifest)
    runtime = _ensure_safe_runtime_root(root, create=False)
    targets, existing_heads = _preflight_existing_sources(manifest, runtime)
    sources = manifest["runtime_sources"]
    expected_heads = {name: str(sources[name]["revision"]) for name in TRUSTED_RUNTIME_SOURCES}

    if set(existing_heads) == set(TRUSTED_RUNTIME_SOURCES) and existing_heads == expected_heads:
        errors = verify_runtime_sources(manifest, root=runtime)
        if not errors:
            return "ready"
    if dry_run:
        print("[dry-run] governed source plan validated")
        return "dry-run"

    runtime = _ensure_safe_runtime_root(runtime, create=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".fastvlm-sources-stage-", dir=runtime))
    try:
        _stage_source_set(manifest, stage_root)
        _promote_source_set(manifest, stage_root=stage_root, runtime=runtime)
    finally:
        _remove_path(stage_root)
    return "installed"


def verify_installed_runtime_sources(manifest: Mapping[str, Any], *, root: Path) -> str:
    """Perform a pure, network-free verification of the installed source pair."""

    require_source_integrity_manifest(manifest)
    runtime = _ensure_safe_runtime_root(root, create=False)
    _preflight_existing_sources(manifest, runtime)
    errors = verify_runtime_sources(manifest, root=runtime)
    if errors:
        raise RuntimeVerificationError("; ".join(errors))
    return "verified"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(Path(args.manifest))
        root = runtime_root(manifest, override=args.runtime_root)
        if args.verify_only:
            status = verify_installed_runtime_sources(manifest, root=root)
        else:
            status = install_runtime_sources(manifest, root=root, dry_run=bool(args.dry_run))
    except ManifestError as exc:
        print(f"FastVLM source manifest invalid: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeVerificationError, UnicodeError) as exc:
        print(f"FastVLM source installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"FastVLM governed sources: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
