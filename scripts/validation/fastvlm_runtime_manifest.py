#!/usr/bin/env python3
"""Shared FastVLM runtime manifest validation helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TRUSTED_FASTVLM_MODEL_REPOS = {
    "apple/FastVLM-0.5B-fp16",
    "apple/FastVLM-1.5B-int8",
    "apple/FastVLM-7B-int4",
}
TRUSTED_RUNTIME_SOURCES = {
    "ml_fastvlm": "https://github.com/apple/ml-fastvlm.git",
    "mlx_vlm": "https://github.com/Blaizzy/mlx-vlm.git",
}
HEX_DIGITS = set("0123456789abcdef")


class ManifestError(RuntimeError):
    """Raised when the FastVLM runtime manifest violates governance."""


class RuntimeVerificationError(RuntimeError):
    """Raised when the local FastVLM runtime does not match the manifest."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return repo_root() / "config" / "fastvlm_runtime_manifest.json"


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"FastVLM runtime manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"FastVLM runtime manifest is invalid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("FastVLM runtime manifest must be a JSON object.")
    return payload


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in HEX_DIGITS for char in text.lower())


def _is_git_revision(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 40 and all(char in HEX_DIGITS for char in text.lower())


def _safe_relative_parts(path_text: Any) -> tuple[str, ...]:
    text = str(path_text or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or "\x00" in text:
        raise ManifestError(f"Unsafe FastVLM manifest path: {path_text!r}")
    parts = tuple(part for part in text.split("/") if part)
    if any(part in {".", ".."} for part in parts):
        raise ManifestError(f"Unsafe FastVLM manifest path: {path_text!r}")
    return parts


def safe_child(root: Path, path_text: Any) -> Path:
    parts = _safe_relative_parts(path_text)
    resolved_root = Path(os.path.realpath(root))
    candidate = Path(os.path.realpath(resolved_root.joinpath(*parts)))
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestError(f"FastVLM manifest path escapes runtime root: {path_text!r}") from exc
    return candidate


def runtime_root(manifest: Mapping[str, Any], *, override: Path | str | None = None) -> Path:
    if override is not None:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else repo_root() / candidate
    root_text = str(manifest.get("runtime_root") or ".runtime/fastvlm")
    return safe_child(repo_root(), root_text)


def selected_model_roles(
    manifest: Mapping[str, Any],
    *,
    models: str | Sequence[str] | None = None,
    all_models: bool = False,
    default_roles: Sequence[str] = ("smoke", "default"),
) -> list[str]:
    model_map = manifest.get("models")
    if not isinstance(model_map, dict) or not model_map:
        raise ManifestError("FastVLM runtime manifest must define models.")
    if all_models:
        roles = list(model_map.keys())
    elif isinstance(models, str) and models.strip():
        roles = [role.strip() for role in models.split(",") if role.strip()]
    elif models:
        roles = [str(role).strip() for role in models if str(role).strip()]
    else:
        roles = list(default_roles)
    unknown = sorted(set(roles) - set(model_map.keys()))
    if unknown:
        raise ManifestError(f"Unknown FastVLM model role(s): {', '.join(unknown)}")
    return roles


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "fastvlm-runtime.v1":
        errors.append("schema_version must be fastvlm-runtime.v1")
    try:
        safe_child(repo_root(), manifest.get("runtime_root") or ".runtime/fastvlm")
    except ManifestError as exc:
        errors.append(str(exc))

    sources = manifest.get("runtime_sources")
    if not isinstance(sources, dict):
        errors.append("runtime_sources must be an object")
    else:
        for name, expected_url in TRUSTED_RUNTIME_SOURCES.items():
            source = sources.get(name)
            if not isinstance(source, dict):
                errors.append(f"runtime_sources.{name} must be an object")
                continue
            if source.get("repo_url") != expected_url:
                errors.append(f"runtime_sources.{name}.repo_url is not allowlisted")
            if not _is_git_revision(source.get("revision")):
                errors.append(f"runtime_sources.{name}.revision must be a pinned 40-hex revision")
            try:
                _safe_relative_parts(source.get("target_dir"))
            except ManifestError as exc:
                errors.append(str(exc))

    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        errors.append("models must be a non-empty object")
    else:
        for role, model in models.items():
            if not isinstance(model, dict):
                errors.append(f"models.{role} must be an object")
                continue
            repo_id = model.get("repo_id")
            if repo_id not in TRUSTED_FASTVLM_MODEL_REPOS:
                errors.append(f"models.{role}.repo_id is not allowlisted")
            if not _is_git_revision(model.get("revision")):
                errors.append(f"models.{role}.revision must be a pinned 40-hex revision")
            try:
                _safe_relative_parts(model.get("target_dir"))
            except ManifestError as exc:
                errors.append(str(exc))
            required_files = model.get("required_files")
            if not isinstance(required_files, list) or not required_files:
                errors.append(f"models.{role}.required_files must be a non-empty list")
                continue
            for index, entry in enumerate(required_files):
                if not isinstance(entry, dict):
                    errors.append(f"models.{role}.required_files[{index}] must be an object")
                    continue
                try:
                    _safe_relative_parts(entry.get("path"))
                except ManifestError as exc:
                    errors.append(str(exc))
                if not _is_sha256(entry.get("sha256")):
                    errors.append(f"models.{role}.required_files[{index}].sha256 must be a SHA-256 hex digest")
                size = entry.get("size_bytes")
                if not isinstance(size, int) or size <= 0:
                    errors.append(f"models.{role}.required_files[{index}].size_bytes must be a positive integer")
    return errors


def require_valid_manifest(manifest: Mapping[str, Any]) -> None:
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError("; ".join(errors))


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_not_symlink(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeVerificationError(f"FastVLM manifest path must not be a symlink: {path}")


def model_target_dir(manifest: Mapping[str, Any], role: str, *, root: Path | None = None) -> Path:
    model_map = manifest.get("models")
    if not isinstance(model_map, dict) or role not in model_map:
        raise ManifestError(f"Unknown FastVLM model role: {role}")
    runtime = root or runtime_root(manifest)
    model = model_map[role]
    if not isinstance(model, dict):
        raise ManifestError(f"models.{role} must be an object")
    return safe_child(runtime, model.get("target_dir"))


def verify_model_role(manifest: Mapping[str, Any], role: str, *, root: Path | None = None) -> list[str]:
    target = model_target_dir(manifest, role, root=root)
    return verify_model_files(manifest, role, target)


def verify_model_files(manifest: Mapping[str, Any], role: str, target: Path) -> list[str]:
    errors: list[str] = []
    model = manifest["models"][role]
    if not target.is_dir():
        return [f"FastVLM model role {role} missing directory: {target}"]
    for entry in model["required_files"]:
        file_path = safe_child(target, entry["path"])
        try:
            _ensure_not_symlink(file_path)
            if not file_path.is_file():
                errors.append(f"FastVLM model role {role} missing required file: {file_path}")
                continue
            actual_size = file_path.stat().st_size
            if actual_size != int(entry["size_bytes"]):
                errors.append(
                    f"FastVLM model role {role} file size mismatch for {entry['path']}: "
                    f"{actual_size} != {entry['size_bytes']}"
                )
                continue
            actual_sha = compute_file_sha256(file_path)
            if actual_sha != str(entry["sha256"]).lower():
                errors.append(
                    f"FastVLM model role {role} SHA-256 mismatch for {entry['path']}: " f"{actual_sha} != {entry['sha256']}"
                )
        except (OSError, ManifestError, RuntimeVerificationError) as exc:
            errors.append(str(exc))
    return errors


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def verify_runtime_sources(manifest: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    runtime = root or runtime_root(manifest)
    errors: list[str] = []
    for name, source in manifest["runtime_sources"].items():
        target = safe_child(runtime, source["target_dir"])
        if not target.is_dir():
            errors.append(f"FastVLM runtime source {name} missing directory: {target}")
            continue
        head = _git_head(target)
        if head is not None and head != source["revision"]:
            errors.append(f"FastVLM runtime source {name} revision mismatch: {head} != {source['revision']}")
    return errors


def verify_python_runtime(manifest: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    runtime = root or runtime_root(manifest)
    python_config = manifest.get("python") or {}
    if not isinstance(python_config, dict):
        return ["FastVLM manifest python section must be an object"]
    venv_dir = safe_child(runtime, python_config.get("venv_dir") or ".venv-fastvlm")
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python_path.is_file():
        return [f"FastVLM Python executable missing: {python_path}"]
    if os.name != "nt" and not os.access(python_path, os.X_OK):
        return [f"FastVLM Python executable is not executable: {python_path}"]
    return []


def verify_runtime(
    manifest: Mapping[str, Any],
    *,
    roles: Iterable[str],
    root: Path | None = None,
    include_sources: bool = True,
    include_python: bool = True,
) -> list[str]:
    require_valid_manifest(manifest)
    errors: list[str] = []
    if include_sources:
        errors.extend(verify_runtime_sources(manifest, root=root))
    if include_python:
        errors.extend(verify_python_runtime(manifest, root=root))
    for role in roles:
        errors.extend(verify_model_role(manifest, role, root=root))
    return errors


def allow_patterns_for_role(manifest: Mapping[str, Any], role: str) -> list[str]:
    model = manifest["models"][role]
    patterns = sorted({str(entry["path"]) for entry in model["required_files"]})
    return patterns


def add_common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="FastVLM runtime manifest path (default: %(default)s)",
    )
    parser.add_argument(
        "--runtime-root",
        default="",
        help="Optional runtime root override (default: manifest runtime_root)",
    )
    parser.add_argument(
        "--models",
        default="smoke,default",
        help="Comma-separated model roles to verify or install (default: %(default)s)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Select all manifest model roles",
    )
