"""Lightweight shared helpers for model-lock manifest resolution and loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_MANIFEST_ENV_VAR = "TP_MODEL_LOCK_MANIFEST"
DEFAULT_MANIFEST_RELATIVE_PATH = Path("config/model_lock_manifest.yaml")


def repo_root() -> Path:
    """Best-effort repository root discovery."""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / DEFAULT_MANIFEST_RELATIVE_PATH).exists():
            return parent
    for parent in this_file.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return this_file.parents[0]


def model_lock_manifest_path(path: Optional[Path] = None) -> Path:
    """Resolve the model-lock manifest path."""
    if path is not None:
        return Path(path)

    env_path = os.getenv(DEFAULT_MANIFEST_ENV_VAR)
    if env_path:
        return Path(env_path)

    repo_candidate = repo_root() / DEFAULT_MANIFEST_RELATIVE_PATH
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = Path.cwd() / DEFAULT_MANIFEST_RELATIVE_PATH
    if cwd_candidate.exists():
        return cwd_candidate

    return repo_candidate


def load_model_lock_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the model-lock manifest from disk."""
    manifest_path = model_lock_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model lock manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        # YAML_GOVERNANCE_EXEMPT: internal attestation-manifest loader, not a preset contract.
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Model lock manifest root must be an object: {manifest_path}")

    repositories = payload.get("repositories")
    if repositories is None:
        payload["repositories"] = {}
    elif not isinstance(repositories, dict):
        raise ValueError(f"Model lock manifest 'repositories' must be an object: {manifest_path}")

    artifact_attestation = payload.get("artifact_attestation")
    if artifact_attestation is None:
        payload["artifact_attestation"] = {}
    elif not isinstance(artifact_attestation, dict):
        raise ValueError(f"Model lock manifest 'artifact_attestation' must be an object: {manifest_path}")

    return payload
