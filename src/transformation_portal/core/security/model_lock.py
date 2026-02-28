"""Model lock manifest helpers for HuggingFace revision pinning."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"NEEDS_VERIFICATION", re.IGNORECASE),
    re.compile(r"PLACEHOLDER", re.IGNORECASE),
    re.compile(r"TODO_REPLACE", re.IGNORECASE),
)
_UNPINNED_KEYWORDS = {"main", "master", "latest", "head", "null", "none"}

_STRICT_ENV_VAR = "TP_STRICT_MODEL_LOCK"
_MANIFEST_ENV_VAR = "TP_MODEL_LOCK_MANIFEST"
_DEFAULT_MANIFEST_RELATIVE_PATH = Path("config/model_lock_manifest.yaml")


class ModelLockError(RuntimeError):
    """Raised when strict model lock policy is violated."""


def _repo_root() -> Path:
    """Best-effort repository root discovery."""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / _DEFAULT_MANIFEST_RELATIVE_PATH).exists():
            return parent
    for parent in this_file.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return this_file.parents[0]


def model_lock_manifest_path(path: Optional[Path] = None) -> Path:
    """Resolve model lock manifest path."""
    if path is not None:
        return Path(path)
    env_path = os.getenv(_MANIFEST_ENV_VAR)
    if env_path:
        return Path(env_path)
    repo_candidate = _repo_root() / _DEFAULT_MANIFEST_RELATIVE_PATH
    if repo_candidate.exists():
        return repo_candidate
    cwd_candidate = Path.cwd() / _DEFAULT_MANIFEST_RELATIVE_PATH
    if cwd_candidate.exists():
        return cwd_candidate
    return repo_candidate


def _parse_bool(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_model_lock_strict_enabled(strict: Optional[bool] = None) -> bool:
    """Return strict-mode status from override or env."""
    if strict is not None:
        return strict
    return _parse_bool(os.getenv(_STRICT_ENV_VAR))


def _normalize_revision(revision: Optional[str]) -> Optional[str]:
    if revision is None:
        return None
    normalized = revision.strip()
    if not normalized:
        return None
    return normalized


def _canonicalize_revision(revision: Optional[str]) -> Optional[str]:
    """Normalize whitespace and canonicalize pinned SHAs to lowercase."""
    normalized = _normalize_revision(revision)
    if normalized is None:
        return None
    if is_pinned_revision(normalized):
        return normalized.lower()
    return normalized


def is_pinned_revision(revision: Optional[str]) -> bool:
    """True if revision is a deterministic pinned commit SHA."""
    normalized = _normalize_revision(revision)
    if normalized is None:
        return False
    lowered = normalized.lower()
    if lowered in _UNPINNED_KEYWORDS:
        return False
    for pattern in _PLACEHOLDER_PATTERNS:
        if pattern.search(normalized):
            return False
    return bool(_HEX40_RE.fullmatch(lowered))


def load_model_lock_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model lock manifest from disk."""
    manifest_path = model_lock_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model lock manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Model lock manifest root must be an object: {manifest_path}")
    repos = payload.get("repositories")
    if repos is None:
        payload["repositories"] = {}
    elif not isinstance(repos, dict):
        raise ValueError(f"Model lock manifest 'repositories' must be an object: {manifest_path}")
    return payload


def manifest_revision_for_repo(
    repo_id: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
) -> Optional[str]:
    """Return manifest revision value for a HuggingFace repo ID."""
    payload = manifest if manifest is not None else load_model_lock_manifest(manifest_path)
    repos = payload.get("repositories", {})
    entry = repos.get(repo_id)
    if not isinstance(entry, dict):
        return None
    revision = entry.get("revision")
    return revision if isinstance(revision, str) else None


def _safe_manifest_revision_for_repo(
    repo_id: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
    strict_manifest_required: bool,
) -> Optional[str]:
    """Best-effort manifest lookup with strict-mode aware failure semantics."""
    try:
        return manifest_revision_for_repo(
            repo_id,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        if strict_manifest_required:
            # Caller supplied a manifest object/path explicitly; treat missing or malformed
            # manifest as strict-mode policy error.
            raise ModelLockError(f"Invalid model lock manifest: {exc}") from exc
        logger.warning("Model lock manifest unavailable for repo '%s': %s", repo_id, exc)
        return None


def resolve_model_lock_revision(
    repo_id: str,
    requested_revision: Optional[str],
    *,
    strict: Optional[bool] = None,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
    context: str = "",
) -> Optional[str]:
    """Resolve effective revision using explicit arg + model lock manifest.

    Resolution precedence:
    1) ``requested_revision`` if set
    2) manifest revision for ``repo_id`` if set and pinned

    Strict mode (``TP_STRICT_MODEL_LOCK=1`` or ``strict=True``):
    - effective revision MUST be pinned (40-char SHA)
    - unpinned/placeholder values are rejected with ``ModelLockError``
    """
    strict_enabled = is_model_lock_strict_enabled(strict)
    manifest_supplied = manifest is not None or manifest_path is not None
    manifest_rev = _safe_manifest_revision_for_repo(
        repo_id,
        manifest=manifest,
        manifest_path=manifest_path,
        strict_manifest_required=strict_enabled and manifest_supplied,
    )
    manifest_rev_normalized = _canonicalize_revision(manifest_rev)
    requested = _canonicalize_revision(requested_revision)

    if requested is None and is_pinned_revision(manifest_rev_normalized):
        requested = manifest_rev_normalized

    if strict_enabled and not is_pinned_revision(requested):
        prefix = f"{context}: " if context else ""
        raise ModelLockError(
            prefix
            + f"repo '{repo_id}' is unpinned in strict model-lock mode. "
            + "Provide a 40-char commit SHA directly or via config/model_lock_manifest.yaml."
        )

    if strict_enabled and requested and is_pinned_revision(manifest_rev_normalized) and requested != manifest_rev_normalized:
        prefix = f"{context}: " if context else ""
        raise ModelLockError(
            prefix
            + f"repo '{repo_id}' revision mismatch with model lock manifest "
            + f"(requested={requested}, manifest={manifest_rev_normalized})."
        )

    if not strict_enabled:
        if requested and not is_pinned_revision(requested):
            logger.warning(
                "Model repo '%s' uses unpinned revision '%s' in non-strict mode",
                repo_id,
                requested,
            )
        if requested is None and manifest_rev_normalized is not None and not is_pinned_revision(manifest_rev_normalized):
            logger.warning(
                "Model lock manifest entry for '%s' is not pinned ('%s')",
                repo_id,
                manifest_rev_normalized,
            )

    return requested
