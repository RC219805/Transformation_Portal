"""Asset path resolution helpers for external APEX eval corpora."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

APEX_EVAL_ASSET_ROOT_ENV = "APEX_EVAL_ASSET_ROOT"

ResolutionStrategy = Literal["absolute", "cli_asset_root", "env_asset_root", "repo_relative"]


@dataclass(frozen=True)
class ResolvedAsset:
    """Resolved view of a manifest path without forcing reports to leak local paths."""

    reported_path: str
    resolved_path: Path | None
    strategy: ResolutionStrategy
    exists: bool
    path_was_absolute: bool
    escaped_asset_root: bool = False

    def to_report_dict(self) -> dict[str, object]:
        """Return portable report metadata without full local filesystem paths."""
        return {
            "strategy": self.strategy,
            "reported_path": self.reported_path,
            "exists": self.exists,
            "path_was_absolute": self.path_was_absolute,
            "escaped_asset_root": self.escaped_asset_root,
        }


def _normalize_optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return Path(normalized).expanduser() if normalized else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_manifest_path(
    reported_path: str | os.PathLike[str],
    *,
    repo_root: Path,
    asset_root: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAsset:
    """Resolve an evalset path using absolute, asset-root, env-root, then repo-relative rules."""
    reported = str(reported_path)
    path = Path(reported)
    if path.is_absolute():
        return ResolvedAsset(
            reported_path=reported,
            resolved_path=path,
            strategy="absolute",
            exists=path.is_file(),
            path_was_absolute=True,
        )

    root = _normalize_optional_path(asset_root)
    strategy: ResolutionStrategy = "cli_asset_root"
    if root is None:
        env_root = (env or os.environ).get(APEX_EVAL_ASSET_ROOT_ENV)
        root = _normalize_optional_path(env_root)
        strategy = "env_asset_root"
    if root is None:
        resolved = (repo_root / path).resolve(strict=False)
        return ResolvedAsset(
            reported_path=reported,
            resolved_path=resolved,
            strategy="repo_relative",
            exists=resolved.is_file(),
            path_was_absolute=False,
        )

    root_resolved = root.resolve(strict=False)
    resolved = (root_resolved / path).resolve(strict=False)
    escaped = not _is_relative_to(resolved, root_resolved)
    return ResolvedAsset(
        reported_path=reported,
        resolved_path=None if escaped else resolved,
        strategy=strategy,
        exists=False if escaped else resolved.is_file(),
        path_was_absolute=False,
        escaped_asset_root=escaped,
    )
