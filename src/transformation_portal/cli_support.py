"""Shared helpers for Transformation Portal CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

from transformation_portal.config_loader import get_recipe_info, load_recipe, validate_recipe

_DEFAULT_RECIPE_DIR = Path("config/recipes")
_FALLBACK_RECIPE_DIR = Path("config")


@dataclass(frozen=True)
class DependencyStatus:
    """Status for an installed distribution."""

    display_name: str
    available: bool
    version: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class FeatureStatus:
    """Status for an optional feature probe."""

    display_name: str
    available: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class RecipeValidationResult:
    """Validation result for a recipe file."""

    recipe: dict[str, Any]
    info: dict[str, Any]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def summarize_exception(exc: BaseException, limit: int = 160) -> str:
    """Return a compact exception summary suitable for CLI output."""

    message = str(exc).strip()
    summary = type(exc).__name__ if not message else f"{type(exc).__name__}: {message}"
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3].rstrip() + "..."


def discover_recipe_files(recipes_dir: Optional[Path] = None) -> list[Path]:
    """Discover UnifiedPipeline recipe files.

    When ``recipes_dir`` is omitted, search ``config/recipes`` first and then
    fall back to a recursive search under ``config`` when no candidates exist in
    the preferred location.
    """

    if recipes_dir is not None:
        return _discover_recipe_candidates(Path(recipes_dir), recursive=True)

    primary_candidates = _discover_recipe_candidates(_DEFAULT_RECIPE_DIR, recursive=False)
    if primary_candidates:
        return primary_candidates

    return _discover_recipe_candidates(_FALLBACK_RECIPE_DIR, recursive=True)


def list_recipe_summaries(recipes_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Return summary dictionaries for discovered recipe files."""

    summaries: list[dict[str, Any]] = []
    for recipe_path in discover_recipe_files(recipes_dir):
        recipe = load_recipe(recipe_path, expand_env=False, resolve_paths=False)
        info = get_recipe_info(recipe)
        info["path"] = str(recipe_path)
        summaries.append(info)
    return summaries


def validate_recipe_file(recipe_path: Path) -> RecipeValidationResult:
    """Load a recipe file and validate it."""

    loaded_recipe = load_recipe(recipe_path, expand_env=False, resolve_paths=False)
    info = get_recipe_info(loaded_recipe)
    is_valid, errors = validate_recipe(loaded_recipe)
    return RecipeValidationResult(
        recipe=loaded_recipe,
        info=info,
        errors=[] if is_valid else errors,
    )


def probe_dependency_versions(
    dependency_specs: Iterable[tuple[str, str]],
) -> list[DependencyStatus]:
    """Probe installed package versions via distribution metadata."""

    statuses: list[DependencyStatus] = []
    for display_name, distribution_name in dependency_specs:
        try:
            version = metadata.version(distribution_name)
            statuses.append(
                DependencyStatus(
                    display_name=display_name,
                    available=True,
                    version=version,
                )
            )
        except metadata.PackageNotFoundError:
            statuses.append(
                DependencyStatus(
                    display_name=display_name,
                    available=False,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            statuses.append(
                DependencyStatus(
                    display_name=display_name,
                    available=False,
                    reason=summarize_exception(exc),
                )
            )
    return statuses


def probe_pipeline_features() -> list[FeatureStatus]:
    """Best-effort probe for optional pipeline feature availability."""

    try:
        from transformation_portal.pipeline_unified import HAS_4K_PIPELINE, HAS_QUALITY_BRIDGE

        return [
            FeatureStatus("RAG Quality Feedback", HAS_QUALITY_BRIDGE),
            FeatureStatus("4K Rendering Pipeline", HAS_4K_PIPELINE),
        ]
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        reason = summarize_exception(exc)
        return [
            FeatureStatus("RAG Quality Feedback", False, reason),
            FeatureStatus("4K Rendering Pipeline", False, reason),
        ]


def _discover_recipe_candidates(root_dir: Path, recursive: bool) -> list[Path]:
    """Return recipe-shaped YAML files from ``root_dir``."""

    if not root_dir.exists() or not root_dir.is_dir():
        return []

    pattern = "**/*.y*ml" if recursive else "*.y*ml"
    candidates: list[Path] = []
    for recipe_path in sorted(path for path in root_dir.glob(pattern) if path.is_file()):
        try:
            payload = load_recipe(recipe_path, expand_env=False, resolve_paths=False)
        except Exception:
            continue
        if _looks_like_recipe(payload):
            candidates.append(recipe_path)
    return candidates


def _looks_like_recipe(payload: Any) -> bool:
    """Return ``True`` when the parsed YAML resembles a UnifiedPipeline recipe."""

    return isinstance(payload, dict) and "name" in payload and "stages" in payload
