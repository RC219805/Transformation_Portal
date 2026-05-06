"""Configuration model for the Spatial AI orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformation_portal.spatial_ai.materials.material_backend import (
    format_backend_resolution_message,
    resolve_material_backend_decision,
)

from .error_handler import ErrorRecoveryStrategy
from .resource_manager import ResourceLimits

_RELOAD_SAFE_PIPELINE_CONFIG_FIELDS = {
    "tier",
    "stages",
    "ingest",
    "segmentation",
    "materials",
    "reconstruction",
    "resource_limits",
    "error_strategy",
    "use_execution_graph",
}
_VALID_SEGMENTATION_CACHE_POLICIES = {"off", "read_write"}
_VALID_STAGES = ["ingest", "segment", "segmentation", "materials", "reconstruction"]
_VALID_TIERS = ["standard", "apex_research", "apex_research_ultra", "experimental"]
_RECONSTRUCTION_TIERS = ["apex_research", "apex_research_ultra", "experimental"]
_ERROR_STRATEGY_ALIASES = {
    "retry": ErrorRecoveryStrategy.RETRY,
    "retry_cpu_fallback": ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
    "retry_with_cpu_fallback": ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
    "skip_stage": ErrorRecoveryStrategy.SKIP_STAGE,
    "fail_fast": ErrorRecoveryStrategy.FAIL_FAST,
    "return_partial": ErrorRecoveryStrategy.RETURN_PARTIAL,
}


def _is_reload_safe_pipeline_config(candidate: object) -> bool:
    """Accept reloaded PipelineConfig objects by structure, not class identity."""
    candidate_type = type(candidate)
    if candidate_type.__name__ != "PipelineConfig":
        return False
    return all(hasattr(candidate, field_name) for field_name in _RELOAD_SAFE_PIPELINE_CONFIG_FIELDS)


def _normalise_segmentation_cache_policy(value: Any) -> str:
    policy = str(value or "read_write").strip().lower()
    return policy if policy in _VALID_SEGMENTATION_CACHE_POLICIES else "off"


def _extract_materials_governance_overrides(data: dict[str, Any]) -> dict[str, bool]:
    """Extract auditable materials governance overrides from preset/config data."""
    governance = data.get("governance", {})
    materials_governance = governance.get("materials", {}) if isinstance(governance, dict) else {}
    pipeline_cfg = data.get("pipeline", {})
    pipeline_materials = pipeline_cfg.get("materials", {}) if isinstance(pipeline_cfg, dict) else {}
    top_level_materials = data.get("materials", {})

    def _coerce_bool(mapping: Any, key: str) -> Optional[bool]:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        value = mapping[key]
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean, got {type(value).__name__}.")
        return value

    resolved: dict[str, bool] = {}
    for key in ("allow_research_materials", "allow_unattested_materials"):
        candidates = (
            _coerce_bool(materials_governance, key),
            _coerce_bool(top_level_materials, key),
            _coerce_bool(pipeline_materials, key),
        )
        resolved[key] = any(value is True for value in candidates)

    return resolved


@dataclass
class PipelineConfig:
    """Configuration for Spatial AI pipeline.

    Attributes:
        tier: Tier level (standard, apex_research, experimental).
        stages: Stages to execute. Defaults to ["ingest", "segment"]. The segmentation
            stage may be specified as either "segment" or "segmentation"; both are
            accepted. Common conceptual stages are ingest/segmentation/materials/reconstruction.
        ingest: Ingest configuration.
        segmentation: Segmentation configuration.
        materials: Materials configuration.
        reconstruction: Reconstruction configuration.
        resource_limits: Resource limits for execution.
        error_strategy: Error recovery strategy.
        use_execution_graph: If True, use ADR-029 graph-based execution (default: False).
    """

    tier: str
    stages: List[str] = field(default_factory=lambda: ["ingest", "segment"])
    ingest: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Optional[ResourceLimits] = None
    error_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY
    use_execution_graph: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        strategy_value_obj: object = getattr(self.error_strategy, "value", self.error_strategy)
        if not isinstance(strategy_value_obj, str):
            strategy_value_obj = getattr(strategy_value_obj, "value", strategy_value_obj)
        if not isinstance(strategy_value_obj, str):
            raise ValueError(f"Invalid error strategy '{strategy_value_obj}'")
        strategy_value = strategy_value_obj
        if strategy_value not in _ERROR_STRATEGY_ALIASES:
            raise ValueError(f"Invalid error strategy '{strategy_value}'")
        self.error_strategy = _ERROR_STRATEGY_ALIASES[strategy_value]

        self.stages = ["reconstruction" if stage == "reconstruct" else stage for stage in self.stages]
        for stage in self.stages:
            if stage not in _VALID_STAGES:
                raise ValueError(f"Invalid stage '{stage}'. Valid: {_VALID_STAGES}")

        raw_segmentation_cfg = self.segmentation or {}
        raw_cache_policy = raw_segmentation_cfg.get("cache_policy") if isinstance(raw_segmentation_cfg, dict) else None
        if raw_cache_policy is not None and str(raw_cache_policy).strip().lower() not in _VALID_SEGMENTATION_CACHE_POLICIES:
            raise ValueError("segmentation.cache_policy must be one of: off, read_write")

        segmentation_cfg = dict(raw_segmentation_cfg)
        segmentation_cfg["cache_policy"] = _normalise_segmentation_cache_policy(
            segmentation_cfg.get("cache_policy", "read_write")
        )
        self.segmentation = segmentation_cfg

        if self.tier not in _VALID_TIERS:
            raise ValueError(f"Invalid tier '{self.tier}'. Valid: {_VALID_TIERS}")

        if "reconstruction" in self.stages and self.tier not in _RECONSTRUCTION_TIERS:
            raise ValueError(f"Reconstruction requires research tier, got '{self.tier}' (Inria 3DGS license restriction)")

        self._validate_materials_config()

    def _validate_materials_config(self) -> None:
        """Validate materials backend selection for the current single-image pipeline contract."""
        if "materials" not in self.stages:
            return

        materials_cfg = dict(self.materials or {})
        requested_backend = materials_cfg.get("backend", "heuristic")
        strict_backend = materials_cfg.get("strict_backend", False)

        if not isinstance(strict_backend, bool):
            raise ValueError(f"materials.strict_backend must be a boolean, got {type(strict_backend).__name__}")

        decision = resolve_material_backend_decision(requested_backend)

        if strict_backend and decision.requested_backend != decision.executed_backend:
            message = format_backend_resolution_message(
                decision,
                context="materials.strict_backend=True forbids fallback in the single-image materials pipeline",
            )
            raise ValueError(f"{message}. Use backend='heuristic' or disable strict_backend for permissive fallback behavior.")


__all__ = [
    "PipelineConfig",
    "_extract_materials_governance_overrides",
    "_is_reload_safe_pipeline_config",
    "_normalise_segmentation_cache_policy",
]
