"""Central Lux Depth V3 model registry and selector metadata.

This module owns the V1 runtime contract for model identity, license policy,
backend capability, and CLI exposure. It is intentionally narrower than the
upstream DA3 capability surface: the current Lux Depth V3 runtime consumes
relative depth plus metadata, so advanced DA3 outputs remain informational.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, Literal, Optional, Tuple


class UsageClass(str, Enum):
    """Runtime usage policy classification."""

    COMMERCIAL_OK = "commercial_ok"
    NON_COMMERCIAL_ONLY = "non_commercial_only"
    UNKNOWN = "unknown"


class BackendKind(str, Enum):
    """Canonical backend type for registry entries."""

    DA3_API = "da3_api"
    DA2_TRANSFORMERS = "da2_transformers"
    DEPTH_PRO = "depth_pro"
    COREML_PUBLISHED = "coreml_published"


class AcceleratorKind(str, Enum):
    """Accelerator lane resolved for a model selection."""

    NONE = "none"
    COREML = "coreml"


class ConsumableOutput(str, Enum):
    """Current Lux Depth V3 outputs enforced by the registry."""

    NORMALIZED_RELATIVE_DEPTH = "normalized_relative_depth"
    DEPTH_METADATA = "depth_metadata"
    COREML_RELATIVE_DEPTH = "coreml_relative_depth"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model contract used by Lux Depth V3 V1."""

    key: str
    repo_id: str
    family: str
    backend_kind: BackendKind
    license_id: str
    usage_class: UsageClass
    requires_non_commercial_ok: bool
    supports_coreml: bool
    consumable_outputs: FrozenSet[ConsumableOutput]
    upstream_capabilities: FrozenSet[str] = frozenset()
    fallback_keys: Tuple[str, ...] = ()
    lock_required: bool = True
    exposed_in_cli: bool = True
    maturity: Literal["stable", "compat", "experimental", "internal"] = "stable"


# Repair 1.2 (#2066, option A): the out-of-the-box default is the
# commercial-safe Apache-2.0 metric model. The research model remains
# available behind an explicit selector + non_commercial_ok acknowledgement.
DEFAULT_MODEL_KEY = "da3_metric"

# Selector label recorded when no model selector was given. Distinct from the
# "da3" alias so manifests stay honest: "da3" still names the research model
# (deprecated, see DEPRECATED_MODEL_KEY_ALIAS_WARNINGS) while the default is
# da3_metric.
DEFAULT_MODEL_SELECTOR = "default"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "da3_research": ModelSpec(
        key="da3_research",
        repo_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        family="da3",
        backend_kind=BackendKind.DA3_API,
        license_id="cc-by-nc-4.0",
        usage_class=UsageClass.NON_COMMERCIAL_ONLY,
        requires_non_commercial_ok=True,
        supports_coreml=False,
        consumable_outputs=frozenset(
            {
                ConsumableOutput.NORMALIZED_RELATIVE_DEPTH,
                ConsumableOutput.DEPTH_METADATA,
            }
        ),
        upstream_capabilities=frozenset(
            {
                "relative_depth",
                "metric_depth",
                "pose_estimation",
                "pose_conditioning",
                "gaussian_splat",
                "sky_segmentation",
            }
        ),
        maturity="stable",
        exposed_in_cli=True,
    ),
    "da3_metric": ModelSpec(
        key="da3_metric",
        repo_id="depth-anything/DA3METRIC-LARGE",
        family="da3",
        backend_kind=BackendKind.DA3_API,
        license_id="apache-2.0",
        usage_class=UsageClass.COMMERCIAL_OK,
        requires_non_commercial_ok=False,
        supports_coreml=False,
        consumable_outputs=frozenset(
            {
                ConsumableOutput.NORMALIZED_RELATIVE_DEPTH,
                ConsumableOutput.DEPTH_METADATA,
            }
        ),
        upstream_capabilities=frozenset(
            {
                "relative_depth",
                "metric_depth",
                "sky_segmentation",
            }
        ),
        maturity="stable",
        exposed_in_cli=True,
    ),
    "da3_base": ModelSpec(
        key="da3_base",
        repo_id="depth-anything/DA3-BASE",
        family="da3",
        backend_kind=BackendKind.DA3_API,
        license_id="apache-2.0",
        usage_class=UsageClass.COMMERCIAL_OK,
        requires_non_commercial_ok=False,
        supports_coreml=False,
        consumable_outputs=frozenset(
            {
                ConsumableOutput.NORMALIZED_RELATIVE_DEPTH,
                ConsumableOutput.DEPTH_METADATA,
            }
        ),
        upstream_capabilities=frozenset(
            {
                "relative_depth",
                "pose_estimation",
                "pose_conditioning",
            }
        ),
        maturity="experimental",
        exposed_in_cli=False,
    ),
    "da3_small": ModelSpec(
        key="da3_small",
        repo_id="depth-anything/DA3-SMALL",
        family="da3",
        backend_kind=BackendKind.DA3_API,
        license_id="apache-2.0",
        usage_class=UsageClass.COMMERCIAL_OK,
        requires_non_commercial_ok=False,
        supports_coreml=False,
        consumable_outputs=frozenset(
            {
                ConsumableOutput.NORMALIZED_RELATIVE_DEPTH,
                ConsumableOutput.DEPTH_METADATA,
            }
        ),
        upstream_capabilities=frozenset(
            {
                "relative_depth",
                "pose_estimation",
                "pose_conditioning",
            }
        ),
        maturity="experimental",
        exposed_in_cli=False,
    ),
    "coreml_depth_anything_v2_small": ModelSpec(
        key="coreml_depth_anything_v2_small",
        repo_id="apple/coreml-depth-anything-v2-small",
        family="depth_anything_v2",
        backend_kind=BackendKind.COREML_PUBLISHED,
        license_id="apache-2.0",
        usage_class=UsageClass.COMMERCIAL_OK,
        requires_non_commercial_ok=False,
        supports_coreml=True,
        consumable_outputs=frozenset({ConsumableOutput.COREML_RELATIVE_DEPTH}),
        maturity="stable",
        exposed_in_cli=False,
    ),
}


LEGACY_MODEL_VARIANT_ALIASES: Dict[str, str] = {
    "METRIC_LARGE": "da3_research",
    "METRIC_BASE": "da3_base",
    "METRIC_SMALL": "da3_small",
}


LEGACY_MODEL_VARIANT_WARNINGS: Dict[str, str] = {
    "METRIC_LARGE": (
        "ModelVariant.METRIC_LARGE is deprecated. It resolves to da3_research, "
        "which uses depth-anything/DA3NESTED-GIANT-LARGE-1.1 and requires "
        "non_commercial_ok=True. Use model_key='da3-research' explicitly for "
        "the research model; the commercial-safe default is da3_metric."
    ),
    "METRIC_BASE": (
        "ModelVariant.METRIC_BASE is deprecated. It resolves to da3_base for "
        "compatibility. The current Lux Depth V3 adapter emits normalized "
        "relative depth, not guaranteed metric-depth output."
    ),
    "METRIC_SMALL": (
        "ModelVariant.METRIC_SMALL is deprecated. It resolves to da3_small for "
        "compatibility. The current Lux Depth V3 adapter emits normalized "
        "relative depth, not guaranteed metric-depth output."
    ),
}


# Repair 1.2 (#2066, option A): deprecated model_key aliases. Each alias keeps
# its historical meaning for the length of the warning cycle — its meaning
# never silently flips — and resolution emits DeprecatedModelSelectorWarning
# steering callers to an explicit selector.
DEPRECATED_MODEL_KEY_ALIAS_WARNINGS: Dict[str, str] = {
    "da3": (
        "model_key='da3' is deprecated as a model selector. It still resolves "
        "to da3_research (cc-by-nc-4.0, requires non_commercial_ok=True) for "
        "compatibility, but no longer matches the default. Use "
        "model_key='da3-research' explicitly for the research model, or "
        "model_key='da3-metric' (Apache-2.0) — the commercial-safe default. "
        "'da3' remains the backend-family identifier (--depth-backend da3), "
        "which is unchanged."
    ),
}


def canonicalize_selector(selector: str) -> str:
    """Normalize a selector for alias lookup."""
    return selector.strip().lower()


def deprecated_model_key_alias_warning(selector: Optional[str]) -> Optional[str]:
    """Return the deprecation warning text for a deprecated model_key alias."""
    if not selector:
        return None
    return DEPRECATED_MODEL_KEY_ALIAS_WARNINGS.get(canonicalize_selector(selector))


def canonicalize_repo_id(repo_id: str) -> str:
    """Normalize a repo identifier for lookup."""
    return repo_id.strip().lower()


MODEL_ALIASES: Dict[str, str] = {
    "da3": "da3_research",
    "da3-research": "da3_research",
    "da3_research": "da3_research",
    "da3-metric": "da3_metric",
    "da3_metric": "da3_metric",
    "da3-base": "da3_base",
    "da3_base": "da3_base",
    "da3-small": "da3_small",
    "da3_small": "da3_small",
    "coreml-depth-anything-v2-small": "coreml_depth_anything_v2_small",
    "coreml_depth_anything_v2_small": "coreml_depth_anything_v2_small",
}
MODEL_ALIASES.update({canonicalize_repo_id(spec.repo_id): spec.key for spec in MODEL_REGISTRY.values()})


def resolve_registry_key(selector: Optional[str]) -> Optional[str]:
    """Resolve a model selector to a canonical registry key."""
    if selector is None:
        return None

    raw_value = selector.strip()
    if not raw_value:
        return None
    if raw_value in MODEL_REGISTRY:
        return raw_value
    return MODEL_ALIASES.get(canonicalize_selector(raw_value))


def get_model_spec(key: str) -> ModelSpec:
    """Return a registry entry by canonical key."""
    return MODEL_REGISTRY[key]


def resolve_model_spec(selector: str) -> Optional[ModelSpec]:
    """Resolve any selector to a registry spec."""
    key = resolve_registry_key(selector)
    if key is None:
        return None
    return get_model_spec(key)


def resolve_legacy_model_variant_key(model_variant: Any) -> Optional[str]:
    """Resolve a legacy ModelVariant enum member to a canonical key."""
    variant_name = getattr(model_variant, "name", None)
    if not isinstance(variant_name, str):
        return None
    return LEGACY_MODEL_VARIANT_ALIASES.get(variant_name)


def legacy_model_variant_warning(model_variant: Any) -> Optional[str]:
    """Return the deprecation warning text for a legacy ModelVariant."""
    variant_name = model_variant if isinstance(model_variant, str) else getattr(model_variant, "name", None)
    if not isinstance(variant_name, str):
        return None
    return LEGACY_MODEL_VARIANT_WARNINGS.get(variant_name)


def visible_cli_model_specs() -> Tuple[ModelSpec, ...]:
    """Return the public CLI-visible model choices."""
    ordered_keys = ("da3_research", "da3_metric")
    return tuple(MODEL_REGISTRY[key] for key in ordered_keys if MODEL_REGISTRY[key].exposed_in_cli)


def lock_required_model_specs() -> Tuple[ModelSpec, ...]:
    """Return all registry specs that require model-lock coverage."""
    return tuple(spec for spec in MODEL_REGISTRY.values() if spec.lock_required)
