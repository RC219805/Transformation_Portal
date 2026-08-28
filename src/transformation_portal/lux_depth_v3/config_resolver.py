"""Configuration resolution for lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition.

This module provides:
- Preset discovery and loading
- Configuration merging (user override > preset > default)
- Configuration fingerprint computation for cache validation
- ResolvedConfig data class for fully resolved configuration state

The configuration resolution follows this precedence:
1. User-provided overrides (explicit CLI/API parameters)
2. Preset defaults (ARCHITECTURAL_INTERIOR, LUXURY_ESTATE, etc.)
3. System defaults (DA3Config, EnhanceConfig defaults)

Usage:
    from transformation_portal.lux_depth_v3.config_resolver import (
        ConfigResolver,
        ResolvedConfig,
        resolve_preset,
        compute_config_fingerprint,
    )

    # Using ConfigResolver class
    resolver = ConfigResolver()
    resolved = resolver.resolve(enhance_config)

    # Using standalone functions
    da3_config, resolved_model_variant = resolve_preset(preset, model_variant_override)
    fingerprint = compute_config_fingerprint(config)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.da3_runtime import REPO_LOCAL_DA3_PYTHON, find_repo_root, repo_local_da3_python_path
from ..core.raw_runtime import RAW_RUNTIME_ENV_VAR, REPO_LOCAL_RAW_PYTHON, repo_local_raw_python_path
from ..ingest.canonical_json import canonicalize_json, dumps_json
from ._backend_contract import normalize_backend_id
from .config import DA3Config, EnhanceConfig, ModelVariant, Preset
from .manifest import ConfigFingerprint
from .model_resolution import (
    ModelRequest,
    ResolvedModel,
    resolve_model_contract,
    validate_authoritative_model_contract,
)

logger = logging.getLogger(__name__)
_REPO_LOCAL_DEPTH_PRO_PYTHON_PARTS = (".venv-depth-pro", "bin", "python")
REPO_LOCAL_DEPTH_PRO_PYTHON = f"./{'/'.join(_REPO_LOCAL_DEPTH_PRO_PYTHON_PARTS)}"
DEPTH_PRO_MODEL_ID = "apple/ml-depth-pro"


def _repo_local_da3_python_path() -> Optional[Path]:
    """Return the canonical repo-local DA3 interpreter path."""
    return repo_local_da3_python_path(Path(__file__))


def _repo_local_depth_pro_python_path() -> Optional[Path]:
    """Return the canonical repo-local Depth Pro interpreter path."""
    repo_root = find_repo_root(Path(__file__))
    if repo_root is None:
        return None
    return repo_root.joinpath(*_REPO_LOCAL_DEPTH_PRO_PYTHON_PARTS)


def _repo_local_raw_python_path() -> Optional[Path]:
    """Return the canonical repo-local RAW interpreter path."""
    return repo_local_raw_python_path(Path(__file__))


def _normalize_python_executable(value: Any) -> Optional[str]:
    """Normalize Python executable configuration values."""
    if value is None:
        return None
    try:
        normalized = os.fspath(value).strip()
    except TypeError:
        normalized = str(value).strip()
    return normalized or None


def resolve_effective_da3_python_executable(
    config: EnhanceConfig,
) -> Optional[str]:
    """Resolve the effective DA3 runtime executable for this config.

    Resolution precedence:
    1. Explicit config.da3_python_executable
    2. TRANSFORMATION_PORTAL_DA3_PYTHON environment override
    3. Repo-local stable contract path when present
    """
    configured = _normalize_python_executable(getattr(config, "da3_python_executable", None))
    if configured:
        return configured

    env_candidate = _normalize_python_executable(os.environ.get("TRANSFORMATION_PORTAL_DA3_PYTHON"))
    if env_candidate:
        return env_candidate

    repo_local_python = _repo_local_da3_python_path()
    if repo_local_python is not None and repo_local_python.exists():
        return REPO_LOCAL_DA3_PYTHON

    return None


def apply_effective_da3_runtime_config(
    config: EnhanceConfig,
) -> EnhanceConfig:
    """Persist the effective DA3 runtime choice onto the config object."""
    config.da3_python_executable = resolve_effective_da3_python_executable(config)
    return config


def resolve_effective_raw_python_executable(
    config: EnhanceConfig,
) -> Optional[str]:
    """Resolve the effective RAW runtime executable for this config.

    Resolution precedence:
    1. Explicit config.raw_python_executable
    2. RAW runtime environment override
    3. Repo-local stable contract path when present
    """
    configured = _normalize_python_executable(getattr(config, "raw_python_executable", None))
    if configured:
        return configured

    env_candidate = _normalize_python_executable(os.environ.get(RAW_RUNTIME_ENV_VAR))
    if env_candidate:
        return env_candidate

    repo_local_python = _repo_local_raw_python_path()
    if repo_local_python is not None and repo_local_python.exists():
        return REPO_LOCAL_RAW_PYTHON

    return None


def apply_effective_raw_runtime_config(
    config: EnhanceConfig,
) -> EnhanceConfig:
    """Persist the effective RAW runtime choice onto the config object."""
    config.raw_python_executable = resolve_effective_raw_python_executable(config)
    return config


def resolve_effective_depth_pro_python_executable(
    config: EnhanceConfig,
) -> Optional[str]:
    """Resolve the effective Depth Pro runtime executable for this config.

    Resolution precedence:
    1. Explicit config.depth_pro_python_executable
    2. TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON environment override
    3. Repo-local stable contract path when present
    """
    configured = _normalize_python_executable(getattr(config, "depth_pro_python_executable", None))
    if configured:
        return configured

    env_candidate = _normalize_python_executable(os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON"))
    if env_candidate:
        return env_candidate

    repo_local_python = _repo_local_depth_pro_python_path()
    if repo_local_python is not None and repo_local_python.exists():
        return REPO_LOCAL_DEPTH_PRO_PYTHON

    return None


def apply_effective_depth_pro_runtime_config(
    config: EnhanceConfig,
) -> EnhanceConfig:
    """Persist the effective Depth Pro runtime choice onto the config object."""
    config.depth_pro_python_executable = resolve_effective_depth_pro_python_executable(config)
    return config


@dataclass
class PresetInfo:
    """Information about an available preset.

    Attributes:
        name: Preset enum value name (e.g., "ARCHITECTURAL_INTERIOR")
        value: Preset string value (e.g., "architectural_interior")
        display_name: Human-readable preset name
        description: Optional description of the preset
        default_model: Default model variant for this preset
        tier: Quality tier (standard, premium, apex)
    """

    name: str
    value: str
    display_name: str
    description: Optional[str] = None
    default_model: Optional[str] = None
    tier: str = "standard"


@dataclass
class ResolvedConfig:
    """Fully resolved configuration state.

    This class holds the result of configuration resolution, capturing:
    - The original user request
    - What was resolved/defaulted
    - The final effective configuration

    Attributes:
        enhance_config: The (possibly mutated) EnhanceConfig
        da3_config: Resolved DA3 pipeline configuration
        preset_requested: Original preset request (may be None)
        preset_resolved: Actually resolved preset value
        model_variant: Resolved model variant
        quality_tier: Resolved quality tier
        fingerprint: Configuration fingerprint for caching
    """

    enhance_config: EnhanceConfig
    da3_config: DA3Config
    preset_requested: Optional[str] = None
    preset_resolved: Optional[str] = None
    model_variant: Optional[ModelVariant] = None
    quality_tier: str = "standard"
    fingerprint: Optional[ConfigFingerprint] = None
    resolved_model_contract: Optional[ResolvedModel] = None


def _compat_model_variant_for_resolved_key(canonical_key: str) -> ModelVariant:
    """Map resolved registry keys back to compatibility model variants."""
    if canonical_key == "da3_base":
        return ModelVariant.METRIC_BASE
    if canonical_key == "da3_small":
        return ModelVariant.METRIC_SMALL
    return ModelVariant.METRIC_LARGE


def resolved_model_identity_for_backend(
    config: EnhanceConfig,
    model_variant: Optional[ModelVariant] = None,
    *,
    backend_id: Optional[Any] = None,
    resolved_model_contract: Optional[ResolvedModel] = None,
) -> str:
    """Return the model identity serialized into replay/cache fingerprints.

    When an authoritative resolved contract is available — passed explicitly
    (plan path) or carried on the config (run path) — the identity is the
    contract's ``canonical_key:repo_id@revision``, which distinguishes models
    the legacy ModelVariant label cannot (da3_metric vs da3_research both map
    to METRIC_LARGE) and changes when the locked revision changes. This keeps
    plan, cache, manifest, and run-card identity on ONE algorithm (P0-1,
    issue #2065). Without a contract, the legacy label is preserved so
    uncarried runs keep their existing fingerprints.
    """
    normalized_backend = normalize_backend_id(
        backend_id if backend_id is not None else getattr(config, "depth_backend", None),
    )
    if normalized_backend == "depth_pro":
        return DEPTH_PRO_MODEL_ID

    contract = resolved_model_contract
    if contract is None:
        invocation = getattr(config, "resolved_invocation", None)
        if invocation is not None:
            contract = getattr(invocation, "resolved_model", None)
    if contract is not None and (normalized_backend is None or normalized_backend == "da3"):
        repo_id = getattr(contract.spec, "repo_id", None)
        return f"{contract.canonical_key}:{repo_id}@{contract.revision}"

    mv = model_variant or config.model_variant or ModelVariant.METRIC_LARGE
    return mv.value.name


def discover_presets(pipeline: str = "lux_depth_v3") -> List[PresetInfo]:
    """Discover available presets for a given pipeline.

    Args:
        pipeline: Pipeline identifier (currently only "lux_depth_v3" supported)

    Returns:
        List of PresetInfo objects describing available presets
    """
    if pipeline != "lux_depth_v3":
        logger.warning(
            "Preset discovery for pipeline '%s' not supported, returning empty list",
            pipeline,
        )
        return []

    presets = []
    preset_metadata = {
        Preset.ARCHITECTURAL_INTERIOR: {
            "display_name": "Architectural Interior",
            "description": "High quality for interior architectural renders",
            "default_model": "METRIC_LARGE",
            "tier": "premium",
        },
        Preset.ARCHITECTURAL_EXTERIOR: {
            "display_name": "Architectural Exterior",
            "description": "Balanced for exterior scenes",
            "default_model": "METRIC_BASE",
            "tier": "standard",
        },
        Preset.LUXURY_ESTATE: {
            "display_name": "Luxury Estate",
            "description": "Premium quality for luxury real estate",
            "default_model": "METRIC_LARGE",
            "tier": "apex",
        },
        Preset.DEFAULT: {
            "display_name": "Default",
            "description": "Standard balanced configuration",
            "default_model": "METRIC_LARGE",
            "tier": "standard",
        },
    }

    for preset in Preset:
        metadata = preset_metadata.get(preset, {})
        presets.append(
            PresetInfo(
                name=preset.name,
                value=preset.value,
                display_name=metadata.get("display_name", preset.name),
                description=metadata.get("description"),
                default_model=metadata.get("default_model"),
                tier=metadata.get("tier", "standard"),
            )
        )

    return presets


def resolve_preset(
    preset: Optional[Preset],
    model_variant_override: Optional[ModelVariant] = None,
) -> Tuple[DA3Config, Optional[ModelVariant]]:
    """Resolve a preset to DA3Config with optional model override.

    Implements the configuration precedence:
    1. User-provided model_variant (if specified)
    2. Preset default model_variant
    3. System default (METRIC_LARGE)

    Args:
        preset: Optional preset to resolve
        model_variant_override: Optional user-specified model variant

    Returns:
        Tuple of (DA3Config, resolved_model_variant)
    """
    if preset is not None:
        da3_config = DA3Config.from_preset(preset)
        if model_variant_override is not None:
            logger.info(
                "Overriding preset model with user choice: %s",
                model_variant_override.value.display_name,
            )
            da3_config.model_variant = model_variant_override
            return da3_config, model_variant_override
        return da3_config, da3_config.model_variant
    else:
        # No preset: use explicit model or default
        model = model_variant_override if model_variant_override is not None else ModelVariant.METRIC_LARGE
        return DA3Config(model_variant=model), model


def build_materials_fingerprint_payload(config: EnhanceConfig) -> Dict[str, Any]:
    """Build Materials V3 fingerprint payload for cache validation.

    Extracts all Materials V3 configuration settings that affect
    output and should invalidate caches when changed.

    Args:
        config: EnhanceConfig instance

    Returns:
        Dictionary of materials configuration for fingerprinting
    """
    # Mirror the segmentation-mask cache key: when the runtime would fall
    # through to SAMVitHBackend.EXPECTED_SHA256 (because the operator did not
    # populate EnhanceConfig.sam_vit_h_expected_sha256), record that effective
    # hash in the manifest fingerprint too. Otherwise full-pipeline manifests
    # written before the backend default became fail-closed would still match
    # a current run — orchestrator.should_skip_depth() could replay the
    # cached run and skip segment_materials() entirely, bypassing
    # _validate_checkpoint_sha256() on the underlying checkpoint bytes.
    raw_sam2_model_config = getattr(config, "sam2_model_config", None)
    raw_sam2_expected_sha256 = getattr(config, "sam2_expected_sha256", None)
    effective_sam2_model_config = raw_sam2_model_config
    effective_sam2_expected_sha256 = raw_sam2_expected_sha256
    if str(getattr(config, "material_segmentation_backend", "")) == "sam2":
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend as SpatialSAM2Backend

        sam2_model_size = str(getattr(config, "sam2_model_size", "base")).lower()
        effective_sam2_model_config = raw_sam2_model_config or SpatialSAM2Backend.MODEL_CONFIGS.get(sam2_model_size)
        effective_sam2_expected_sha256 = raw_sam2_expected_sha256 or SpatialSAM2Backend.CHECKPOINT_SHA256.get(sam2_model_size)

    raw_sam_vit_h_expected_sha256 = getattr(config, "sam_vit_h_expected_sha256", None)
    effective_sam_vit_h_expected_sha256 = raw_sam_vit_h_expected_sha256
    if str(getattr(config, "material_segmentation_backend", "")) == "sam_vit_h":
        # Lazy import: avoid expanding the config_resolver import surface and
        # keep the segmentation package out of orchestrator-init hot paths.
        from .segmentation.sam_vit_h import SAMVitHBackend

        effective_sam_vit_h_expected_sha256 = raw_sam_vit_h_expected_sha256 or SAMVitHBackend.EXPECTED_SHA256

    return {
        "enable_materials_v3": bool(config.enable_materials_v3),
        "apply_pixel_ops": bool(config.apply_pixel_ops),
        # APEX Materials V3 strict-gate policy version. Bumped when the gate's
        # blocker semantics change so cache replays don't serve verdicts under
        # old policy. v2 introduces soft-passthrough on confidence-only blocks.
        "pixel_ops_strict_policy_version": "v2",
        "enable_material_segmentation": bool(config.enable_material_segmentation),
        "material_segmentation_backend": str(config.material_segmentation_backend),
        "strict_backend": bool(config.strict_backend),
        "refinement_strategy": str(config.refinement_strategy),
        "min_coverage_px": int(config.min_coverage_px),
        "min_mean_conf": float(config.min_mean_conf),
        "glass_response_enabled": bool(config.glass_response_enabled),
        "mask_feather_sigma_default": float(config.mask_feather_sigma_default),
        "mask_feather_sigma_overrides": {
            key: float(value) for key, value in sorted(config.mask_feather_sigma_overrides.items())
        },
        "mask_feather_disabled_materials": sorted(str(value) for value in config.mask_feather_disabled_materials),
        # Seam-safe guard knobs for large low-texture materials. These must be
        # in the fingerprint so that changing any of them invalidates cached
        # materials_v3 artifacts — otherwise operators tuning these will keep
        # seeing pre-change output from the reuse cache.
        "pixel_ops_low_grad_threshold": float(getattr(config, "pixel_ops_low_grad_threshold", 0.01)),
        "pixel_ops_low_tex_min_bbox_frac": float(getattr(config, "pixel_ops_low_tex_min_bbox_frac", 0.05)),
        "pixel_ops_low_tex_feather_multiplier": float(getattr(config, "pixel_ops_low_tex_feather_multiplier", 8.0)),
        "pixel_ops_low_tex_delta_ceiling": float(getattr(config, "pixel_ops_low_tex_delta_ceiling", 0.04)),
        "sky_top_region_fraction": float(config.sky_top_region_fraction),
        "sky_gradient_threshold": float(config.sky_gradient_threshold),
        "sky_brightness_threshold": float(config.sky_brightness_threshold),
        "sam2_model_size": str(config.sam2_model_size),
        "sam2_checkpoint_path": config.sam2_checkpoint_path,
        "sam2_model_config": effective_sam2_model_config,
        "sam2_expected_sha256": effective_sam2_expected_sha256,
        "sam2_tiling_enabled": bool(config.sam2_tiling_enabled),
        "sam2_tile_size_px": int(config.sam2_tile_size_px),
        "sam2_overlap_px": int(config.sam2_overlap_px),
        "sam2_global_pass_longest_side": int(config.sam2_global_pass_longest_side),
        "sam2_max_concurrency": int(config.sam2_max_concurrency),
        "sam2_points_per_side": int(config.sam2_points_per_side),
        "sam2_points_per_batch": int(config.sam2_points_per_batch),
        "sam2_pred_iou_thresh": float(config.sam2_pred_iou_thresh),
        "sam2_stability_score_thresh": float(config.sam2_stability_score_thresh),
        "sam2_crop_n_layers": int(config.sam2_crop_n_layers),
        "sam_vit_h_checkpoint_path": getattr(config, "sam_vit_h_checkpoint_path", None),
        "sam_vit_h_points_per_side": int(getattr(config, "sam_vit_h_points_per_side", 32)),
        "sam_vit_h_pred_iou_thresh": float(getattr(config, "sam_vit_h_pred_iou_thresh", 0.88)),
        "sam_vit_h_confidence_threshold": float(getattr(config, "sam_vit_h_confidence_threshold", 0.85)),
        "sam_vit_h_expected_sha256": effective_sam_vit_h_expected_sha256,
    }


def build_pbr_fingerprint_payload(config: EnhanceConfig) -> Dict[str, Any]:
    """Build PBR fingerprint payload for cache validation.

    Extracts all PBR configuration settings that affect output
    and should invalidate caches when changed.

    Args:
        config: EnhanceConfig instance

    Returns:
        Dictionary of PBR configuration for fingerprinting
    """
    return {
        "generate_pbr": bool(config.generate_pbr),
        "save_float_depth": bool(getattr(config, "save_float_depth", False)),
        "normal_strength": float(config.pbr_normal_strength),
        "normal_blur_radius": int(config.pbr_normal_blur_radius),
        "roughness_strength": float(config.pbr_roughness_strength),
        "roughness_blur_radius": int(config.pbr_roughness_blur_radius),
        "ao_strength": float(config.pbr_ao_strength),
        "ao_blur_radius": int(config.pbr_ao_blur_radius),
        "ao_bias": float(config.pbr_ao_bias),
    }


def build_apex_depth_gate_fingerprint_payload(config: EnhanceConfig) -> Dict[str, Any]:
    """Build APEX depth gate fingerprint payload for cache validation.

    Extracts all APEX depth gate configuration settings that affect
    output and should invalidate caches when changed.

    Args:
        config: EnhanceConfig instance

    Returns:
        Dictionary of APEX gate configuration for fingerprinting
    """
    return {
        "quality_tier": str(config.quality_tier),
        "min_finite_pct": float(config.apex_depth_min_finite_pct),
        "min_upper_iqr": float(config.apex_depth_min_upper_iqr),
        "max_high_saturation_fraction": float(config.apex_depth_max_high_saturation_fraction),
        "max_low_saturation_fraction": float(config.apex_depth_max_low_saturation_fraction),
        "scaled_saturation_margin": float(config.apex_depth_scaled_saturation_margin),
        "low_saturation_warning_band": float(
            config.apex_depth_low_saturation_warning_band,
        ),
        "saturation_high_value": float(config.apex_depth_saturation_high_value),
        "saturation_low_value": float(config.apex_depth_saturation_low_value),
        "min_gradient_energy": float(config.apex_depth_min_gradient_energy),
        "threshold_epsilon": float(config.apex_depth_threshold_epsilon),
        "hist_bins": int(config.apex_depth_hist_bins),
        # `depth_fallback` is part of the gate policy: APEX auto-upgrades
        # "fail" → "v2-auto" (different recovery on flat scenes), and the
        # operator can opt back into strict-fail with `apex-strict`. Two runs
        # with the same depth thresholds but different fallbacks produce
        # observably different outputs, so cache replays must invalidate.
        "depth_fallback": str(config.depth_fallback),
    }


def build_depth_cache_payload(
    config: EnhanceConfig,
    model_variant: Optional[ModelVariant] = None,
) -> Dict[str, Any]:
    """Build depth cache fingerprint payload.

    Extracts depth-related configuration settings for cache key
    computation. The model_variant is required for consistent cache keys.

    Args:
        config: EnhanceConfig instance
        model_variant: Resolved model variant (uses config.model_variant if not provided)

    Returns:
        Dictionary of depth configuration for cache fingerprinting
    """
    mv = model_variant or config.model_variant
    if mv is None:
        mv = ModelVariant.METRIC_LARGE
    effective_da3_python = resolve_effective_da3_python_executable(config)
    effective_raw_python = resolve_effective_raw_python_executable(config)

    return {
        "model_variant": resolved_model_identity_for_backend(config, mv),
        "model_key": getattr(config, "model_key", None),
        "raw_model_id": getattr(config, "raw_model_id", None),
        "depth_device": config.depth_device,
        "preset": config.preset.value if config.preset else None,
        "depth_backend": config.depth_backend,
        "depth_pro_checkpoint_path": config.depth_pro_checkpoint_path,
        "depth_pro_python_executable": config.depth_pro_python_executable,
        "raw_python_executable": effective_raw_python,
        "da3_python_executable": effective_da3_python,
    }


def require_model_variant(config: EnhanceConfig) -> ModelVariant:
    """Return the resolved model variant required by orchestrator helpers."""
    mv = config.model_variant
    assert mv is not None, "model_variant must be set"
    return mv


def build_depth_cache_fingerprint(
    config: EnhanceConfig,
    model_variant: ModelVariant,
    backend_id: str,
    output_depth_units: str,
) -> str:
    """Build backend-scoped depth cache fingerprint."""
    cache_payload = build_depth_cache_payload(config, model_variant)
    base_fp = hashlib.sha256(
        dumps_json(
            cache_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = f"{base_fp}|backend={backend_id}|units={output_depth_units}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_config_fingerprint(
    config: EnhanceConfig,
    model_variant: Optional[ModelVariant] = None,
    *,
    resolved_model_contract: Optional[ResolvedModel] = None,
) -> ConfigFingerprint:
    """Compute configuration fingerprint for cache validation.

    Creates a ConfigFingerprint capturing all settings that affect
    pipeline output, enabling accurate cache invalidation.

    Args:
        config: EnhanceConfig instance
        model_variant: Resolved model variant (defaults to config.model_variant)

    Returns:
        ConfigFingerprint instance
    """
    mv = model_variant or config.model_variant
    if mv is None:
        mv = ModelVariant.METRIC_LARGE
    effective_da3_python = resolve_effective_da3_python_executable(config)
    effective_raw_python = resolve_effective_raw_python_executable(config)

    return ConfigFingerprint(
        model_variant=resolved_model_identity_for_backend(
            config,
            mv,
            resolved_model_contract=resolved_model_contract,
        ),
        depth_quantization=config.depth_quantization,
        depth_device=config.depth_device,
        preset=config.preset.value if config.preset else None,
        v2_preset=config.v2_preset,
        v2_device=config.v2_device,
        v2_upscaler_backend=config.v2_upscaler_backend,
        depth_backend=config.depth_backend,
        depth_pro_checkpoint_path=config.depth_pro_checkpoint_path,
        depth_pro_python_executable=config.depth_pro_python_executable,
        raw_python_executable=effective_raw_python,
        da3_python_executable=effective_da3_python,
        quality_tier=str(config.quality_tier),
        materials_config=build_materials_fingerprint_payload(config),
        pbr_config=build_pbr_fingerprint_payload(config),
        apex_depth_gate_config=build_apex_depth_gate_fingerprint_payload(config),
        emit_master16=bool(config.emit_master16),
        emit_upscaled16=bool(config.emit_upscaled16),
        enable_v2=bool(config.enable_v2),
    )


def build_run_card_config_fingerprint(
    config: EnhanceConfig,
    model_variant: Optional[ModelVariant] = None,
    backend_metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build run-card config fingerprint with full provenance.

    Creates the extended fingerprint payload used in run cards,
    including resolution metadata and canonical JSON hash.

    Args:
        config: EnhanceConfig instance
        model_variant: Resolved model variant
        backend_metadata: Optional backend selection metadata

    Returns:
        Dictionary with fingerprint payload and SHA-256 hash
    """
    from .ingest_adapter import raw_ingest_summary

    base = compute_config_fingerprint(config, model_variant)
    raw_summary = raw_ingest_summary(
        config,
        raw_python_executable=base.raw_python_executable,
    )

    preset_requested = getattr(config, "preset_requested", None) or (config.preset.value if config.preset else None)
    preset_resolved = config.preset.value if config.preset else f"quality_tier:{config.quality_tier}"

    # Extract backend resolution info
    requested_backend = "auto"
    resolved_backend = None
    requested_device = config.depth_device
    resolved_device = config.depth_device

    if backend_metadata is not None:
        requested_backend = getattr(backend_metadata, "requested_backend", None) or "auto"
        resolved_backend = getattr(backend_metadata, "resolved_backend", None)
        resolved_device = getattr(backend_metadata, "device", config.depth_device)

    payload = {
        "model_variant": base.model_variant,
        "depth_quantization": base.depth_quantization,
        "depth_device": base.depth_device,
        "preset": base.preset,
        "v2_preset": base.v2_preset,
        "v2_device": base.v2_device,
        "v2_upscaler_backend": base.v2_upscaler_backend,
        "depth_pro_python_executable": base.depth_pro_python_executable,
        "raw_python_executable": base.raw_python_executable,
        "da3_python_executable": base.da3_python_executable,
        "preset_requested": preset_requested,
        "preset_resolved": preset_resolved,
        "backend_requested": requested_backend,
        "backend_resolved": resolved_backend,
        "device_requested": requested_device,
        "device_resolved": resolved_device,
        "quality_tier": config.quality_tier,
        "strict_inputs": bool(config.strict_inputs),
        "strict_segmentation": bool(config.strict_backend),
        "apex_strict_mode": config.quality_tier == "apex",
        "raw_ingest_profile": str(raw_summary.get("profile", "")),
        "raw_ingest_settings_hash": str(
            raw_summary.get("settings_hash", ""),
        ),
    }

    canonical_json_bytes = canonicalize_json(payload)
    canonical_json_str = canonical_json_bytes.decode("utf-8")

    return {
        **payload,
        "hash_algorithm": "sha256",
        "canonical_json": canonical_json_str,
        "sha256": hashlib.sha256(canonical_json_bytes).hexdigest(),
    }


def finalize_run_card_config_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach canonical JSON and SHA-256 over the resolved fingerprint payload."""
    canonical_payload = {
        key: value for key, value in payload.items() if key not in {"hash_algorithm", "canonical_json", "sha256"}
    }
    canonical_json_bytes = canonicalize_json(canonical_payload)
    return {
        **canonical_payload,
        "hash_algorithm": "sha256",
        "canonical_json": canonical_json_bytes.decode("utf-8"),
        "sha256": hashlib.sha256(canonical_json_bytes).hexdigest(),
    }


def build_orchestrator_run_card_config_fingerprint(
    config: EnhanceConfig,
    model_variant: ModelVariant,
    backend_metadata: Optional[Any],
    *,
    backend_selection: Optional[Dict[str, Any]] = None,
    run_card_version: Optional[str] = None,
    include_proofs: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the orchestrator run-card config fingerprint payload."""
    fingerprint = build_run_card_config_fingerprint(
        config,
        model_variant,
        backend_metadata,
    )
    payload = {key: value for key, value in fingerprint.items() if key not in {"hash_algorithm", "canonical_json", "sha256"}}
    resolved_backend = None
    if isinstance(backend_selection, dict):
        resolved_backend = backend_selection.get("resolved")
        payload.update(
            {
                "resolved_model_id": backend_selection.get("model_id"),
                "model_artifact_filename": backend_selection.get("model_artifact_filename"),
                "model_artifact_sha256": backend_selection.get("model_artifact_sha256"),
            }
        )
    if resolved_backend is None:
        resolved_backend = getattr(backend_metadata, "resolved_backend", None)
    normalized_resolved_backend = normalize_backend_id(resolved_backend)
    if normalized_resolved_backend == "depth_pro":
        payload["model_variant"] = "apple/ml-depth-pro"
        payload["preset"] = "depth_pro"
        payload["preset_requested"] = "depth_pro"
        payload["preset_resolved"] = "backend:depth_pro"
    payload["output_depth_units"] = "meters" if normalized_resolved_backend == "depth_pro" else "relative"
    payload["depth_png_encoding"] = "normalized_u16_png"
    materials_v3_enabled = bool(getattr(config, "enable_materials_v3", False))
    material_segmentation_enabled = bool(
        getattr(
            config,
            "enable_material_segmentation",
            False,
        )
    )
    payload["materials_v3_enabled"] = materials_v3_enabled
    payload["material_segmentation_enabled"] = material_segmentation_enabled
    payload["material_segmentation_backend"] = getattr(config, "material_segmentation_backend", None)
    payload["strict_segmentation"] = bool(
        materials_v3_enabled and material_segmentation_enabled and getattr(config, "strict_backend", False)
    )
    payload["pbr_enabled"] = bool(getattr(config, "generate_pbr", False))
    payload["vlm_captioning_enabled"] = bool(getattr(config, "vlm_captioning_enabled", False))
    payload["vlm_captioning_backend"] = getattr(config, "vlm_captioning_backend", "fastvlm")
    payload["vlm_captioning_model"] = getattr(config, "vlm_captioning_model", "default")
    payload["vlm_captioning_proxy_format"] = getattr(config, "vlm_captioning_proxy_format", "png")
    payload["vlm_captioning_max_side_px"] = int(getattr(config, "vlm_captioning_max_side_px", 1600) or 1600)
    payload["fastvlm_timeout_seconds"] = int(getattr(config, "fastvlm_timeout_seconds", 180) or 180)
    if run_card_version is not None:
        payload["run_card_version"] = run_card_version
    if include_proofs is not None:
        payload["run_card_include_proofs"] = bool(include_proofs)
    payload["emit_run_card"] = bool(getattr(config, "emit_run_card", False))
    return finalize_run_card_config_fingerprint(payload)


class ConfigResolver:
    """Configuration resolution and management.

    Provides a unified interface for:
    - Resolving presets to full configurations
    - Merging user overrides with defaults
    - Computing configuration fingerprints
    - Discovering available presets

    This class is the primary interface for configuration resolution
    per ADR-043.

    Example:
        resolver = ConfigResolver()

        # Discover available presets
        presets = resolver.discover_presets()

        # Resolve configuration
        resolved = resolver.resolve(enhance_config)

        # Access resolved values
        print(resolved.da3_config.model_variant)
        print(resolved.fingerprint.to_sha256())
    """

    def __init__(self) -> None:
        """Initialize config resolver."""
        self._preset_cache: Dict[Preset, DA3Config] = {}

    def resolve(self, config: EnhanceConfig) -> ResolvedConfig:
        """Resolve EnhanceConfig to ResolvedConfig.

        Applies the configuration precedence hierarchy:
        1. User-provided overrides
        2. Preset defaults
        3. System defaults

        Mutates config.model_variant if not explicitly set.

        Args:
            config: EnhanceConfig to resolve

        Returns:
            ResolvedConfig with fully resolved settings
        """
        # Resolve preset and model variant
        apply_effective_da3_runtime_config(config)
        apply_effective_raw_runtime_config(config)
        da3_config, resolved_model = resolve_preset(
            config.preset,
            config.model_variant,
        )
        # P0-1 (issue #2065): when the CLI has already performed the single
        # license-enforcing resolution, consume its authoritative contract
        # instead of re-resolving (which would use enforce_license=False and
        # could drift through the legacy model_variant compatibility mapping).
        authoritative_contract = None
        invocation = getattr(config, "resolved_invocation", None)
        if invocation is not None:
            authoritative_contract = getattr(invocation, "resolved_model", None)
        if authoritative_contract is not None:
            resolved_model_contract = validate_authoritative_model_contract(
                authoritative_contract,
                non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
            )
            resolved_model = _compat_model_variant_for_resolved_key(
                resolved_model_contract.canonical_key,
            )
        else:
            # A pure default carries no selection on any plane: the
            # resolved default must then be pinned onto the config BEFORE
            # the compat model_variant mutation below, or downstream
            # re-resolutions (DA3Backend, engine) would read the mutated
            # METRIC_LARGE as an explicit legacy selection and resolve the
            # research model — a split identity between run-card metadata
            # (da3_metric) and the executed model (repair 1.2, #2066).
            pure_default_selection = (
                getattr(config, "model_key", None) is None
                and getattr(config, "raw_model_id", None) is None
                and config.model_variant is None
                and config.preset is None
            )
            resolved_model_contract = resolve_model_contract(
                ModelRequest(
                    model_key=getattr(config, "model_key", None),
                    raw_model_id=getattr(config, "raw_model_id", None),
                    model_variant=config.model_variant,
                    use_coreml_backend=bool(getattr(config, "use_coreml_backend", False)),
                    non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
                    enforce_license=False,
                )
            )
            if pure_default_selection:
                config.model_key = resolved_model_contract.canonical_key
            if getattr(config, "model_key", None) or getattr(config, "raw_model_id", None):
                resolved_model = _compat_model_variant_for_resolved_key(
                    resolved_model_contract.canonical_key,
                )

        # Apply device configuration
        da3_config.device.device = config.depth_device
        da3_config.model_key = getattr(config, "model_key", None)
        da3_config.raw_model_id = getattr(config, "raw_model_id", None)
        da3_config.non_commercial_ok = bool(getattr(config, "non_commercial_ok", False))

        # Update config with resolved model variant
        config.model_variant = resolved_model

        # Determine preset resolution metadata
        preset_requested = getattr(config, "preset_requested", None) or (config.preset.value if config.preset else None)
        preset_resolved = config.preset.value if config.preset else f"quality_tier:{config.quality_tier}"

        # Compute fingerprint
        fingerprint = compute_config_fingerprint(config, resolved_model)

        return ResolvedConfig(
            enhance_config=config,
            da3_config=da3_config,
            preset_requested=preset_requested,
            preset_resolved=preset_resolved,
            model_variant=resolved_model,
            quality_tier=config.quality_tier,
            fingerprint=fingerprint,
            resolved_model_contract=resolved_model_contract,
        )

    def discover_presets(self, pipeline: str = "lux_depth_v3") -> List[PresetInfo]:
        """Discover available presets for a pipeline.

        Args:
            pipeline: Pipeline identifier

        Returns:
            List of PresetInfo describing available presets
        """
        return discover_presets(pipeline)

    def get_preset_config(self, preset: Preset) -> DA3Config:
        """Get cached DA3Config for a preset.

        Args:
            preset: Preset to get configuration for

        Returns:
            DA3Config for the preset (cached)
        """
        if preset not in self._preset_cache:
            self._preset_cache[preset] = DA3Config.from_preset(preset)
        return self._preset_cache[preset]

    def compute_fingerprint(
        self,
        config: EnhanceConfig,
        model_variant: Optional[ModelVariant] = None,
    ) -> ConfigFingerprint:
        """Compute configuration fingerprint.

        Args:
            config: EnhanceConfig instance
            model_variant: Optional resolved model variant

        Returns:
            ConfigFingerprint for cache validation
        """
        return compute_config_fingerprint(config, model_variant)

    def build_run_card_fingerprint(
        self,
        config: EnhanceConfig,
        model_variant: Optional[ModelVariant] = None,
        backend_metadata: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build run-card fingerprint with provenance.

        Args:
            config: EnhanceConfig instance
            model_variant: Resolved model variant
            backend_metadata: Backend selection metadata

        Returns:
            Dictionary with fingerprint and SHA-256 hash
        """
        return build_run_card_config_fingerprint(
            config,
            model_variant,
            backend_metadata,
        )
