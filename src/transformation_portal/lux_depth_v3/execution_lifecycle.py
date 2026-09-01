"""Native, carried Lux execution-plan preparation and consumption.

This module is the execution-authority boundary for ``tp.execution.plan.v1``.
It prepares one immutable canonical carrier before execution and reconstructs
runtime configuration only from that carrier at every later boundary.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..core.execution_plan import (
    EXECUTION_COMPLETE,
    EXECUTION_PLAN_SCHEMA,
    MAX_DECODED_PIXELS_PER_INPUT,
    MAX_INPUT_DECOMPRESSION_RATIO,
    MAX_TOTAL_DECODED_PIXELS,
    BackendCandidateIntent,
    BackendModelIntent,
    CanonicalExecutionPlan,
    ExecutionPlanError,
    ResolvedModelIntent,
    parse_execution_plan_json,
    with_execution_plan_fingerprint,
)
from ..depth.backends.registry import DepthBackendRegistry
from ..ingest.canonical_json import TP_CANONICAL_JSON_PROFILE
from ..stage_graph.registry import StageRegistryIdentifier, get_output_definition, get_stage_definition
from ._backend_contract import normalize_backend_id
from .config import EnhanceConfig, ModelVariant, PostprocessingConfig, Preset, deprecated_output_flag_notices
from .config_resolver import (
    apply_effective_da3_runtime_config,
    apply_effective_depth_pro_runtime_config,
    apply_effective_raw_runtime_config,
    build_materials_fingerprint_payload,
    compute_config_fingerprint,
    preset_model_key_for_selection,
    resolve_effective_depth_pro_checkpoint_path,
    resolve_preset,
    with_typed_preset_provenance,
)
from .execution_plan_adapter import LuxExecutionPlanAuthorityError, revalidate_lux_execution_plan_authority
from .model_registry import AcceleratorKind, get_model_spec
from .model_resolution import (
    ModelRequest,
    ResolvedModel,
    direct_model_contract,
    direct_model_source_selector_state,
    model_selection_migration_notices,
    refresh_direct_model_acknowledgement,
    resolve_model_contract,
    restore_stale_direct_model_selection,
    validate_authoritative_model_contract,
)
from .pipeline_coordinator import resolve_requested_backend, resolve_runtime_backend_chain
from .security import HashMode

_STAGE_ORDER = (
    "preprocess",
    "depth",
    "materials_v3",
    "pbr",
    "v2",
    "reconstruction",
    "output",
)
_STAGE_REGISTRY_IDS = {
    "preprocess": StageRegistryIdentifier.LUX_PREPROCESS,
    "depth": StageRegistryIdentifier.LUX_DEPTH,
    "materials_v3": StageRegistryIdentifier.LUX_MATERIALS_V3,
    "pbr": StageRegistryIdentifier.LUX_PBR,
    "v2": StageRegistryIdentifier.LUX_V2,
    "reconstruction": StageRegistryIdentifier.LUX_RECONSTRUCTION,
    "output": StageRegistryIdentifier.LUX_OUTPUT,
}
_NODE_IDS = {stage: f"lux.{stage}" for stage in _STAGE_ORDER}
_OUTPUT_OWNER = {
    "depth_u16_png": "depth",
    "depth_metadata_json": "depth",
    "depth_float_npy": "depth",
    "materials_v3_masks": "materials_v3",
    "pbr_maps": "pbr",
    "v2_enhanced_image": "v2",
    "reconstruction_bundle": "reconstruction",
    "combined_manifest_json": "output",
    "batch_manifest_json": "output",
    "run_card": "output",
}
_INTERNAL_OUTPUTS = {"preprocess": ("preprocessed_image",), "depth": ("depth_map",)}


@dataclass(frozen=True)
class _ExecutionPlanModelCarrier:
    """Minimal legacy-compatible carrier consumed by ConfigResolver/DA3."""

    resolved_model: Optional[ResolvedModel]


@dataclass(frozen=True)
class BackendCandidateAuthority:
    """Exact carried authority for one candidate or ensemble constituent."""

    plan_fingerprint_sha256: str
    candidate_id: str
    backend_id: str
    constituent_backend_id: Optional[str]
    candidate: BackendCandidateIntent
    model_contract: Optional[BackendModelIntent]
    resolved_model_contract: Optional[ResolvedModel]
    device: str
    weight: Optional[float]


@dataclass(frozen=True)
class PreparedLuxExecution:
    """Canonical plan plus a source-independent runtime projection."""

    plan: CanonicalExecutionPlan
    canonical_plan_bytes: bytes
    runtime_config: EnhanceConfig
    input_root: Path
    input_files: tuple[Path, ...]

    @property
    def plan_fingerprint_sha256(self) -> str:
        """Return the carried plan identity without reserializing it."""

        return self.plan.plan_fingerprint_sha256


def _canonical_input_key(path: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", path).casefold())


def _resolve_input_selection(
    input_root: Path,
    input_files: Sequence[Path],
) -> tuple[Path, tuple[Path, ...], tuple[str, ...]]:
    """Resolve one contained, deterministic input selection."""

    try:
        root = Path(input_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExecutionPlanError(f"Authorized input root cannot be resolved: {input_root}") from exc
    if not root.is_dir():
        raise ExecutionPlanError(f"Authorized input root is not a directory: {root}")

    selected: dict[str, Path] = {}
    collision_keys: dict[str, str] = {}
    for raw_path in input_files:
        lexical = Path(raw_path).expanduser()
        candidate = lexical if lexical.is_absolute() else root / lexical
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExecutionPlanError(f"Planned input cannot be resolved: {raw_path}") from exc
        if not resolved.is_file():
            raise ExecutionPlanError(f"Planned input is not a regular file: {resolved}")
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ExecutionPlanError(f"Planned input escapes the authorized root: {raw_path}") from exc
        if relative in selected:
            raise ExecutionPlanError(f"Duplicate planned input path: {relative!r}")
        collision_key = _canonical_input_key(relative)
        prior = collision_keys.get(collision_key)
        if prior is not None:
            raise ExecutionPlanError(
                f"Planned input paths collide under portable Unicode/case normalization: {prior!r}, {relative!r}"
            )
        collision_keys[collision_key] = relative
        selected[relative] = resolved

    if not selected:
        raise ExecutionPlanError("A Lux execution plan requires at least one input file")
    relative_paths = tuple(sorted(selected))
    resolved_paths = tuple(selected[relative] for relative in relative_paths)
    return root, resolved_paths, relative_paths


def _bind_plan_inputs(
    plan: CanonicalExecutionPlan,
    authorized_input_root: Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Bind carried relative inputs to one externally authorized real root."""

    try:
        authorized_root = Path(authorized_input_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LuxExecutionPlanAuthorityError(f"Authorized input root cannot be resolved: {authorized_input_root}") from exc
    if not authorized_root.is_dir():
        raise LuxExecutionPlanAuthorityError(f"Authorized input root is not a directory: {authorized_root}")

    carried_root = Path(plan.input_root)
    if not carried_root.is_absolute():
        raise LuxExecutionPlanAuthorityError("Execution plan input root must be absolute")
    try:
        carried_real_root = carried_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LuxExecutionPlanAuthorityError(f"Carried input root cannot be resolved: {carried_root}") from exc
    if carried_root != carried_real_root:
        raise LuxExecutionPlanAuthorityError("Execution plan input root must be a canonical real path")
    if carried_real_root != authorized_root:
        raise LuxExecutionPlanAuthorityError(
            f"Execution plan input root {carried_real_root} is not the authorized root {authorized_root}"
        )

    bound: list[Path] = []
    for item in plan.inputs:
        lexical = authorized_root / item.path
        try:
            resolved = lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise LuxExecutionPlanAuthorityError(f"Carried input cannot be resolved: {item.path!r}") from exc
        if not resolved.is_file():
            raise LuxExecutionPlanAuthorityError(f"Carried input is not a regular file: {item.path!r}")
        try:
            resolved_relative = resolved.relative_to(authorized_root).as_posix()
        except ValueError as exc:
            raise LuxExecutionPlanAuthorityError(f"Carried input escapes the authorized root: {item.path!r}") from exc
        if resolved_relative != item.path:
            raise LuxExecutionPlanAuthorityError(
                f"Carried input path changed through a symlink or alias: {item.path!r} -> {resolved_relative!r}"
            )
        bound.append(resolved)
    return authorized_root, tuple(bound)


def _resolved_model_payload(contract: ResolvedModel) -> dict[str, Any]:
    spec = contract.spec
    return {
        "requested_selector": contract.requested_selector,
        "resolution_reason": contract.resolution_reason,
        "canonical_key": contract.canonical_key,
        "repo_id": spec.repo_id,
        "revision": contract.revision,
        "license_id": spec.license_id,
        "usage_class": spec.usage_class.value,
        "requires_non_commercial_ok": spec.requires_non_commercial_ok,
        "accelerator_kind": contract.accelerator_kind.value,
        "legacy_model_variant_name": contract.legacy_model_variant_name,
    }


def _static_model_payload(
    *,
    requested_selector: str,
    canonical_key: str,
    repo_id: str,
    revision: Optional[str],
    license_id: str,
    usage_class: str,
    requires_non_commercial_ok: bool,
) -> dict[str, Any]:
    return {
        "requested_selector": requested_selector,
        "resolution_reason": "native execution plan carried pinned backend authority",
        "canonical_key": canonical_key,
        "repo_id": repo_id,
        "revision": revision,
        "license_id": license_id,
        "usage_class": usage_class,
        "requires_non_commercial_ok": requires_non_commercial_ok,
        "accelerator_kind": "none",
        "legacy_model_variant_name": None,
    }


def _resolve_da3_model_once(config: EnhanceConfig) -> ResolvedModel:
    """Resolve or revalidate the one DA3 authority used by every candidate."""

    restore_stale_direct_model_selection(config)
    refresh_direct_model_acknowledgement(config, stacklevel=4)
    carried = direct_model_contract(config)
    if carried is not None:
        return validate_authoritative_model_contract(
            carried,
            non_commercial_ok=bool(config.non_commercial_ok),
        )

    source_selector_state = direct_model_source_selector_state(config)
    _, resolved_variant = resolve_preset(config.preset, config.model_variant)
    preset_model_key = preset_model_key_for_selection(config, resolved_variant)
    resolved = resolve_model_contract(
        ModelRequest(
            model_key=config.model_key or preset_model_key,
            raw_model_id=config.raw_model_id,
            model_variant=config.model_variant,
            use_coreml_backend=bool(config.use_coreml_backend),
            non_commercial_ok=bool(config.non_commercial_ok),
            enforce_license=True,
        )
    )
    resolved = with_typed_preset_provenance(config, resolved, preset_model_key)
    # Retaining the direct carrier on the private working copy prevents any
    # compatibility resolver invoked while preparing metadata from resolving
    # the selector a second time.
    from .model_resolution import carry_direct_model_contract

    carry_direct_model_contract(config, resolved, source_selector_state=source_selector_state)
    return resolved


def _effective_worker_counts(config: EnhanceConfig) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    if config.max_workers is not None:
        max_workers = int(config.max_workers)
    elif config.depth_device in {"mps", "cuda"}:
        max_workers = min(2, cpu_count)
    else:
        max_workers = int(config.max_parallel_workers or max(1, cpu_count - 1))
    if max_workers < 1 or max_workers > 256:
        raise ExecutionPlanError("max_workers must be between 1 and 256")

    if config.max_gpu_workers is not None:
        max_gpu_workers = int(config.max_gpu_workers)
    elif config.depth_device in {"mps", "cuda"}:
        max_gpu_workers = max_workers
    else:
        max_gpu_workers = 0
    if max_gpu_workers < 0 or max_gpu_workers > 8:
        raise ExecutionPlanError("max_gpu_workers must be between 0 and 8")
    return max_workers, max_gpu_workers


def _file_sha256(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExecutionPlanError(f"Referenced sidecar cannot be resolved: {path_value}") from exc
    if not path.is_file():
        raise ExecutionPlanError(f"Referenced sidecar is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backend_model_contract(
    backend_id: str,
    config: EnhanceConfig,
    da3_model: Optional[ResolvedModel],
    *,
    role: str = "primary",
    weight: Optional[float] = None,
) -> dict[str, Any]:
    device = str(config.depth_device)
    if backend_id == "da3":
        if da3_model is None:
            raise ExecutionPlanError("DA3 execution authority is missing its resolved model")
        model = _resolved_model_payload(da3_model)
        artifact_path = None
        artifact_sha256 = None
    elif backend_id == "depth_pro":
        from ..depth.backends.depth_pro import DepthProBackend
        from .config_resolver import DEPTH_PRO_MODEL_ID

        model = _static_model_payload(
            requested_selector="backend:depth_pro",
            canonical_key="depth_pro",
            repo_id=DEPTH_PRO_MODEL_ID,
            revision=None,
            license_id="apple_amlr",
            usage_class="non_commercial_only",
            requires_non_commercial_ok=True,
        )
        artifact_path = config.depth_pro_checkpoint_path
        artifact_sha256 = DepthProBackend.EXPECTED_SHA256
    elif backend_id == "da2":
        from ..core.security.model_lock import manifest_revision_for_repo
        from ..depth.models.depth_anything_v2 import ModelVariant as DA2ModelVariant

        repo_id = DA2ModelVariant.SMALL.value
        revision = manifest_revision_for_repo(repo_id)
        if revision is None:
            raise ExecutionPlanError("The governed model lock has no DA2 Small revision")
        model = _static_model_payload(
            requested_selector="backend:da2",
            canonical_key="da2_small",
            repo_id=repo_id,
            revision=revision,
            license_id="apache-2.0",
            usage_class="commercial_ok",
            requires_non_commercial_ok=False,
        )
        artifact_path = None
        artifact_sha256 = None
        # The live DA2 implementation supports CPU/MPS only. Preserve the
        # historical safe CPU fallback instead of emitting an authority that
        # its canonical backend must reject at construction time.
        if device not in {"cpu", "mps"}:
            device = "cpu"
    elif backend_id == "depthcrafter":
        raise LuxExecutionPlanAuthorityError(
            "DepthCrafter cannot enter an execution-complete plan until it has a pinned executable identity"
        )
    else:
        raise ExecutionPlanError(f"Backend {backend_id!r} has no carried model-contract shape")

    return {
        "role": role,
        "backend_id": backend_id,
        "model": model,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "enabled": True,
        "weight": weight,
        "device": device,
    }


def _backend_candidates_payload(
    candidate_chain: tuple[str, ...],
    config: EnhanceConfig,
    da3_model: Optional[ResolvedModel],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for backend_id in candidate_chain:
        if backend_id == "synthetic":
            contracts: list[dict[str, Any]] = []
        elif backend_id == "ensemble":
            # The current live ensemble enables Depth Pro and DA3. Its
            # DepthCrafter member is disabled and unpinned, so it cannot cross
            # this execution-authority boundary. Normalize the two enabled
            # weights exactly as the live backend does (0.5/0.8, 0.3/0.8).
            contracts = [
                _backend_model_contract(
                    "depth_pro",
                    config,
                    da3_model,
                    role="ensemble_constituent",
                    weight=0.625,
                ),
                _backend_model_contract(
                    "da3",
                    config,
                    da3_model,
                    role="ensemble_constituent",
                    weight=0.375,
                ),
            ]
        else:
            contracts = [_backend_model_contract(backend_id, config, da3_model)]
        candidates.append({"backend_id": backend_id, "model_contracts": contracts})
    return candidates


def _requested_outputs(config: EnhanceConfig) -> list[str]:
    outputs = [
        "depth_u16_png",
        "depth_metadata_json",
        "combined_manifest_json",
        "batch_manifest_json",
    ]
    if config.save_float_depth:
        outputs.append("depth_float_npy")
    if config.enable_materials_v3:
        outputs.append("materials_v3_masks")
    if config.generate_pbr:
        outputs.append("pbr_maps")
    if config.enable_v2 and config.v2_preset is not None:
        outputs.append("v2_enhanced_image")
    if config.enable_reconstruction:
        outputs.append("reconstruction_bundle")
    if config.emit_run_card:
        outputs.append("run_card")
    return outputs


def _planned_stages(config: EnhanceConfig) -> list[str]:
    stages = ["preprocess", "depth"]
    if config.enable_materials_v3:
        stages.append("materials_v3")
    if config.generate_pbr:
        stages.append("pbr")
    if config.enable_v2 and config.v2_preset is not None:
        stages.append("v2")
    if config.enable_reconstruction:
        stages.append("reconstruction")
    stages.append("output")
    return stages


def _stage_optional(stage: str, quality_tier: str) -> bool:
    return stage == "pbr" or (stage == "materials_v3" and quality_tier.lower() != "apex")


def _output_declaration(
    stage: str,
    artifact_kind: str,
    *,
    disposition: str,
    required: bool,
) -> dict[str, Any]:
    definition = get_output_definition(artifact_kind)
    return {
        "id": f"{_NODE_IDS[stage]}.output.{artifact_kind}",
        "artifact_kind": artifact_kind,
        "scope": definition.scope.value,
        "cardinality": definition.cardinality.value,
        "required": required,
        "disposition": disposition,
    }


def _postprocessing_payload(config: EnhanceConfig) -> dict[str, Any]:
    if config.depth_postprocessing is not None:
        postprocessing = config.depth_postprocessing
    else:
        da3_config, _ = resolve_preset(config.preset, config.model_variant)
        postprocessing = da3_config.postprocessing
    return {
        "apply_metric_scaling": bool(postprocessing.apply_metric_scaling),
        "scale_factor": float(postprocessing.scale_factor),
        "apply_median_filter": bool(postprocessing.apply_median_filter),
        "median_kernel_size": int(postprocessing.median_kernel_size),
        "apply_bilateral_filter": bool(postprocessing.apply_bilateral_filter),
        "bilateral_sigma_color": float(postprocessing.bilateral_sigma_color),
        "bilateral_sigma_space": float(postprocessing.bilateral_sigma_space),
        "preserve_edges": bool(postprocessing.preserve_edges),
        "edge_threshold": float(postprocessing.edge_threshold),
        "fusion_mode": str(postprocessing.fusion_mode),
        # The current live refinement hook is not a serializable executor
        # contract; execution-complete v1 therefore accepts only its null form.
        "refinement": None,
    }


def _stage_configuration(
    stage: str,
    config: EnhanceConfig,
    *,
    planned_backend: str,
    candidate_chain: tuple[str, ...],
    da3_model: Optional[ResolvedModel],
    requested_outputs: list[str],
    max_workers: int,
    max_gpu_workers: int,
) -> dict[str, Any]:
    schema = get_stage_definition(_STAGE_REGISTRY_IDS[stage]).configuration_schema
    base: dict[str, Any] = {"schema": schema, "configuration_completeness": EXECUTION_COMPLETE}
    if stage == "preprocess":
        base.update(
            {
                "verify_images": bool(config.verify_images or config.strict_inputs),
                "raw_ingest_mode": str(config.raw_ingest_mode),
                "raw_wb_mode": str(config.raw_wb_mode),
                "raw_demosaic": str(config.raw_demosaic),
                "raw_python_executable": config.raw_python_executable,
                "raw_preview_escape_enabled": bool(config.raw_preview_escape_enabled),
                "output_key_hash_algorithm": "xxhash" if config.use_xxhash else "sha1",
                "parallel_enabled": bool(config.enable_parallel_processing),
                "max_workers": max_workers,
                "max_gpu_workers": max_gpu_workers,
            }
        )
    elif stage == "depth":
        model_payload = None if da3_model is None else _resolved_model_payload(da3_model)
        base.update(
            {
                "planned_backend": planned_backend,
                "candidate_fallback_chain": list(candidate_chain),
                "resolved_model_key": None if model_payload is None else model_payload["canonical_key"],
                "resolved_model_revision": None if model_payload is None else model_payload["revision"],
                "device": str(config.depth_device),
                "quantization": str(config.depth_quantization),
                "verify_writes": bool(config.verify_depth_writes),
                "save_float_depth": bool(config.save_float_depth),
                "force": bool(config.force_depth),
                "fallback_mode": str(config.depth_fallback),
                "allow_semantic_fallback": bool(config.allow_semantic_fallback),
                "allow_synthetic_fallback": "synthetic" in candidate_chain,
                "da3_python_executable": config.da3_python_executable,
                "da3_subprocess_timeout_seconds": int(config.da3_subprocess_timeout_seconds),
                "depth_pro_checkpoint_path": config.depth_pro_checkpoint_path,
                "depth_pro_python_executable": config.depth_pro_python_executable,
                "hash_mode": config.hash_mode.value,
                "manifest_cache_enabled": bool(config.enable_manifest_cache),
                "depth_cache_enabled": bool(config.enable_depth_cache),
                "depth_cache_max_size_gb": float(config.depth_cache_max_size_gb),
                "postprocessing": _postprocessing_payload(config),
                "apex_gate": {
                    "quality_tier": str(config.quality_tier),
                    "min_finite_pct": float(config.apex_depth_min_finite_pct),
                    "min_upper_iqr": float(config.apex_depth_min_upper_iqr),
                    "max_high_saturation_fraction": float(config.apex_depth_max_high_saturation_fraction),
                    "max_low_saturation_fraction": float(config.apex_depth_max_low_saturation_fraction),
                    "scaled_saturation_margin": float(config.apex_depth_scaled_saturation_margin),
                    "low_saturation_warning_band": float(config.apex_depth_low_saturation_warning_band),
                    "saturation_high_value": float(config.apex_depth_saturation_high_value),
                    "saturation_low_value": float(config.apex_depth_saturation_low_value),
                    "min_gradient_energy": float(config.apex_depth_min_gradient_energy),
                    "threshold_epsilon": float(config.apex_depth_threshold_epsilon),
                    "hist_bins": int(config.apex_depth_hist_bins),
                    "depth_fallback": str(config.depth_fallback),
                },
                "ensemble": (
                    {
                        "fusion_method": str(config.ensemble_fusion_method),
                        "max_variance_threshold": float(config.ensemble_max_variance_threshold),
                        "temporal_post_filter": {
                            "mode": str(config.ensemble_temporal_filter_mode),
                            "alpha": float(config.ensemble_temporal_filter_alpha),
                        },
                    }
                    if "ensemble" in candidate_chain
                    else None
                ),
            }
        )
    elif stage == "materials_v3":
        materials = build_materials_fingerprint_payload(config)
        base.update(
            {
                "apply_pixel_ops": materials["apply_pixel_ops"],
                "pixel_ops_strict_policy_version": materials["pixel_ops_strict_policy_version"],
                "enable_material_segmentation": materials["enable_material_segmentation"],
                "material_segmentation_backend": materials["material_segmentation_backend"],
                "strict_backend": materials["strict_backend"],
                "segmentation_cache_policy": str(config.material_segmentation_cache_policy),
                "refinement_strategy": materials["refinement_strategy"],
                "min_coverage_px": materials["min_coverage_px"],
                "min_mean_conf": materials["min_mean_conf"],
                "glass_response_enabled": materials["glass_response_enabled"],
                "mask_feather_sigma_default": materials["mask_feather_sigma_default"],
                "mask_feather_sigma_overrides": [
                    {"material": key, "sigma": value}
                    for key, value in sorted(materials["mask_feather_sigma_overrides"].items())
                ],
                "mask_feather_disabled_materials": materials["mask_feather_disabled_materials"],
                "pixel_ops_low_grad_threshold": materials["pixel_ops_low_grad_threshold"],
                "pixel_ops_low_tex_min_bbox_frac": materials["pixel_ops_low_tex_min_bbox_frac"],
                "pixel_ops_low_tex_feather_multiplier": materials["pixel_ops_low_tex_feather_multiplier"],
                "pixel_ops_low_tex_delta_ceiling": materials["pixel_ops_low_tex_delta_ceiling"],
                "sky_top_region_fraction": materials["sky_top_region_fraction"],
                "sky_gradient_threshold": materials["sky_gradient_threshold"],
                "sky_brightness_threshold": materials["sky_brightness_threshold"],
                "sam2_model_size": materials["sam2_model_size"],
                "sam2_checkpoint_path": materials["sam2_checkpoint_path"],
                "sam2_model_config": materials["sam2_model_config"],
                "sam2_expected_sha256": materials["sam2_expected_sha256"],
                "sam2_tiling_enabled": materials["sam2_tiling_enabled"],
                "sam2_tile_size_px": materials["sam2_tile_size_px"],
                "sam2_overlap_px": materials["sam2_overlap_px"],
                "sam2_global_pass_longest_side": materials["sam2_global_pass_longest_side"],
                "sam2_max_concurrency": materials["sam2_max_concurrency"],
                "sam2_points_per_side": materials["sam2_points_per_side"],
                "sam2_points_per_batch": materials["sam2_points_per_batch"],
                "sam2_pred_iou_thresh": materials["sam2_pred_iou_thresh"],
                "sam2_stability_score_thresh": materials["sam2_stability_score_thresh"],
                "sam2_crop_n_layers": materials["sam2_crop_n_layers"],
                "sam_vit_h_checkpoint_path": materials["sam_vit_h_checkpoint_path"],
                "sam_vit_h_points_per_side": materials["sam_vit_h_points_per_side"],
                "sam_vit_h_pred_iou_thresh": materials["sam_vit_h_pred_iou_thresh"],
                "sam_vit_h_confidence_threshold": materials["sam_vit_h_confidence_threshold"],
                "sam_vit_h_expected_sha256": materials["sam_vit_h_expected_sha256"],
                "device": str(config.depth_device),
                "output_bit_depth": int(config.output_bit_depth),
            }
        )
    elif stage == "pbr":
        base.update(
            {
                "normal_strength": float(config.pbr_normal_strength),
                "normal_blur_radius": int(config.pbr_normal_blur_radius),
                "roughness_strength": float(config.pbr_roughness_strength),
                "roughness_blur_radius": int(config.pbr_roughness_blur_radius),
                "ao_strength": float(config.pbr_ao_strength),
                "ao_blur_radius": int(config.pbr_ao_blur_radius),
                "ao_bias": float(config.pbr_ao_bias),
            }
        )
    elif stage == "v2":
        base.update(
            {
                "preset": str(config.v2_preset),
                "device": str(config.v2_device),
                "upscaler_backend": str(config.v2_upscaler_backend),
                "timeout_seconds": int(config.v2_timeout),
                "force": bool(config.force_v2),
                "input_bit_depth": int(config.output_bit_depth),
                "output_bit_depth": int(config.output_bit_depth),
                "materials_mask_handoff": (
                    "required"
                    if config.enable_materials_v3 and config.quality_tier.lower() == "apex"
                    else "optional" if config.enable_materials_v3 else "none"
                ),
                "keep_intermediate": bool(config.keep_intermediates),
            }
        )
    elif stage == "reconstruction":
        base.update(
            {
                "grouping_mode": str(config.grouping_mode),
                "cameras_sidecar_path": config.cameras_sidecar_path,
                "cameras_sidecar_sha256": config.cameras_sidecar_sha256,
                "iterations": int(config.reconstruction_iterations),
                "tier": str(config.reconstruction_tier),
                "emit_scene_debug_bundle": bool(config.emit_scene_debug_bundle),
                "risk_threshold": float(config.reconstruction_risk_threshold),
            }
        )
    elif stage == "output":
        base.update(
            {
                "requested_outputs": list(requested_outputs),
                "output_bit_depth": int(config.output_bit_depth),
                "hash_mode": config.hash_mode.value,
                "run_card_enabled": bool(config.emit_run_card),
                "run_card_version": str(config.run_card_version),
                "run_card_include_proofs": bool(config.run_card_include_proofs),
                "keep_intermediates": bool(config.keep_intermediates),
                "captioning": {
                    "enabled": bool(config.vlm_captioning_enabled),
                    "backend": str(config.vlm_captioning_backend),
                    "selector": str(config.vlm_captioning_model),
                    "model_id": None,
                    "model_revision": None,
                    "model_path": config.fastvlm_model_path,
                    "review_model_path": config.fastvlm_review_model_path,
                    "proxy_format": str(config.vlm_captioning_proxy_format),
                    "max_side_px": int(config.vlm_captioning_max_side_px),
                    "python_executable": config.fastvlm_python_executable,
                    "mlx_vlm_dir": config.fastvlm_mlx_vlm_dir,
                    "max_tokens": config.fastvlm_max_tokens,
                    "temperature": config.fastvlm_temperature,
                    "timeout_seconds": int(config.fastvlm_timeout_seconds),
                },
            }
        )
    return base


def _native_plan(
    config: EnhanceConfig,
    *,
    input_root: Path,
    relative_inputs: tuple[str, ...],
    planned_backend: str,
    candidate_chain: tuple[str, ...],
    da3_model: Optional[ResolvedModel],
) -> CanonicalExecutionPlan:
    requested_outputs = _requested_outputs(config)
    stages = _planned_stages(config)
    max_workers, max_gpu_workers = _effective_worker_counts(config)
    config.max_workers = max_workers
    config.max_gpu_workers = max_gpu_workers

    requested_by_stage: dict[str, list[str]] = {stage: [] for stage in stages}
    for artifact_kind in requested_outputs:
        requested_by_stage[_OUTPUT_OWNER[artifact_kind]].append(artifact_kind)

    nodes: list[dict[str, Any]] = []
    for stage in stages:
        optional = _stage_optional(stage, config.quality_tier)
        outputs = [
            _output_declaration(stage, artifact, disposition="intermediate", required=True)
            for artifact in _INTERNAL_OUTPUTS.get(stage, ())
        ]
        for artifact in requested_by_stage[stage]:
            outputs.append(
                _output_declaration(
                    stage,
                    artifact,
                    disposition="requested",
                    required=not optional and artifact != "run_card",
                )
            )
        definition = get_stage_definition(_STAGE_REGISTRY_IDS[stage])
        nodes.append(
            {
                "id": _NODE_IDS[stage],
                "stage_registry_id": definition.identifier.value,
                "configuration": _stage_configuration(
                    stage,
                    config,
                    planned_backend=planned_backend,
                    candidate_chain=candidate_chain,
                    da3_model=da3_model,
                    requested_outputs=requested_outputs,
                    max_workers=max_workers,
                    max_gpu_workers=max_gpu_workers,
                ),
                "resources": definition.resources.to_payload(),
                "outputs": outputs,
                "optional": optional,
                "failure_policy": "omit_outputs" if optional else "abort_plan",
            }
        )

    config_fingerprint = compute_config_fingerprint(
        config,
        resolved_model_contract=da3_model,
    ).to_sha256()
    warnings = list(deprecated_output_flag_notices(config))
    warnings.extend(
        model_selection_migration_notices(
            da3_model,
            non_commercial_ok=bool(config.non_commercial_ok),
        )
    )
    resolved_model_payload = None if da3_model is None else _resolved_model_payload(da3_model)
    payload = with_execution_plan_fingerprint(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "canonicalization": TP_CANONICAL_JSON_PROFILE,
            "configuration_completeness": EXECUTION_COMPLETE,
            "planned_backend": planned_backend,
            "candidate_fallback_chain": list(candidate_chain),
            "backend_candidates": _backend_candidates_payload(candidate_chain, config, da3_model),
            "resolved_model": resolved_model_payload,
            "license_acknowledgements": {
                "non_commercial_ok": bool(config.non_commercial_ok),
                "apple_depth_pro_research": bool(config.accept_apple_depth_pro_research_license),
                "research_tools": bool(config.accept_research_tools_license),
            },
            "license_evaluation": {"enforced": True, "status": "allowed"},
            "quality_tier": str(config.quality_tier),
            "preset_requested": config.preset_requested,
            "preset_resolved": config.preset.value if config.preset else f"quality_tier:{config.quality_tier}",
            "input_selection": {
                "root": input_root.as_posix(),
                "files": [{"id": f"input-{index:06d}", "path": path} for index, path in enumerate(relative_inputs, start=1)],
            },
            "input_limits": {
                "max_decoded_pixels_per_input": MAX_DECODED_PIXELS_PER_INPUT,
                "max_total_decoded_pixels": MAX_TOTAL_DECODED_PIXELS,
                "max_decompression_ratio": MAX_INPUT_DECOMPRESSION_RATIO,
            },
            "config_fingerprint_sha256": config_fingerprint,
            "nodes": nodes,
            "edges": [{"from": _NODE_IDS[source], "to": _NODE_IDS[target]} for source, target in zip(stages, stages[1:])],
            "requested_outputs": requested_outputs,
            "warnings": warnings,
        }
    )
    return revalidate_lux_execution_plan_authority(CanonicalExecutionPlan.from_payload(payload))


def _prepare_working_config(config: EnhanceConfig) -> EnhanceConfig:
    if config.execution_plan_authority is not None or config.execution_plan_canonical_bytes is not None:
        raise ExecutionPlanError("Cannot prepare a new plan from a config that already carries execution authority")
    working = copy.deepcopy(config)
    working.execution_plan_authority = None
    working.execution_plan_canonical_bytes = None
    working.resolved_invocation = None
    apply_effective_da3_runtime_config(working)
    apply_effective_raw_runtime_config(working)
    apply_effective_depth_pro_runtime_config(working)
    working.depth_pro_checkpoint_path = resolve_effective_depth_pro_checkpoint_path(working)

    if working.vlm_captioning_enabled:
        for field_name, environment_name in (
            ("fastvlm_model_path", "TP_FASTVLM_MODEL"),
            ("fastvlm_review_model_path", "TP_FASTVLM_REVIEW_MODEL"),
            ("fastvlm_python_executable", "TP_FASTVLM_PYTHON"),
            ("fastvlm_mlx_vlm_dir", "TP_FASTVLM_MLX_VLM_DIR"),
        ):
            configured = getattr(working, field_name)
            configured_value = None if configured is None else os.fspath(configured).strip() or None
            environment_value = os.environ.get(environment_name)
            frozen_value = configured_value
            if frozen_value is None and environment_value is not None:
                frozen_value = environment_value.strip() or None
            setattr(working, field_name, frozen_value)

        if working.fastvlm_max_tokens is None:
            max_tokens_value = os.environ.get("TP_FASTVLM_MAX_TOKENS", "120")
            try:
                working.fastvlm_max_tokens = int(max_tokens_value)
            except ValueError as exc:
                raise ExecutionPlanError("TP_FASTVLM_MAX_TOKENS must be an integer") from exc
        if type(working.fastvlm_max_tokens) is not int or working.fastvlm_max_tokens <= 0:
            raise ExecutionPlanError("fastvlm_max_tokens must be a positive integer")

        if working.fastvlm_temperature is None:
            temperature_value = os.environ.get("TP_FASTVLM_TEMPERATURE", "0.0")
            try:
                working.fastvlm_temperature = float(temperature_value)
            except ValueError as exc:
                raise ExecutionPlanError("TP_FASTVLM_TEMPERATURE must be a number") from exc
        if (
            isinstance(working.fastvlm_temperature, bool)
            or not isinstance(working.fastvlm_temperature, (int, float))
            or not math.isfinite(float(working.fastvlm_temperature))
            or float(working.fastvlm_temperature) < 0
        ):
            raise ExecutionPlanError("fastvlm_temperature must be a finite non-negative number")
        working.fastvlm_temperature = float(working.fastvlm_temperature)
    else:
        # Disabled optional work must not acquire hidden runtime authority from
        # process state, and its unused overrides do not belong in the plan.
        working.fastvlm_model_path = None
        working.fastvlm_review_model_path = None
        working.fastvlm_python_executable = None
        working.fastvlm_mlx_vlm_dir = None
        working.fastvlm_max_tokens = None
        working.fastvlm_temperature = None

    preview_value = os.environ.get("TP_ALLOW_RAW_PREVIEW", "0").strip().lower()
    working.raw_preview_escape_enabled = bool(working.raw_preview_escape_enabled) or preview_value in {
        "1",
        "true",
        "yes",
        "on",
    }
    risk_value = os.environ.get("TP_RECONSTRUCTION_RISK_THRESHOLD")
    if risk_value is not None:
        try:
            working.reconstruction_risk_threshold = float(risk_value)
        except ValueError as exc:
            raise ExecutionPlanError("TP_RECONSTRUCTION_RISK_THRESHOLD must be a number") from exc
    if not 0 <= working.reconstruction_risk_threshold <= 1:
        raise ExecutionPlanError("reconstruction_risk_threshold must be between 0 and 1")
    if working.enable_reconstruction:
        working.cameras_sidecar_sha256 = _file_sha256(working.cameras_sidecar_path)
    defaults = EnhanceConfig()
    if not working.enable_v2 or working.v2_preset is None:
        working.enable_v2 = False
        working.v2_preset = None
        working.v2_device = defaults.v2_device
        working.v2_upscaler_backend = defaults.v2_upscaler_backend
    if not working.generate_pbr:
        for field_name in (
            "pbr_normal_strength",
            "pbr_normal_blur_radius",
            "pbr_roughness_strength",
            "pbr_roughness_blur_radius",
            "pbr_ao_strength",
            "pbr_ao_blur_radius",
            "pbr_ao_bias",
        ):
            setattr(working, field_name, copy.deepcopy(getattr(defaults, field_name)))
    if not working.enable_materials_v3:
        for field_name in (
            "apply_pixel_ops",
            "refinement_strategy",
            "min_coverage_px",
            "min_mean_conf",
            "glass_response_enabled",
            "mask_feather_sigma_default",
            "mask_feather_sigma_overrides",
            "mask_feather_disabled_materials",
            "pixel_ops_low_grad_threshold",
            "pixel_ops_low_tex_min_bbox_frac",
            "pixel_ops_low_tex_feather_multiplier",
            "pixel_ops_low_tex_delta_ceiling",
            "sky_top_region_fraction",
            "sky_gradient_threshold",
            "sky_brightness_threshold",
            "enable_material_segmentation",
            "material_segmentation_backend",
            "strict_backend",
            "material_segmentation_cache_policy",
            "sam2_model_size",
            "sam2_checkpoint_path",
            "sam2_model_config",
            "sam2_expected_sha256",
            "sam2_tiling_enabled",
            "sam2_tile_size_px",
            "sam2_overlap_px",
            "sam2_global_pass_longest_side",
            "sam2_max_concurrency",
            "sam2_points_per_side",
            "sam2_points_per_batch",
            "sam2_pred_iou_thresh",
            "sam2_stability_score_thresh",
            "sam2_crop_n_layers",
            "sam_vit_h_checkpoint_path",
            "sam_vit_h_points_per_side",
            "sam_vit_h_pred_iou_thresh",
            "sam_vit_h_confidence_threshold",
            "sam_vit_h_expected_sha256",
        ):
            setattr(working, field_name, copy.deepcopy(getattr(defaults, field_name)))
    return working


def prepare_lux_execution(
    config: EnhanceConfig,
    input_root: Path,
    input_files: Sequence[Path],
) -> PreparedLuxExecution:
    """Resolve once and freeze a native execution-complete Lux plan."""

    root, resolved_inputs, relative_inputs = _resolve_input_selection(input_root, input_files)
    working = _prepare_working_config(config)

    explicit_backend = normalize_backend_id(working.depth_backend) is not None
    planned_backend = resolve_requested_backend(working.depth_backend, working)
    registry = DepthBackendRegistry()
    planned_backend = registry.validate_backend_request(planned_backend, working)
    candidate_chain = (
        (planned_backend,) if explicit_backend else tuple(resolve_runtime_backend_chain(planned_backend, working))
    )
    if not candidate_chain or candidate_chain[0] != planned_backend:
        raise ExecutionPlanError("Native candidate chain must begin with the planned backend")
    if len(candidate_chain) != len(set(candidate_chain)):
        raise ExecutionPlanError("Native candidate chain contains duplicate backends")
    if "depthcrafter" in candidate_chain:
        raise LuxExecutionPlanAuthorityError(
            "DepthCrafter cannot enter an execution-complete plan until it has a pinned executable identity"
        )
    for backend_id in candidate_chain:
        registry.validate_backend_request(backend_id, working)
    if "ensemble" in candidate_chain:
        # Constituents are independently authorized; the ensemble umbrella
        # acknowledgement alone cannot grant Depth Pro model authority.
        registry.validate_backend_request("depth_pro", working)

    da3_needed = "da3" in candidate_chain or "ensemble" in candidate_chain
    da3_model = _resolve_da3_model_once(working) if da3_needed else None
    if da3_model is None:
        # Model selectors do not affect a plan whose complete candidate set has
        # no DA3 execution path; discard them from the canonical projection so
        # an ignored mutable selector cannot perturb runtime identity.
        working.model_key = None
        working.raw_model_id = None
        working.model_variant = None

    # Canonical runtime semantics use the carried planned backend and exact
    # candidate list rather than retaining the mutable/ambient request form.
    working.depth_backend = planned_backend
    working.depth_operational_fallback_chain = candidate_chain
    working.allow_synthetic_fallback = "synthetic" in candidate_chain
    plan = _native_plan(
        working,
        input_root=root,
        relative_inputs=relative_inputs,
        planned_backend=planned_backend,
        candidate_chain=candidate_chain,
        da3_model=da3_model,
    )
    canonical_plan_bytes = plan.to_canonical_json().encode("utf-8")
    runtime_config = runtime_config_from_execution_plan(plan)
    bound_root, bound_inputs = _bind_plan_inputs(plan, root)
    prepared = PreparedLuxExecution(
        plan=plan,
        canonical_plan_bytes=canonical_plan_bytes,
        runtime_config=runtime_config,
        input_root=bound_root,
        input_files=bound_inputs,
    )
    return validate_prepared_lux_execution(prepared)


def consume_lux_execution_plan(
    data: str | bytes,
    *,
    authorized_input_root: Path,
) -> PreparedLuxExecution:
    """Consume canonical bytes under an externally authorized input root."""

    if isinstance(data, str):
        try:
            serialized = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ExecutionPlanError("Execution plan JSON contains invalid Unicode") from exc
    elif isinstance(data, bytes):
        serialized = data
    else:
        raise TypeError("Execution plan data must be str or bytes")
    structural_plan = parse_execution_plan_json(serialized)
    canonical = structural_plan.to_canonical_json().encode("utf-8")
    if serialized != canonical:
        raise ExecutionPlanError("Execution plan bytes are not the exact canonical serialization")
    plan = revalidate_lux_execution_plan_authority(structural_plan)
    root, inputs = _bind_plan_inputs(plan, authorized_input_root)
    runtime_config = runtime_config_from_execution_plan(plan)
    prepared = PreparedLuxExecution(
        plan=plan,
        canonical_plan_bytes=canonical,
        runtime_config=runtime_config,
        input_root=root,
        input_files=inputs,
    )
    return validate_prepared_lux_execution(prepared)


def consume_lux_worker_execution_plan(data: str | bytes) -> CanonicalExecutionPlan:
    """Revalidate canonical authority already root-bound by a parent process.

    This entrypoint is only for an isolated worker whose parent passed the
    exact prepared bytes. It deliberately omits filesystem authorization;
    workers that receive an input root must call ``consume_lux_execution_plan``.
    """

    if isinstance(data, str):
        try:
            serialized = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ExecutionPlanError("Execution plan JSON contains invalid Unicode") from exc
    elif isinstance(data, bytes):
        serialized = data
    else:
        raise TypeError("Execution plan data must be str or bytes")
    structural_plan = parse_execution_plan_json(serialized)
    canonical = structural_plan.to_canonical_json().encode("utf-8")
    if serialized != canonical:
        raise ExecutionPlanError("Execution plan bytes are not the exact canonical serialization")
    return revalidate_lux_execution_plan_authority(structural_plan)


def _node_configuration(
    plan: CanonicalExecutionPlan,
    registry_id: StageRegistryIdentifier,
    *,
    required: bool = True,
) -> Optional[dict[str, Any]]:
    matches = [node for node in plan.nodes if node.stage_registry_id is registry_id]
    if not matches:
        if required:
            raise LuxExecutionPlanAuthorityError(f"Execution plan is missing required node {registry_id.value!r}")
        return None
    if len(matches) != 1:
        raise LuxExecutionPlanAuthorityError(f"Execution plan has duplicate node {registry_id.value!r}")
    return matches[0].to_payload()["configuration"]


def _runtime_da3_contract(
    model: Optional[ResolvedModelIntent],
    *,
    non_commercial_ok: bool,
) -> Optional[ResolvedModel]:
    if model is None:
        return None
    try:
        spec = get_model_spec(model.canonical_key)
        accelerator = AcceleratorKind(model.accelerator_kind)
    except (KeyError, TypeError, ValueError) as exc:
        raise LuxExecutionPlanAuthorityError("Carried DA3 model identity is not present in current authority") from exc
    candidate = ResolvedModel(
        requested_selector=model.requested_selector,
        resolution_reason=model.resolution_reason,
        canonical_key=model.canonical_key,
        spec=spec,
        revision=model.revision,
        fallback_chain=(),
        accelerator_kind=accelerator,
        legacy_model_variant_name=model.legacy_model_variant_name,
    )
    return validate_authoritative_model_contract(candidate, non_commercial_ok=non_commercial_ok)


def _model_selector_projection(
    model: Optional[ResolvedModelIntent],
) -> tuple[Optional[str], Optional[str], Optional[ModelVariant]]:
    if model is None or model.requested_selector == "default" or model.requested_selector.startswith("preset:"):
        return None, None, None
    if model.legacy_model_variant_name is not None:
        for variant in ModelVariant:
            if variant.value.name == model.legacy_model_variant_name:
                return None, None, variant
        raise LuxExecutionPlanAuthorityError(
            f"Carried legacy model variant {model.legacy_model_variant_name!r} is no longer supported"
        )
    if model.requested_selector == model.repo_id:
        return None, model.requested_selector, None
    return model.requested_selector, None, None


def _preset_projection(plan: CanonicalExecutionPlan) -> Optional[Preset]:
    if plan.preset_resolved is None:
        return None
    try:
        return Preset(plan.preset_resolved)
    except ValueError:
        return None


def runtime_config_from_execution_plan(
    plan: CanonicalExecutionPlan,
    *,
    candidate_authority: Optional[BackendCandidateAuthority] = None,
) -> EnhanceConfig:
    """Project a new mutable runtime config solely from typed plan nodes."""

    source_plan = plan
    plan = revalidate_lux_execution_plan_authority(plan)
    if candidate_authority is not None:
        if candidate_authority.plan_fingerprint_sha256 != plan.plan_fingerprint_sha256:
            raise LuxExecutionPlanAuthorityError("Candidate authority belongs to a different execution plan")
        expected_authority = backend_candidate_authority(
            plan,
            candidate_authority.candidate_id,
            model_backend_id=candidate_authority.constituent_backend_id,
        )
        if expected_authority != candidate_authority:
            raise LuxExecutionPlanAuthorityError("Candidate authority does not match the carried execution plan")

    preprocess = _node_configuration(plan, StageRegistryIdentifier.LUX_PREPROCESS)
    depth = _node_configuration(plan, StageRegistryIdentifier.LUX_DEPTH)
    output = _node_configuration(plan, StageRegistryIdentifier.LUX_OUTPUT)
    assert preprocess is not None and depth is not None and output is not None
    materials = _node_configuration(plan, StageRegistryIdentifier.LUX_MATERIALS_V3, required=False)
    pbr = _node_configuration(plan, StageRegistryIdentifier.LUX_PBR, required=False)
    v2 = _node_configuration(plan, StageRegistryIdentifier.LUX_V2, required=False)
    reconstruction = _node_configuration(plan, StageRegistryIdentifier.LUX_RECONSTRUCTION, required=False)

    acknowledgements = plan.license_acknowledgements
    runtime_model = _runtime_da3_contract(
        plan.resolved_model,
        non_commercial_ok=acknowledgements.non_commercial_ok,
    )
    model_key, raw_model_id, model_variant = _model_selector_projection(plan.resolved_model)
    if candidate_authority is not None:
        runtime_model = candidate_authority.resolved_model_contract
        if candidate_authority.backend_id != "da3":
            model_key = None
            raw_model_id = None
            model_variant = None
        depth_backend = candidate_authority.backend_id
        depth_device = candidate_authority.device
    else:
        depth_backend = plan.planned_backend
        depth_device = depth["device"]

    postprocessing = PostprocessingConfig(**depth["postprocessing"])
    kwargs: dict[str, Any] = {
        "model_variant": model_variant,
        "model_key": model_key,
        "raw_model_id": raw_model_id,
        "preset": _preset_projection(plan),
        "preset_requested": plan.preset_requested,
        "depth_device": depth_device,
        "depth_quantization": depth["quantization"],
        "depth_postprocessing": postprocessing,
        "force_depth": depth["force"],
        "non_commercial_ok": acknowledgements.non_commercial_ok,
        "verify_depth_writes": depth["verify_writes"],
        "strict_inputs": preprocess["verify_images"],
        "verify_images": preprocess["verify_images"],
        "keep_intermediates": output["keep_intermediates"],
        "accept_apple_depth_pro_research_license": acknowledgements.apple_depth_pro_research,
        "accept_research_tools_license": acknowledgements.research_tools,
        "raw_ingest_mode": preprocess["raw_ingest_mode"],
        "raw_wb_mode": preprocess["raw_wb_mode"],
        "raw_demosaic": preprocess["raw_demosaic"],
        "raw_preview_escape_enabled": preprocess["raw_preview_escape_enabled"],
        "depth_backend": depth_backend,
        "depth_pro_checkpoint_path": depth["depth_pro_checkpoint_path"],
        "depth_pro_python_executable": depth["depth_pro_python_executable"],
        "raw_python_executable": preprocess["raw_python_executable"],
        "da3_python_executable": depth["da3_python_executable"],
        "da3_subprocess_timeout_seconds": depth["da3_subprocess_timeout_seconds"],
        "depth_fallback": depth["fallback_mode"],
        "v2_timeout": v2["timeout_seconds"] if v2 is not None else 300,
        "allow_synthetic_fallback": depth["allow_synthetic_fallback"],
        "allow_semantic_fallback": depth["allow_semantic_fallback"],
        "depth_operational_fallback_chain": tuple(plan.candidate_fallback_chain),
        "hash_mode": HashMode(depth["hash_mode"]),
        "enable_manifest_cache": depth["manifest_cache_enabled"],
        "enable_parallel_processing": preprocess["parallel_enabled"],
        "max_parallel_workers": preprocess["max_workers"],
        "max_workers": preprocess["max_workers"],
        "max_gpu_workers": preprocess["max_gpu_workers"],
        "enable_depth_cache": depth["depth_cache_enabled"],
        "depth_cache_max_size_gb": depth["depth_cache_max_size_gb"],
        "use_coreml_backend": bool(plan.resolved_model and plan.resolved_model.accelerator_kind == "coreml"),
        "use_xxhash": preprocess["output_key_hash_algorithm"] == "xxhash",
        "save_float_depth": depth["save_float_depth"],
        "generate_pbr": pbr is not None,
        "quality_tier": plan.quality_tier,
        "output_bit_depth": output["output_bit_depth"],
        "emit_run_card": output["run_card_enabled"],
        "run_card_version": output["run_card_version"],
        "run_card_include_proofs": output["run_card_include_proofs"],
        "vlm_captioning_enabled": output["captioning"]["enabled"],
        "vlm_captioning_backend": output["captioning"]["backend"],
        "vlm_captioning_model": output["captioning"]["selector"],
        "vlm_captioning_proxy_format": output["captioning"]["proxy_format"],
        "vlm_captioning_max_side_px": output["captioning"]["max_side_px"],
        "fastvlm_model_path": output["captioning"].get("model_path"),
        "fastvlm_review_model_path": output["captioning"].get("review_model_path"),
        "fastvlm_python_executable": output["captioning"]["python_executable"],
        "fastvlm_mlx_vlm_dir": output["captioning"]["mlx_vlm_dir"],
        "fastvlm_max_tokens": output["captioning"].get("max_tokens", 120),
        "fastvlm_temperature": output["captioning"].get("temperature", 0.0),
        "fastvlm_timeout_seconds": output["captioning"]["timeout_seconds"],
        "enable_v2": v2 is not None,
        "v2_preset": v2["preset"] if v2 is not None else None,
        "v2_device": v2["device"] if v2 is not None else "cpu",
        "v2_upscaler_backend": v2["upscaler_backend"] if v2 is not None else "default",
        "force_v2": v2["force"] if v2 is not None else False,
        "enable_materials_v3": materials is not None,
        "enable_reconstruction": reconstruction is not None,
        "resolved_invocation": _ExecutionPlanModelCarrier(runtime_model),
        # Retain the caller's exact immutable object while using the freshly
        # revalidated value above for projection.  The paired canonical bytes
        # are the closed carrier forwarded to backend and worker boundaries.
        "execution_plan_authority": source_plan,
        "execution_plan_canonical_bytes": source_plan.to_canonical_json().encode("utf-8"),
    }
    apex = depth["apex_gate"]
    kwargs.update(
        {
            "apex_depth_min_finite_pct": apex["min_finite_pct"],
            "apex_depth_min_upper_iqr": apex["min_upper_iqr"],
            "apex_depth_max_high_saturation_fraction": apex["max_high_saturation_fraction"],
            "apex_depth_max_low_saturation_fraction": apex["max_low_saturation_fraction"],
            "apex_depth_scaled_saturation_margin": apex["scaled_saturation_margin"],
            "apex_depth_low_saturation_warning_band": apex["low_saturation_warning_band"],
            "apex_depth_saturation_high_value": apex["saturation_high_value"],
            "apex_depth_saturation_low_value": apex["saturation_low_value"],
            "apex_depth_min_gradient_energy": apex["min_gradient_energy"],
            "apex_depth_threshold_epsilon": apex["threshold_epsilon"],
            "apex_depth_hist_bins": apex["hist_bins"],
        }
    )
    ensemble = depth["ensemble"]
    if ensemble is not None:
        kwargs.update(
            {
                "ensemble_fusion_method": ensemble["fusion_method"],
                "ensemble_max_variance_threshold": ensemble["max_variance_threshold"],
                "ensemble_temporal_filter_mode": ensemble["temporal_post_filter"]["mode"],
                "ensemble_temporal_filter_alpha": ensemble["temporal_post_filter"]["alpha"],
            }
        )
    if materials is not None:
        kwargs.update(
            {
                "apply_pixel_ops": materials["apply_pixel_ops"],
                "enable_material_segmentation": materials["enable_material_segmentation"],
                "material_segmentation_backend": materials["material_segmentation_backend"],
                "strict_backend": materials["strict_backend"],
                "material_segmentation_cache_policy": materials["segmentation_cache_policy"],
                "refinement_strategy": materials["refinement_strategy"],
                "min_coverage_px": materials["min_coverage_px"],
                "min_mean_conf": materials["min_mean_conf"],
                "glass_response_enabled": materials["glass_response_enabled"],
                "mask_feather_sigma_default": materials["mask_feather_sigma_default"],
                "mask_feather_sigma_overrides": {
                    item["material"]: item["sigma"] for item in materials["mask_feather_sigma_overrides"]
                },
                "mask_feather_disabled_materials": list(materials["mask_feather_disabled_materials"]),
                "pixel_ops_low_grad_threshold": materials["pixel_ops_low_grad_threshold"],
                "pixel_ops_low_tex_min_bbox_frac": materials["pixel_ops_low_tex_min_bbox_frac"],
                "pixel_ops_low_tex_feather_multiplier": materials["pixel_ops_low_tex_feather_multiplier"],
                "pixel_ops_low_tex_delta_ceiling": materials["pixel_ops_low_tex_delta_ceiling"],
                "sky_top_region_fraction": materials["sky_top_region_fraction"],
                "sky_gradient_threshold": materials["sky_gradient_threshold"],
                "sky_brightness_threshold": materials["sky_brightness_threshold"],
                "sam2_model_size": materials["sam2_model_size"],
                "sam2_checkpoint_path": materials["sam2_checkpoint_path"],
                "sam2_model_config": materials["sam2_model_config"],
                "sam2_expected_sha256": materials["sam2_expected_sha256"],
                "sam2_tiling_enabled": materials["sam2_tiling_enabled"],
                "sam2_tile_size_px": materials["sam2_tile_size_px"],
                "sam2_overlap_px": materials["sam2_overlap_px"],
                "sam2_global_pass_longest_side": materials["sam2_global_pass_longest_side"],
                "sam2_max_concurrency": materials["sam2_max_concurrency"],
                "sam2_points_per_side": materials["sam2_points_per_side"],
                "sam2_points_per_batch": materials["sam2_points_per_batch"],
                "sam2_pred_iou_thresh": materials["sam2_pred_iou_thresh"],
                "sam2_stability_score_thresh": materials["sam2_stability_score_thresh"],
                "sam2_crop_n_layers": materials["sam2_crop_n_layers"],
                "sam_vit_h_checkpoint_path": materials["sam_vit_h_checkpoint_path"],
                "sam_vit_h_points_per_side": materials["sam_vit_h_points_per_side"],
                "sam_vit_h_pred_iou_thresh": materials["sam_vit_h_pred_iou_thresh"],
                "sam_vit_h_confidence_threshold": materials["sam_vit_h_confidence_threshold"],
                "sam_vit_h_expected_sha256": materials["sam_vit_h_expected_sha256"],
            }
        )
    if pbr is not None:
        kwargs.update(
            {
                "pbr_normal_strength": pbr["normal_strength"],
                "pbr_normal_blur_radius": pbr["normal_blur_radius"],
                "pbr_roughness_strength": pbr["roughness_strength"],
                "pbr_roughness_blur_radius": pbr["roughness_blur_radius"],
                "pbr_ao_strength": pbr["ao_strength"],
                "pbr_ao_blur_radius": pbr["ao_blur_radius"],
                "pbr_ao_bias": pbr["ao_bias"],
            }
        )
    if reconstruction is not None:
        kwargs.update(
            {
                "grouping_mode": reconstruction["grouping_mode"],
                "cameras_sidecar_path": reconstruction["cameras_sidecar_path"],
                "cameras_sidecar_sha256": reconstruction["cameras_sidecar_sha256"],
                "reconstruction_iterations": reconstruction["iterations"],
                "reconstruction_tier": reconstruction["tier"],
                "emit_scene_debug_bundle": reconstruction["emit_scene_debug_bundle"],
                "reconstruction_risk_threshold": reconstruction["risk_threshold"],
            }
        )

    runtime = EnhanceConfig(**kwargs)
    # Preserve an explicit fail-closed APEX policy after EnhanceConfig's legacy
    # constructor normalization, which otherwise cannot distinguish the old
    # `apex-strict` request from a default `fail` input.
    runtime.depth_fallback = depth["fallback_mode"]
    if candidate_authority is None:
        projected_fingerprint = compute_config_fingerprint(
            runtime,
            resolved_model_contract=runtime_model,
        ).to_sha256()
        if projected_fingerprint != plan.config_fingerprint_sha256:
            raise LuxExecutionPlanAuthorityError(
                "Typed runtime configuration projection does not match the carried config fingerprint"
            )
    return runtime


def backend_candidate_authority(
    plan: CanonicalExecutionPlan,
    candidate_id: str,
    *,
    model_backend_id: Optional[str] = None,
) -> BackendCandidateAuthority:
    """Select one exact carried candidate/constituent without resolution."""

    plan = revalidate_lux_execution_plan_authority(plan)
    matches = [candidate for candidate in plan.backend_candidates if candidate.backend_id == candidate_id]
    if len(matches) != 1:
        raise LuxExecutionPlanAuthorityError(f"Candidate id {candidate_id!r} is absent or ambiguous")
    candidate = matches[0]
    enabled_contracts = tuple(contract for contract in candidate.model_contracts if contract.enabled)
    selected_contract: Optional[BackendModelIntent]
    if model_backend_id is not None:
        selected = [contract for contract in enabled_contracts if contract.backend_id == model_backend_id]
        if len(selected) != 1:
            raise LuxExecutionPlanAuthorityError(
                f"Candidate {candidate_id!r} has no unique enabled model {model_backend_id!r}"
            )
        selected_contract = selected[0]
    elif len(enabled_contracts) == 0:
        selected_contract = None
    elif len(enabled_contracts) == 1:
        selected_contract = enabled_contracts[0]
    elif candidate.backend_id == "ensemble":
        selected_contract = None
    else:
        raise LuxExecutionPlanAuthorityError(
            f"Candidate {candidate_id!r} has multiple enabled models; model_backend_id is required"
        )

    depth = _node_configuration(plan, StageRegistryIdentifier.LUX_DEPTH)
    assert depth is not None
    runtime_model = None
    device = str(depth["device"])
    weight = None
    constituent_backend_id = None
    if selected_contract is not None:
        constituent_backend_id = selected_contract.backend_id if candidate.backend_id == "ensemble" else None
        device = selected_contract.device
        weight = selected_contract.weight
        if selected_contract.backend_id == "da3":
            runtime_model = _runtime_da3_contract(
                selected_contract.model,
                non_commercial_ok=plan.license_acknowledgements.non_commercial_ok,
            )
    executed_backend_id = selected_contract.backend_id if selected_contract is not None else candidate.backend_id
    return BackendCandidateAuthority(
        plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
        candidate_id=candidate.backend_id,
        backend_id=executed_backend_id,
        constituent_backend_id=constituent_backend_id,
        candidate=candidate,
        model_contract=selected_contract,
        resolved_model_contract=runtime_model,
        device=device,
        weight=weight,
    )


def runtime_model_contract_from_candidate(authority: BackendCandidateAuthority) -> ResolvedModel:
    """Return the exact re-anchored DA3 contract carried by an authority."""

    if authority.resolved_model_contract is None:
        raise LuxExecutionPlanAuthorityError(
            f"Candidate authority {authority.candidate_id!r}/{authority.backend_id!r} does not carry a DA3 model"
        )
    return authority.resolved_model_contract


def validate_prepared_lux_execution(
    prepared: PreparedLuxExecution,
) -> PreparedLuxExecution:
    """Revalidate every coupled invariant of a prepared execution carrier."""

    if not isinstance(prepared, PreparedLuxExecution):
        raise TypeError("prepared must be a PreparedLuxExecution")
    if type(prepared.canonical_plan_bytes) is not bytes:
        raise LuxExecutionPlanAuthorityError("Prepared execution canonical bytes must be immutable bytes")

    canonical = prepared.plan.to_canonical_json().encode("utf-8")
    if prepared.canonical_plan_bytes != canonical:
        raise LuxExecutionPlanAuthorityError("Prepared execution does not carry the plan's exact canonical bytes")
    validated_plan = revalidate_lux_execution_plan_authority(prepared.plan)
    if validated_plan != prepared.plan:
        raise LuxExecutionPlanAuthorityError("Prepared execution plan changed during authority revalidation")

    runtime = prepared.runtime_config
    if runtime.execution_plan_authority is not prepared.plan:
        raise LuxExecutionPlanAuthorityError("Prepared runtime config does not retain the exact plan object")
    if runtime.execution_plan_canonical_bytes != canonical:
        raise LuxExecutionPlanAuthorityError("Prepared runtime config does not retain the exact canonical bytes")
    expected_runtime = runtime_config_from_execution_plan(prepared.plan)
    if runtime != expected_runtime:
        raise LuxExecutionPlanAuthorityError("Prepared runtime config is not the exact typed plan projection")

    bound_root, bound_inputs = _bind_plan_inputs(prepared.plan, prepared.input_root)
    if prepared.input_root != bound_root or prepared.input_files != bound_inputs:
        raise LuxExecutionPlanAuthorityError("Prepared inputs do not exactly match the plan-derived filesystem binding")
    return prepared


def authorize_prepared_input(
    prepared: PreparedLuxExecution,
    input_path: Path,
) -> Path:
    """Recheck that an accessed file is one exact frozen prepared input."""

    validate_prepared_lux_execution(prepared)
    root, current_inputs = _bind_plan_inputs(prepared.plan, prepared.input_root)
    lexical = Path(input_path).expanduser()
    candidate = lexical if lexical.is_absolute() else root / lexical
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LuxExecutionPlanAuthorityError(f"Input cannot be resolved at access time: {input_path}") from exc
    if resolved not in current_inputs:
        raise LuxExecutionPlanAuthorityError(f"Input is not present in the prepared execution: {input_path}")
    return resolved


__all__ = [
    "BackendCandidateAuthority",
    "PreparedLuxExecution",
    "authorize_prepared_input",
    "backend_candidate_authority",
    "consume_lux_execution_plan",
    "consume_lux_worker_execution_plan",
    "prepare_lux_execution",
    "runtime_config_from_execution_plan",
    "runtime_model_contract_from_candidate",
    "validate_prepared_lux_execution",
]
