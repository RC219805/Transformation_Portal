"""Integrity verification helpers for Lux run cards."""

from __future__ import annotations

import hashlib
import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.artifact_tree import verify_artifact_tree_payload
from transformation_portal.lux_depth_v3.run_card_contract import (
    RunCardPathValidationError,
    infer_run_card_version,
    normalize_run_card_relative_path,
    with_inferred_run_card_version,
)
from transformation_portal.schemas.run_card import load_run_card_schema

from .jsonschema_formats import build_jsonschema_format_checker
from .run_card_validator import _default_schema_path

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - runtime guard for environments missing optional deps
    Draft202012Validator = None  # type: ignore[assignment]
    FormatChecker = None  # type: ignore[assignment]


SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
DEFAULT_SCHEMA_V1_PATH = _default_schema_path("v1")
DEFAULT_SCHEMA_V2_PATH = _default_schema_path("v2")


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, f"JSON file not found: {path}"
    except PermissionError:
        return None, f"Permission denied reading JSON file: {path}"
    except JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    except OSError as exc:
        return None, f"Failed to read JSON file {path}: {exc}"


def canonical_json_text(payload: Any) -> str:
    return dumps_json(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def format_error_path(error_path: Any) -> str:
    parts = [str(item) for item in error_path]
    return ".".join(parts) if parts else "<root>"


def resolve_artifact_path(
    *,
    run_card_root: Path,
    relative_path: str,
    context: str,
) -> tuple[Path | None, str | None]:
    """Resolve an artifact path while keeping reads confined to the run-card root."""
    try:
        normalized_relative_path = normalize_run_card_relative_path(relative_path)
    except RunCardPathValidationError as exc:
        return None, f"{context} {exc}"

    root_resolved = run_card_root.resolve()
    artifact_path = (root_resolved / Path(*Path(normalized_relative_path).parts)).resolve()
    try:
        artifact_path.relative_to(root_resolved)
    except ValueError:
        return None, f"{context} relative_path escapes run card root: {normalized_relative_path}"
    return artifact_path, None


def _verify_backend_semantics(run_card_payload: dict[str, Any], errors: list[str]) -> None:
    backend_selection = run_card_payload.get("backend_selection")
    backend_summary = run_card_payload.get("backend_summary")
    if not isinstance(backend_selection, dict) or not isinstance(backend_summary, dict):
        return

    final_backends_used = backend_summary.get("final_backends_used")
    if not isinstance(final_backends_used, list):
        errors.append("backend_summary.final_backends_used must be an array")
        return

    success_count = run_card_payload.get("success_count")
    if not isinstance(success_count, int):
        success_count = 0

    if not final_backends_used:
        if success_count > 0:
            errors.append("backend_summary.final_backends_used must be non-empty when success_count > 0")
        return

    primary_backend = final_backends_used[0]
    if not isinstance(primary_backend, str) or not primary_backend:
        errors.append("backend_summary.final_backends_used[0] must be a non-empty string")
        return

    summary_primary = backend_summary.get("primary_backend")
    if summary_primary != primary_backend:
        errors.append("backend_summary.primary_backend must equal backend_summary.final_backends_used[0]")

    resolved = backend_selection.get("resolved")
    if not isinstance(resolved, str) or not resolved:
        errors.append("backend_selection.resolved must be a non-empty string")
        return
    if resolved != primary_backend:
        errors.append("backend_selection.resolved must match backend_summary.final_backends_used[0]")

    logical_backend = backend_selection.get("logical_backend")
    resolved_engine = backend_selection.get("resolved_engine")
    wrapper_declared = logical_backend is not None or resolved_engine is not None
    if not wrapper_declared:
        return

    if not isinstance(logical_backend, str) or not logical_backend:
        errors.append("backend_selection.logical_backend must be a non-empty string when wrapper semantics are declared")
    if not isinstance(resolved_engine, str) or not resolved_engine:
        errors.append("backend_selection.resolved_engine must be a non-empty string when wrapper semantics are declared")
    if isinstance(logical_backend, str) and isinstance(resolved_engine, str):
        if logical_backend == resolved_engine:
            errors.append("backend_selection.logical_backend and backend_selection.resolved_engine must differ")
        if resolved_engine != primary_backend:
            errors.append("backend_selection.resolved_engine must match backend_summary.final_backends_used[0]")

    fallback_images = backend_summary.get("fallback_images")
    if isinstance(fallback_images, int) and fallback_images != 0:
        errors.append("wrapper semantics are only valid when backend_summary.fallback_images == 0")


def _first_combined_manifest_fallback_reason(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    errors: list[str] | None = None,
) -> str | None:
    manifest_artifacts = [
        artifact
        for artifact in run_card_payload.get("artifact_index", [])
        if isinstance(artifact, dict) and artifact.get("artifact_type") == "combined_manifest"
    ]
    for artifact in manifest_artifacts:
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        manifest_path, path_error = resolve_artifact_path(
            run_card_root=run_card_root,
            relative_path=relative_path,
            context="combined_manifest artifact",
        )
        if path_error:
            if errors is not None:
                errors.append(path_error)
            continue
        assert manifest_path is not None
        payload, load_error = _load_json(manifest_path)
        if load_error or not isinstance(payload, dict):
            continue
        backend_selection = payload.get("backend_selection")
        if not isinstance(backend_selection, dict):
            continue
        resolution_reason = backend_selection.get("resolution_reason")
        if isinstance(resolution_reason, str) and resolution_reason.strip():
            return f"{relative_path}: {resolution_reason.strip()}"
        attempts = backend_selection.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "failed":
                continue
            error_message = attempt.get("error_message") or attempt.get("error_code")
            if isinstance(error_message, str) and error_message.strip():
                return f"{relative_path}: {error_message.strip()}"
    return None


def _verify_requested_depth_pro_fulfillment(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    errors: list[str],
) -> None:
    backend_selection = run_card_payload.get("backend_selection")
    backend_summary = run_card_payload.get("backend_summary")
    if not isinstance(backend_selection, dict) or not isinstance(backend_summary, dict):
        return

    requested_backend = backend_selection.get("requested") or backend_summary.get("requested_backend")
    if requested_backend != "depth_pro":
        return

    success_count = run_card_payload.get("success_count")
    fallback_images = backend_summary.get("fallback_images")
    primary_backend = backend_summary.get("primary_backend")
    total_images = run_card_payload.get("total_images")
    error_count = run_card_payload.get("error_count")
    has_error_failures = isinstance(error_count, int) and error_count > 0
    has_incomplete_successes = (
        isinstance(success_count, int) and isinstance(total_images, int) and success_count < total_images
    )
    run_failed = has_error_failures or has_incomplete_successes
    if (
        not isinstance(success_count, int)
        or success_count <= 0
        or not isinstance(fallback_images, int)
        or fallback_images != success_count
        or not isinstance(primary_backend, str)
        or primary_backend == requested_backend
        or run_failed
    ):
        return

    error = (
        "requested backend 'depth_pro' was not honored: "
        f"all successful images ({success_count}/{success_count}) used fallback backend '{primary_backend}'"
    )
    fallback_reason = _first_combined_manifest_fallback_reason(
        run_card_payload,
        run_card_root=run_card_root,
        errors=errors,
    )
    if fallback_reason:
        error = f"{error}. First fallback reason: {fallback_reason}"
    errors.append(error)


def _verify_config_fingerprint(run_card_payload: dict[str, Any], errors: list[str]) -> None:
    config_fingerprint = run_card_payload.get("config_fingerprint")
    if not isinstance(config_fingerprint, dict):
        return

    canonical_json = config_fingerprint.get("canonical_json")
    hash_algorithm = config_fingerprint.get("hash_algorithm")
    sha256_hex = config_fingerprint.get("sha256")
    if not isinstance(canonical_json, str) or not canonical_json:
        errors.append("config_fingerprint.canonical_json must be a non-empty string")
        return
    if hash_algorithm != "sha256":
        errors.append("config_fingerprint.hash_algorithm must be 'sha256'")
        return
    if not isinstance(sha256_hex, str) or not SHA256_HEX_RE.fullmatch(sha256_hex):
        errors.append("config_fingerprint.sha256 must be a lowercase 64-char hex digest")
        return

    fields = (
        "model_variant",
        "depth_quantization",
        "depth_device",
        "preset",
        "v2_preset",
        "v2_device",
        "v2_upscaler_backend",
        "preset_requested",
        "preset_resolved",
        "backend_requested",
        "backend_resolved",
        "device_requested",
        "device_resolved",
        "quality_tier",
        "strict_inputs",
        "strict_segmentation",
        "apex_strict_mode",
    )
    optional_fields = (
        "raw_ingest_profile",
        "raw_ingest_settings_hash",
        "depth_pro_python_executable",
        "raw_python_executable",
        "da3_python_executable",
    )
    present_optional_fields = tuple(field for field in optional_fields if field in config_fingerprint)
    fingerprint_fields = (*fields, *present_optional_fields)
    expected_canonical_json = dumps_json(
        {field: config_fingerprint.get(field) for field in fingerprint_fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if canonical_json != expected_canonical_json:
        errors.append("config_fingerprint.canonical_json does not match canonicalized config fingerprint fields")

    recomputed_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if recomputed_sha != sha256_hex:
        errors.append("config_fingerprint.sha256 mismatch: " f"got={sha256_hex}, expected={recomputed_sha}")


def _verify_reconstruction_scene_manifests(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    artifact_index_by_relative_path: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    manifest_artifacts = [
        artifact
        for artifact in run_card_payload.get("artifact_index", [])
        if isinstance(artifact, dict) and artifact.get("artifact_type") == "reconstruction_scene_manifest"
    ]
    if not manifest_artifacts:
        return

    try:
        from transformation_portal.lux_depth_v3.scene_integrity import verify_scene_integrity
    except Exception as exc:  # pragma: no cover - import guard for script-only environments
        errors.append(f"Unable to import reconstruction scene integrity helpers: {exc}")
        return

    for artifact in manifest_artifacts:
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append("reconstruction_scene_manifest artifact is missing relative_path")
            continue
        manifest_path, path_error = resolve_artifact_path(
            run_card_root=run_card_root,
            relative_path=relative_path,
            context="reconstruction_scene_manifest artifact",
        )
        if path_error:
            errors.append(path_error)
            continue
        assert manifest_path is not None
        payload, load_error = _load_json(manifest_path)
        if load_error:
            errors.append(load_error)
            continue
        if not isinstance(payload, dict):
            errors.append(f"Reconstruction scene manifest root must be a JSON object: {manifest_path}")
            continue
        try:
            verify_scene_integrity(
                payload,
                artifact_index=artifact_index_by_relative_path,
                base_dir=run_card_root,
            )
        except Exception as exc:
            errors.append(f"Reconstruction scene manifest validation failed ({relative_path}): {exc}")


def _verify_reconstruction_diagnostics(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    errors: list[str],
) -> None:
    diagnostics_artifacts = [
        artifact
        for artifact in run_card_payload.get("artifact_index", [])
        if isinstance(artifact, dict) and artifact.get("artifact_type") == "reconstruction_diagnostics"
    ]
    if not diagnostics_artifacts:
        return

    for artifact in diagnostics_artifacts:
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append("reconstruction_diagnostics artifact is missing relative_path")
            continue
        diagnostics_path, path_error = resolve_artifact_path(
            run_card_root=run_card_root,
            relative_path=relative_path,
            context="reconstruction_diagnostics artifact",
        )
        if path_error:
            errors.append(path_error)
            continue
        assert diagnostics_path is not None
        payload, load_error = _load_json(diagnostics_path)
        if load_error:
            errors.append(load_error)
            continue
        if not isinstance(payload, dict):
            errors.append(f"Reconstruction diagnostics root must be a JSON object: {diagnostics_path}")
            continue
        if payload.get("schema") != "tp.reconstruction_diagnostics.v1":
            errors.append(f"Reconstruction diagnostics schema mismatch ({relative_path})")
            continue
        cameras = payload.get("cameras")
        if not isinstance(cameras, list):
            errors.append(f"Reconstruction diagnostics cameras must be a list ({relative_path})")
            continue
        if payload.get("camera_count") != len(cameras):
            errors.append(f"Reconstruction diagnostics camera_count mismatch ({relative_path})")
        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                errors.append(f"Reconstruction diagnostics camera entry must be object ({relative_path} #{index})")
                continue
            for field in ("reprojection_p50", "reprojection_p95", "reprojection_p99"):
                if field not in camera:
                    errors.append(f"Reconstruction diagnostics missing {field} ({relative_path} #{index})")


def infer_schema_path_for_payload(payload: dict[str, Any], explicit_schema_path: Path | None = None) -> Path:
    if explicit_schema_path is not None:
        return explicit_schema_path
    return DEFAULT_SCHEMA_V2_PATH if infer_run_card_version(payload) == "v2" else DEFAULT_SCHEMA_V1_PATH


def verify_run_card_integrity(
    run_card_path: Path,
    *,
    schema_path: Path | None = None,
    check_canonical_json: bool = False,
) -> list[str]:
    """Return list of integrity errors for a run card."""
    if not run_card_path.exists():
        return [f"Run card not found: {run_card_path}"]
    if Draft202012Validator is None or FormatChecker is None:
        return ["jsonschema dependency is required (install jsonschema>=4.21.0,<5)"]

    raw_run_card_payload, run_card_load_error = _load_json(run_card_path)
    if run_card_load_error:
        return [run_card_load_error]
    if not isinstance(raw_run_card_payload, dict):
        return [f"Run card root must be a JSON object: {run_card_path}"]
    run_card_payload = with_inferred_run_card_version(raw_run_card_payload)

    effective_schema_path = infer_schema_path_for_payload(run_card_payload, explicit_schema_path=schema_path)
    if schema_path is not None:
        if not effective_schema_path.exists():
            return [f"Run card schema not found: {effective_schema_path}"]
        schema_payload, schema_load_error = _load_json(effective_schema_path)
        if schema_load_error:
            return [schema_load_error]
    else:
        schema_payload = load_run_card_schema(infer_run_card_version(run_card_payload))

    errors: list[str] = []
    validator = Draft202012Validator(schema_payload, format_checker=build_jsonschema_format_checker())
    schema_errors = sorted(validator.iter_errors(run_card_payload), key=lambda item: list(item.path))
    for item in schema_errors:
        errors.append(f"Schema validation failed at {format_error_path(item.path)}: {item.message}")

    artifact_index = run_card_payload.get("artifact_index")
    if not isinstance(artifact_index, list):
        errors.append("artifact_index must be an array")
        return errors

    relative_paths: list[str] = []
    for index, artifact in enumerate(artifact_index):
        if not isinstance(artifact, dict):
            errors.append(f"artifact_index[{index}] must be an object")
            continue

        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"artifact_index[{index}].relative_path must be a non-empty string")
        else:
            try:
                normalized_relative_path = normalize_run_card_relative_path(relative_path)
            except RunCardPathValidationError as exc:
                errors.append(f"artifact_index[{index}].relative_path {exc}")
            else:
                if normalized_relative_path != relative_path:
                    errors.append("artifact_index" f"[{index}].relative_path must already be normalized POSIX-style")
                relative_paths.append(normalized_relative_path)

        sha256_hex = artifact.get("sha256")
        if not isinstance(sha256_hex, str) or not SHA256_HEX_RE.fullmatch(sha256_hex):
            errors.append(f"artifact_index[{index}].sha256 must be a lowercase 64-char hex digest")

    if relative_paths:
        sorted_paths = sorted(relative_paths)
        if relative_paths != sorted_paths:
            errors.append("artifact_index ordering is non-deterministic (relative_path must be lexicographically sorted)")
        if len(set(relative_paths)) != len(relative_paths):
            errors.append("artifact_index contains duplicate relative_path entries")

    artifact_tree = run_card_payload.get("artifact_tree")
    if artifact_tree is not None:
        if isinstance(artifact_tree, dict) and not any("artifact_index[" in error for error in errors):
            errors.extend(verify_artifact_tree_payload(artifact_tree, artifact_index=artifact_index))
        elif not isinstance(artifact_tree, dict):
            errors.append("artifact_tree must be an object")
    else:
        expected_merkle_root = run_card_payload.get("artifact_merkle_root")
        if not isinstance(expected_merkle_root, str) or not SHA256_HEX_RE.fullmatch(expected_merkle_root):
            errors.append("artifact_merkle_root must be a lowercase 64-char hex digest")
        elif not any("artifact_index[" in error for error in errors):
            recomputed_merkle_root = compute_artifact_merkle_root(artifact_index)
            if recomputed_merkle_root != expected_merkle_root:
                errors.append(
                    f"artifact_merkle_root mismatch: expected={expected_merkle_root}, recomputed={recomputed_merkle_root}"
                )

    artifact_index_by_relative_path = {}
    for artifact in artifact_index:
        if not isinstance(artifact, dict):
            continue
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        try:
            normalized_relative_path = normalize_run_card_relative_path(relative_path)
        except RunCardPathValidationError:
            continue
        artifact_index_by_relative_path[normalized_relative_path] = artifact
    _verify_reconstruction_scene_manifests(
        run_card_payload,
        run_card_root=run_card_path.parent,
        artifact_index_by_relative_path=artifact_index_by_relative_path,
        errors=errors,
    )
    _verify_reconstruction_diagnostics(
        run_card_payload,
        run_card_root=run_card_path.parent,
        errors=errors,
    )

    _verify_backend_semantics(run_card_payload, errors)
    _verify_requested_depth_pro_fulfillment(
        run_card_payload,
        run_card_root=run_card_path.parent,
        errors=errors,
    )
    _verify_config_fingerprint(run_card_payload, errors)

    if check_canonical_json:
        raw_text = run_card_path.read_text(encoding="utf-8")
        canonical = canonical_json_text(raw_run_card_payload)
        if raw_text not in (canonical, canonical + "\n"):
            errors.append("JSON canonical serialization drift detected (expected sort_keys=True, indent=2)")

    return errors
