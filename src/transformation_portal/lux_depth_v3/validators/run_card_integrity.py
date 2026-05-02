"""Integrity verification helpers for Lux run cards."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.artifact_tree import verify_artifact_tree_payload
from transformation_portal.lux_depth_v3.manifest import compute_file_sha256 as _shared_compute_file_sha256
from transformation_portal.lux_depth_v3.run_card_contract import (
    RunCardPathValidationError,
    infer_run_card_version,
    normalize_run_card_relative_path,
    with_inferred_run_card_version,
)

from .run_card_backend_semantics import collect_run_card_backend_semantic_errors
from .run_card_validator import _default_schema_path, _load_validator

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


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"Text file not found: {path}"
    except PermissionError:
        return None, f"Permission denied reading text file: {path}"
    except OSError as exc:
        return None, f"Failed to read text file {path}: {exc}"


def _compute_file_sha256(path: Path) -> tuple[str | None, str | None]:
    try:
        return _shared_compute_file_sha256(path), None
    except FileNotFoundError:
        return None, f"File not found while hashing: {path}"
    except PermissionError:
        return None, f"Permission denied while hashing file: {path}"
    except OSError as exc:
        return None, f"Failed to hash file {path}: {exc}"


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


def _verify_backend_semantics(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    errors: list[str],
) -> None:
    for error in collect_run_card_backend_semantic_errors(run_card_payload):
        if error.startswith("requested backend 'depth_pro' was not honored"):
            fallback_reason = _first_combined_manifest_fallback_reason(
                run_card_payload,
                run_card_root=run_card_root,
                errors=errors,
            )
            if fallback_reason:
                error = f"{error}. First fallback reason: {fallback_reason}"
        errors.append(error)


def _verify_captioning_status(run_card_payload: dict[str, Any], errors: list[str]) -> None:
    """Fail closed if advisory captions are marked as quality-gate evidence."""
    top_level_status = run_card_payload.get("captioning_status")
    if isinstance(top_level_status, dict) and top_level_status.get("used_for_quality_gate") is True:
        errors.append("captioning_status.used_for_quality_gate must be false")

    result_summary = run_card_payload.get("result_summary")
    if not isinstance(result_summary, list):
        return
    for index, row in enumerate(result_summary):
        if not isinstance(row, dict):
            continue
        status = row.get("captioning_status")
        if isinstance(status, dict) and status.get("used_for_quality_gate") is True:
            errors.append(f"result_summary[{index}].captioning_status.used_for_quality_gate must be false")


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

    fingerprint_fields = tuple(
        field for field in config_fingerprint if field not in {"hash_algorithm", "canonical_json", "sha256"}
    )
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


def _verify_indexed_artifact_files(
    run_card_payload: dict[str, Any],
    *,
    run_card_root: Path,
    errors: list[str],
) -> None:
    """Recompute indexed artifact size/hash from disk."""
    artifact_index = run_card_payload.get("artifact_index")
    if not isinstance(artifact_index, list):
        return

    for index, artifact in enumerate(artifact_index):
        if not isinstance(artifact, dict):
            continue
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        artifact_path, path_error = resolve_artifact_path(
            run_card_root=run_card_root,
            relative_path=relative_path,
            context=f"artifact_index[{index}]",
        )
        if path_error:
            errors.append(path_error)
            continue
        assert artifact_path is not None
        try:
            artifact_stat = artifact_path.stat()
        except FileNotFoundError:
            errors.append(f"artifact_index[{index}] file is missing: {relative_path}")
            continue
        except PermissionError:
            errors.append(f"artifact_index[{index}] file is not readable: {relative_path}")
            continue
        except OSError as exc:
            errors.append(f"artifact_index[{index}] file stat failed for {relative_path}: {exc}")
            continue
        if not stat.S_ISREG(artifact_stat.st_mode):
            errors.append(f"artifact_index[{index}] is not a regular file: {relative_path}")
            continue
        expected_size = artifact.get("size_bytes")
        actual_size = artifact_stat.st_size
        if isinstance(expected_size, int) and expected_size != actual_size:
            errors.append(
                f"artifact_index[{index}].size_bytes mismatch for {relative_path}: "
                f"got={expected_size}, actual={actual_size}"
            )
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and SHA256_HEX_RE.fullmatch(expected_sha):
            actual_sha, hash_error = _compute_file_sha256(artifact_path)
            if hash_error:
                errors.append(f"artifact_index[{index}] file hash failed for {relative_path}: {hash_error}")
                continue
            if expected_sha != actual_sha:
                errors.append(
                    f"artifact_index[{index}].sha256 mismatch for {relative_path}: " f"got={expected_sha}, actual={actual_sha}"
                )


def _verify_run_card_self_integrity(
    raw_run_card_payload: dict[str, Any],
    *,
    run_card_path: Path,
    errors: list[str],
) -> None:
    """Validate non-recursive run-card self-integrity evidence when present."""
    integrity = raw_run_card_payload.get("run_card_integrity")
    if integrity is None:
        return
    if not isinstance(integrity, dict):
        errors.append("run_card_integrity must be an object")
        return

    if integrity.get("self_indexing") != "excluded_self_hash_cycle":
        errors.append("run_card_integrity.self_indexing must be 'excluded_self_hash_cycle'")
    expected_relative_path = run_card_path.name
    if integrity.get("path") != expected_relative_path:
        errors.append(
            "run_card_integrity.path mismatch: " f"got={integrity.get('path')!r}, expected={expected_relative_path!r}"
        )

    expected_payload_sha = integrity.get("canonical_payload_sha256")
    if not isinstance(expected_payload_sha, str) or not SHA256_HEX_RE.fullmatch(expected_payload_sha):
        errors.append("run_card_integrity.canonical_payload_sha256 must be a lowercase 64-char hex digest")
    else:
        integrity_without_hash = {key: value for key, value in integrity.items() if key != "canonical_payload_sha256"}
        payload_without_hash = {
            **raw_run_card_payload,
            "run_card_integrity": integrity_without_hash,
        }
        actual_payload_sha = hashlib.sha256(canonicalize_json(payload_without_hash)).hexdigest()
        if actual_payload_sha != expected_payload_sha:
            errors.append(
                "run_card_integrity.canonical_payload_sha256 mismatch: "
                f"got={expected_payload_sha}, actual={actual_payload_sha}"
            )

    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar_payload, sidecar_error = _load_json(sidecar_path)
    if sidecar_error:
        errors.append(f"run card self-integrity sidecar missing or unreadable: {sidecar_error}")
        return
    if not isinstance(sidecar_payload, dict):
        errors.append(f"run card self-integrity sidecar must be a JSON object: {sidecar_path}")
        return
    if sidecar_payload.get("run_card_path") != expected_relative_path:
        errors.append(
            "run card self-integrity sidecar run_card_path mismatch: "
            f"got={sidecar_payload.get('run_card_path')!r}, expected={expected_relative_path!r}"
        )
    if sidecar_payload.get("self_indexing") != "excluded_self_hash_cycle":
        errors.append("run card self-integrity sidecar self_indexing must be 'excluded_self_hash_cycle'")
    if sidecar_payload.get("hash_algorithm") != "sha256":
        errors.append("run card self-integrity sidecar hash_algorithm must be 'sha256'")
    final_sha = sidecar_payload.get("final_run_card_sha256")
    if not isinstance(final_sha, str) or not SHA256_HEX_RE.fullmatch(final_sha):
        errors.append("run card self-integrity sidecar final_run_card_sha256 must be a lowercase 64-char hex digest")
        return
    actual_file_sha, run_card_hash_error = _compute_file_sha256(run_card_path)
    if run_card_hash_error:
        errors.append(f"run card self-integrity sidecar hash check failed: {run_card_hash_error}")
        return
    if final_sha != actual_file_sha:
        errors.append(
            "run card self-integrity sidecar final_run_card_sha256 mismatch: " f"got={final_sha}, actual={actual_file_sha}"
        )


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

    errors: list[str] = []
    try:
        validator = _load_validator(
            str(effective_schema_path) if schema_path is not None else None,
            infer_run_card_version(run_card_payload),
        )
    except RuntimeError as exc:
        errors.append(str(exc))
        return errors
    except (JSONDecodeError, OSError) as exc:
        errors.append(f"Failed to load run card schema {effective_schema_path}: {exc}")
        return errors

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

    _verify_indexed_artifact_files(
        run_card_payload,
        run_card_root=run_card_path.parent,
        errors=errors,
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

    _verify_backend_semantics(
        run_card_payload,
        run_card_root=run_card_path.parent,
        errors=errors,
    )
    _verify_captioning_status(run_card_payload, errors)
    _verify_config_fingerprint(run_card_payload, errors)
    _verify_run_card_self_integrity(
        raw_run_card_payload,
        run_card_path=run_card_path,
        errors=errors,
    )

    if check_canonical_json:
        raw_text, raw_text_error = _read_text(run_card_path)
        if raw_text_error:
            errors.append(raw_text_error)
        else:
            canonical = canonical_json_text(raw_run_card_payload)
            if raw_text not in (canonical, canonical + "\n"):
                errors.append("JSON canonical serialization drift detected (expected sort_keys=True, indent=2)")

    return errors
