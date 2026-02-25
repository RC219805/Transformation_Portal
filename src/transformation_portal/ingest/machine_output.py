"""Deterministic machine-output serializers for ingest domain results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import IngestError, IngestExitCode
from .metadata_service import BatchExtractResult, BatchItemResult, ExtractResult, ValidateResult

MACHINE_SCHEMA_VERSION = "tp.meta.machine.v1"
_CANONICAL_JSON_KWARGS: Dict[str, Any] = {"sort_keys": True, "ensure_ascii": True}


def exit_code_to_dict(code: IngestExitCode) -> Dict[str, Any]:
    """Serialize exit-code enum to stable machine shape."""
    return {"name": code.name, "value": int(code)}


def error_to_dict(error: IngestError) -> Dict[str, Any]:
    """Serialize typed ingest errors without repr artifacts."""
    return {
        "type": error.__class__.__name__,
        "message": error.message,
        "exit_code": exit_code_to_dict(error.exit_code),
        "priority": error.priority,
    }


def extract_result_to_dict(result: ExtractResult, *, preset: Optional[str] = None) -> Dict[str, Any]:
    """Serialize single-item extract result (includes volatile elapsed_seconds telemetry)."""
    return {
        "input_path": str(result.path),
        "success": result.success,
        "output_path": str(result.output_path) if result.output_path is not None else None,
        "elapsed_seconds": result.elapsed_seconds,
        "preset": preset,
        "error": error_to_dict(result.error) if result.error is not None else None,
    }


def validate_result_to_dict(
    result: ValidateResult,
    *,
    sidecar_path: Path,
    strict: bool,
) -> Dict[str, Any]:
    """Serialize validation result including typed errors."""
    return {
        "sidecar_path": str(sidecar_path),
        "strict": strict,
        "success": result.success,
        "errors": [error_to_dict(error) for error in result.errors],
        "dominant_error": error_to_dict(result.dominant_error) if result.dominant_error is not None else None,
    }


def batch_item_to_dict(item: BatchItemResult) -> Dict[str, Any]:
    """Serialize a single batch item result (includes volatile elapsed_seconds telemetry)."""
    return {
        "path": str(item.path),
        "success": item.success,
        "output_path": str(item.output_path) if item.output_path is not None else None,
        "elapsed_seconds": item.elapsed_seconds,
        "error": error_to_dict(item.error) if item.error is not None else None,
    }


def _stable_by_exit_code(raw_by_exit_code: Dict[str, Any]) -> Dict[str, int]:
    ordered: Dict[str, int] = {
        code.name: int(raw_by_exit_code.get(code.name, 0))
        for code in sorted(IngestExitCode, key=lambda code: code.value)
        if code != IngestExitCode.SUCCESS
    }
    for name in sorted(raw_by_exit_code.keys()):
        if name not in ordered:
            ordered[name] = int(raw_by_exit_code[name])
    return ordered


def batch_result_to_dict(
    result: BatchExtractResult,
    *,
    input_root: Path,
    output_dir: Path,
    fail_fast: bool,
    preserve_structure: bool,
) -> Dict[str, Any]:
    """Serialize batch extraction result with stable summary ordering."""
    summary_counts = {
        "total": int(result.summary_counts.get("total", len(result.items))),
        "success": int(result.summary_counts.get("success", 0)),
        "failure": int(result.summary_counts.get("failure", 0)),
        "by_exit_code": _stable_by_exit_code(
            result.summary_counts.get("by_exit_code", {})
            if isinstance(result.summary_counts.get("by_exit_code"), dict)
            else {}
        ),
    }
    return {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "fail_fast": fail_fast,
        "preserve_structure": preserve_structure,
        "success": result.success,
        "items": [batch_item_to_dict(item) for item in result.items],
        "summary_counts": summary_counts,
        "dominant_error": error_to_dict(result.dominant_error) if result.dominant_error is not None else None,
    }


def dump_json(payload: Dict[str, Any], *, pretty: bool) -> str:
    """Dump payload with lexicographic key ordering (`sort_keys=True`)."""
    if pretty:
        return json.dumps(payload, indent=2, **_CANONICAL_JSON_KWARGS)
    return json.dumps(payload, separators=(",", ":"), **_CANONICAL_JSON_KWARGS)
