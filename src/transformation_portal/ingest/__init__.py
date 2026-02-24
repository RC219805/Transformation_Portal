"""Ingest contract enforcement for RAW/TIFF provenance capture.

This module provides audit-grade provenance tracking and schema validation
for the Phase I linear ingest contract.

Components:
- schemas: Versioned JSON schemas (Pydantic models)
- provenance: Full metadata extraction and capture
- validator: Schema validation with drift detection
- sidecar: Deterministic sidecar JSON writing

Contract version: 1.0.0
"""

from __future__ import annotations

__all__ = [
    "IngestManifest",
    "ProvenanceSidecar",
    "validate_schema",
    "classify_validation_exit_code",
    "classify_validation_error",
    "classify_validation_errors",
    "aggregate_exit_codes",
    "EXIT_SUCCESS",
    "EXIT_SCHEMA_VALIDATION_FAILED",
    "EXIT_8BIT_CONVERSION",
    "EXIT_GAMMA_VIOLATION",
    "EXIT_SCHEMA_DRIFT",
    "EXIT_OTHER_FAILURE",
    "capture_provenance",
    "write_sidecar",
    "load_sidecar",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "IngestManifest":
        from .schemas import IngestManifest

        return IngestManifest
    elif name == "ProvenanceSidecar":
        from .schemas import ProvenanceSidecar

        return ProvenanceSidecar
    elif name == "validate_schema":
        from .validator import validate_schema

        return validate_schema
    elif name == "classify_validation_exit_code":
        from .validator import classify_validation_exit_code

        return classify_validation_exit_code
    elif name == "classify_validation_error":
        from .validator import classify_validation_error

        return classify_validation_error
    elif name == "classify_validation_errors":
        from .validator import classify_validation_errors

        return classify_validation_errors
    elif name == "aggregate_exit_codes":
        from .validator import aggregate_exit_codes

        return aggregate_exit_codes
    elif name == "EXIT_SUCCESS":
        from .validator import EXIT_SUCCESS

        return EXIT_SUCCESS
    elif name == "EXIT_SCHEMA_VALIDATION_FAILED":
        from .validator import EXIT_SCHEMA_VALIDATION_FAILED

        return EXIT_SCHEMA_VALIDATION_FAILED
    elif name == "EXIT_8BIT_CONVERSION":
        from .validator import EXIT_8BIT_CONVERSION

        return EXIT_8BIT_CONVERSION
    elif name == "EXIT_GAMMA_VIOLATION":
        from .validator import EXIT_GAMMA_VIOLATION

        return EXIT_GAMMA_VIOLATION
    elif name == "EXIT_SCHEMA_DRIFT":
        from .validator import EXIT_SCHEMA_DRIFT

        return EXIT_SCHEMA_DRIFT
    elif name == "EXIT_OTHER_FAILURE":
        from .validator import EXIT_OTHER_FAILURE

        return EXIT_OTHER_FAILURE
    elif name == "capture_provenance":
        from .provenance import capture_provenance

        return capture_provenance
    elif name == "write_sidecar":
        from .sidecar import write_sidecar

        return write_sidecar
    elif name == "load_sidecar":
        from .sidecar import load_sidecar

        return load_sidecar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
