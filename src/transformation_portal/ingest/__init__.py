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
