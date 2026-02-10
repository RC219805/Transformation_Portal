"""Sidecar writer for deterministic JSON output.

Provides atomic, deterministic writing of provenance sidecars and manifests.

Key features:
- Atomic writes (temp file + rename)
- Deterministic JSON (sorted keys, stable formatting)
- Safe error handling (cleanup on failure)
- Fsync support for durability

Usage:
    from transformation_portal.ingest import write_sidecar

    write_sidecar(
        sidecar=provenance_sidecar,
        output_path=Path("input_provenance.json"),
        fsync=True,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from .schemas import IngestManifest, ProvenanceSidecar

logger = logging.getLogger(__name__)


def write_sidecar(
    sidecar: Union[ProvenanceSidecar, IngestManifest],
    output_path: Path,
    fsync: bool = False,
) -> None:
    """Write sidecar JSON with atomic write pattern.

    Uses temp file + rename for atomic writes to prevent corruption.

    Args:
        sidecar: ProvenanceSidecar or IngestManifest object
        output_path: Output path for JSON file
        fsync: If True, call fsync before rename (slower but more durable)

    Raises:
        IOError: If write fails
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use temp file for atomic write
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        # Write to temp file
        with open(temp_path, "w") as f:
            f.write(sidecar.to_json_deterministic())

            # Fsync if requested
            if fsync:
                f.flush()
                import os

                os.fsync(f.fileno())

        # Atomic rename
        temp_path.replace(output_path)

        logger.debug(f"Wrote sidecar: {output_path}")

    except Exception as e:
        # Cleanup temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write sidecar to {output_path}: {e}") from e


def load_sidecar(
    sidecar_path: Path,
    schema_type: str = "provenance",
) -> Union[ProvenanceSidecar, IngestManifest]:
    """Load sidecar from JSON file.

    Args:
        sidecar_path: Path to sidecar JSON file
        schema_type: "provenance" or "manifest"

    Returns:
        ProvenanceSidecar or IngestManifest object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema_type invalid or JSON invalid
    """
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    with open(sidecar_path, "r") as f:
        data = json.load(f)

    if schema_type == "provenance":
        return ProvenanceSidecar(**data)
    elif schema_type == "manifest":
        return IngestManifest(**data)
    else:
        raise ValueError(f"Invalid schema_type: {schema_type}. Must be 'provenance' or 'manifest'.")
