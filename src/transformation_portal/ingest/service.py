"""Phase 3.7 ingest orchestration surface.

This module intentionally adds a contract skeleton only. Runtime behavior
remains unchanged until later commits wire CLI and orchestration flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metadata_service import MetadataExtractionService


@dataclass(frozen=True)
class ServiceRunRequest:
    """Input contract for orchestration entrypoints."""

    command: str
    input_path: Path | None = None
    input_paths: Sequence[Path] = ()
    output_dir: Path | None = None
    machine_mode: bool = False
    strict: bool = True
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceRunResult:
    """Output contract for orchestration entrypoints."""

    success: bool
    exit_code: int
    payload: dict[str, Any] | None = None


class MetadataExtractionOrchestrationService:
    """Skeleton orchestration layer for future Phase 3.7 integration."""

    def __init__(self, *, metadata_service: MetadataExtractionService | None = None) -> None:
        self._metadata_service = metadata_service or MetadataExtractionService()

    @property
    def metadata_service(self) -> MetadataExtractionService:
        """Expose delegated metadata service dependency."""
        return self._metadata_service

    def run(self, request: ServiceRunRequest) -> ServiceRunResult:
        """Execute orchestration flow.

        Placeholder only for commit 1. Later commits will implement command
        dispatch and machine-mode output composition.
        """
        raise NotImplementedError(
            "Phase 3.7 orchestration skeleton: run() is not implemented yet " f"for command '{request.command}'."
        )
