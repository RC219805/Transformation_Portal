"""Ingest orchestration service for provenance extraction and validation."""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import IngestError, IngestExitCode, OtherIngestFailure, aggregate_errors
from .provenance import capture_provenance
from .sidecar import write_sidecar
from .validator import validate_schema_errors


@dataclass(frozen=True)
class ExtractRequest:
    input_path: Path
    output_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    preset: Optional[str] = None
    cli_args: Sequence[str] = ()
    config_dict: Optional[Dict[str, Any]] = None
    fsync: bool = False


@dataclass(frozen=True)
class ExtractResult:
    path: Path
    success: bool
    output_path: Optional[Path]
    elapsed_seconds: float
    error: Optional[IngestError] = None


@dataclass(frozen=True)
class ValidateRequest:
    sidecar_path: Path
    schema_type: str = "provenance"
    strict: bool = True


@dataclass(frozen=True)
class ValidateResult:
    success: bool
    errors: List[IngestError]
    dominant_error: Optional[IngestError]


@dataclass(frozen=True)
class BatchExtractRequest:
    input_paths: Sequence[Path]
    output_dir: Path
    preset: Optional[str] = None
    cli_args: Sequence[str] = ()
    config_dict: Optional[Dict[str, Any]] = None
    fsync: bool = False
    deterministic_order: bool = True
    fail_fast: bool = False
    preserve_structure: bool = True
    input_root: Optional[Path] = None


@dataclass(frozen=True)
class BatchItemResult:
    path: Path
    success: bool
    output_path: Optional[Path]
    elapsed_seconds: float
    error: Optional[IngestError] = None


@dataclass(frozen=True)
class BatchExtractResult:
    items: List[BatchItemResult]
    total_elapsed: float
    summary_counts: Dict[str, Any]
    dominant_error: Optional[IngestError]

    @property
    def success(self) -> bool:
        return self.dominant_error is None


class MetadataExtractionService:
    """Canonical ingest orchestration API for extraction and validation."""

    def __init__(
        self,
        *,
        capture_provenance_fn: Callable[..., Any] = capture_provenance,
        write_sidecar_fn: Callable[..., Any] = write_sidecar,
        validate_schema_errors_fn: Callable[..., List[IngestError]] = validate_schema_errors,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._capture_provenance = capture_provenance_fn
        self._write_sidecar = write_sidecar_fn
        self._validate_schema_errors = validate_schema_errors_fn
        self._clock = clock_fn

    def extract(self, req: ExtractRequest) -> ExtractResult:
        """Extract provenance for a single input and write sidecar."""
        start = self._clock()
        try:
            if not req.input_path.exists():
                error = OtherIngestFailure(f"Input not found: {req.input_path}")
                return ExtractResult(
                    path=req.input_path,
                    success=False,
                    output_path=None,
                    elapsed_seconds=self._clock() - start,
                    error=error,
                )

            output_path = self._derive_output_path(req)
            config_dict = req.config_dict if req.config_dict is not None else {"mode": "metadata_service", "phase": "3.7"}

            sidecar = self._capture_provenance(
                input_path=req.input_path,
                cli_args=list(req.cli_args),
                config_dict=config_dict,
                preset=req.preset,
            )
            self._write_sidecar(sidecar, output_path, fsync=req.fsync)

            return ExtractResult(
                path=req.input_path,
                success=True,
                output_path=output_path,
                elapsed_seconds=self._clock() - start,
            )
        except IngestError as error:
            return ExtractResult(
                path=req.input_path,
                success=False,
                output_path=None,
                elapsed_seconds=self._clock() - start,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - service API returns typed failures
            error = OtherIngestFailure(str(exc))
            return ExtractResult(
                path=req.input_path,
                success=False,
                output_path=None,
                elapsed_seconds=self._clock() - start,
                error=error,
            )

    def validate(self, req: ValidateRequest) -> ValidateResult:
        """Validate a sidecar against the configured schema type."""
        try:
            errors = self._validate_schema_errors(
                data=req.sidecar_path,
                schema_type=req.schema_type,
                strict_mode=req.strict,
            )
            dominant_error = aggregate_errors(errors)
            return ValidateResult(success=dominant_error is None, errors=errors, dominant_error=dominant_error)
        except IngestError as error:
            return ValidateResult(success=False, errors=[error], dominant_error=error)
        except Exception as exc:  # noqa: BLE001 - service API returns typed failures
            error = OtherIngestFailure(str(exc))
            return ValidateResult(success=False, errors=[error], dominant_error=error)

    def batch_extract(self, req: BatchExtractRequest) -> BatchExtractResult:
        """Run extract for each file and aggregate typed outcomes."""
        start = self._clock()
        paths = list(req.input_paths)
        if req.deterministic_order:
            paths = sorted(paths, key=lambda path: str(path))

        items: List[BatchItemResult] = []
        errors: List[IngestError] = []
        by_exit = Counter()
        success_count = 0
        failure_count = 0

        default_batch_config = {"mode": "batch_extraction", "phase": "3.7"}
        config_dict = req.config_dict if req.config_dict is not None else default_batch_config

        effective_input_root = req.input_root
        if req.preserve_structure and effective_input_root is None and paths:
            effective_input_root = self._common_input_root(paths)

        output_paths = self._derive_batch_output_paths(
            paths=paths,
            output_dir=req.output_dir,
            preserve_structure=req.preserve_structure,
            input_root=effective_input_root,
        )

        # Ordering of results is deterministic and matches input path order.
        for path, output_path in zip(paths, output_paths):
            result = self.extract(
                ExtractRequest(
                    input_path=path,
                    output_path=output_path,
                    preset=req.preset,
                    cli_args=req.cli_args,
                    config_dict=config_dict,
                    fsync=req.fsync,
                )
            )
            item = BatchItemResult(
                path=result.path,
                success=result.success,
                output_path=result.output_path,
                elapsed_seconds=result.elapsed_seconds,
                error=result.error,
            )
            items.append(item)

            if item.success:
                success_count += 1
                continue

            failure_count += 1
            if item.error is not None:
                errors.append(item.error)
                by_exit[item.error.exit_code] += 1
            else:
                by_exit[IngestExitCode.OTHER_FAILURE] += 1
            if req.fail_fast:
                break

        dominant_error = aggregate_errors(errors)
        summary = {
            "total": len(items),
            "success": success_count,
            "failure": failure_count,
            "by_exit_code": {
                code.name: by_exit.get(code, 0)
                for code in sorted(IngestExitCode, key=lambda code: code.value)
                if code != IngestExitCode.SUCCESS
            },
        }
        return BatchExtractResult(
            items=items,
            total_elapsed=self._clock() - start,
            summary_counts=summary,
            dominant_error=dominant_error,
        )

    def _derive_batch_output_paths(
        self,
        *,
        paths: Sequence[Path],
        output_dir: Path,
        preserve_structure: bool,
        input_root: Optional[Path],
    ) -> List[Path]:
        base_paths: List[Path] = []
        for input_path in paths:
            base_paths.append(
                self._derive_batch_output_path(
                    input_path=input_path,
                    output_dir=output_dir,
                    preserve_structure=preserve_structure,
                    input_root=input_root,
                )
            )

        counts = Counter(base_paths)
        chosen_paths: List[Path] = []
        seen: set[Path] = set()
        for input_path, base_path in zip(paths, base_paths):
            if counts[base_path] == 1:
                candidate = base_path
            else:
                candidate = base_path.with_name(f"{input_path.name}.provenance.json")
                if candidate in seen:
                    digest = self._stable_path_digest(input_path)
                    candidate = base_path.with_name(f"{input_path.name}.{digest}.provenance.json")
                    counter = 0
                    while candidate in seen:
                        counter += 1
                        candidate = base_path.with_name(f"{input_path.name}.{digest}.{counter}.provenance.json")
            seen.add(candidate)
            chosen_paths.append(candidate)

        return chosen_paths

    def _stable_path_digest(self, input_path: Path) -> str:
        try:
            canonical_path = input_path.resolve(strict=False).as_posix()
        except (OSError, RuntimeError):
            canonical_path = input_path.absolute().as_posix()
        return hashlib.blake2s(canonical_path.encode("utf-8"), digest_size=4).hexdigest()

    def _common_input_root(self, paths: Sequence[Path]) -> Optional[Path]:
        if not paths:
            return None
        try:
            common = os.path.commonpath([str(path) for path in paths])
            return Path(common)
        except ValueError:
            return None

    def _derive_batch_output_path(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        preserve_structure: bool,
        input_root: Optional[Path],
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem_name = f"{input_path.stem}.provenance.json"
        if not preserve_structure or input_root is None:
            return output_dir / stem_name

        try:
            relative = input_path.relative_to(input_root)
        except ValueError:
            return output_dir / stem_name

        target_dir = output_dir / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / stem_name

    def _derive_output_path(self, req: ExtractRequest) -> Path:
        if req.output_path is not None:
            return req.output_path

        stem_name = f"{req.input_path.stem}.provenance.json"
        if req.output_dir is not None:
            req.output_dir.mkdir(parents=True, exist_ok=True)
            return req.output_dir / stem_name

        return req.input_path.with_name(stem_name)
