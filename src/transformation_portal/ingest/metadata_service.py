"""Ingest orchestration service for provenance extraction and validation."""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import IngestError, IngestExitCode, OtherIngestFailure, aggregate_errors
from .provenance import capture_provenance
from .raw_sidecar import generate_raw_sidecar, is_raw_image_path
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
    emit_raw_sidecar: bool = True
    raw_sidecar_output_path: Optional[Path] = None
    raw_sidecar_strict: bool = False


@dataclass(frozen=True)
class ExtractResult:
    path: Path
    success: bool
    output_path: Optional[Path]
    elapsed_seconds: float
    error: Optional[IngestError] = None
    raw_sidecar_path: Optional[Path] = None
    raw_sidecar_error: Optional[str] = None


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
    emit_raw_sidecar: bool = True
    raw_sidecar_strict: bool = False


@dataclass(frozen=True)
class BatchItemResult:
    path: Path
    success: bool
    output_path: Optional[Path]
    elapsed_seconds: float
    error: Optional[IngestError] = None
    raw_sidecar_path: Optional[Path] = None
    raw_sidecar_error: Optional[str] = None


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
        generate_raw_sidecar_fn: Callable[..., Any] = generate_raw_sidecar,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._capture_provenance = capture_provenance_fn
        self._write_sidecar = write_sidecar_fn
        self._validate_schema_errors = validate_schema_errors_fn
        self._generate_raw_sidecar = generate_raw_sidecar_fn
        self._clock = clock_fn

    def extract(self, req: ExtractRequest) -> ExtractResult:
        """Extract provenance for a single input and write sidecar."""
        start = self._clock()
        raw_sidecar_path: Optional[Path] = None
        raw_sidecar_error: Optional[str] = None
        raw_sidecar_written = False
        try:
            if not req.input_path.exists():
                not_found_err = OtherIngestFailure(f"Input not found: {req.input_path}")
                return ExtractResult(
                    path=req.input_path,
                    success=False,
                    output_path=None,
                    elapsed_seconds=self._clock() - start,
                    error=not_found_err,
                )

            output_path = self._derive_output_path(req)
            should_emit_raw_sidecar = req.emit_raw_sidecar and is_raw_image_path(req.input_path)
            resolved_raw_sidecar_path = (
                self._derive_raw_sidecar_output_path(
                    provenance_output_path=output_path,
                    explicit_output_path=req.raw_sidecar_output_path,
                )
                if should_emit_raw_sidecar
                else None
            )
            config_dict = (
                req.config_dict
                if req.config_dict is not None
                else {
                    "mode": "metadata_service",
                    "phase": "3.7",
                }
            )

            sidecar = self._capture_provenance(
                input_path=req.input_path,
                cli_args=list(req.cli_args),
                config_dict=config_dict,
                preset=req.preset,
            )

            if should_emit_raw_sidecar and resolved_raw_sidecar_path is not None:
                precomputed_file_sha256, precomputed_file_size = self._extract_precomputed_file_integrity(sidecar)
                precomputed_exiftool_payload = self._extract_precomputed_exiftool_payload(sidecar)
                precomputed_exiftool_version = self._extract_precomputed_exiftool_version(sidecar)
                try:
                    raw_result = self._generate_raw_sidecar(
                        req.input_path,
                        output_path=resolved_raw_sidecar_path,
                        file_sha256=precomputed_file_sha256,
                        file_size_bytes=precomputed_file_size,
                        precomputed_exiftool_payload=precomputed_exiftool_payload,
                        precomputed_exiftool_version=precomputed_exiftool_version,
                        fsync=req.fsync,
                    )
                    raw_sidecar_path = raw_result.output_path
                    raw_sidecar_error = raw_result.rawpy_error
                    raw_sidecar_written = True
                except Exception as exc:  # noqa: BLE001
                    raw_sidecar_error = str(exc)
                    if req.raw_sidecar_strict:
                        raise OtherIngestFailure(f"RAW sidecar generation failed for {req.input_path}: {exc}") from exc

            try:
                self._write_sidecar(sidecar, output_path, fsync=req.fsync)
            except Exception:
                if raw_sidecar_written and raw_sidecar_path is not None:
                    self._remove_if_exists(raw_sidecar_path)
                    raw_sidecar_path = None
                raise

            return ExtractResult(
                path=req.input_path,
                success=True,
                output_path=output_path,
                elapsed_seconds=self._clock() - start,
                raw_sidecar_path=raw_sidecar_path,
                raw_sidecar_error=raw_sidecar_error,
            )
        except IngestError as error:
            return ExtractResult(
                path=req.input_path,
                success=False,
                output_path=None,
                elapsed_seconds=self._clock() - start,
                error=error,
                raw_sidecar_path=raw_sidecar_path,
                raw_sidecar_error=raw_sidecar_error,
            )
        except Exception as exc:  # noqa: BLE001
            wrapped_error = OtherIngestFailure(str(exc))
            return ExtractResult(
                path=req.input_path,
                success=False,
                output_path=None,
                elapsed_seconds=self._clock() - start,
                error=wrapped_error,
                raw_sidecar_path=raw_sidecar_path,
                raw_sidecar_error=raw_sidecar_error,
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
            return ValidateResult(
                success=dominant_error is None,
                errors=errors,
                dominant_error=dominant_error,
            )
        except IngestError as ie:
            return ValidateResult(
                success=False,
                errors=[ie],
                dominant_error=ie,
            )
        except Exception as exc:  # noqa: BLE001
            wrapped = OtherIngestFailure(str(exc))
            return ValidateResult(
                success=False,
                errors=[wrapped],
                dominant_error=wrapped,
            )

    def batch_extract(self, req: BatchExtractRequest) -> BatchExtractResult:
        """Run extract for each file and aggregate typed outcomes."""
        start = self._clock()
        paths = list(req.input_paths)
        if req.deterministic_order:
            paths = sorted(paths, key=str)

        items: List[BatchItemResult] = []
        errors: List[IngestError] = []
        by_exit: Counter[IngestExitCode] = Counter()
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
                    emit_raw_sidecar=req.emit_raw_sidecar,
                    raw_sidecar_strict=req.raw_sidecar_strict,
                )
            )
            item = BatchItemResult(
                path=result.path,
                success=result.success,
                output_path=result.output_path,
                elapsed_seconds=result.elapsed_seconds,
                error=result.error,
                raw_sidecar_path=result.raw_sidecar_path,
                raw_sidecar_error=result.raw_sidecar_error,
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
                    candidate = base_path.with_name(f"{input_path.name}" f".{digest}.provenance.json")
                    counter = 0
                    while candidate in seen:
                        counter += 1
                        candidate = base_path.with_name(f"{input_path.name}" f".{digest}.{counter}" ".provenance.json")
            seen.add(candidate)
            chosen_paths.append(candidate)

        return chosen_paths

    def _stable_path_digest(self, input_path: Path) -> str:
        try:
            canonical_path = input_path.resolve(strict=False).as_posix()
        except (OSError, RuntimeError):
            canonical_path = input_path.absolute().as_posix()
        return hashlib.blake2s(
            canonical_path.encode("utf-8"),
            digest_size=4,
        ).hexdigest()

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

    def _derive_raw_sidecar_output_path(
        self,
        *,
        provenance_output_path: Path,
        explicit_output_path: Optional[Path],
    ) -> Path:
        if explicit_output_path is not None:
            return explicit_output_path

        provenance_name = provenance_output_path.name
        suffix = ".provenance.json"
        if provenance_name.endswith(suffix):
            base_name = provenance_name[: -len(suffix)]
            return provenance_output_path.with_name(f"{base_name}.raw.sidecar.json")
        return provenance_output_path.with_name(f"{provenance_output_path.stem}.raw.sidecar.json")

    def _remove_if_exists(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return

    def _extract_precomputed_file_integrity(
        self,
        sidecar: Any,
    ) -> tuple[Optional[str], Optional[int]]:
        file_integrity = getattr(sidecar, "file_integrity", None)
        if file_integrity is None:
            return None, None

        sha256 = getattr(file_integrity, "sha256", None)
        size_bytes = getattr(file_integrity, "size_bytes", None)
        return (
            str(sha256) if isinstance(sha256, str) else None,
            int(size_bytes) if isinstance(size_bytes, int) else None,
        )

    def _extract_precomputed_exiftool_payload(
        self,
        sidecar: Any,
    ) -> Optional[Dict[str, Any]]:
        exif = getattr(sidecar, "exif", None)
        if exif is None:
            return None

        all_tags = getattr(exif, "all_tags", None)
        if isinstance(all_tags, Mapping):
            return dict(all_tags)
        return None

    def _extract_precomputed_exiftool_version(
        self,
        sidecar: Any,
    ) -> Optional[str]:
        toolchain = getattr(sidecar, "toolchain", None)
        if not isinstance(toolchain, (list, tuple)):
            return None

        for tool in toolchain:
            if isinstance(tool, Mapping):
                tool_name = tool.get("name")
                tool_version = tool.get("version")
            else:
                tool_name = getattr(tool, "name", None)
                tool_version = getattr(tool, "version", None)

            if tool_name == "exiftool" and tool_version is not None:
                return str(tool_version)
        return None
