"""Phase 3.7 ingest orchestration surface.

This module centralizes command-level orchestration while preserving existing
contracts. CLI wiring is handled in later commits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import IngestExitCode, OtherIngestFailure
from .metadata_service import BatchExtractRequest, BatchExtractResult
from .metadata_service import ExtractRequest as CoreExtractRequest
from .metadata_service import ExtractResult as CoreExtractResult
from .metadata_service import MetadataExtractionService as CoreMetadataExtractionService
from .metadata_service import ValidateRequest as CoreValidateRequest
from .metadata_service import ValidateResult as CoreValidateResult
from .sidecar import load_sidecar

SUPPORTED_EXTENSIONS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".arw",
    ".srf",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
}


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


class MetadataExtractionService:
    """Orchestration layer for command-level ingest flows."""

    def __init__(self, *, metadata_service: CoreMetadataExtractionService | None = None) -> None:
        self._metadata_service = metadata_service or CoreMetadataExtractionService()

    @property
    def metadata_service(self) -> CoreMetadataExtractionService:
        """Expose delegated metadata service dependency."""
        return self._metadata_service

    def run(self, request: ServiceRunRequest) -> ServiceRunResult:
        """Execute orchestration flow for supported ingest commands."""
        if request.command == "extract":
            return self._run_extract(request)
        if request.command == "extract-batch":
            return self._run_extract_batch(request)
        if request.command == "validate":
            return self._run_validate(request)

        return ServiceRunResult(
            success=False,
            exit_code=int(IngestExitCode.OTHER_FAILURE),
            payload={"error": f"Unsupported command: {request.command}"},
        )

    def _run_extract(self, request: ServiceRunRequest) -> ServiceRunResult:
        if request.input_path is None:
            error = OtherIngestFailure("Input path required for extract command")
            return ServiceRunResult(
                success=False,
                exit_code=int(error.exit_code),
                payload={"extract_result": None, "sidecar": None, "error": str(error)},
            )

        input_path = self._as_path(request.input_path)
        if input_path is None:
            error = OtherIngestFailure(
                f"Input path for extract command must be str or Path, got {type(request.input_path).__name__}"
            )
            return ServiceRunResult(
                success=False,
                exit_code=int(error.exit_code),
                payload={"extract_result": None, "sidecar": None, "error": str(error)},
            )

        config_dict = request.args.get("config_dict")
        if not isinstance(config_dict, dict):
            config_dict = None

        output_path = self._as_path(request.args.get("output_path"))
        output_dir = self._as_path(request.output_dir)
        cli_args, cli_args_error = self._normalize_cli_args(
            request.args.get("cli_args"),
            command_name="extract",
            payload_key="extract_result",
            sidecar_key="sidecar",
        )
        if cli_args_error is not None:
            return cli_args_error

        extracted = self._metadata_service.extract(
            CoreExtractRequest(
                input_path=input_path,
                output_path=output_path,
                output_dir=output_dir,
                preset=request.args.get("preset"),
                cli_args=cli_args,
                config_dict=config_dict,
                fsync=bool(request.args.get("fsync", False)),
            )
        )

        sidecar = None
        if extracted.output_path is not None:
            try:
                sidecar = load_sidecar(extracted.output_path, schema_type="provenance")
            except Exception:  # noqa: BLE001 - optional payload enrichment
                sidecar = None

        if extracted.success:
            exit_code = int(IngestExitCode.SUCCESS)
        else:
            exit_code = int(extracted.error.exit_code) if extracted.error is not None else int(IngestExitCode.OTHER_FAILURE)

        return ServiceRunResult(
            success=extracted.success,
            exit_code=exit_code,
            payload={"extract_result": extracted, "sidecar": sidecar},
        )

    def _run_extract_batch(self, request: ServiceRunRequest) -> ServiceRunResult:
        if request.input_path is None:
            return self._batch_setup_failure("Input directory required for extract-batch command")

        input_dir = self._as_path(request.input_path)
        if input_dir is None:
            return self._batch_setup_failure(
                f"Input directory for extract-batch command must be str or Path, got {type(request.input_path).__name__}"
            )

        if not input_dir.exists():
            return self._batch_setup_failure(f"Directory not found: {input_dir}")
        if not input_dir.is_dir():
            return self._batch_setup_failure(f"Not a directory: {input_dir}")

        recursive = bool(request.args.get("recursive", True))
        if request.input_paths:
            images, invalid_input_paths = self._as_paths(request.input_paths)
            if invalid_input_paths:
                return self._batch_setup_failure(
                    "input_paths must contain only str or Path values; " f"invalid entries: {invalid_input_paths!r}"
                )
        else:
            images = self._find_images(input_dir, recursive=recursive)

        resolved_output_dir = self._as_path(request.output_dir) or input_dir / "provenance_sidecars"
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = request.args.get("config_dict")
        if not isinstance(config_dict, dict):
            config_dict = None
        cli_args, cli_args_error = self._normalize_cli_args(
            request.args.get("cli_args"),
            command_name="extract-batch",
            payload_key="batch_result",
            sidecar_key=None,
        )
        if cli_args_error is not None:
            return cli_args_error

        result = self._metadata_service.batch_extract(
            BatchExtractRequest(
                input_paths=images,
                output_dir=resolved_output_dir,
                preset=request.args.get("preset"),
                cli_args=cli_args,
                config_dict=config_dict,
                fsync=bool(request.args.get("fsync", False)),
                deterministic_order=True,
                fail_fast=bool(request.args.get("fail_fast", False)),
                preserve_structure=True,
                input_root=input_dir,
            )
        )

        exit_code = int(IngestExitCode.SUCCESS)
        if result.dominant_error is not None:
            exit_code = int(result.dominant_error.exit_code)

        return ServiceRunResult(
            success=result.dominant_error is None,
            exit_code=exit_code,
            payload={
                "batch_result": result,
                "input_dir": input_dir,
                "output_dir": resolved_output_dir,
            },
        )

    def _run_validate(self, request: ServiceRunRequest) -> ServiceRunResult:
        if request.input_path is None:
            error = OtherIngestFailure("Sidecar path required for validate command")
            return ServiceRunResult(
                success=False,
                exit_code=int(error.exit_code),
                payload={"validate_result": None, "sidecar_data": None, "error": str(error)},
            )

        sidecar_path = self._as_path(request.input_path)
        if sidecar_path is None:
            error = OtherIngestFailure(
                f"Sidecar path for validate command must be str or Path, got {type(request.input_path).__name__}"
            )
            return ServiceRunResult(
                success=False,
                exit_code=int(error.exit_code),
                payload={"validate_result": None, "sidecar_data": None, "error": str(error)},
            )

        validated = self._metadata_service.validate(
            CoreValidateRequest(
                sidecar_path=sidecar_path,
                schema_type=str(request.args.get("schema_type", "provenance")),
                strict=request.strict,
            )
        )

        if not validated.success:
            exit_code = (
                int(validated.dominant_error.exit_code)
                if validated.dominant_error is not None
                else int(IngestExitCode.OTHER_FAILURE)
            )
            return ServiceRunResult(
                success=False,
                exit_code=exit_code,
                payload={"validate_result": validated, "sidecar_data": None},
            )

        sidecar_data: dict[str, Any] | None = None
        try:
            with open(sidecar_path, encoding="utf-8") as handle:
                sidecar_data = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - preserve existing fallback semantics
            fallback_error = OtherIngestFailure(str(exc))
            fallback_validate = CoreValidateResult(
                success=False,
                errors=[fallback_error],
                dominant_error=fallback_error,
            )
            return ServiceRunResult(
                success=False,
                exit_code=int(fallback_error.exit_code),
                payload={"validate_result": fallback_validate, "sidecar_data": None},
            )

        return ServiceRunResult(
            success=True,
            exit_code=int(IngestExitCode.SUCCESS),
            payload={"validate_result": validated, "sidecar_data": sidecar_data},
        )

    def _batch_setup_failure(self, message: str) -> ServiceRunResult:
        error = OtherIngestFailure(message)
        batch_result = BatchExtractResult(
            items=[],
            total_elapsed=0.0,
            summary_counts=self._empty_batch_summary(),
            dominant_error=error,
        )
        return ServiceRunResult(
            success=False,
            exit_code=int(error.exit_code),
            payload={"batch_result": batch_result, "input_dir": None, "output_dir": None},
        )

    def _empty_batch_summary(self) -> dict[str, Any]:
        return {
            "total": 0,
            "success": 0,
            "failure": 0,
            "by_exit_code": {
                code.name: 0 for code in sorted(IngestExitCode, key=lambda code: code.value) if code != IngestExitCode.SUCCESS
            },
        }

    def _find_images(self, directory: Path, *, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        images = [
            candidate
            for candidate in directory.glob(pattern)
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(images)

    def _as_path(self, value: Any) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        return None

    def _as_paths(self, values: Sequence[Any]) -> tuple[list[Path], list[Any]]:
        normalized: list[Path] = []
        invalid: list[Any] = []
        for value in values:
            path = self._as_path(value)
            if path is not None:
                normalized.append(path)
            else:
                invalid.append(value)
        return normalized, invalid

    def _normalize_cli_args(
        self,
        raw_cli_args: Any,
        *,
        command_name: str,
        payload_key: str,
        sidecar_key: str | None,
    ) -> tuple[list[Any], ServiceRunResult | None]:
        if raw_cli_args is None:
            return [], None
        if isinstance(raw_cli_args, Sequence) and not isinstance(raw_cli_args, (str, bytes, bytearray)):
            return list(raw_cli_args), None
        error = OtherIngestFailure(
            f"cli_args for {command_name} command must be a sequence or None, got {type(raw_cli_args).__name__}"
        )
        payload: dict[str, Any] = {payload_key: None, "error": str(error)}
        if sidecar_key is not None:
            payload[sidecar_key] = None
        return [], ServiceRunResult(success=False, exit_code=int(error.exit_code), payload=payload)


# Backward-compatible alias while this layer is rolling in.
MetadataExtractionOrchestrationService = MetadataExtractionService
