"""Unit tests for ingest MetadataExtractionService orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.ingest.errors import (
    BitDepthViolation,
    IngestExitCode,
    OtherIngestFailure,
    SchemaDriftFailure,
    SchemaValidationFailure,
)
from transformation_portal.ingest.metadata_service import (
    BatchExtractRequest,
    ExtractRequest,
    ExtractResult,
    MetadataExtractionService,
    ValidateRequest,
)


def _clock() -> float:
    return 100.0


def test_batch_extract_delegates_per_file_to_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "b.cr2", tmp_path / "a.cr2", tmp_path / "c.cr2"]
    for path in input_paths:
        path.touch()

    service = MetadataExtractionService(clock_fn=_clock)
    calls: list[ExtractRequest] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        calls.append(req)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path))

    assert len(calls) == len(input_paths)
    assert [call.input_path for call in calls] == sorted(input_paths, key=lambda path: str(path))
    assert all(call.output_path is not None for call in calls)
    assert result.success


def test_batch_extract_uses_priority_to_choose_dominant_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "one.tif", tmp_path / "two.tif", tmp_path / "three.tif"]
    for path in input_paths:
        path.touch()

    error_by_name = {
        "one.tif": OtherIngestFailure("fallback failure"),
        "two.tif": SchemaValidationFailure("schema failure"),
        "three.tif": SchemaDriftFailure("drift failure"),
    }

    service = MetadataExtractionService(clock_fn=_clock)

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        error = error_by_name[req.input_path.name]
        return ExtractResult(
            path=req.input_path,
            success=False,
            output_path=None,
            elapsed_seconds=0.2,
            error=error,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path, deterministic_order=False)
    )

    assert not result.success
    assert isinstance(result.dominant_error, SchemaDriftFailure)


def test_batch_extract_summary_counts_are_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "a.cr2", tmp_path / "b.cr2", tmp_path / "c.cr2", tmp_path / "d.cr2"]
    for path in input_paths:
        path.touch()

    response_by_name = {
        "a.cr2": ExtractResult(
            path=input_paths[0], success=True, output_path=tmp_path / "a.provenance.json", elapsed_seconds=0.1
        ),
        "b.cr2": ExtractResult(
            path=input_paths[1],
            success=False,
            output_path=None,
            elapsed_seconds=0.1,
            error=BitDepthViolation("8-bit violation"),
        ),
        "c.cr2": ExtractResult(
            path=input_paths[2],
            success=False,
            output_path=None,
            elapsed_seconds=0.1,
            error=OtherIngestFailure("other failure"),
        ),
        "d.cr2": ExtractResult(
            path=input_paths[3], success=True, output_path=tmp_path / "d.provenance.json", elapsed_seconds=0.1
        ),
    }

    service = MetadataExtractionService(clock_fn=_clock)

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        return response_by_name[req.input_path.name]

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path))

    expected_exit_keys = [code.name for code in IngestExitCode if code != IngestExitCode.SUCCESS]
    assert list(result.summary_counts["by_exit_code"].keys()) == expected_exit_keys
    assert result.summary_counts["total"] == 4
    assert result.summary_counts["success"] == 2
    assert result.summary_counts["failure"] == 2
    assert result.summary_counts["by_exit_code"]["BIT_DEPTH_VIOLATION"] == 1
    assert result.summary_counts["by_exit_code"]["OTHER_FAILURE"] == 1


def test_batch_extract_preserves_relative_directory_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    nested_a = input_root / "a"
    nested_b = input_root / "b"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    first = nested_a / "image.cr2"
    second = nested_b / "image.cr2"
    first.touch()
    second.touch()

    output_dir = tmp_path / "out"
    service = MetadataExtractionService(clock_fn=_clock)
    output_paths: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        assert req.output_path is not None
        output_paths.append(req.output_path)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=output_dir,
            input_root=input_root,
            preserve_structure=True,
        )
    )

    assert result.success
    assert output_paths == [output_dir / "a" / "image.provenance.json", output_dir / "b" / "image.provenance.json"]


def test_batch_extract_duplicate_stem_nested_dirs_do_not_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    first_dir = input_root / "first"
    second_dir = input_root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "image.cr2"
    second = second_dir / "image.cr2"
    first.touch()
    second.touch()

    output_dir = tmp_path / "out"
    service = MetadataExtractionService(clock_fn=_clock)
    output_paths: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        assert req.output_path is not None
        output_paths.append(req.output_path)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=output_dir,
            input_root=input_root,
            preserve_structure=True,
            deterministic_order=False,
        )
    )

    assert result.success
    assert len(output_paths) == 2
    assert len(set(output_paths)) == 2
    assert output_paths == [
        output_dir / "first" / "image.provenance.json",
        output_dir / "second" / "image.provenance.json",
    ]


def test_batch_extract_infers_input_root_for_structure_preservation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    nested_a = input_root / "a"
    nested_b = input_root / "b"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    first = nested_a / "image.cr2"
    second = nested_b / "image.cr2"
    first.touch()
    second.touch()

    output_dir = tmp_path / "out"
    service = MetadataExtractionService(clock_fn=_clock)
    output_paths: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        assert req.output_path is not None
        output_paths.append(req.output_path)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=output_dir,
            preserve_structure=True,
        )
    )

    assert result.success
    assert output_paths == [output_dir / "a" / "image.provenance.json", output_dir / "b" / "image.provenance.json"]


def test_batch_extract_disambiguates_same_stem_collisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir(parents=True)
    first = input_root / "image.cr2"
    second = input_root / "image.jpg"
    first.touch()
    second.touch()

    output_dir = tmp_path / "out"
    service = MetadataExtractionService(clock_fn=_clock)
    output_paths: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        assert req.output_path is not None
        output_paths.append(req.output_path)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=output_dir,
            input_root=input_root,
            preserve_structure=True,
            deterministic_order=False,
        )
    )

    assert result.success
    assert output_paths == [
        output_dir / "image.cr2.provenance.json",
        output_dir / "image.jpg.provenance.json",
    ]


def test_batch_extract_disambiguates_collisions_without_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    nested_a = input_root / "a"
    nested_b = input_root / "b"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    first = nested_a / "image.cr2"
    second = nested_b / "image.cr2"
    first.touch()
    second.touch()

    output_dir = tmp_path / "out"
    service = MetadataExtractionService(clock_fn=_clock)
    output_paths: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        assert req.output_path is not None
        output_paths.append(req.output_path)
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=output_dir,
            preserve_structure=False,
            deterministic_order=False,
        )
    )

    assert result.success
    assert output_paths[0] == output_dir / "image.cr2.provenance.json"
    assert output_paths[1] != output_paths[0]
    assert output_paths[1].name.startswith("image.cr2.")
    assert output_paths[1].name.endswith(".provenance.json")


def test_batch_extract_fail_fast_stops_after_first_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "one.cr2", tmp_path / "two.cr2", tmp_path / "three.cr2"]
    for path in input_paths:
        path.touch()

    service = MetadataExtractionService(clock_fn=_clock)
    calls: list[Path] = []

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        calls.append(req.input_path)
        if req.input_path.name == "two.cr2":
            return ExtractResult(
                path=req.input_path,
                success=False,
                output_path=req.output_path,
                elapsed_seconds=0.1,
                error=OtherIngestFailure("stop here"),
            )
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=input_paths,
            output_dir=tmp_path,
            deterministic_order=False,
            fail_fast=True,
        )
    )

    assert calls == [tmp_path / "one.cr2", tmp_path / "two.cr2"]
    assert result.summary_counts["total"] == 2
    assert result.summary_counts["success"] == 1
    assert result.summary_counts["failure"] == 1
    assert not result.success


def test_extract_wraps_unknown_exception_as_other_ingest_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "input.cr2"
    input_path.touch()

    def capture_raises(**_: object) -> object:
        raise RuntimeError("boom")

    service = MetadataExtractionService(
        capture_provenance_fn=capture_raises,
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=tmp_path))

    assert not result.success
    assert result.error is not None
    assert isinstance(result.error, OtherIngestFailure)
    assert "boom" in str(result.error)


def test_batch_extract_sorts_items_when_deterministic_order_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "z.cr2", tmp_path / "a.cr2", tmp_path / "m.cr2"]
    for path in input_paths:
        path.touch()

    service = MetadataExtractionService(clock_fn=_clock)

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    result = service.batch_extract(BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path, deterministic_order=True))

    assert [item.path for item in result.items] == sorted(input_paths, key=lambda path: str(path))


def test_batch_extract_result_items_order_is_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_paths = [tmp_path / "z.cr2", tmp_path / "a.cr2", tmp_path / "m.cr2"]
    for path in input_paths:
        path.touch()

    service = MetadataExtractionService(clock_fn=_clock)

    def spy_extract(req: ExtractRequest) -> ExtractResult:
        return ExtractResult(
            path=req.input_path,
            success=True,
            output_path=req.output_path,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(service, "extract", spy_extract)

    first = service.batch_extract(BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path, deterministic_order=True))
    second = service.batch_extract(BatchExtractRequest(input_paths=input_paths, output_dir=tmp_path, deterministic_order=True))

    assert [item.path for item in first.items] == [item.path for item in second.items]
    assert [item.path for item in first.items] == sorted(input_paths, key=lambda path: str(path))


def test_validate_returns_typed_result_with_aggregated_dominance() -> None:
    expected_errors = [SchemaValidationFailure("schema issue"), SchemaDriftFailure("drift issue")]
    service = MetadataExtractionService(validate_schema_errors_fn=lambda **_: expected_errors, clock_fn=_clock)

    result = service.validate(ValidateRequest(sidecar_path=Path("/tmp/sidecar.json")))

    assert not result.success
    assert result.errors == expected_errors
    assert isinstance(result.dominant_error, SchemaDriftFailure)
