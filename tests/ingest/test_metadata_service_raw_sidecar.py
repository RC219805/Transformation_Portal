"""Tests for RAW-sidecar integration in MetadataExtractionService."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.ingest.errors import OtherIngestFailure
from transformation_portal.ingest.metadata_service import BatchExtractRequest, ExtractRequest, MetadataExtractionService
from transformation_portal.ingest.raw_sidecar import RawSidecarResult

pytestmark = pytest.mark.unit


def _clock() -> float:
    return 100.0


def test_extract_generates_raw_sidecar_for_raw_inputs_and_records_path(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr2"
    input_path.touch()
    output_dir = tmp_path / "out"

    written_paths: list[Path] = []
    raw_sidecar_calls: list[tuple[Path, Path, bool, str | None, int | None]] = []

    def fake_write_sidecar(_sidecar: object, output_path: Path, fsync: bool = False) -> None:
        written_paths.append(output_path)

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append((input_path_arg, output_path, fsync, file_sha256, file_size_bytes))
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
            rawpy_error=None,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=fake_write_sidecar,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=output_dir))

    assert result.success is True
    assert result.output_path == output_dir / "sample.provenance.json"
    assert result.raw_sidecar_path == output_dir / "sample.raw.sidecar.json"
    assert result.raw_sidecar_error is None
    assert written_paths == [output_dir / "sample.provenance.json"]
    assert raw_sidecar_calls == [(input_path, output_dir / "sample.raw.sidecar.json", False, None, None)]


def test_extract_non_raw_input_does_not_attempt_raw_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jpg"
    input_path.touch()
    raw_sidecar_calls: list[Path] = []

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append(input_path_arg)
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=tmp_path / "out"))

    assert result.success is True
    assert result.raw_sidecar_path is None
    assert result.raw_sidecar_error is None
    assert raw_sidecar_calls == []


def test_extract_can_disable_raw_sidecar_generation(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.touch()
    raw_sidecar_calls: list[Path] = []

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append(input_path_arg)
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_dir=tmp_path / "out",
            emit_raw_sidecar=False,
        )
    )

    assert result.success is True
    assert result.raw_sidecar_path is None
    assert result.raw_sidecar_error is None
    assert raw_sidecar_calls == []


def test_extract_raw_sidecar_failure_soft_fails_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.touch()

    def fail_generate_raw_sidecar(
        _input_path: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raise RuntimeError(f"boom: {output_path.name}")

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fail_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=tmp_path / "out"))

    assert result.success is True
    assert result.raw_sidecar_path is None
    assert result.raw_sidecar_error is not None
    assert "boom: sample.raw.sidecar.json" in result.raw_sidecar_error


def test_extract_raw_sidecar_failure_can_be_strict_and_writes_no_provenance(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.touch()
    output_dir = tmp_path / "out"
    written_paths: list[Path] = []

    def fake_write_sidecar(_sidecar: object, output_path: Path, fsync: bool = False) -> None:
        written_paths.append(output_path)
        output_path.write_text("should-not-exist", encoding="utf-8")

    def fail_generate_raw_sidecar(
        _input_path: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raise RuntimeError("strict failure")

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=fake_write_sidecar,
        generate_raw_sidecar_fn=fail_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_dir=output_dir,
            raw_sidecar_strict=True,
        )
    )

    assert result.success is False
    assert isinstance(result.error, OtherIngestFailure)
    assert "RAW sidecar generation failed" in str(result.error)
    assert result.raw_sidecar_path is None
    assert result.raw_sidecar_error == "strict failure"
    assert written_paths == []
    assert not (output_dir / "sample.provenance.json").exists()


def test_extract_provenance_write_failure_removes_generated_raw_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.touch()
    output_dir = tmp_path / "out"

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
            rawpy_error=None,
        )

    def fake_write_sidecar_raises(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=fake_write_sidecar_raises,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_dir=output_dir,
            raw_sidecar_strict=True,
        )
    )

    assert result.success is False
    assert isinstance(result.error, OtherIngestFailure)
    assert result.raw_sidecar_path is None
    assert not (output_dir / "sample.raw.sidecar.json").exists()


def test_extract_reuses_precomputed_file_integrity_for_raw_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.write_bytes(b"raw-bytes")
    output_dir = tmp_path / "out"
    raw_sidecar_calls: list[tuple[str | None, int | None]] = []

    class _FileIntegrity:
        sha256 = "a" * 64
        size_bytes = 12345

    class _Sidecar:
        file_integrity = _FileIntegrity()

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append((file_sha256, file_size_bytes))
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: _Sidecar(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=output_dir))

    assert result.success is True
    assert raw_sidecar_calls == [("a" * 64, 12345)]


def test_extract_reuses_precomputed_exiftool_metadata_for_raw_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.write_bytes(b"raw-bytes")
    output_dir = tmp_path / "out"
    raw_sidecar_calls: list[tuple[dict[str, object] | None, str | None]] = []

    class _Exif:
        all_tags = {"EXIF:ISO": 100}

    class _ToolchainEntry:
        name = "exiftool"
        version = "13.55"

    class _Sidecar:
        exif = _Exif()
        toolchain = [_ToolchainEntry()]

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append((precomputed_exiftool_payload, precomputed_exiftool_version))
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: _Sidecar(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(ExtractRequest(input_path=input_path, output_dir=output_dir))

    assert result.success is True
    assert raw_sidecar_calls == [({"EXIF:ISO": 100}, "13.55")]


def test_extract_respects_explicit_raw_sidecar_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr3"
    input_path.touch()
    explicit_output_path = tmp_path / "custom" / "sample.custom.raw.sidecar.json"
    raw_sidecar_calls: list[Path] = []

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append(output_path)
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_dir=tmp_path / "out",
            raw_sidecar_output_path=explicit_output_path,
        )
    )

    assert result.success is True
    assert result.raw_sidecar_path == explicit_output_path
    assert raw_sidecar_calls == [explicit_output_path]


def test_extract_derives_raw_sidecar_name_from_custom_provenance_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr3"
    input_path.touch()
    custom_provenance_output_path = tmp_path / "out" / "custom.prov.json"
    raw_sidecar_calls: list[Path] = []

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        raw_sidecar_calls.append(output_path)
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_path=custom_provenance_output_path,
        )
    )

    assert result.success is True
    assert result.output_path == custom_provenance_output_path
    assert result.raw_sidecar_path == tmp_path / "out" / "custom.prov.raw.sidecar.json"
    assert raw_sidecar_calls == [tmp_path / "out" / "custom.prov.raw.sidecar.json"]


def test_batch_extract_preserves_disambiguated_raw_sidecar_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    first_dir = input_root / "first"
    second_dir = input_root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "image.cr2"
    second = second_dir / "image.cr2"
    first.touch()
    second.touch()

    def fake_generate_raw_sidecar(
        input_path_arg: Path,
        *,
        output_path: Path,
        fsync: bool = False,
        file_sha256: str | None = None,
        file_size_bytes: int | None = None,
        precomputed_exiftool_payload: dict[str, object] | None = None,
        precomputed_exiftool_version: str | None = None,
    ) -> RawSidecarResult:
        return RawSidecarResult(
            input_path=input_path_arg,
            output_path=output_path,
            rawpy_available=True,
            rawpy_ok=True,
        )

    service = MetadataExtractionService(
        capture_provenance_fn=lambda **_: object(),
        write_sidecar_fn=lambda *_args, **_kwargs: None,
        generate_raw_sidecar_fn=fake_generate_raw_sidecar,
        clock_fn=_clock,
    )

    result = service.batch_extract(
        BatchExtractRequest(
            input_paths=[first, second],
            output_dir=tmp_path / "out",
            preserve_structure=False,
            deterministic_order=False,
        )
    )

    assert result.success is True
    assert len(result.items) == 2
    raw_names = [item.raw_sidecar_path.name if item.raw_sidecar_path is not None else None for item in result.items]
    assert raw_names[0] == "image.cr2.raw.sidecar.json"
    assert raw_names[1] is not None
    assert raw_names[1] != raw_names[0]
    assert raw_names[1].startswith("image.cr2.")
    assert raw_names[1].endswith(".raw.sidecar.json")
