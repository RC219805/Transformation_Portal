"""Sidecar file I/O is UTF-8 regardless of the process locale (finding D-12).

``ProvenanceSidecar.to_json_deterministic`` keeps non-ASCII characters and the
ingest contract promises UTF-8 sidecars, so the bytes on disk must not depend on
the locale encoding. The tests simulate an ASCII locale by giving the module an
``open`` whose default encoding is ASCII, which is what a C / POSIX locale
without UTF-8 mode gives the builtin.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest import sidecar as sidecar_module
from transformation_portal.ingest import validator as validator_module
from transformation_portal.ingest.schemas import (
    ExifMetadata,
    FileIntegrity,
    HostEnvironment,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
)

pytestmark = pytest.mark.unit


def _ascii_locale_open(file: Any, mode: str = "r", *args: Any, encoding: str | None = None, **kwargs: Any) -> Any:
    if "b" not in mode and encoding is None:
        encoding = "ascii"
    return builtins.open(file, mode, *args, encoding=encoding, **kwargs)


def _non_ascii_sidecar() -> ProvenanceSidecar:
    return ProvenanceSidecar(
        file_integrity=FileIntegrity(sha256="a" * 64, size_bytes=1, path="/in/x.cr2"),
        exif=ExifMetadata(all_tags={"EXIF:Artist": "José Müller", "EXIF:Copyright": "© 2026 山田"}),
        toolchain=[],
        host=HostEnvironment(hostname="h", os="Linux", os_version="1", python_version="3.11", arch="x86_64"),
        timestamps=IngestTimestamps(ingest_start="2026-01-01T00:00:00+00:00", ingest_end="2026-01-01T00:00:01+00:00"),
        pipeline_config=PipelineConfig(config_sha256="b" * 64),
        run_id="r",
    )


def test_write_sidecar_writes_utf8_under_ascii_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sidecar_module, "open", _ascii_locale_open, raising=False)
    sidecar = _non_ascii_sidecar()
    out = tmp_path / "x.provenance.json"

    sidecar_module.write_sidecar(sidecar, out)

    assert out.read_bytes() == sidecar.to_json_deterministic().encode("utf-8")
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_load_sidecar_reads_utf8_under_ascii_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sidecar = _non_ascii_sidecar()
    out = tmp_path / "x.provenance.json"
    out.write_bytes(sidecar.to_json_deterministic().encode("utf-8"))
    monkeypatch.setattr(sidecar_module, "open", _ascii_locale_open, raising=False)

    loaded = sidecar_module.load_sidecar(out)

    assert loaded.exif.all_tags["EXIF:Copyright"] == "© 2026 山田"
    assert loaded.to_json_deterministic() == sidecar.to_json_deterministic()


def test_validate_schema_reads_utf8_sidecar_under_ascii_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sidecar = _non_ascii_sidecar()
    out = tmp_path / "x.provenance.json"
    out.write_bytes(sidecar.to_json_deterministic().encode("utf-8"))
    monkeypatch.setattr(validator_module, "open", _ascii_locale_open, raising=False)

    assert validator_module.validate_schema(out) == []
