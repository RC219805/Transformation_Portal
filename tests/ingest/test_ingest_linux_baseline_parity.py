"""Linux baseline parity checks for normalized ingest sidecar payloads."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformation_portal.ingest.metadata_service import ExtractRequest, ExtractResult, MetadataExtractionService
from transformation_portal.ingest.normalize_machine_json import canonical_json_bytes, normalize_machine_payload
import pytest


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ingest" / "linux_baseline"


def _mime_type_for_suffix(path: Path) -> str:
    if path.suffix.lower() == ".mov":
        return "video/quicktime"
    return "image/x-adobe-dng"


def _write_dict_sidecar(sidecar: dict[str, Any], output_path: Path, fsync: bool = False) -> None:  # noqa: ARG001
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(sidecar))


def _fake_capture_provenance(
    *,
    input_path: Path,
    cli_args: list[str] | None = None,  # noqa: ARG001
    config_dict: dict[str, Any] | None = None,  # noqa: ARG001
    preset: str | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    file_bytes = input_path.read_bytes()
    return {
        "schema_version": "1.0.2",
        "file_integrity": {
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
            "size_bytes": len(file_bytes),
            "path": f"/fixtures/{input_path.name}",
            "mime_type": _mime_type_for_suffix(input_path),
        },
        "exif": {
            "all_tags": {
                "SourceFile": input_path.name,
                "File:FileType": input_path.suffix.lstrip(".").upper(),
            },
            "camera_make": "DJI",
            "camera_model": "Mavic 3",
        },
        "toolchain": [
            {"name": "python", "version": sys.version.split()[0]},
            {"name": "exiftool", "version": "dynamic-fixture"},
        ],
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": sys.version.split()[0],
            "arch": platform.machine(),
        },
        "timestamps": {
            "ingest_start": datetime.now(timezone.utc).isoformat(),
            "ingest_end": datetime.now(timezone.utc).isoformat(),
        },
        "pipeline_config": {
            "config_sha256": hashlib.sha256(f"linux-baseline:{input_path.name}".encode("utf-8")).hexdigest(),
            "preset": "linux-baseline",
        },
        "git_commit": "f" * 40,
        "run_id": str(uuid.uuid4()),
    }


def _extract_and_normalize(input_path: Path, output_path: Path) -> bytes:
    service = MetadataExtractionService(
        capture_provenance_fn=_fake_capture_provenance,
        write_sidecar_fn=_write_dict_sidecar,
        clock_fn=lambda: 1.0,
    )
    result: ExtractResult = service.extract(
        ExtractRequest(
            input_path=input_path,
            output_path=output_path,
            preset="linux-baseline",
            cli_args=["--determinism-parity-fixture"],
            config_dict={"mode": "linux-baseline"},
        )
    )

    assert result.success
    produced_payload = json.loads(output_path.read_text(encoding="utf-8"))
    normalized = normalize_machine_payload(produced_payload, profile="ingest_v1")
    return canonical_json_bytes(normalized)


def test_linux_baseline_parity_for_dng_and_mov(tmp_path: Path) -> None:
    fixtures = {
        "DJI_0018.DNG": b"fixture-dng-content\n",
        "DJI_0361.MOV": b"fixture-mov-content\n",
    }

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "sidecars"
    output_dir.mkdir(parents=True)

    for filename, file_bytes in fixtures.items():
        input_path = input_dir / filename
        input_path.write_bytes(file_bytes)

        normalized_bytes = _extract_and_normalize(input_path, output_dir / f"{filename}.provenance.json")
        expected_payload = json.loads((BASELINE_DIR / f"{filename}.normalized.json").read_text(encoding="utf-8"))
        expected_bytes = canonical_json_bytes(expected_payload)

        assert normalized_bytes == expected_bytes
