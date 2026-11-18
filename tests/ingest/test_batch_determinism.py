"""Determinism tests for mixed-media ingest batch normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest.batch import BATCH_MANIFEST_FILENAME, compute_batch_root_sha256, run_ingest_batch
from transformation_portal.ingest.normalize_machine_json import canonical_json_bytes

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_INPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ingest" / "batch_inputs"


def _fake_ingest_payload_factory(run_marker: str):
    def _factory(input_path: Path) -> dict[str, Any]:
        file_bytes = input_path.read_bytes()
        source_name = input_path.name
        return {
            "schema_version": "1.0.2",
            "file_integrity": {
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
                "size_bytes": len(file_bytes),
                "path": f"/batch/{source_name}",
                "mime_type": "application/octet-stream",
            },
            "exif": {
                "all_tags": {
                    "SourceFile": source_name,
                }
            },
            "pipeline_config": {
                "config_sha256": hashlib.sha256(f"batch:{source_name}".encode("utf-8")).hexdigest(),
                "preset": "batch-determinism",
            },
            "toolchain": [{"name": "python", "version": run_marker}],
            "host": {
                "hostname": f"host-{run_marker}",
                "os": "Linux",
                "os_version": "volatile",
                "python_version": run_marker,
                "arch": "x86_64",
            },
            "timestamps": {
                "ingest_start": f"{run_marker}-start",
                "ingest_end": f"{run_marker}-end",
            },
            "git_commit": hashlib.sha256(run_marker.encode("utf-8")).hexdigest()[:40],
            "run_id": run_marker,
        }

    return _factory


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_mixed_media_batch_determinism_run_twice_identical(tmp_path: Path) -> None:
    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"

    manifest_a = run_ingest_batch(
        input_dir=FIXTURE_INPUT_DIR,
        output_dir=run_a_dir,
        profile="ingest_v1",
        ingest_payload_factory=_fake_ingest_payload_factory("run_a"),
    )
    manifest_b = run_ingest_batch(
        input_dir=FIXTURE_INPUT_DIR,
        output_dir=run_b_dir,
        profile="ingest_v1",
        ingest_payload_factory=_fake_ingest_payload_factory("run_b"),
    )

    persisted_manifest_a = _load_manifest(run_a_dir / BATCH_MANIFEST_FILENAME)
    persisted_manifest_b = _load_manifest(run_b_dir / BATCH_MANIFEST_FILENAME)

    assert manifest_a == manifest_b
    assert persisted_manifest_a == persisted_manifest_b
    assert manifest_a["batch_root_sha256"] == manifest_b["batch_root_sha256"]

    hashes_a = [item["normalized_json_sha256"] for item in manifest_a["items"]]
    hashes_b = [item["normalized_json_sha256"] for item in manifest_b["items"]]
    assert hashes_a == hashes_b

    for item in manifest_a["items"]:
        relpath = Path(item["normalized_json_relpath"])
        output_a = run_a_dir / relpath
        output_b = run_b_dir / relpath
        assert output_a.read_bytes() == output_b.read_bytes()
        assert item["normalized_json_sha256"] == hashlib.sha256(output_a.read_bytes()).hexdigest()

    assert canonical_json_bytes(persisted_manifest_a) == canonical_json_bytes(persisted_manifest_b)


def test_failed_rerun_keeps_previous_outputs_consistent(tmp_path: Path) -> None:
    """D-19: an ingest failure must not leave a manifest describing rewritten files."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "a.jpg").write_bytes(b"a1")
    (input_dir / "b.jpg").write_bytes(b"b1")
    output_dir = tmp_path / "out"
    good_factory = _fake_ingest_payload_factory("run-a")

    run_ingest_batch(input_dir=input_dir, output_dir=output_dir, ingest_payload_factory=good_factory)
    (input_dir / "a.jpg").write_bytes(b"a2-changed")

    def failing_factory(input_path: Path) -> dict[str, Any]:
        if input_path.name == "b.jpg":
            raise RuntimeError("exiftool extraction failed")
        return good_factory(input_path)

    with pytest.raises(RuntimeError, match="exiftool extraction failed"):
        run_ingest_batch(input_dir=input_dir, output_dir=output_dir, ingest_payload_factory=failing_factory)

    manifest = _load_manifest(output_dir / BATCH_MANIFEST_FILENAME)
    for item in manifest["items"]:
        on_disk = (output_dir / item["normalized_json_relpath"]).read_bytes()
        assert hashlib.sha256(on_disk).hexdigest() == item["normalized_json_sha256"]
    assert sorted(path.name for path in output_dir.rglob("*.tmp")) == []


def test_failed_write_withdraws_previous_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-19: once rewriting starts, a write failure leaves no stale manifest behind."""
    import transformation_portal.ingest.batch as batch_module

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "a.jpg").write_bytes(b"a1")
    output_dir = tmp_path / "out"
    factory = _fake_ingest_payload_factory("run-a")
    run_ingest_batch(input_dir=input_dir, output_dir=output_dir, ingest_payload_factory=factory)
    (input_dir / "a.jpg").write_bytes(b"a2-changed")

    def failing_write(output_path: Path, data: bytes) -> Path:
        raise OSError(f"disk full writing {output_path.name}")

    monkeypatch.setattr(batch_module, "atomic_write_bytes", failing_write, raising=False)
    monkeypatch.setattr(Path, "write_bytes", lambda self, data: failing_write(self, data))

    with pytest.raises(OSError, match="disk full"):
        run_ingest_batch(input_dir=input_dir, output_dir=output_dir, ingest_payload_factory=factory)

    assert not (output_dir / BATCH_MANIFEST_FILENAME).exists()


def test_compute_batch_root_sha256_is_order_independent() -> None:
    items_a = [
        {"relative_path": "b/file.mov", "normalized_json_sha256": "2" * 64},
        {"relative_path": "a/file.dng", "normalized_json_sha256": "1" * 64},
    ]
    items_b = list(reversed(items_a))

    assert compute_batch_root_sha256(items_a, profile="ingest_v1") == compute_batch_root_sha256(
        items_b,
        profile="ingest_v1",
    )
