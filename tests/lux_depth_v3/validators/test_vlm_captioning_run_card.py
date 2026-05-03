from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.manifest import compute_file_sha256
from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

pytestmark = pytest.mark.unit


def _config_fingerprint() -> dict[str, Any]:
    payload = {
        "model_variant": "METRIC_LARGE",
        "depth_quantization": "u16",
        "depth_device": "cpu",
        "preset": "premium",
        "preset_requested": "premium",
        "preset_resolved": "premium",
        "backend_requested": "da3",
        "backend_resolved": "da3",
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "quality_tier": "premium",
        "strict_inputs": False,
        "strict_segmentation": False,
        "apex_strict_mode": False,
        "v2_preset": "premium",
        "v2_device": "cpu",
        "v2_upscaler_backend": "realesrgan",
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "hash_algorithm": "sha256",
        "canonical_json": canonical_json,
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def _write_run_card(tmp_path: Path, *, used_for_quality_gate: bool) -> Path:
    sidecar_path = tmp_path / "captioning" / "image.vlm_captioning.sidecar.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text('{"vlm_captioning":{"role":"advisory"}}\n', encoding="utf-8")
    artifact_index = [
        {
            "artifact_type": "vlm_caption_sidecar",
            "path": "captioning/image.vlm_captioning.sidecar.json",
            "relative_path": "captioning/image.vlm_captioning.sidecar.json",
            "size_bytes": sidecar_path.stat().st_size,
            "sha256": compute_file_sha256(sidecar_path),
        }
    ]
    status = {
        "enabled": True,
        "backend": "fastvlm",
        "model_role": "default",
        "model_id": "apple/FastVLM-1.5B-int8",
        "role": "advisory",
        "sidecar_count": 1,
        "failed_count": 0,
        "used_for_quality_gate": used_for_quality_gate,
    }
    run_card = {
        "run_card_version": "v1",
        "batch_id": "2026-05-02_120000",
        "start_time": "2026-05-02T12:00:00Z",
        "end_time": "2026-05-02T12:01:00Z",
        "config_fingerprint": _config_fingerprint(),
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3-METRIC-LARGE-1.0",
        },
        "backend_summary": {
            "requested_backend": "da3",
            "primary_backend": "da3",
            "final_backends_used": ["da3"],
            "fallback_images": 0,
            "semantic_fallback_images": 0,
            "operational_fallback_images": 0,
        },
        "inputs": [],
        "effective_config": {},
        "result_summary": [
            {
                "image": "image.tif",
                "status": "ok",
                "backend": "da3",
                "runtime_s": 1.0,
                "captioning_status": status,
            }
        ],
        "captioning_status": status,
        "environment": {
            "python_version": "3.12.0",
            "platform": "darwin",
            "machine": "arm64",
            "hostname": "test-host",
        },
        "git_revision": {"v3": "test", "v2": "test"},
        "runtime_stats": {
            "count": 1,
            "total": 1.0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
        },
        "outliers": [],
        "total_images": 1,
        "success_count": 1,
        "error_count": 0,
        "artifact_index": artifact_index,
        "artifact_merkle_root": compute_artifact_merkle_root(artifact_index),
    }
    run_card_path = tmp_path / "run_card.json"
    run_card_path.write_text(json.dumps(run_card, indent=2, sort_keys=True), encoding="utf-8")
    return run_card_path


def test_vlm_captioning_status_false_passes_integrity(tmp_path: Path) -> None:
    run_card_path = _write_run_card(tmp_path, used_for_quality_gate=False)

    assert verify_run_card_integrity(run_card_path) == []


def test_vlm_captioning_status_true_fails_closed(tmp_path: Path) -> None:
    run_card_path = _write_run_card(tmp_path, used_for_quality_gate=True)

    errors = verify_run_card_integrity(run_card_path)

    assert any("captioning_status.used_for_quality_gate must be false" in error for error in errors)
    assert any("result_summary[0].captioning_status.used_for_quality_gate must be false" in error for error in errors)
