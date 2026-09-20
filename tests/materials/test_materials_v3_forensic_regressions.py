"""Regressions from the Materials V3 evidence and execution boundary audit."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.materials_v3 import MaterialsV3Engine, authorized_v2_material_masks
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.pixel_ops_executor import _resolve_overlaps, apply_pixel_ops
from transformation_portal.lux_depth_v3.segmentation import _cache
from transformation_portal.lux_depth_v3.segmentation.sam2 import SAM2SegmentationBackend

pytestmark = pytest.mark.unit


def _config(**overrides):
    options = {
        "enable_materials_v3": True,
        "apply_pixel_ops": True,
        "mask_feather_sigma_default": 0.0,
        "pixel_ops_low_tex_min_bbox_frac": 2.0,
    }
    options.update(overrides)
    return EnhanceConfig(**options)


def _process(materials, config=None, image=None):
    if image is None:
        image = np.full((64, 64, 3), 0.5, np.float32)
    return MaterialsV3Engine(config or _config()).process(image, {"materials": materials})


@pytest.mark.parametrize("material", ["wood", "metal", "fabric", "stucco", "unknown"])
def test_unsupported_material_is_a_total_no_implementation_decision(material):
    result = _process({material: (np.ones((64, 64), np.float32), 0.99)})
    assert result["materials_v3_pixel_ops"]["applied"] == []
    assert result["materials_v3_pixel_ops"]["blocked"][0]["reason"] == "no_implementation"
    np.testing.assert_array_equal(result["enhanced_image"], np.full((64, 64, 3), 0.5, np.float32))


def test_delta_telemetry_measures_actual_changes_and_full_feather_write_scope():
    image = np.full((64, 64, 3), 0.5, np.float32)
    mask = np.zeros((64, 64), np.float32)
    mask[6:58, 6:58] = 1
    result = _process({"water": (mask, 0.99)}, _config(mask_feather_sigma_default=2.0), image)
    delta = np.abs(result["enhanced_image"] - image)
    stats = result["materials_v3_pixel_ops"]["applied"][0]["delta_stats"]
    assert stats["inside_mask_mean_abs"] == pytest.approx(float(delta[mask > 0.5].mean()), abs=5e-7)
    assert stats["outside_mask_mean_abs"] == pytest.approx(float(delta[mask <= 0.5].mean()), abs=5e-7)
    assert stats["inside_mask_mean_abs"] > 0
    assert stats["outside_mask_mean_abs"] > 0


def test_stock_water_operation_obeys_actual_soft_mask_p99_ceiling():
    image = np.full((64, 64, 3), 0.9, np.float32)
    mask = np.full((64, 64), 0.6, np.float32)
    config = _config(pixel_ops_low_tex_min_bbox_frac=0.05, pixel_ops_low_tex_delta_ceiling=0.04)
    result = _process({"water": (mask, 0.99)}, config, image)
    delta = np.abs(result["enhanced_image"] - image).max(axis=2)
    assert np.percentile(delta, 99) <= 0.04 + 1e-7
    assert result["materials_v3_pixel_ops"]["applied"][0]["low_tex_guard"]["delta_scale_applied"] < 1.0


def test_post_overlap_support_is_rechecked_without_reauthorizing_uncertain_region():
    sky = np.ones((64, 64), np.float32)
    sky[0, 0] = 0
    water = np.ones((64, 64), np.float32)
    result = _process({"sky": (sky, 0.01), "water": (water, 0.99)})
    blocked = {entry["material"]: entry["reason"] for entry in result["materials_v3_pixel_ops"]["blocked"]}
    assert blocked == {"sky": "below_confidence_threshold", "water": "below_coverage_threshold"}
    assert result["materials_v3_pixel_ops"]["applied"] == []
    np.testing.assert_array_equal(result["enhanced_image"], np.full((64, 64, 3), 0.5, np.float32))
    assert authorized_v2_material_masks(result["material_masks"], result, _config()) == {}


def test_stale_plan_bbox_cannot_truncate_current_mask_support():
    image = np.full((64, 64, 3), 0.5, np.float32)
    mask = np.zeros((64, 64), np.float32)
    mask[8:56, 8:56] = 1
    config = _config()
    initial = _process({"water": (mask, 0.99)}, config, image)
    plan = copy.deepcopy(initial["materials_v3_response_plan"])
    plan["per_class"]["water"]["bbox"] = [8, 8, 12, 12]
    output, _ = apply_pixel_ops(image, {"materials": {"water": mask}}, plan, config)
    assert np.count_nonzero(np.any(output != image, axis=2)) == 48 * 48
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])


def test_equal_priority_overlap_resolution_is_mapping_order_independent():
    mask = np.ones((8, 8), np.float32)
    metadata = {"a": {"priority": 1}, "b": {"priority": 1}}
    first, _ = _resolve_overlaps({"a": mask, "b": mask}, metadata, mask.shape)
    second, _ = _resolve_overlaps({"b": mask, "a": mask}, metadata, mask.shape)
    for name in metadata:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize(
    "condition",
    ["low_confidence", "missing_confidence", "disabled", "missing_receipt", "unexecuted", "invalid_operation", "bool_score"],
)
def test_v2_projection_cannot_reauthorize_refused_or_missing_evidence(condition):
    config = _config(apply_pixel_ops=condition != "disabled")
    mask = np.ones((64, 64), np.float32)
    value = mask if condition == "missing_confidence" else (mask, 0.01 if condition == "low_confidence" else 0.99)
    result = _process({"glass": value}, config)
    if condition == "missing_receipt":
        result.pop("materials_v3_response_plan")
    elif condition == "unexecuted":
        result["materials_v3_pixel_ops"]["applied"] = []
    elif condition == "invalid_operation":
        result["materials_v3_pixel_ops"]["applied"][0]["ops"] = ["unimplemented"]
    elif condition == "bool_score":
        result["materials_v3_response_plan"]["per_class"]["glass"]["material_confidence"] = True
    assert authorized_v2_material_masks(result["material_masks"], result, config) == {}
    np.testing.assert_array_equal(result["material_masks"]["glass"], mask)


def _orchestrator(tmp_path, config):
    orchestrator = EnhanceOrchestrator.__new__(EnhanceOrchestrator)
    orchestrator.config = config
    orchestrator.output_root = tmp_path
    orchestrator.v2_dir = tmp_path / "v2"
    orchestrator.depth_dir = tmp_path / "depth"
    orchestrator.should_skip_v2 = Mock(return_value=False)
    orchestrator._enforce_v2_depth_handoff = Mock()
    orchestrator.v2_runner = Mock()
    return orchestrator


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("confidence", [0.01, 0.99])
def test_orchestrator_projects_authorization_without_rewriting_segmentation_artifact(tmp_path, restored, confidence):
    config = _config(enable_v2=True)
    orchestrator = _orchestrator(tmp_path, config)
    result = _process({"glass": (np.ones((64, 64), np.float32), confidence)}, config)
    artifact = orchestrator._serialize_material_masks(result["material_masks"], Path("photo"), tmp_path / "segmentation")
    before = artifact.read_bytes()
    result["materials_v3_metadata"]["segmentation_metadata"] = {"mask_artifact_path": str(artifact)}
    if restored:
        result.pop("material_masks")
    captured = {}

    def run(**kwargs):
        path = kwargs["masks_file"]
        captured["path"] = path
        if path is not None:
            assert path != artifact
            with np.load(path, allow_pickle=False) as archive:
                captured["masks"] = {name: archive[name] for name in archive.files}
        return {"status": "ok", "runtime_s": 0.0}

    orchestrator.v2_runner.run.side_effect = run
    orchestrator._run_v2_stage(
        ImageInput(path=tmp_path / "source.tif"),
        depth_path=None,
        output_key=Path("photo"),
        v2_log_path=tmp_path / "v2.log",
        manifest_path=tmp_path / "manifest.json",
        skip_depth=False,
        materials_v3_result=result,
    )
    assert artifact.read_bytes() == before
    if confidence < 0.5:
        assert captured["path"] is None
    else:
        assert set(captured["masks"]) == {"glass"}
        assert not captured["path"].exists()


@pytest.mark.parametrize("confidence", [None, float("nan"), -1.0, 2.0, True])
def test_sam2_missing_semantic_score_never_borrows_geometric_confidence(monkeypatch, confidence):
    backend = SAM2SegmentationBackend()
    backend._model_loaded = True
    backend._sam2_backend = SimpleNamespace(
        segment=lambda _request: SimpleNamespace(
            masks=np.ones((1, 32, 32), dtype=bool),
            scores=np.array([0.99], np.float32),
            metadata=[SimpleNamespace(bbox=(0, 0, 32, 32), material_label="glass", material_confidence=confidence)],
        )
    )
    monkeypatch.setattr(backend, "_resolve_effective_tiling", lambda _image: (None, False))
    monkeypatch.setattr(backend, "_record_runtime_metadata", lambda *_args, **_kwargs: None)
    masks = backend.segment(np.full((32, 32, 3), 128, np.uint8))
    assert masks["glass"][1] == 0.0
    evidence = backend._material_confidence_evidence["glass"]
    assert evidence["confidence_score_type"] == "missing_material_confidence"
    assert evidence["calibration_version"] is None
    result = MaterialsV3Engine(_config()).process(
        np.full((32, 32, 3), 0.5, np.float32),
        {"materials": masks, "material_confidence_evidence": backend._material_confidence_evidence},
    )
    assert result["materials_v3_pixel_ops"]["applied"] == []


def test_legacy_segmentation_cache_schema_cannot_restore_promoted_semantic_scores(tmp_path):
    key = "a" * 64
    payload = {"schema_version": _cache.SEGMENTATION_CACHE_SCHEMA_VERSION}
    _cache._write_cached_material_masks(
        cache_dir=tmp_path,
        cache_key=key,
        key_payload=payload,
        results={"glass": (np.ones((8, 8), np.float32), 0.99)},
        runtime_metadata={},
    )
    assert _cache._read_cached_material_masks(cache_dir=tmp_path, cache_key=key, expected_payload=payload) is not None
    _masks_path, metadata_path = _cache._segmentation_cache_paths(tmp_path, key)
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = "materials-segmentation-cache.v1"
    metadata_path.write_text(json.dumps(metadata))
    assert _cache._read_cached_material_masks(cache_dir=tmp_path, cache_key=key, expected_payload=payload) is None


def test_sam2_class_union_cannot_borrow_confidence_from_a_different_region(monkeypatch):
    backend = SAM2SegmentationBackend()
    backend._model_loaded = True
    masks = np.zeros((2, 32, 32), dtype=bool)
    masks[0, :16] = True
    masks[1, 16:] = True
    backend._sam2_backend = SimpleNamespace(
        segment=lambda _request: SimpleNamespace(
            masks=masks,
            scores=np.array([0.99, 0.99], np.float32),
            metadata=[
                SimpleNamespace(bbox=(0, 0, 32, 16), material_label="glass", material_confidence=0.99),
                SimpleNamespace(bbox=(0, 16, 32, 16), material_label="glass", material_confidence=None),
            ],
        )
    )
    monkeypatch.setattr(backend, "_resolve_effective_tiling", lambda _image: (None, False))
    monkeypatch.setattr(backend, "_record_runtime_metadata", lambda *_args, **_kwargs: None)
    result = backend.segment(np.full((32, 32, 3), 128, np.uint8))
    np.testing.assert_array_equal(result["glass"][0], np.ones((32, 32), np.float32))
    assert result["glass"][1] == 0.0
    assert backend._material_confidence_evidence["glass"]["confidence_score_type"] == "missing_material_confidence"


@pytest.mark.parametrize("limit, expected_shape", [(100, [64, 64]), (4096, [32, 32])])
def test_restored_masks_obey_prepared_geometry_before_array_allocation(tmp_path, monkeypatch, limit, expected_shape):
    orchestrator = _orchestrator(tmp_path, _config())
    artifact = orchestrator._serialize_material_masks(
        {"glass": np.ones((64, 64), np.float32)}, Path("photo"), tmp_path / "segmentation"
    )
    orchestrator._prepared_execution = SimpleNamespace(
        plan=SimpleNamespace(input_limits=SimpleNamespace(max_decoded_pixels_per_input=limit))
    )

    def unexpected_allocation(*_args, **_kwargs):
        pytest.fail("Rejected geometry must not construct an array")

    monkeypatch.setattr(np, "frombuffer", unexpected_allocation)
    assert (
        orchestrator._load_persisted_material_masks_for_v2(
            {
                "materials_v3_metadata": {
                    "segmentation_metadata": {"mask_artifact_path": str(artifact), "mask_artifact_shape": expected_shape}
                }
            }
        )
        == {}
    )


@pytest.mark.parametrize("version, length_bytes", [((1, 0), 2), ((2, 0), 4)])
def test_restored_masks_reject_oversized_header_before_numpy_reads_it(tmp_path, monkeypatch, version, length_bytes):
    orchestrator = _orchestrator(tmp_path, _config())
    artifact = tmp_path / "hostile.npz"
    payload = b"\x93NUMPY" + bytes(version) + (60000).to_bytes(length_bytes, "little")
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("glass.npy", payload)

    def unexpected_header_read(*_args, **_kwargs):
        pytest.fail("Oversized header reached the NumPy parser")

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", unexpected_header_read)
    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", unexpected_header_read)
    assert (
        orchestrator._load_persisted_material_masks_for_v2(
            {"materials_v3_metadata": {"segmentation_metadata": {"mask_artifact_path": str(artifact)}}}
        )
        == {}
    )
