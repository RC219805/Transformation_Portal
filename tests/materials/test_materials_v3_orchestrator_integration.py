"""Integration tests for Materials V3 with orchestrator.

Tests that Materials V3 Engine is properly wired into the orchestrator
and processes images when enabled.
"""

# pytest fixture injection uses function args that match fixture names.
# pylint: disable=redefined-outer-name

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


@pytest.fixture
def mock_depth_backend():
    """Mock depth backend to avoid ML dependencies in integration tests."""
    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
        yield


@pytest.fixture
def mock_da3_available():
    """Mock DA3Backend.ensure_available() to succeed in offline CI."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.ensure_available"):
        yield


def test_materials_v3_engine_initialization_when_enabled(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that MaterialsV3Engine is initialized when enable_materials_v3=True."""
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Check that Materials V3 engine was initialized
    assert hasattr(orchestrator, "materials_v3_engine")
    assert orchestrator.materials_v3_engine is not None
    assert orchestrator.materials_v3_engine.config == config


def test_materials_v3_engine_not_initialized_when_disabled(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that MaterialsV3Engine is not initialized when enable_materials_v3=False."""
    config = EnhanceConfig(
        enable_materials_v3=False,
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Check that Materials V3 engine was not initialized
    assert hasattr(orchestrator, "materials_v3_engine")
    assert orchestrator.materials_v3_engine is None


def test_materials_v3_process_integration(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that Materials V3 process method can be called with expected inputs."""
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Create mock inputs
    image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    segmentation_result = {
        "materials": {},
        "segmentation_metadata": {
            "clip_runtime": {
                "offline_mode": True,
                "weights_source": "cache_path",
                "weights_sha256": "a" * 64,
            }
        },
    }
    depth_map = np.ones((256, 256), dtype=np.float32) * 0.5

    # Call the Materials V3 engine directly
    result = orchestrator.materials_v3_engine.process(
        image=image, segmentation_result=segmentation_result, depth_map=depth_map
    )

    # Verify result structure
    assert isinstance(result, dict)
    assert "materials_v3_response_plan" in result
    assert "materials_v3_pixel_ops" in result
    assert "materials_v3_metadata" in result

    # Verify metadata
    assert result["materials_v3_metadata"]["version"] == "3.1"
    assert result["materials_v3_metadata"]["segmentation_metadata"]["clip_runtime"]["weights_source"] == "cache_path"

    # Verify pixel ops structure (should be telemetry even with empty materials)
    pixel_ops = result["materials_v3_pixel_ops"]
    assert "enabled" in pixel_ops
    assert "applied" in pixel_ops
    assert "blocked" in pixel_ops
    assert "timing_ms" in pixel_ops


def test_run_materials_v3_stage_persists_mask_artifact_and_sets_metadata(tmp_path, mock_depth_backend, mock_da3_available):
    """Materials V3 stage should persist segmentation mask artifacts and merge metadata."""
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    preprocessed_array = np.zeros((8, 8, 3), dtype=np.float32)
    depth_map = np.ones((8, 8), dtype=np.float32)
    glass_mask = np.ones((8, 8), dtype=np.float32)
    output_key = Path("nested/image_01")

    materials_result = {
        "enhanced_image": None,
        "materials_v3_response_plan": {"per_class": {}},
        "materials_v3_pixel_ops": {"applied": [], "blocked": []},
        "materials_v3_metadata": {
            "version": "3.1",
            "segmentation_metadata": {"clip_runtime": {"weights_source": "cache_path"}},
        },
        "material_masks": {"glass": glass_mask},
    }

    with patch(
        "transformation_portal.lux_depth_v3.segmentation_backend.segment_materials", return_value={"glass": glass_mask}
    ):
        with patch(
            "transformation_portal.lux_depth_v3.segmentation_backend.get_last_segmentation_runtime_metadata",
            return_value={"clip_runtime": {"weights_source": "cache_path"}},
        ):
            with patch.object(orchestrator.materials_v3_engine, "process", return_value=materials_result):
                result, _, enhanced_path = orchestrator._run_materials_v3_stage(
                    preprocessed_array=preprocessed_array,
                    depth_map=depth_map,
                    output_key=output_key,
                )

    assert result is materials_result
    assert enhanced_path is None

    segmentation_metadata = result["materials_v3_metadata"]["segmentation_metadata"]
    assert segmentation_metadata["clip_runtime"]["weights_source"] == "cache_path"
    assert segmentation_metadata["mask_artifact_format"] == "npz"

    mask_artifact_path = Path(segmentation_metadata["mask_artifact_path"])
    assert mask_artifact_path == orchestrator._segmentation_mask_artifact_path(output_key)
    assert mask_artifact_path.exists()

    with np.load(mask_artifact_path) as data:
        assert set(data.files) == {"glass"}
        loaded_mask = np.asarray(data["glass"])

    assert loaded_mask.shape == glass_mask.shape
    assert loaded_mask.dtype == np.float32
    assert np.array_equal(loaded_mask, glass_mask)


def test_materials_v3_manifest_integration(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that Materials V3 results can be stored in manifest."""
    from transformation_portal.lux_depth_v3.manifest import CombinedManifest, MaterialsV3Metadata

    # Create Materials V3 metadata
    materials_v3_metadata = MaterialsV3Metadata(
        enabled=True,
        version="3.1",
        response_plan={"per_class": {}},
        pixel_ops={"enabled": True, "applied": [], "blocked": []},
        segmentation_metadata={"clip_runtime": {"offline_mode": True, "weights_source": "cache_path"}},
        runtime_seconds=0.123,
    )

    # Create and save manifest
    manifest = CombinedManifest(materials_v3=materials_v3_metadata)

    manifest_path = tmp_path / "test_manifest.json"
    manifest.save(manifest_path)

    # Load and verify
    loaded_manifest = CombinedManifest.load(manifest_path)

    assert loaded_manifest.materials_v3 is not None
    assert loaded_manifest.materials_v3.enabled is True
    assert loaded_manifest.materials_v3.version == "3.1"
    assert loaded_manifest.materials_v3.runtime_seconds == 0.123
    assert loaded_manifest.materials_v3.segmentation_metadata is not None
    assert loaded_manifest.materials_v3.segmentation_metadata["clip_runtime"]["offline_mode"] is True
    assert loaded_manifest.materials_v3.schema_version == "1.1"  # Updated to 1.1 for bit depth tracking


def test_materials_v3_disabled_returns_empty(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that Materials V3 engine is not initialized when disabled."""
    config = EnhanceConfig(
        enable_materials_v3=False,  # Materials V3 disabled
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # When enable_materials_v3=False, the engine should not be initialized
    assert orchestrator.materials_v3_engine is None


def test_materials_v3_masks_exposed_to_v2():
    """Verify material masks are exposed in Materials V3 result for future V2 integration.

    This test verifies that:
    1. MaterialsV3Engine.process() returns material_masks in the result
    2. The masks are properly formatted (dict mapping material names to numpy arrays)
    3. Infrastructure is ready for V2 subprocess integration (future work)

    Note: Full V2 integration requires mask serialization (see _run_v2_stage comments).
    """
    from transformation_portal.lux_depth_v3.materials_v3 import MaterialsV3Engine

    # Create minimal config
    config_mock = MagicMock()
    config_mock.enabled = True
    config_mock.enable_materials_v3 = True
    config_mock.apply_pixel_ops = True
    config_mock.min_coverage_px = 100
    config_mock.min_mean_conf = 0.2
    config_mock.refinement_strategy = "canary"
    config_mock.glass_response_enabled = True

    engine = MaterialsV3Engine(config_mock)

    # Create test inputs with material masks
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100
    glass_mask = np.zeros((64, 64), dtype=np.float32)
    glass_mask[10:50, 10:50] = 0.8

    segmentation_result = {
        "materials": {
            "glass": glass_mask,
        }
    }

    # Process and get result
    result = engine.process(image, segmentation_result, depth_map=None)

    # Verify material_masks are exposed
    assert "material_masks" in result, "material_masks should be in result"
    assert isinstance(result["material_masks"], dict), "material_masks should be a dict"
    assert "glass" in result["material_masks"], "glass mask should be in material_masks"

    # Verify mask is the same as input
    assert np.array_equal(result["material_masks"]["glass"], glass_mask)

    # Verify other expected keys are present
    assert "materials_v3_response_plan" in result
    assert "materials_v3_pixel_ops" in result
    assert "materials_v3_metadata" in result


def test_apex_strict_gate_requires_segmentation_enabled(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX + Materials V3 must explicitly enable segmentation."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=False,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    with pytest.raises(RuntimeError, match="enable-segmentation"):
        orchestrator._enforce_apex_materials_gate()


def test_apex_strict_gate_rejects_stub_backend(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX + Materials V3 must not use stub segmentation backend."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="stub",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    with pytest.raises(RuntimeError, match="segmentation-backend efficientsam"):
        orchestrator._enforce_apex_materials_gate()


def test_apex_strict_gate_requires_strict_backend(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX + Materials V3 must run strict segmentation mode."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=False,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    with pytest.raises(RuntimeError, match="strict-segmentation"):
        orchestrator._enforce_apex_materials_gate()


def test_apex_strict_gate_accepts_sam2_backend_without_stub(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX + Materials V3 should allow SAM2 backend when strict mode is enabled."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # No segmentation_result provided: this should validate config-only gates and pass.
    orchestrator._enforce_apex_materials_gate()


def test_apex_strict_gate_requires_non_empty_material_masks(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX + Materials V3 should fail when segmentation returns no materials."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    with pytest.raises(RuntimeError, match="no material masks"):
        orchestrator._enforce_apex_materials_gate({"materials": {}})


def test_apex_strict_gate_not_applied_outside_apex(tmp_path, mock_depth_backend, mock_da3_available):
    """Standard tier should not enforce apex-only gate constraints."""
    config = EnhanceConfig(
        quality_tier="standard",
        enable_materials_v3=True,
        enable_material_segmentation=False,
        material_segmentation_backend="stub",
        strict_backend=False,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Should not raise outside apex tier
    orchestrator._enforce_apex_materials_gate({"materials": {}})


def test_apex_v2_preflight_rejects_non_canonical_fastpath(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX strict mode should fail before V2 when cached fast-path drifts from canonical stem."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth_path = tmp_path / "depth" / "image_depth.png"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.write_bytes(b"depth")

    non_canonical_input = tmp_path / "input" / "image.png"
    non_canonical_input.parent.mkdir(parents=True, exist_ok=True)
    non_canonical_input.write_bytes(b"raw")

    with pytest.raises(RuntimeError, match="fast-path stem divergence"):
        orchestrator._enforce_apex_v2_canonical_input_preflight(
            depth_path=depth_path,
            output_key=Path("image_01"),
            v2_input_path=non_canonical_input,
            enhanced_image_path=None,
            materials_v3_result={"materials_v3_metadata": {"version": "3.1"}},
        )


def test_apex_v2_preflight_accepts_canonical_handoff(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX strict mode should allow V2 preflight only with canonical enhanced input + masks."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    output_key = Path("image_01")
    depth_path = tmp_path / "depth" / "image_depth.png"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.write_bytes(b"depth")

    expected_path = orchestrator._expected_materials_v3_enhanced_path(output_key)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_bytes(b"enhanced")

    orchestrator._enforce_apex_v2_canonical_input_preflight(
        depth_path=depth_path,
        output_key=output_key,
        v2_input_path=expected_path,
        enhanced_image_path=expected_path,
        materials_v3_result={"material_masks": {"glass": np.ones((2, 2), dtype=np.float32)}},
    )


def test_apex_cached_depth_recomputes_materials_for_canonical_handoff(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX strict mode should recompute Materials V3 when cached depth lacks canonical handoff artifacts."""
    config = EnhanceConfig(
        quality_tier="apex",
        enable_materials_v3=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    input_path = tmp_path / "inputs" / "image.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"input")

    output_key = Path("image_01")
    depth_path = tmp_path / "depth" / "image_01_depth.png"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.write_bytes(b"depth")
    float_depth_path = tmp_path / "depth" / "image_01_depth.npy"
    np.save(float_depth_path, np.ones((4, 4), dtype=np.float32))

    expected_path = orchestrator._expected_materials_v3_enhanced_path(output_key)
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    recomputed_result = {
        "material_masks": {"glass": np.ones((4, 4), dtype=np.float32)},
        "materials_v3_metadata": {"version": "3.1"},
    }

    def _run_stage(*, preprocessed_array, depth_map, output_key):  # noqa: ARG001
        expected_path.write_bytes(b"enhanced")
        return recomputed_result, 0.321, expected_path

    with patch("transformation_portal.lux_depth_v3.preprocessing.validate_image_format", return_value=input_path):
        with patch(
            "transformation_portal.lux_depth_v3.preprocessing.preprocess_image",
            return_value=(np.zeros((4, 4, 3), dtype=np.float32), (4, 4)),
        ):
            with patch.object(orchestrator, "_load_cached_depth", return_value=np.ones((4, 4), dtype=np.float32)):
                with patch.object(orchestrator, "_run_materials_v3_stage", side_effect=_run_stage) as run_stage:
                    result, runtime_s, enhanced_path = orchestrator._ensure_apex_canonical_materials_execution(
                        image_input=ImageInput(path=input_path),
                        output_key=output_key,
                        depth_path=depth_path,
                        float_depth_path=float_depth_path,
                        materials_v3_result={"materials_v3_metadata": {"version": "3.1"}},
                        materials_v3_runtime_s=0.0,
                        enhanced_image_path=None,
                    )

    assert result is recomputed_result
    assert runtime_s == pytest.approx(0.321, rel=1e-6, abs=1e-6)
    assert enhanced_path == expected_path
    assert run_stage.call_count == 1


def test_apex_depth_validity_gate_rejects_upper_quartile_plateau(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX depth gate should fail on p95≈p75 plateau in upper quantiles."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.concatenate(
        [
            np.linspace(0.05, 0.65, 70 * 100, dtype=np.float32),
            np.full((30 * 100,), 0.7549, dtype=np.float32),
        ]
    ).reshape(100, 100)

    with pytest.raises(RuntimeError, match="APEX_DEPTH_PLATEAU"):
        orchestrator._enforce_apex_depth_validity_gate(depth)


def test_apex_depth_validity_gate_rejects_high_saturation(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX depth gate should fail when too many pixels saturate near high end."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.linspace(0.0, 0.95, 100 * 100, dtype=np.float32).reshape(100, 100)
    depth[:20, :] = 1.0  # 20% saturation > default 2% threshold

    with pytest.raises(RuntimeError, match="APEX_DEPTH_SATURATION_HIGH"):
        orchestrator._enforce_apex_depth_validity_gate(depth)


def test_apex_depth_validity_gate_rejects_low_saturation(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX depth gate should fail when too many pixels collapse near zero."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.linspace(0.05, 1.0, 100 * 100, dtype=np.float32).reshape(100, 100)
    depth[:25, :] = 0.0  # 25% low-end saturation > default 2% threshold

    with pytest.raises(RuntimeError, match="APEX_DEPTH_SATURATION_LOW"):
        orchestrator._enforce_apex_depth_validity_gate(depth)


def test_apex_depth_validity_gate_rejects_nonfinite(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX depth gate should fail when finite percentage is below floor."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.linspace(0.1, 0.9, 100 * 100, dtype=np.float32).reshape(100, 100)
    depth[:2, :] = np.nan  # 2% NaN => finite_pct 0.98 < 0.999

    with pytest.raises(RuntimeError, match="APEX_DEPTH_NONFINITE"):
        orchestrator._enforce_apex_depth_validity_gate(depth)


def test_apex_depth_validity_gate_normalizes_metric_depth_for_saturation_checks(
    tmp_path, mock_depth_backend, mock_da3_available
):
    """Metric-depth ranges should be normalized before saturation checks in APEX gate."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Metric-style depth in meters (not pre-normalized to [0,1]).
    depth = np.linspace(20.0, 40.0, 100 * 100, dtype=np.float32).reshape(100, 100)

    verdict = orchestrator._enforce_apex_depth_validity_gate(depth, depth_units="meters")
    assert verdict is not None
    assert verdict["passed"] is True
    assert verdict["metrics"]["source_unit"] == "meters"
    assert verdict["metrics"]["gate_unit"] == "relative_0_1"
    assert verdict["metrics"]["saturation_high_fraction"] < config.apex_depth_max_high_saturation_fraction


def test_apex_depth_validity_gate_preserves_relative_depth_semantics(tmp_path, mock_depth_backend, mock_da3_available):
    """Relative depth should keep identity normalization to preserve threshold semantics."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.linspace(0.0, 1.0, 100 * 100, dtype=np.float32).reshape(100, 100)
    verdict = orchestrator._enforce_apex_depth_validity_gate(depth, depth_units="relative")
    assert verdict is not None
    assert verdict["passed"] is True
    assert verdict["metrics"]["source_unit"] == "relative"
    assert verdict["metrics"]["gate_normalization"]["scaled"] is False
    assert verdict["metrics"]["gate_normalization"]["mode"] == "identity_relative"


def test_apex_depth_validity_gate_returns_thresholds_and_metrics_on_pass(tmp_path, mock_depth_backend, mock_da3_available):
    """APEX depth gate should emit structured decision payload on success."""
    config = EnhanceConfig(
        quality_tier="apex",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    h, w = 256, 256
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    depth = np.clip(0.08 + 0.84 * (0.55 * y + 0.45 * x), 0.08, 0.92).astype(np.float32)

    verdict = orchestrator._enforce_apex_depth_validity_gate(depth)
    assert verdict is not None
    assert verdict["passed"] is True
    assert "thresholds" in verdict
    assert "metrics" in verdict
    assert "failure_codes" in verdict
    assert verdict["failure_codes"] == []


def test_apex_depth_validity_gate_noop_outside_apex(tmp_path, mock_depth_backend, mock_da3_available):
    """Depth validity gate should be inactive for non-APEX tiers."""
    config = EnhanceConfig(
        quality_tier="standard",
        depth_device="cpu",
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    depth = np.ones((32, 32), dtype=np.float32)
    metrics = orchestrator._enforce_apex_depth_validity_gate(depth)
    assert metrics is None
