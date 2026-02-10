"""Integration tests for Materials V3 with orchestrator.

Tests that Materials V3 Engine is properly wired into the orchestrator
and processes images when enabled.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
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
    segmentation_result = {"materials": {}}
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

    # Verify pixel ops structure (should be telemetry even with empty materials)
    pixel_ops = result["materials_v3_pixel_ops"]
    assert "enabled" in pixel_ops
    assert "applied" in pixel_ops
    assert "blocked" in pixel_ops
    assert "timing_ms" in pixel_ops


def test_materials_v3_manifest_integration(tmp_path, mock_depth_backend, mock_da3_available):
    """Test that Materials V3 results can be stored in manifest."""
    from transformation_portal.lux_depth_v3.manifest import CombinedManifest, MaterialsV3Metadata

    # Create Materials V3 metadata
    materials_v3_metadata = MaterialsV3Metadata(
        enabled=True,
        version="3.1",
        response_plan={"per_class": {}},
        pixel_ops={"enabled": True, "applied": [], "blocked": []},
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
    assert loaded_manifest.materials_v3.schema_version == "1.0"


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
