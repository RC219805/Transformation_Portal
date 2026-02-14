"""Integration tests for orchestrator backend registry integration.

Tests that the orchestrator correctly uses the DepthBackendRegistry
and implements fallback logic.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.depth.backends.protocol import LicenseRestrictionError
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Mark all tests as ML tier - they test backend registry behavior with real backends
pytestmark = pytest.mark.ml


@pytest.fixture
def mock_da3_available():
    """Mock DA3Backend.ensure_available() to succeed in offline CI."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.ensure_available"):
        yield


def test_orchestrator_uses_registry(tmp_path, mock_da3_available):
    """Orchestrator uses DepthBackendRegistry."""
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert hasattr(orchestrator, "depth_backend")
    assert orchestrator.depth_backend.name == "da3"


def test_orchestrator_default_backend(tmp_path, mock_da3_available):
    """Orchestrator defaults to DA3 if no backend specified."""
    config = EnhanceConfig(
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.resolution_status == "success"


def test_orchestrator_fallback_logic(tmp_path):
    """Orchestrator falls back to DA3 if requested backend unavailable."""
    config = EnhanceConfig(
        depth_backend="nonexistent_backend",
        depth_device="cpu",
        enable_v2=False,
    )

    # Should raise ValueError for unknown backend
    with pytest.raises(ValueError, match="Unknown depth backend"):
        EnhanceOrchestrator(config, tmp_path)


def test_orchestrator_backend_metadata_capture(tmp_path, mock_da3_available):
    """Orchestrator captures backend selection metadata."""
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    metadata = orchestrator._capture_backend_metadata()

    assert metadata.requested_backend == "da3"
    assert metadata.resolved_backend == "da3"
    assert metadata.resolution_status == "success"
    assert metadata.device == "cpu"


@pytest.mark.skipif(
    not Path("checkpoints/depth_pro.pt").exists(),
    reason="Depth Pro checkpoint not available",
)
def test_orchestrator_depth_pro_selection(tmp_path):
    """Orchestrator selects Depth Pro when available and licensed."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        depth_pro_checkpoint_path="checkpoints/depth_pro.pt",
        accept_apple_depth_pro_research_license=True,
        non_commercial_ok=True,
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "depth_pro"
    assert orchestrator._backend_metadata.resolution_status == "success"


def test_orchestrator_depth_pro_license_enforcement(tmp_path):
    """Orchestrator enforces license restrictions for Depth Pro."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        accept_apple_depth_pro_research_license=False,  # Not accepted
        non_commercial_ok=True,
        enable_v2=False,
    )

    with pytest.raises(LicenseRestrictionError):
        EnhanceOrchestrator(config, tmp_path)


def test_orchestrator_depth_pro_non_commercial_enforcement(tmp_path):
    """Orchestrator enforces non_commercial_ok for Depth Pro."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        accept_apple_depth_pro_research_license=True,
        non_commercial_ok=False,  # Not accepted
        enable_v2=False,
    )

    with pytest.raises(LicenseRestrictionError):
        EnhanceOrchestrator(config, tmp_path)


@pytest.mark.skipif(
    not Path("checkpoints/depth_pro.pt").exists(),
    reason="Depth Pro checkpoint not available",
)
def test_orchestrator_depth_pro_checkpoint_missing(tmp_path):
    """Orchestrator falls back to DA3 if Depth Pro checkpoint missing."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        depth_pro_checkpoint_path="checkpoints/nonexistent.pt",
        accept_apple_depth_pro_research_license=True,
        non_commercial_ok=True,
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Should fallback to DA3
    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.resolution_status == "fallback"
    assert "not found" in orchestrator._backend_metadata.resolution_reason


def test_depth_metadata_uses_resolved_backend_not_config_default(tmp_path, mock_da3_available):
    """REGRESSION TEST for ADR-023: depth.model must use resolved backend, not config default.

    Bug: Previously used self.config.model_variant.value.name which shows config default
    Fix: Now uses self._backend_metadata.resolved_backend which shows actual execution

    This prevents manifest mismatches like:
    - depth.model = "depth-anything-v3-metric-large" (config)
    - backend_selection.resolved_backend = "depth_pro" (reality)

    Critical for production debugging when fallbacks occur.
    """
    import json
    from unittest.mock import patch

    import numpy as np
    from PIL import Image

    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    # Create test image
    test_image = tmp_path / "test.png"
    img = Image.new("RGB", (64, 64), color="white")
    img.save(test_image)

    # Configure for DA3 backend
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        enable_materials_v3=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # Verify backend metadata was captured correctly
    assert orchestrator._backend_metadata.requested_backend == "da3"
    assert orchestrator._backend_metadata.resolved_backend == "da3"
    assert orchestrator._backend_metadata.resolution_status == "success"

    # Mock the depth backend compute to return synthetic result (fast test)
    mock_depth_result = DepthResult(
        depth_map=np.random.rand(64, 64).astype(np.float32),
        original_image=np.array(img),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )

    with patch.object(orchestrator.depth_backend, "compute", return_value=mock_depth_result):
        # Process single image to trigger depth metadata creation
        image_input = ImageInput(path=test_image)
        result = orchestrator.enhance_image(image_input)

    # Verify manifest was created
    manifest_path = result["manifest"]
    assert Path(manifest_path).exists()

    # Load and verify manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # CRITICAL ASSERTION: depth.model must match backend_selection.resolved_backend
    assert "depth" in manifest
    assert "backend_selection" in manifest

    depth_model = manifest["depth"]["model"]
    resolved_backend = manifest["backend_selection"]["resolved_backend"]

    # This is the regression test: they must match!
    assert depth_model == resolved_backend, (
        f"ADR-023 violation: depth.model='{depth_model}' != "
        f"backend_selection.resolved_backend='{resolved_backend}'. "
        f"Depth metadata must use resolved backend, not config default."
    )

    # For DA3 backend, both should be "da3"
    assert depth_model == "da3"
    assert resolved_backend == "da3"
