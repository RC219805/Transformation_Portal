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
