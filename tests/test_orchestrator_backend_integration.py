"""Integration tests for orchestrator backend registry integration.

Tests that the orchestrator correctly uses the DepthBackendRegistry
and implements fallback logic.
"""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from transformation_portal.depth.backends.protocol import LicenseRestrictionError
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import ApexStrictGateError, EnhanceOrchestrator

# Mark all tests as ML tier - they test backend registry behavior with real backends
pytestmark = pytest.mark.ml
DEPTH_PRO_PKG_AVAILABLE = importlib.util.find_spec("depth_pro") is not None


@pytest.fixture(name="mock_da3_available")
def fixture_mock_da3_available():
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
    not Path("checkpoints/depth_pro.pt").exists() or not DEPTH_PRO_PKG_AVAILABLE,
    reason="Depth Pro checkpoint or package not available",
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
    reason = (orchestrator._backend_metadata.resolution_reason or "").lower()
    assert ("not found" in reason) or ("not installed" in reason)


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


def _make_depth_result(width: int = 64, height: int = 64):
    """Create synthetic depth result for orchestrator fallback tests."""
    import numpy as np
    from PIL import Image

    from transformation_portal.depth.backends.protocol import DepthResult

    image = np.array(Image.new("RGB", (width, height), color="white"))
    depth = np.linspace(0.0, 1.0, width * height, dtype=np.float32).reshape(height, width)
    return DepthResult(
        depth_map=depth,
        original_image=image,
        metadata={},
        depth_units="relative",
        backend_id="mock",
        device="cpu",
    )


def test_runtime_operational_failure_falls_back_to_da2_with_attempt_provenance(tmp_path):
    """Operational failure should fallback to DA2 and persist attempts in metadata."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "runtime_fallback.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
    )

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.side_effect = RuntimeError("Torch not compiled with CUDA enabled")

    da2_backend = Mock()
    da2_backend.name = "da2"
    da2_backend.license_type = Mock(value="commercial")
    da2_backend.ensure_available.return_value = None
    da2_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "da3": da3_backend,
        "da2": da2_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["backend"] == "da2"
    assert len(result["attempts"]) >= 2
    assert result["attempts"][0]["failure_kind"] == "operational"
    assert result["attempts"][0]["error_code"] == "CUDA_HARDCODED_IN_BACKEND"
    assert result["attempts"][1]["status"] == "success"
    assert result["selected_attempt_index"] == 1
    selected_attempt_index = result["selected_attempt_index"]
    assert any(
        attempt.get("attempt") == selected_attempt_index and attempt.get("backend") == result["backend"]
        for attempt in result["attempts"]
    )

    manifest = json.loads(Path(result["manifest"]).read_text())
    backend_selection = manifest["backend_selection"]
    assert backend_selection["resolved_backend"] == "da2"
    assert backend_selection["resolution_status"] == "fallback"
    assert len(backend_selection["attempts"]) >= 2
    assert backend_selection["attempts"][0]["failure_kind"] == "operational"


def test_runtime_semantic_fallback_retries_when_enabled(tmp_path):
    """Semantic-gate failures should retry fallback backend when enabled."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "semantic_fallback.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        quality_tier="apex",
        allow_semantic_fallback=True,
    )

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = _make_depth_result()

    da2_backend = Mock()
    da2_backend.name = "da2"
    da2_backend.license_type = Mock(value="commercial")
    da2_backend.ensure_available.return_value = None
    da2_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "da3": da3_backend,
        "da2": da2_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with patch.object(
            orchestrator,
            "_enforce_apex_depth_validity_gate",
            side_effect=[
                ApexStrictGateError(
                    "APEX_DEPTH_PLATEAU",
                    "APEX depth validity gate failed: APEX_DEPTH_PLATEAU",
                    details={"passed": False},
                ),
                {"passed": True, "failure_codes": [], "warnings": [], "metrics": {}, "thresholds": {}},
            ],
        ):
            result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["backend"] == "da2"
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["failure_kind"] == "semantic"
    assert result["attempts"][0]["error_code"] == "APEX_DEPTH_PLATEAU"
    assert result["attempts"][1]["status"] == "success"
    assert result["selected_attempt_index"] == 1
    selected_attempt_index = result["selected_attempt_index"]
    assert any(
        attempt.get("attempt") == selected_attempt_index and attempt.get("backend") == result["backend"]
        for attempt in result["attempts"]
    )

    manifest = json.loads(Path(result["manifest"]).read_text())
    backend_selection = manifest["backend_selection"]
    assert backend_selection["resolved_backend"] == "da2"
    assert backend_selection["resolution_status"] == "fallback"
    assert backend_selection["attempts"][0]["failure_kind"] == "semantic"


def test_runtime_multilevel_operational_fallback_chain_is_deterministic(tmp_path):
    """Depth Pro -> DA3 -> DA2 fallback should produce deterministic attempt indices [0,1,2]."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "multilevel_fallback.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        enable_v2=False,
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )

    depth_pro_backend = Mock()
    depth_pro_backend.name = "depth_pro"
    depth_pro_backend.license_type = Mock(value="research_only")
    depth_pro_backend.ensure_available.return_value = None
    depth_pro_backend.compute.side_effect = FileNotFoundError("depth_pro checkpoint not found")

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.side_effect = RuntimeError("Torch not compiled with CUDA enabled")

    da2_backend = Mock()
    da2_backend.name = "da2"
    da2_backend.license_type = Mock(value="commercial")
    da2_backend.ensure_available.return_value = None
    da2_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "depth_pro": depth_pro_backend,
        "da3": da3_backend,
        "da2": da2_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["backend"] == "da2"
    assert [attempt["attempt"] for attempt in result["attempts"]] == [0, 1, 2]
    assert [attempt["backend"] for attempt in result["attempts"]] == ["depth_pro", "da3", "da2"]
    assert result["selected_attempt_index"] == 2

    manifest = json.loads(Path(result["manifest"]).read_text())
    attempts = manifest["backend_selection"]["attempts"]
    assert [attempt["attempt"] for attempt in attempts] == [0, 1, 2]
    assert attempts[0]["failure_kind"] == "operational"
    assert attempts[1]["failure_kind"] == "operational"
    assert attempts[2]["status"] == "success"
