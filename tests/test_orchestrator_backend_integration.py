"""Integration tests for orchestrator backend registry integration.

Tests that the orchestrator correctly uses the DepthBackendRegistry
and implements fallback logic.
"""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from transformation_portal.core.platform_matrix import PlatformAccel, PlatformISA, PlatformMatrix, PlatformOS
from transformation_portal.depth.backends.protocol import LicenseRestrictionError
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
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
        non_commercial_ok=True,
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert hasattr(orchestrator, "depth_backend")
    assert orchestrator.depth_backend.name == "da3"


def test_orchestrator_default_backend(tmp_path, mock_da3_available):
    """Orchestrator defaults to DA3 if no backend specified."""
    config = EnhanceConfig(
        depth_device="cpu",
        non_commercial_ok=True,
        enable_v2=False,
    )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.resolution_status == "success"


def test_orchestrator_explicit_da3_auto_discovers_repo_runtime(tmp_path, monkeypatch):
    """Explicit DA3 should persist the repo-local subprocess contract when available."""
    from transformation_portal.lux_depth_v3.config_resolver import REPO_LOCAL_DA3_PYTHON

    discovered_python = tmp_path / ".venv-da3" / "bin" / "python"
    discovered_python.parent.mkdir(parents=True)
    discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
        lambda: discovered_python,
    )

    backend_calls = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def ensure_available(self):
            return None

    def fake_get_backend(self, backend_name, config):
        del self
        backend_calls.append((backend_name, config.da3_python_executable))
        return FakeBackend(backend_name)

    with patch(
        "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
        new=fake_get_backend,
    ):
        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator.config.da3_python_executable == REPO_LOCAL_DA3_PYTHON
    assert backend_calls == [("da3", REPO_LOCAL_DA3_PYTHON)]


def test_orchestrator_explicit_depth_pro_auto_discovers_repo_runtime(tmp_path, monkeypatch):
    """Explicit Depth Pro should persist the repo-local subprocess contract when available."""
    from transformation_portal.lux_depth_v3.config_resolver import REPO_LOCAL_DEPTH_PRO_PYTHON

    discovered_python = tmp_path / ".venv-depth-pro" / "bin" / "python"
    discovered_python.parent.mkdir(parents=True)
    discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON", raising=False)
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.config_resolver._repo_local_depth_pro_python_path",
        lambda: discovered_python,
    )

    backend_calls = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def ensure_available(self):
            return None

    def fake_get_backend(self, backend_name, config):
        del self
        backend_calls.append((backend_name, config.depth_pro_python_executable))
        return FakeBackend(backend_name)

    with patch(
        "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
        new=fake_get_backend,
    ):
        config = EnhanceConfig(
            depth_backend="depth_pro",
            depth_device="cpu",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            enable_v2=False,
        )

        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "depth_pro"
    assert orchestrator.config.depth_pro_python_executable == REPO_LOCAL_DEPTH_PRO_PYTHON
    assert backend_calls == [("depth_pro", REPO_LOCAL_DEPTH_PRO_PYTHON)]


def test_orchestrator_auto_discovers_repo_raw_runtime(tmp_path, monkeypatch):
    """Orchestrator should persist the repo-local RAW subprocess contract when available."""
    from transformation_portal.lux_depth_v3.config_resolver import REPO_LOCAL_RAW_PYTHON

    discovered_python = tmp_path / ".venv-raw" / "bin" / "python"
    discovered_python.parent.mkdir(parents=True)
    discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
        lambda: discovered_python,
    )

    class FakeBackend:
        name = "da3"

        def ensure_available(self):
            return None

    with patch(
        "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
        return_value=FakeBackend(),
    ):
        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )
        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.config.raw_python_executable == REPO_LOCAL_RAW_PYTHON


def test_orchestrator_explicit_da3_unavailable_fails_without_fallback(tmp_path):
    """Explicit DA3 should fail fast instead of silently selecting DA2."""
    backend_calls = []

    class FakeBackend:
        def __init__(self, name, error=None):
            self.name = name
            self._error = error

        def ensure_available(self):
            if self._error is not None:
                raise self._error
            return None

    def fake_get_backend(self, backend_name, config):
        del self, config
        backend_calls.append(backend_name)
        if backend_name == "da3":
            return FakeBackend(
                "da3",
                ImportError("DA3 subprocess environment is not ready."),
            )
        return FakeBackend(backend_name)

    with patch(
        "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
        new=fake_get_backend,
    ):
        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        with pytest.raises(ImportError, match="DA3 subprocess environment is not ready"):
            EnhanceOrchestrator(config, tmp_path)

    assert backend_calls == ["da3"]


def test_orchestrator_default_backend_falls_back_to_da2_when_da3_unavailable(tmp_path):
    """Auto/default backend selection should still fall back to DA2."""
    backend_calls = []

    class FakeBackend:
        def __init__(self, name, error=None):
            self.name = name
            self._error = error

        def ensure_available(self):
            if self._error is not None:
                raise self._error
            return None

    def fake_get_backend(self, backend_name, config):
        del self, config
        backend_calls.append(backend_name)
        if backend_name == "da3":
            return FakeBackend("da3", ImportError("depth_anything_3 package not installed"))
        return FakeBackend("da2")

    with patch(
        "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
        new=fake_get_backend,
    ):
        config = EnhanceConfig(
            depth_device="cpu",
            enable_v2=False,
        )

        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert backend_calls == ["da3", "da2"]
    assert orchestrator.depth_backend.name == "da2"
    assert orchestrator._backend_metadata.requested_backend == "da3"
    assert orchestrator._backend_metadata.resolved_backend == "da2"
    assert orchestrator._backend_metadata.resolution_status == "fallback"


def test_orchestrator_prefers_depth_pro_only_on_apple_silicon(tmp_path):
    """Apple Silicon should auto-select Depth Pro only when explicitly opted in."""
    backend_calls = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def ensure_available(self):
            return None

    def fake_get_backend(self, backend_name, config):
        del self, config
        backend_calls.append(backend_name)
        return FakeBackend(backend_name)

    with (
        patch(
            "transformation_portal.lux_depth_v3.pipeline_coordinator.CURRENT_PLATFORM",
            PlatformMatrix(PlatformOS.DARWIN, PlatformISA.ARM64, PlatformAccel.MPS),
        ),
        patch(
            "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
            new=fake_get_backend,
        ),
    ):
        config = EnhanceConfig(
            depth_device="cpu",
            depth_pro_python_executable="/usr/bin/python3",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            enable_v2=False,
        )

        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert backend_calls[0] == "depth_pro"
    assert orchestrator.depth_backend.name == "depth_pro"
    assert orchestrator._backend_metadata.requested_backend == "depth_pro"
    assert orchestrator._backend_metadata.resolved_backend == "depth_pro"


def test_orchestrator_keeps_da3_default_off_apple_silicon(tmp_path):
    """Intel macOS should not inherit the Apple Silicon Depth Pro auto-promotion."""
    backend_calls = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def ensure_available(self):
            return None

    def fake_get_backend(self, backend_name, config):
        del self, config
        backend_calls.append(backend_name)
        return FakeBackend(backend_name)

    with (
        patch(
            "transformation_portal.lux_depth_v3.pipeline_coordinator.CURRENT_PLATFORM",
            PlatformMatrix(PlatformOS.DARWIN, PlatformISA.X86_64, PlatformAccel.CPU),
        ),
        patch(
            "transformation_portal.depth.backends.registry.DepthBackendRegistry.get_backend",
            new=fake_get_backend,
        ),
    ):
        config = EnhanceConfig(
            depth_device="cpu",
            depth_pro_python_executable="/usr/bin/python3",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            enable_v2=False,
            use_coreml_backend=False,
        )

        orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert backend_calls[0] == "da3"
    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.requested_backend == "da3"


def test_orchestrator_canonicalizes_legacy_backend_alias_in_metadata(tmp_path, mock_da3_available):
    """Legacy backend aliases should not leak into emitted backend metadata."""
    with pytest.warns(FutureWarning, match="depth_anything_v3"):
        config = EnhanceConfig(
            depth_backend="depth_anything_v3",
            depth_device="cpu",
            non_commercial_ok=True,
            enable_v2=False,
        )

    orchestrator = EnhanceOrchestrator(config, tmp_path)

    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.requested_backend == "da3"
    assert orchestrator._backend_metadata.resolved_backend == "da3"


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
        non_commercial_ok=True,
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
def test_orchestrator_depth_pro_checkpoint_missing(tmp_path, mock_da3_available):
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


def test_startup_selection_fallback_is_persisted_in_attempt_history(tmp_path):
    """Selection-time fallback should be visible in manifest attempt history."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "startup_fallback.png"
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
    depth_pro_backend.ensure_available.side_effect = ImportError("depth_pro package not installed in the active environment.")

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "depth_pro": depth_pro_backend,
        "da3": da3_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["backend"] == "da3"
    assert [attempt["backend"] for attempt in result["attempts"]] == ["depth_pro", "da3"]
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][0]["failure_kind"] == "operational"
    assert "depth_pro package not installed" in result["attempts"][0]["error_message"]
    assert result["attempts"][1]["status"] == "success"
    assert result["selected_attempt_index"] == 1

    manifest = json.loads(Path(result["manifest"]).read_text())
    backend_selection = manifest["backend_selection"]
    assert backend_selection["resolved_backend"] == "da3"
    assert backend_selection["resolution_status"] == "fallback"
    assert "Requested 'depth_pro' unavailable" in backend_selection["resolution_reason"]
    assert [attempt["backend"] for attempt in backend_selection["attempts"]] == ["depth_pro", "da3"]
    assert backend_selection["attempts"][0]["status"] == "failed"


def test_startup_selection_fallback_preserves_depth_pro_mps_unavailable_diagnostic(tmp_path):
    """Selection-time Depth Pro device failures should remain visible in fallback metadata."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "startup_mps_fallback.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="mps",
        enable_v2=False,
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )

    depth_pro_backend = Mock()
    depth_pro_backend.name = "depth_pro"
    depth_pro_backend.license_type = Mock(value="research_only")
    depth_pro_backend.ensure_available.side_effect = ImportError(
        '{"device":"mps","mps_available":false,"reason":"PyTorch MPS backend is not available in this runtime."}'
    )

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "depth_pro": depth_pro_backend,
        "da3": da3_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["backend"] == "da3"
    assert [attempt["backend"] for attempt in result["attempts"]] == ["depth_pro", "da3"]
    assert '"mps_available":false' in result["attempts"][0]["error_message"]

    manifest = json.loads(Path(result["manifest"]).read_text())
    backend_selection = manifest["backend_selection"]
    assert backend_selection["resolved_backend"] == "da3"
    assert backend_selection["resolution_status"] == "fallback"
    assert '"mps_available":false' in backend_selection["resolution_reason"]


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
        non_commercial_ok=True,
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


def test_get_or_create_depth_backend_prefers_active_instance_over_stale_cache(tmp_path, mock_da3_available):
    """When active backend matches backend id, orchestrator should prefer active instance."""
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        non_commercial_ok=True,
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    stale_cached = Mock()
    stale_cached.name = "da3"
    orchestrator._depth_backend_cache["da3"] = stale_cached

    active_backend = Mock()
    active_backend.name = "da3"
    orchestrator.depth_backend = active_backend

    resolved = orchestrator._get_or_create_depth_backend("da3")
    assert resolved is active_backend
    assert orchestrator._depth_backend_cache["da3"] is active_backend


def _make_depth_result(width: int = 64, height: int = 64):
    """Create synthetic depth result for orchestrator fallback tests."""
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


def test_enhance_image_reuses_initialized_backend_metadata_without_recapture(tmp_path, mock_da3_available):
    """enhance_image should not recapture backend metadata when metadata already exists."""
    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "backend_metadata_reuse.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        non_commercial_ok=True,
        enable_v2=False,
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)
    orchestrator.postprocessor = Mock(process=lambda result: result)

    with patch.object(orchestrator.depth_backend, "compute", return_value=_make_depth_result()):
        with patch.object(
            orchestrator,
            "_capture_backend_metadata",
            side_effect=AssertionError("should not recapture backend metadata"),
        ) as capture_mock:
            result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    capture_mock.assert_not_called()


def test_apex_gate_evaluates_native_depth_grid_before_artifact_resize(tmp_path, mock_da3_available):
    """APEX gate should inspect the backend/native grid, not the resized depth artifact."""
    import json

    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "native_grid_gate.png"
    Image.new("RGB", (65, 65), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        non_commercial_ok=True,
        enable_v2=False,
        quality_tier="apex",
    )
    orchestrator = EnhanceOrchestrator(config, tmp_path)
    orchestrator.postprocessor = Mock(process=lambda result: result)

    seen: dict[str, tuple[int, int] | None] = {
        "gate_shape": None,
        "native_shape": None,
        "artifact_shape": None,
    }

    def _capture_gate(depth_map, depth_units=None, *, native_shape=None, artifact_shape=None):  # noqa: ARG001
        seen["gate_shape"] = tuple(int(value) for value in depth_map.shape[:2])
        seen["native_shape"] = tuple(int(value) for value in native_shape)
        seen["artifact_shape"] = tuple(int(value) for value in artifact_shape)
        return {
            "passed": True,
            "failure_codes": [],
            "warnings": [],
            "metrics": {},
            "thresholds": {},
            "shape_context": {
                "gate_evaluated_shape": list(seen["gate_shape"]),
                "native_shape": list(seen["native_shape"]),
                "artifact_shape": list(seen["artifact_shape"]),
            },
        }

    with patch.object(orchestrator.depth_backend, "compute", return_value=_make_depth_result(width=56, height=56)):
        with patch.object(orchestrator, "_enforce_apex_depth_validity_gate", side_effect=_capture_gate):
            result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert seen["gate_shape"] == (56, 56)
    assert seen["native_shape"] == (56, 56)
    assert seen["artifact_shape"] == (65, 65)

    manifest = json.loads(Path(result["manifest"]).read_text())
    depth_stats = manifest["depth"]["stats"]
    assert depth_stats["native_shape"] == [56, 56]
    assert depth_stats["artifact_shape"] == [65, 65]
    assert depth_stats["gate_evaluated_shape"] == [56, 56]


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
    # Repair 1.2 (#2066): a bare default (no model selector) resolves the
    # commercial-safe da3_metric model, and attempt provenance records the
    # repo that would actually execute — not the mutated compat variant.
    assert result["attempts"][0]["model_id"] == "depth-anything/DA3METRIC-LARGE"
    assert result["attempts"][1]["status"] == "success"
    assert result["attempts"][1]["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert result["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
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
    assert backend_selection["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
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
    # Repair 1.2 (#2066): a bare default (no model selector) resolves the
    # commercial-safe da3_metric model, and attempt provenance records the
    # repo that would actually execute — not the mutated compat variant.
    assert result["attempts"][0]["model_id"] == "depth-anything/DA3METRIC-LARGE"
    assert result["attempts"][1]["status"] == "success"
    assert result["attempts"][1]["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert result["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
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
    assert backend_selection["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert backend_selection["attempts"][0]["failure_kind"] == "semantic"


def test_runtime_license_restriction_does_not_fallback(tmp_path):
    """LicenseRestrictionError should fail fast instead of continuing fallback chain."""
    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "license_fail_fast.png"
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
    depth_pro_backend.compute.side_effect = LicenseRestrictionError("restricted backend")

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, _config: {
        "depth_pro": depth_pro_backend,
        "da3": da3_backend,
    }[backend_id]

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with pytest.raises(LicenseRestrictionError):
            orchestrator.enhance_image(ImageInput(path=test_image))

    assert da3_backend.compute.call_count == 0


def test_unprepared_runtime_rejects_enabled_depth_cache_before_backend_access(tmp_path):
    """Direct construction must reject cache use without plan authority."""
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        enable_depth_cache=True,
    )

    output_root = tmp_path / "output"
    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as registry_cls:
        with pytest.raises(LuxExecutionPlanAuthorityError, match="from_prepared"):
            EnhanceOrchestrator(config, output_root)

    registry_cls.assert_not_called()
    assert not output_root.exists()


def test_unprepared_runtime_cache_rejection_is_fail_closed(tmp_path):
    """An unprepared caller cannot inject a legacy cache hit after construction."""
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        enable_depth_cache=True,
    )

    with pytest.raises(LuxExecutionPlanAuthorityError, match="ExecutionIdentity v3"):
        EnhanceOrchestrator(config, tmp_path / "output")


def test_runtime_attempt_device_records_actual_backend_device(tmp_path):
    """Attempt provenance should record the device actually used by backend output."""
    from PIL import Image

    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "device_provenance.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cuda",
        enable_v2=False,
    )

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.return_value = da3_backend

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["attempts"][0]["backend"] == "da3"
    assert result["attempts"][0]["device"] == "cpu"


def test_runtime_metric_depth_resize_preserves_metric_units(tmp_path):
    """Resizing depth back to original shape must preserve metric-value scale."""
    from PIL import Image

    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    test_image = tmp_path / "metric_resize.png"
    Image.new("RGB", (97, 103), color="white").save(test_image)

    metric_depth = np.linspace(20.0, 40.0, 98 * 98, dtype=np.float32).reshape(98, 98)
    metric_result = DepthResult(
        depth_map=metric_depth,
        original_image=np.array(Image.new("RGB", (98, 98), color="white")),
        metadata={
            "source_depth_units": "meters",
            "output_depth_units": "meters",
            "output_normalization": "native_metric",
        },
        depth_units="meters",
        backend_id="da3",
        device="cpu",
    )

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        save_float_depth=True,
    )

    da3_backend = Mock()
    da3_backend.name = "da3"
    da3_backend.license_type = Mock(value="commercial")
    da3_backend.ensure_available.return_value = None
    da3_backend.compute.return_value = metric_result

    registry = Mock()
    registry.get_backend.return_value = da3_backend

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(ImageInput(path=test_image))

    assert result["status"] == "ok"
    assert result["depth_float_path"] is not None

    resized_depth = np.load(result["depth_float_path"])
    assert resized_depth.shape == (103, 97)
    assert float(np.max(resized_depth)) > 5.0
    assert float(np.min(resized_depth)) >= 15.0


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
    depth_pro_backend._checkpoint_path = Path("/tmp/depth_pro_custom.pt")
    depth_pro_backend._checkpoint_hash_cached = "A" * 64
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
    assert [attempt["model_id"] for attempt in result["attempts"]] == [
        "apple/ml-depth-pro",
        # Repair 1.2 (#2066): the defaulted da3 attempt records the
        # commercial-safe metric repo, not the mutated compat variant.
        "depth-anything/DA3METRIC-LARGE",
        "depth-anything/Depth-Anything-V2-Small-hf",
    ]
    assert result["attempts"][0]["model_artifact_filename"] == "depth_pro_custom.pt"
    assert result["attempts"][0]["model_artifact_sha256"] == "a" * 64
    assert result["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert result["selected_attempt_index"] == 2

    manifest = json.loads(Path(result["manifest"]).read_text())
    attempts = manifest["backend_selection"]["attempts"]
    assert [attempt["attempt"] for attempt in attempts] == [0, 1, 2]
    assert attempts[0]["failure_kind"] == "operational"
    assert attempts[1]["failure_kind"] == "operational"
    assert attempts[2]["status"] == "success"
    assert attempts[0]["model_artifact_filename"] == "depth_pro_custom.pt"
    assert attempts[0]["model_artifact_sha256"] == "a" * 64
    assert manifest["backend_selection"]["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
