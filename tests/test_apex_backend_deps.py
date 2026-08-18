"""Tests for APEX backend-aware dependency validation (Phase 3).

Tests that the check_ml_dependencies() function correctly validates host
dependencies based on backend runtime requirements. Legacy/in-process backends
default to Torch plus backend-specific packages, while isolated backends may
declare that their host process needs no ML stack.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_backend_registry():
    """Create a mock backend registry with test backends."""

    # Mock DA3 backend class
    class MockDA3:
        name = "da3"

        @classmethod
        def required_packages(cls):
            return ["transformers"]

    # Mock non-HF backend class (e.g., ONNX)
    class MockONNX:
        name = "onnx"

        @classmethod
        def required_packages(cls):
            return ["onnxruntime"]

    # Mock minimal backend (no extra deps)
    class MockMinimal:
        name = "mock"

        @classmethod
        def required_packages(cls):
            return []

    registry = MagicMock()
    # Support both old _backends access and new public API
    registry._backends = {
        "da3": MockDA3,
        "onnx": MockONNX,
        "mock": MockMinimal,
    }
    registry.get_backend_class = lambda backend_id: registry._backends.get(backend_id)
    registry.available_backend_ids = lambda: sorted(registry._backends.keys())
    registry.has_backend = lambda backend_id: backend_id in registry._backends
    return registry


def test_da3_backend_requires_torch_and_transformers(mock_backend_registry):
    """Test that DA3 backend requires both torch and transformers."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    # Mock find_spec to return valid specs for all packages
    def find_spec_side_effect(name):
        return MagicMock()  # All packages exist

    def import_module_side_effect(name):
        # Return mocks for all "available" packages
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("da3")

                assert all_available is True
                assert missing == []


def test_da3_backend_fails_when_transformers_missing(mock_backend_registry):
    """Test that DA3 backend fails when transformers is missing."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def find_spec_side_effect(name):
        if name == "transformers":
            return None  # Simulate package not found
        return MagicMock()  # Package exists

    def import_module_side_effect(name):
        if name == "torch":
            return MagicMock()
        raise ModuleNotFoundError(f"No module named '{name}'")

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("da3")

                assert all_available is False
                assert "transformers" in missing


def test_non_hf_backend_does_not_require_transformers(mock_backend_registry):
    """Test that non-HF backends don't require transformers."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def find_spec_side_effect(name):
        if name == "transformers":
            return None  # Simulate package not found
        elif name == "onnxruntime":
            return MagicMock()  # ONNX runtime is available
        elif name == "torch":
            return MagicMock()  # Torch is available
        return MagicMock()

    def import_module_side_effect(name):
        # Return mocks for packages that should be "available"
        if name in ("torch", "onnxruntime"):
            return MagicMock()
        # transformers isn't available, will be caught by find_spec
        raise ModuleNotFoundError(f"No module named '{name}'")

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            # Patch importlib.import_module where it's USED (in apex_matrix_runner namespace)
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("onnx")

                # Should pass because ONNX backend doesn't require transformers
                assert all_available is True
                assert missing == []


def test_torch_required_without_runtime_specific_contract(mock_backend_registry):
    """Backends without a runtime-specific contract require host Torch."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def find_spec_side_effect(name):
        if name == "torch":
            return None  # Simulate torch not found
        return MagicMock()

    def import_module_side_effect(name):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("mock")

                assert all_available is False
                assert "torch" in missing


def test_torch_broken_install_treated_as_missing(mock_backend_registry):
    """Test that broken torch install (OSError/RuntimeError) is treated as missing."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    # Mock find_spec to return valid spec, but then fail on actual import
    def find_spec_side_effect(name):
        return MagicMock()  # Spec exists

    def import_module_side_effect(name):
        if name == "torch":
            raise OSError("Missing CUDA libraries")
        # For all other imports, use the real import
        import importlib

        return importlib.import_module(name)

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("mock")

                assert all_available is False
                assert "torch" in missing


def test_backend_specific_dep_missing_reported_correctly(mock_backend_registry):
    """Test that backend-specific missing dep is reported with backend ID."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def find_spec_side_effect(name):
        if name == "onnxruntime":
            return None  # Package not found
        return MagicMock()  # Other packages exist

    def import_module_side_effect(name):
        if name == "torch":
            return MagicMock()
        raise ModuleNotFoundError(f"No module named '{name}'")

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("onnx")

                assert all_available is False
                assert "onnxruntime" in missing


def test_unknown_backend_fails_fast_with_clear_message(mock_backend_registry):
    """Test that unknown backend raises ApexConfigError with available backends listed."""
    from scripts.apex_matrix_runner import ApexConfigError, check_ml_dependencies

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with pytest.raises(ApexConfigError) as exc_info:
            check_ml_dependencies("unknown_backend")

        error_msg = str(exc_info.value)
        assert "Unknown backend_id 'unknown_backend'" in error_msg
        assert "Available backends:" in error_msg
        assert "da3" in error_msg  # Should list available backends
        assert "onnx" in error_msg
        assert "mock" in error_msg


def test_minimal_backend_requires_only_torch(mock_backend_registry):
    """Test that minimal backend (no extra deps) only requires torch."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def find_spec_side_effect(name):
        return MagicMock()  # All packages exist

    def import_module_side_effect(name):
        return MagicMock()  # All imports succeed

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                # Mock backend declares no extra requirements
                all_available, missing = check_ml_dependencies("mock")

                assert all_available is True
                assert missing == []


def test_da3_isolated_runtime_does_not_require_local_ml_stack(monkeypatch, tmp_path):
    """DA3 subprocess mode should not require host Torch or Transformers."""
    from scripts.apex_matrix_runner import check_ml_dependencies
    from transformation_portal.depth.backends.da3 import DA3Backend

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", str(python_executable))

    def find_spec_side_effect(name):
        if name in {"torch", "transformers"}:
            return None
        return MagicMock()

    def import_module_side_effect(name):
        raise ModuleNotFoundError(f"No module named '{name}'")

    registry = MagicMock()
    registry.get_backend_class = lambda backend_id: DA3Backend if backend_id == "da3" else None
    registry.available_backend_ids = lambda: ["da3"]

    with patch("transformation_portal.depth.backends.get_registry", return_value=registry):
        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            with patch("scripts.apex_matrix_runner.importlib.import_module", side_effect=import_module_side_effect):
                all_available, missing = check_ml_dependencies("da3")

    assert all_available is True
    assert missing == []


def test_da3_in_process_runtime_requires_host_ml_stack(monkeypatch):
    """DA3 in-process mode requires host Torch and Transformers."""
    from transformation_portal.depth.backends.da3 import DA3Backend

    monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)

    assert DA3Backend().runtime_required_packages() == ["torch", "transformers"]
