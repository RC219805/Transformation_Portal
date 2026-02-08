"""Tests for APEX backend-aware dependency validation (Phase 3).

Tests that the check_ml_dependencies() function correctly validates
dependencies based on backend requirements, always requiring torch
while making transformers optional for backends that don't need it.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_backend_registry():
    """Create a mock backend registry with test backends."""

    # Mock DA3 backend class
    class MockDA3:
        name = "da3"

        def required_packages(self):
            return ["transformers"]

    # Mock non-HF backend class (e.g., ONNX)
    class MockONNX:
        name = "onnx"

        def required_packages(self):
            return ["onnxruntime"]

    # Mock minimal backend (no extra deps)
    class MockMinimal:
        name = "mock"

        def required_packages(self):
            return []

    registry = MagicMock()
    registry._backends = {
        "da3": MockDA3,
        "onnx": MockONNX,
        "mock": MockMinimal,
    }
    return registry


def test_da3_backend_requires_torch_and_transformers(mock_backend_registry):
    """Test that DA3 backend requires both torch and transformers."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch.dict(sys.modules, {"torch": MagicMock(), "transformers": MagicMock()}):
            all_available, missing = check_ml_dependencies("da3")

            assert all_available is True
            assert missing == []


def test_da3_backend_fails_when_transformers_missing(mock_backend_registry):
    """Test that DA3 backend fails when transformers is missing."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("da3")

            assert all_available is False
            assert "transformers" in missing


def test_non_hf_backend_does_not_require_transformers(mock_backend_registry):
    """Test that non-HF backends don't require transformers."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("onnx")

            # Should pass because ONNX backend doesn't require transformers
            assert all_available is True
            assert missing == []


def test_torch_always_required(mock_backend_registry):
    """Test that torch is always required regardless of backend."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("mock")

            assert all_available is False
            assert "torch" in missing


def test_torch_broken_install_treated_as_missing(mock_backend_registry):
    """Test that broken torch install (OSError/RuntimeError) is treated as missing."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "torch":
            raise OSError("Missing CUDA libraries")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("mock")

            assert all_available is False
            assert "torch" in missing


def test_backend_specific_dep_missing_reported_correctly(mock_backend_registry):
    """Test that backend-specific missing dep is reported with backend ID."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("onnx")

            assert all_available is False
            assert "onnxruntime" in missing


def test_unknown_backend_fallback_strict_check(mock_backend_registry):
    """Test that unknown backend falls back to strict torch+transformers check."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    def import_side_effect(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return MagicMock()

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch("builtins.__import__", side_effect=import_side_effect):
            all_available, missing = check_ml_dependencies("unknown_backend")

            assert all_available is False
            assert "transformers" in missing


def test_minimal_backend_requires_only_torch(mock_backend_registry):
    """Test that minimal backend (no extra deps) only requires torch."""
    from scripts.apex_matrix_runner import check_ml_dependencies

    with patch("transformation_portal.depth.backends.get_registry", return_value=mock_backend_registry):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            # Mock backend declares no extra requirements
            all_available, missing = check_ml_dependencies("mock")

            assert all_available is True
            assert missing == []
