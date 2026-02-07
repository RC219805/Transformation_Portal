#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest configuration for Transformation Portal tests.

NOTE: This file previously contained sys.path manipulation to add src/
to the Python path. This approach violates PR 162 guidelines for proper
package development practices.

PROPER SETUP:
  Option 1 (Recommended): Install package in editable mode
    pip install -e .

  Option 2: Set PYTHONPATH environment variable
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    # or for this test run only:
    PYTHONPATH="$(pwd)/src" pytest

Tests will automatically find the package if installed via pip install -e .
or if PYTHONPATH is set correctly.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures use other fixtures as params

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

# =============================================================================
# TIER 1: Pure Fixtures (no IO, no heavy deps)
# =============================================================================


@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    """Provide deterministic RNG for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Minimal valid config dictionary for EnhanceConfig."""
    return {
        "model_variant": "DA3-Large",
        "preset": "max_quality",
        "depth_device": "cpu",
        "depth_quantization": "none",
        "depth_fallback": "fail",
        "verify_depth_writes": True,
        "force_v2": False,
        "v2_timeout": 300,
    }


@pytest.fixture
def sample_pbr_config_dict() -> dict[str, Any]:
    """Minimal valid config dictionary for PBRConfig."""
    return {
        "normal_strength": 1.0,
        "roughness_base": 0.5,
        "metallic_threshold": 0.5,
        "ao_strength": 1.0,
    }


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Ensure tests don't leak environment state."""
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


# =============================================================================
# TIER 2: IO Fixtures (temp files, small test assets)
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create structured temporary workspace for tests.

    Returns:
        Dictionary with keys: root, input_dir, output_dir, cache_dir
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"

    input_dir.mkdir()
    output_dir.mkdir()
    cache_dir.mkdir()

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "root": tmp_path,
    }


@pytest.fixture
def sample_rgb_image(deterministic_rng) -> np.ndarray:
    """Create small deterministic RGB image (100x100x3, uint8)."""
    arr = (deterministic_rng.random((100, 100, 3)) * 255).astype(np.uint8)
    return arr


@pytest.fixture
def sample_rgb_pil(sample_rgb_image) -> Image.Image:
    """Create PIL Image from sample RGB array."""
    return Image.fromarray(sample_rgb_image, mode="RGB")


@pytest.fixture
def sample_depth_map(deterministic_rng) -> np.ndarray:
    """Create small deterministic depth map (100x100, uint16)."""
    arr = (deterministic_rng.random((100, 100)) * 65535).astype(np.uint16)
    return arr


@pytest.fixture
def sample_depth_pil(sample_depth_map) -> Image.Image:
    """Create PIL Image from sample depth array."""
    return Image.fromarray(sample_depth_map, mode="I;16")


@pytest.fixture
def sample_image_file(temp_workspace, sample_rgb_pil) -> Path:
    """Save sample RGB image to temp file and return path."""
    path = temp_workspace["input_dir"] / "test_image.png"
    sample_rgb_pil.save(path)
    return path


@pytest.fixture
def sample_depth_file(temp_workspace, sample_depth_pil) -> Path:
    """Save sample depth map to temp file and return path."""
    path = temp_workspace["input_dir"] / "test_depth.png"
    sample_depth_pil.save(path)
    return path


@pytest.fixture
def sample_yaml_config(temp_workspace, sample_config_dict) -> Path:
    """Create minimal YAML config file for testing."""
    import yaml

    path = temp_workspace["root"] / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(sample_config_dict, f)
    return path


# =============================================================================
# TIER 3: Optional/ML Fixtures (guarded by importorskip)
# =============================================================================


@pytest.fixture
def transformers_offline():
    """Set environment for offline transformers testing."""
    old_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    old_hf = os.environ.get("HF_HUB_OFFLINE")

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    yield

    if old_transformers is None:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = old_transformers

    if old_hf is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = old_hf


@pytest.fixture
def mock_depth_model(deterministic_rng):
    """Mock depth estimation model for testing without ML dependencies."""
    pytest.importorskip("unittest.mock")
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.infer.return_value = deterministic_rng.random((100, 100)).astype(np.float32)
    return mock


# =============================================================================
# Dependency Availability Helpers (for skip guards)
# =============================================================================


# Cache for expensive availability checks (avoid repeated imports / lookups)
_AVAILABILITY_CACHE: dict[str, bool] = {}


def _cached_bool(key: str, compute: "Callable[[], bool]") -> bool:
    """Return cached boolean for `key`, computing it once if needed."""
    if key not in _AVAILABILITY_CACHE:
        _AVAILABILITY_CACHE[key] = bool(compute())
    return _AVAILABILITY_CACHE[key]


def has_depth_anything_v3() -> bool:
    """Check if the optional `depth_anything_3` package is available.

    This provides the `depth_anything_3` Python module used by DA3 nested models.
    Uses importlib.util.find_spec for fast, side-effect-free checking.

    Returns:
        True if the module spec exists, False otherwise.
    """

    def _compute() -> bool:
        import importlib.util

        return importlib.util.find_spec("depth_anything_3") is not None

    return _cached_bool("depth_anything_v3", _compute)


def has_torch() -> bool:
    """Check if PyTorch is available.

    We use a two-step probe:
      1) find_spec("torch") to avoid expensive imports when torch isn't present
      2) import torch to ensure it actually loads (guards against broken wheels /
         missing shared libraries, which can raise OSError)

    Returns:
        True if torch can be imported successfully, False otherwise.
    """

    def _compute() -> bool:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    return _cached_bool("torch", _compute)


def has_transformers() -> bool:
    """Check if transformers is available (module spec exists).

    Uses find_spec to avoid import-time side effects during test collection.

    Returns:
        True if transformers appears installable, False otherwise.
    """

    def _compute() -> bool:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None

    return _cached_bool("transformers", _compute)


def is_offline_mode() -> bool:
    """Check if running in offline mode (no HuggingFace downloads).

    Returns:
        True if TRANSFORMERS_OFFLINE=1 or HF_HUB_OFFLINE=1, False otherwise.
    """
    return os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def can_run_da3_compute() -> bool:
    """Check if DA3 compute tests can run (all dependencies available).

    DA3 compute tests require:
    - depth_anything_3 package installed
    - torch library available
    - transformers library available
    - Not in offline mode (would fail to download models)

    Returns:
        True if all DA3 compute requirements are met, False otherwise.
    """
    return has_depth_anything_v3() and has_torch() and has_transformers() and not is_offline_mode()
