"""Shared fixtures for reconstruction numeric stability and determinism."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pytest


def _import_torch():
    try:
        import torch  # type: ignore
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - exercised in no-torch/broken-torch lanes
        pytest.skip(f"torch required for reconstruction ML fixture: {exc}")
    return torch


@pytest.fixture
def seed_all_rngs():
    """Seed python/numpy/torch RNGs and return a reseed helper."""

    def _seed(seed: int = 42) -> int:
        random.seed(seed)
        np.random.seed(seed)
        torch = _import_torch()
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    _seed(42)
    return _seed


@pytest.fixture
def torch_dtype_policy():
    """Central dtype policy for reconstruction tests."""
    torch = _import_torch()
    return {
        "default": torch.float32,
        "high_precision": torch.float64,
    }


@pytest.fixture
def device_tolerance() -> Dict[str, Dict[str, float]]:
    """Tolerance table by device for numeric comparisons."""
    return {
        "cpu": {"rtol": 1e-7, "atol": 1e-9},
        "mps": {"rtol": 1e-5, "atol": 1e-7},
        "cuda": {"rtol": 1e-6, "atol": 1e-8},
    }
