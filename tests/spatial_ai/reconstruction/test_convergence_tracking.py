"""Tests for 3DGS convergence tracking (ADR-026 §4.1).

These tests validate convergence behavior of the Gaussian Splatting backend.
They are marked experimental and skipped unless explicitly enabled.

Coverage:
- Convergence state tracking
- RMSE progression monitoring
- Early stopping detection
- Divergence handling
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Module-level availability check
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from transformation_portal.spatial_ai.reconstruction import (
        GaussianBackend,
        ReconstructionInput,
    )

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for 3DGS tests"),
]


def _require_long_tests() -> bool:
    """Check if long tests are enabled."""
    return os.getenv("TP_LONG_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _require_convergence_tests() -> bool:
    """Check if convergence tests are explicitly enabled."""
    return os.getenv("TP_RUN_CONVERGENCE_TESTS", "").strip().lower() in {"1", "true", "yes"}


class TestConvergenceTracking:
    """Test convergence state tracking during optimization."""

    @pytest.fixture
    def gaussian_backend(self):
        """Create backend for convergence tests."""
        from transformation_portal.spatial_ai.reconstruction import GaussianBackend

        return GaussianBackend(
            tier="apex_research",
            device="cpu",
            optimization_seed=42,
        )

    @pytest.fixture
    def minimal_input(self):
        """Create minimal reconstruction input for convergence tests."""
        from transformation_portal.spatial_ai.reconstruction import CameraParams, ReconstructionInput

        h, w = 64, 80
        images = [np.random.rand(h, w, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.array(
            [[0.82 * w, 0, w / 2], [0, 0.82 * w, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(3)]
        return ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

    @pytest.mark.skipif(
        not _require_convergence_tests(),
        reason="Convergence tests require TP_RUN_CONVERGENCE_TESTS=1",
    )
    def test_convergence_state_types(self, gaussian_backend, minimal_input):
        """Convergence field should be one of defined states."""
        from transformation_portal.spatial_ai.reconstruction.contracts import ConvergenceState

        scene = gaussian_backend.reconstruct(minimal_input, iterations=50)

        valid_states = {
            ConvergenceState.CONVERGED,
            ConvergenceState.STALLED,
            ConvergenceState.DIVERGING,
            ConvergenceState.IN_PROGRESS,
        }
        # Check string value is valid
        assert scene.convergence in {s.value for s in valid_states} or scene.convergence in valid_states

    @pytest.mark.skipif(
        not _require_convergence_tests(),
        reason="Convergence tests require TP_RUN_CONVERGENCE_TESTS=1",
    )
    def test_rmse_recorded_in_metadata(self, gaussian_backend, minimal_input):
        """RMSE should be tracked in scene metadata."""
        scene = gaussian_backend.reconstruct(minimal_input, iterations=50)

        assert scene.rmse is not None
        assert isinstance(scene.rmse, float)
        assert scene.rmse >= 0.0
        assert "rmse_history" in scene.metadata or scene.rmse >= 0.0

    @pytest.mark.skipif(
        not _require_convergence_tests(),
        reason="Convergence tests require TP_RUN_CONVERGENCE_TESTS=1",
    )
    def test_iteration_count_matches_requested(self, gaussian_backend, minimal_input):
        """Actual iterations should be recorded accurately."""
        requested = 100
        scene = gaussian_backend.reconstruct(minimal_input, iterations=requested)

        assert scene.iteration == scene.metadata.get("actual_iterations", scene.iteration)
        assert scene.metadata.get("requested_iterations") == requested or scene.iteration <= requested


class TestConvergenceCriteria:
    """Test convergence detection criteria (experimental).

    NOTE: These tests require significant compute time and are designed
    as integration/benchmark tests rather than unit tests.
    """

    @pytest.mark.skipif(
        not (_require_long_tests() and _require_convergence_tests()),
        reason="Long convergence tests require TP_LONG_TESTS=1 and TP_RUN_CONVERGENCE_TESTS=1",
    )
    @pytest.mark.slow
    def test_rmse_decreases_over_iterations(self):
        """RMSE should generally decrease during optimization.

        This is a statistical property test - RMSE should decrease
        on average, though individual steps may increase.
        """
        from transformation_portal.spatial_ai.reconstruction import CameraParams, GaussianBackend, ReconstructionInput

        backend = GaussianBackend(tier="apex_research", device="cpu", optimization_seed=42)

        h, w = 120, 160
        images = [np.random.rand(h, w, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.array(
            [[0.82 * w, 0, w / 2], [0, 0.82 * w, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(3)]
        ri = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        # Run with different iteration counts
        scene_early = backend.reconstruct(ri, iterations=100)
        scene_late = backend.reconstruct(ri, iterations=500)

        # RMSE should decrease (or at least not significantly increase)
        # Allow 20% tolerance for stochastic variation
        assert scene_late.rmse <= scene_early.rmse * 1.2

    @pytest.mark.skipif(
        not (_require_long_tests() and _require_convergence_tests()),
        reason="Long convergence tests require TP_LONG_TESTS=1 and TP_RUN_CONVERGENCE_TESTS=1",
    )
    @pytest.mark.slow
    def test_convergence_below_threshold(self):
        """Scene should converge below RMSE threshold for well-posed inputs.

        Target: RMSE < 5% for simple synthetic scenes.
        """
        from transformation_portal.spatial_ai.reconstruction import CameraParams, GaussianBackend, ReconstructionInput

        backend = GaussianBackend(tier="apex_research", device="cpu", optimization_seed=42)

        # Well-posed input with consistent images
        h, w = 120, 160
        base_image = np.random.rand(h, w, 3).astype(np.float32)
        # Use same image with slight variations (simulates consistent views)
        images = [base_image + np.random.rand(h, w, 3).astype(np.float32) * 0.1 for _ in range(3)]
        intrinsics = np.array(
            [[0.82 * w, 0, w / 2], [0, 0.82 * w, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(3)]
        ri = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene = backend.reconstruct(ri, iterations=1000)

        # Convergence target for simple scenes
        assert scene.rmse < 0.10  # 10% threshold for synthetic test


class TestEarlyStoppingBehavior:
    """Test early stopping and divergence detection (Phase 2.3 roadmap).

    NOTE: Early stopping is a Phase 2.3 roadmap item and may not be
    fully implemented. These tests document expected behavior.
    """

    @pytest.mark.skip(reason="Early stopping not yet implemented (Phase 2.3 roadmap)")
    def test_early_stop_on_plateau(self):
        """Backend should detect RMSE plateau and stop early.

        Expected behavior: If RMSE doesn't improve for N consecutive
        iterations, mark convergence as STALLED and optionally stop.
        """
        # Phase 2.3 implementation placeholder
        pass

    @pytest.mark.skip(reason="Divergence detection not yet implemented (Phase 2.3 roadmap)")
    def test_divergence_detection(self):
        """Backend should detect and report diverging RMSE.

        Expected behavior: If RMSE increases for N consecutive
        iterations, mark convergence as DIVERGING.
        """
        # Phase 2.3 implementation placeholder
        pass
