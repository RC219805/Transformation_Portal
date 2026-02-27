"""Tests for timing context with GPU synchronization."""

import os
import time

import pytest

from transformation_portal.metrics.timing import TimingContext, timing_context


class TestTimingContext:
    """Tests for TimingContext class."""

    def test_basic_timing(self):
        """Test basic timing without device sync."""
        with timing_context("test") as timer:
            time.sleep(0.01)

        assert timer.elapsed_sec >= 0.01

    def test_accumulate_into_dict(self):
        """Test accumulating timings into dict."""
        timings = {}

        with timing_context("phase1", timings):
            time.sleep(0.01)

        with timing_context("phase2", timings):
            time.sleep(0.01)

        assert "phase1" in timings
        assert "phase2" in timings
        assert timings["phase1"] >= 0.01
        assert timings["phase2"] >= 0.01

    def test_timing_context_cpu_device(self):
        """Test timing with CPU device (no sync needed)."""
        with timing_context("test", device="cpu") as timer:
            time.sleep(0.01)

        assert timer.elapsed_sec >= 0.01

    def test_timing_context_mps_device_graceful_fallback(self):
        """Test MPS device is safe by default (sync opt-in)."""
        with timing_context("test", device="mps") as timer:
            time.sleep(0.01)

        # Should still measure time correctly
        assert timer.elapsed_sec >= 0.01
        assert timer.elapsed_sec < 0.02

    def test_timing_context_cuda_device_graceful_fallback(self):
        """Test CUDA device falls back gracefully if torch unavailable."""
        # This should not crash even if torch.cuda not available
        with timing_context("test", device="cuda") as timer:
            time.sleep(0.01)

        # Should still measure time correctly
        assert timer.elapsed_sec >= 0.01
        assert timer.elapsed_sec < 0.02

    def test_timing_context_invalid_device(self):
        """Test that invalid device strings are safely ignored."""
        with timing_context("test", device="quantum_computer") as timer:
            time.sleep(0.01)

        # Should not sync, but should still time correctly
        assert timer.elapsed_sec >= 0.01

    def test_elapsed_sec_available_after_exit(self):
        """Test that elapsed_sec is available after context exit."""
        ctx = TimingContext("test")

        assert ctx.elapsed_sec == 0.0

        with ctx:
            time.sleep(0.01)

        assert ctx.elapsed_sec >= 0.01

    def test_multiple_phases_same_dict(self):
        """Test multiple phases accumulating into same dict."""
        timings = {}

        with timing_context("load", timings):
            time.sleep(0.01)

        with timing_context("process", timings):
            time.sleep(0.01)

        with timing_context("save", timings):
            time.sleep(0.01)

        assert len(timings) == 3
        assert all(v >= 0.01 for v in timings.values())
        total = sum(timings.values())
        assert total >= 0.03


@pytest.mark.ml
class TestTimingContextWithTorch:
    """Tests for timing context with torch (requires ML deps)."""

    def test_mps_sync_if_available(self):
        """Test MPS synchronization only when explicitly enabled."""
        try:
            import torch

            has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            pytest.skip("torch not available")

        if not has_mps:
            pytest.skip("MPS not available")

        if os.getenv("TP_TIMING_SYNC_MPS", "").strip().lower() not in {"1", "true", "yes"}:
            pytest.skip("MPS sync is opt-in; set TP_TIMING_SYNC_MPS=1 to exercise synchronization")

        # This test verifies no crash with explicit MPS sync
        with timing_context("test", device="mps") as timer:
            time.sleep(0.01)

        assert timer.elapsed_sec >= 0.01

    def test_cuda_sync_if_available(self):
        """Test CUDA synchronization if torch available."""
        try:
            import torch

            has_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        except ImportError:
            pytest.skip("torch not available")

        if not has_cuda:
            pytest.skip("CUDA not available")

        # This test just verifies no crash with actual CUDA sync
        with timing_context("test", device="cuda") as timer:
            time.sleep(0.01)

        assert timer.elapsed_sec >= 0.01
