#!/usr/bin/env python3
"""
Tests for Batch Statistics Module
==================================

Tests for lux_depth_v3/enhance/batch_stats.py covering:
- Normal case: mixed ok/error/skipped results
- Edge case: All errors (ok=0) → avg_runtime_s=0.0
- Edge case: Zero runtime → images_per_hour=0.0
- Edge case: Empty results list
- Edge case: Results with missing runtime_s field
- Throughput calculation using only "ok" results
"""

import pytest
from lux_depth_v3.enhance.batch_stats import compute_batch_runtime_stats


class TestComputeBatchRuntimeStats:
    """Test compute_batch_runtime_stats function."""

    def test_normal_mixed_results(self):
        """Normal case: mixed ok/error/skipped results."""
        results = [
            {"status": "ok", "runtime_s": 1.0},
            {"status": "ok", "runtime_s": 2.0},
            {"status": "error", "runtime_s": 0.5},
            {"status": "skipped", "runtime_s": 0.0},
            {"status": "ok", "runtime_s": 3.0},
        ]
        stats = compute_batch_runtime_stats(results)

        # Total runtime includes all results
        assert stats["total_runtime_s"] == 6.5

        # Average: total_runtime / ok_count = 6.5 / 3
        assert stats["avg_runtime_s"] == pytest.approx(6.5 / 3.0, abs=1e-9)

        # Throughput: 3 ok images / 6.5s * 3600s/hour
        expected_throughput = (3.0 / 6.5) * 3600.0
        assert stats["images_per_hour"] == pytest.approx(expected_throughput, abs=1e-9)

    def test_all_errors_no_division_by_zero(self):
        """When all results are errors, avg should be 0 (no division by zero)."""
        results = [
            {"status": "error", "runtime_s": 1.0},
            {"status": "error", "runtime_s": 2.0},
            {"status": "skipped", "runtime_s": 1.5},
        ]
        stats = compute_batch_runtime_stats(results)

        # No ok results → avg should be 0.0, not raise ZeroDivisionError
        assert stats["avg_runtime_s"] == 0.0

        # Total runtime still accumulates
        assert stats["total_runtime_s"] == 4.5

        # No successful images → throughput is 0
        assert stats["images_per_hour"] == 0.0

    def test_zero_total_runtime(self):
        """When total runtime is zero, images_per_hour should be 0.0."""
        results = [
            {"status": "ok", "runtime_s": 0.0},
            {"status": "ok", "runtime_s": 0.0},
        ]
        stats = compute_batch_runtime_stats(results)

        # Total runtime is 0
        assert stats["total_runtime_s"] == 0.0

        # Average over 2 ok results with 0.0 runtime each
        assert stats["avg_runtime_s"] == 0.0

        # Cannot compute throughput with zero runtime
        assert stats["images_per_hour"] == 0.0

    def test_empty_results_list(self):
        """Edge case: empty results list."""
        results = []
        stats = compute_batch_runtime_stats(results)

        # All stats should be 0.0
        assert stats["total_runtime_s"] == 0.0
        assert stats["avg_runtime_s"] == 0.0
        assert stats["images_per_hour"] == 0.0

    def test_missing_runtime_field(self):
        """Results with missing runtime_s field should default to 0.0."""
        results = [
            {"status": "ok", "runtime_s": 2.0},
            {"status": "ok"},  # Missing runtime_s
            {"status": "error"},  # Missing runtime_s
        ]
        stats = compute_batch_runtime_stats(results)

        # Total runtime: 2.0 + 0.0 + 0.0 = 2.0
        assert stats["total_runtime_s"] == 2.0

        # Average over 2 ok results: 2.0 / 2 = 1.0
        assert stats["avg_runtime_s"] == 1.0

        # Throughput: 2 ok / 2.0s * 3600
        assert stats["images_per_hour"] == pytest.approx(3600.0, abs=1e-9)

    def test_none_runtime_field(self):
        """Results with None runtime_s should default to 0.0."""
        results = [
            {"status": "ok", "runtime_s": 3.0},
            {"status": "ok", "runtime_s": None},  # None runtime
        ]
        stats = compute_batch_runtime_stats(results)

        # Total runtime: 3.0 + 0.0 = 3.0
        assert stats["total_runtime_s"] == 3.0

        # Average over 2 ok results: 3.0 / 2 = 1.5
        assert stats["avg_runtime_s"] == 1.5

        # Throughput: 2 ok / 3.0s * 3600
        assert stats["images_per_hour"] == pytest.approx(2400.0, abs=1e-9)

    def test_throughput_uses_only_ok_results(self):
        """Verify throughput calculation uses only 'ok' results, not total count."""
        results = [
            {"status": "ok", "runtime_s": 1.0},
            {"status": "ok", "runtime_s": 1.0},
            {"status": "error", "runtime_s": 10.0},  # Counted in total runtime but not in ok count
            {"status": "skipped", "runtime_s": 5.0},  # Counted in total runtime but not in ok count
        ]
        stats = compute_batch_runtime_stats(results)

        # Total runtime includes all: 1 + 1 + 10 + 5 = 17
        assert stats["total_runtime_s"] == 17.0

        # Average: total_runtime / ok_count = 17.0 / 2 = 8.5
        assert stats["avg_runtime_s"] == 8.5

        # Throughput: ONLY 2 ok images / 17s total * 3600
        expected_throughput = (2.0 / 17.0) * 3600.0
        assert stats["images_per_hour"] == pytest.approx(expected_throughput, abs=1e-9)

    def test_single_ok_result(self):
        """Test with a single ok result."""
        results = [
            {"status": "ok", "runtime_s": 5.0},
        ]
        stats = compute_batch_runtime_stats(results)

        assert stats["total_runtime_s"] == 5.0
        assert stats["avg_runtime_s"] == 5.0
        assert stats["images_per_hour"] == pytest.approx(720.0, abs=1e-9)  # 3600/5

    def test_all_ok_results(self):
        """Test batch with all successful results."""
        results = [
            {"status": "ok", "runtime_s": 1.0},
            {"status": "ok", "runtime_s": 2.0},
            {"status": "ok", "runtime_s": 3.0},
            {"status": "ok", "runtime_s": 4.0},
        ]
        stats = compute_batch_runtime_stats(results)

        # Total: 10.0
        assert stats["total_runtime_s"] == 10.0

        # Average: 10.0 / 4 = 2.5
        assert stats["avg_runtime_s"] == 2.5

        # Throughput: 4 / 10.0 * 3600 = 1440
        assert stats["images_per_hour"] == pytest.approx(1440.0, abs=1e-9)

    def test_large_batch_realistic(self):
        """Test with realistic large batch (100 images)."""
        # Simulate 100 images: 90 ok, 8 errors, 2 skipped
        # Average runtime for ok images: ~0.5s
        results = [{"status": "ok", "runtime_s": 0.5} for _ in range(90)]
        results.extend([{"status": "error", "runtime_s": 0.1} for _ in range(8)])
        results.extend([{"status": "skipped", "runtime_s": 0.0} for _ in range(2)])

        stats = compute_batch_runtime_stats(results)

        # Total: 90 * 0.5 + 8 * 0.1 = 45.0 + 0.8 = 45.8
        assert stats["total_runtime_s"] == pytest.approx(45.8, abs=1e-9)

        # Average: total_runtime / ok_count = 45.8 / 90
        assert stats["avg_runtime_s"] == pytest.approx(45.8 / 90.0, abs=1e-9)

        # Throughput: 90 / 45.8 * 3600 ≈ 7074.7
        expected_throughput = (90.0 / 45.8) * 3600.0
        assert stats["images_per_hour"] == pytest.approx(expected_throughput, abs=1e-6)

    def test_status_case_sensitivity(self):
        """Test that status matching is case-sensitive (only 'ok' counts)."""
        results = [
            {"status": "ok", "runtime_s": 1.0},
            {"status": "OK", "runtime_s": 1.0},  # Should NOT count
            {"status": "Ok", "runtime_s": 1.0},  # Should NOT count
            {"status": "success", "runtime_s": 1.0},  # Should NOT count
        ]
        stats = compute_batch_runtime_stats(results)

        # Only 1 result with status="ok" exactly
        assert stats["total_runtime_s"] == 4.0
        assert stats["avg_runtime_s"] == 4.0  # total_runtime / 1 ok = 4.0 / 1
        assert stats["images_per_hour"] == pytest.approx(900.0, abs=1e-9)  # 1 / 4 * 3600
