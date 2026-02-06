"""Property-based tests for pure Python math (Condition #2).

Validates that pure Python implementations produce equivalent results to NumPy.
"""

from __future__ import annotations

import random
from typing import List

import pytest

# Import both implementations
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    pytest.skip("NumPy required for math validation tests", allow_module_level=True)

from tools.performance_ledger import (
    _bootstrap_confidence_interval,
    _pure_python_mean,
    _pure_python_percentile,
    _pure_python_std,
    compute_statistics,
)

# Import hypothesis for property-based testing
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("Hypothesis required for property-based tests", allow_module_level=True)


class TestPurePythonMean:
    """Test pure Python mean implementation."""

    @given(st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=1, max_size=100))
    @settings(max_examples=100, deadline=None)
    def test_mean_matches_numpy(self, values: List[float]):
        """Pure Python mean should match NumPy within tolerance."""
        numpy_result = float(np.mean(values))
        python_result = _pure_python_mean(values)

        # Allow small floating point differences
        assert (
            abs(numpy_result - python_result) < 1e-9
        ), f"NumPy: {numpy_result}, Python: {python_result}, values: {values[:5]}..."

    def test_mean_empty_list_raises(self):
        """Mean of empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _pure_python_mean([])

    def test_mean_single_value(self):
        """Mean of single value should be that value."""
        assert _pure_python_mean([42.0]) == 42.0

    def test_mean_known_values(self):
        """Test mean with known values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pure_python_mean(values) - 3.0) < 1e-9


class TestPurePythonStd:
    """Test pure Python standard deviation implementation."""

    @given(st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=2, max_size=100))
    @settings(max_examples=100, deadline=None)
    def test_std_matches_numpy(self, values: List[float]):
        """Pure Python std should match NumPy within tolerance."""
        numpy_result = float(np.std(values, ddof=1))
        python_result = _pure_python_std(values)

        # Allow small relative error for floating point
        if numpy_result > 0:
            relative_error = abs(numpy_result - python_result) / numpy_result
            assert relative_error < 1e-8, f"NumPy: {numpy_result}, Python: {python_result}, rel_err: {relative_error}"
        else:
            assert abs(python_result) < 1e-9

    def test_std_empty_list_raises(self):
        """Std of empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _pure_python_std([])

    def test_std_single_value(self):
        """Std of single value should be zero."""
        assert _pure_python_std([42.0]) == 0.0

    def test_std_constant_values(self):
        """Std of constant values should be zero."""
        values = [5.0, 5.0, 5.0, 5.0]
        assert abs(_pure_python_std(values)) < 1e-9

    def test_std_known_values(self):
        """Test std with known values."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        numpy_std = float(np.std(values, ddof=1))
        python_std = _pure_python_std(values)
        assert abs(numpy_std - python_std) < 1e-9


class TestPurePythonPercentile:
    """Test pure Python percentile implementation."""

    @given(
        st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=2, max_size=100),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_percentile_matches_numpy(self, values: List[float], percentile: float):
        """Pure Python percentile should match NumPy within tolerance."""
        numpy_result = float(np.percentile(values, percentile))
        python_result = _pure_python_percentile(values, percentile)

        # Allow small relative error
        if numpy_result > 0:
            relative_error = abs(numpy_result - python_result) / numpy_result
            assert relative_error < 1e-8, f"Percentile {percentile}: NumPy={numpy_result}, Python={python_result}"
        else:
            assert abs(python_result) < 1e-9

    def test_percentile_empty_list_raises(self):
        """Percentile of empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _pure_python_percentile([], 50)

    def test_percentile_single_value(self):
        """Percentile of single value should be that value."""
        assert _pure_python_percentile([42.0], 50) == 42.0
        assert _pure_python_percentile([42.0], 0) == 42.0
        assert _pure_python_percentile([42.0], 100) == 42.0

    def test_percentile_edge_cases(self):
        """Test percentile edge cases."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        # 0th percentile should be min
        p0_numpy = float(np.percentile(values, 0))
        p0_python = _pure_python_percentile(values, 0)
        assert abs(p0_numpy - p0_python) < 1e-9

        # 100th percentile should be max
        p100_numpy = float(np.percentile(values, 100))
        p100_python = _pure_python_percentile(values, 100)
        assert abs(p100_numpy - p100_python) < 1e-9

        # 50th percentile (median)
        p50_numpy = float(np.percentile(values, 50))
        p50_python = _pure_python_percentile(values, 50)
        assert abs(p50_numpy - p50_python) < 1e-9

    def test_percentile_known_values(self):
        """Test percentile with known values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        # p90
        p90_numpy = float(np.percentile(values, 90))
        p90_python = _pure_python_percentile(values, 90)
        assert abs(p90_numpy - p90_python) < 1e-9

        # p95
        p95_numpy = float(np.percentile(values, 95))
        p95_python = _pure_python_percentile(values, 95)
        assert abs(p95_numpy - p95_python) < 1e-9


class TestBootstrapConfidenceInterval:
    """Test bootstrap CI correctness."""

    def test_bootstrap_ci_contains_mean(self):
        """Bootstrap CI should usually contain the true mean."""
        random.seed(42)

        # Generate samples from known distribution
        true_mean = 10.0
        samples = [true_mean + random.gauss(0, 2) for _ in range(100)]

        lower, upper = _bootstrap_confidence_interval(samples, iterations=1000)

        # 95% CI should contain true mean most of the time
        assert lower <= true_mean <= upper, f"True mean {true_mean} not in CI [{lower}, {upper}]"

    def test_bootstrap_ci_wider_with_more_variance(self):
        """Higher variance should produce wider CI."""
        random.seed(42)

        # Low variance sample
        low_var = [10.0 + random.gauss(0, 0.1) for _ in range(50)]
        lower_low, upper_low = _bootstrap_confidence_interval(low_var, iterations=500)
        width_low = upper_low - lower_low

        # High variance sample
        high_var = [10.0 + random.gauss(0, 5.0) for _ in range(50)]
        lower_high, upper_high = _bootstrap_confidence_interval(high_var, iterations=500)
        width_high = upper_high - lower_high

        # High variance should have wider CI
        assert width_high > width_low, f"High var width {width_high} not > low var width {width_low}"

    def test_bootstrap_ci_deterministic_with_seed(self):
        """Same seed should produce same CI."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        lower1, upper1 = _bootstrap_confidence_interval(values, iterations=100, seed=42)
        lower2, upper2 = _bootstrap_confidence_interval(values, iterations=100, seed=42)

        assert lower1 == lower2
        assert upper1 == upper2

    def test_bootstrap_ci_single_value(self):
        """Single value should have degenerate CI."""
        lower, upper = _bootstrap_confidence_interval([42.0], iterations=100)
        assert lower == upper == 42.0

    def test_bootstrap_ci_empty_list(self):
        """Empty list should return (0, 0)."""
        lower, upper = _bootstrap_confidence_interval([], iterations=100)
        assert lower == upper == 0.0


class TestComputeStatisticsNumPyVsPython:
    """Test compute_statistics with NumPy vs pure Python mode."""

    @given(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=5, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_statistics_match_with_numpy(self, values: List[float]):
        """Statistics computed with NumPy should be self-consistent."""
        # This test runs with NumPy available
        stats = compute_statistics(values, bootstrap_iterations=0, enable_bootstrap=False)

        # Verify against direct NumPy computation
        assert abs(stats.mean_sec - float(np.mean(values))) < 1e-9
        assert abs(stats.median_sec - float(np.median(values))) < 1e-9
        assert abs(stats.p90_sec - float(np.percentile(values, 90))) < 1e-9
        assert abs(stats.p95_sec - float(np.percentile(values, 95))) < 1e-9
        assert abs(stats.min_sec - float(np.min(values))) < 1e-9
        assert abs(stats.max_sec - float(np.max(values))) < 1e-9

        if len(values) > 1:
            assert abs(stats.std_sec - float(np.std(values, ddof=1))) < 1e-8

    def test_statistics_with_bootstrap_enabled(self):
        """Test statistics with bootstrap CI enabled."""
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]

        stats = compute_statistics(values, bootstrap_iterations=500, enable_bootstrap=True)

        # CI should exist
        assert stats.bootstrap_ci_95_lower is not None
        assert stats.bootstrap_ci_95_upper is not None

        # CI should contain mean
        assert stats.bootstrap_ci_95_lower <= stats.mean_sec <= stats.bootstrap_ci_95_upper

        # CI should be reasonable
        assert stats.bootstrap_ci_95_upper > stats.bootstrap_ci_95_lower

    def test_statistics_without_bootstrap(self):
        """Test statistics with bootstrap disabled."""
        values = [10.0, 11.0, 12.0, 13.0, 14.0]

        stats = compute_statistics(values, bootstrap_iterations=0, enable_bootstrap=False)

        # CI should be None
        assert stats.bootstrap_ci_95_lower is None
        assert stats.bootstrap_ci_95_upper is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_values(self):
        """Test with very small values (near zero)."""
        values = [0.001, 0.002, 0.003, 0.004, 0.005]

        numpy_mean = float(np.mean(values))
        python_mean = _pure_python_mean(values)

        assert abs(numpy_mean - python_mean) / numpy_mean < 1e-6

    def test_very_large_values(self):
        """Test with very large values."""
        values = [1e6, 2e6, 3e6, 4e6, 5e6]

        numpy_mean = float(np.mean(values))
        python_mean = _pure_python_mean(values)

        assert abs(numpy_mean - python_mean) / numpy_mean < 1e-6

    def test_mixed_scale_values(self):
        """Test with values spanning multiple orders of magnitude."""
        values = [0.1, 1.0, 10.0, 100.0, 1000.0]

        numpy_mean = float(np.mean(values))
        python_mean = _pure_python_mean(values)

        # Still should match
        assert abs(numpy_mean - python_mean) / numpy_mean < 1e-6

    def test_identical_values(self):
        """Test with all identical values."""
        values = [42.0] * 20

        stats = compute_statistics(values, bootstrap_iterations=0)

        assert stats.mean_sec == 42.0
        assert stats.median_sec == 42.0
        assert stats.std_sec == 0.0
        assert stats.min_sec == 42.0
        assert stats.max_sec == 42.0


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_precision_with_small_differences(self):
        """Test precision when values differ by small amounts."""
        base = 1000000.0
        values = [base + i * 0.001 for i in range(100)]

        numpy_std = float(np.std(values, ddof=1))
        python_std = _pure_python_std(values)

        # Should still be accurate
        relative_error = abs(numpy_std - python_std) / numpy_std
        assert relative_error < 1e-6

    def test_no_overflow_with_large_values(self):
        """Test that large values don't cause overflow."""
        values = [1e10 + i for i in range(100)]

        # Should not raise overflow error
        stats = compute_statistics(values, bootstrap_iterations=0)

        assert stats.mean_sec > 1e10
        assert stats.std_sec > 0

    @given(st.lists(st.floats(min_value=1.0, max_value=100.0), min_size=10, max_size=100))
    @settings(max_examples=20, deadline=None)
    def test_sum_stability(self, values: List[float]):
        """Test that sum computation is stable."""
        # Pure Python sum
        python_sum = sum(values)

        # NumPy sum
        numpy_sum = float(np.sum(values))

        # Should be very close
        if numpy_sum > 0:
            relative_error = abs(python_sum - numpy_sum) / numpy_sum
            assert relative_error < 1e-9
