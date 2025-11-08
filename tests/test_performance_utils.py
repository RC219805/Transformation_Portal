#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for performance utilities."""
import time

import pytest

# Use proper package imports (assumes package is installed or PYTHONPATH is set)
# For development: pip install -e . or set PYTHONPATH to include src/
from transformation_portal.utils.performance import (
    timing_decorator,
    cache_result,
    retry_on_failure,
    PerformanceMonitor,
)


class TestTimingDecorator:
    """Tests for timing decorator."""

    def test_timing_decorator_success(self):
        """Test timing decorator logs execution time."""
        @timing_decorator
        def fast_function():
            return 42

        result = fast_function()
        assert result == 42

    def test_timing_decorator_with_exception(self):
        """Test timing decorator handles exceptions."""
        @timing_decorator
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_function()


class TestCacheResult:
    """Tests for caching decorator."""

    def test_cache_basic(self):
        """Test basic caching functionality."""
        call_count = {'count': 0}

        @cache_result(maxsize=4)
        def expensive_function(x):
            call_count['count'] += 1
            return x * 2

        # First call - cache miss
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count['count'] == 1

        # Second call with same arg - cache hit
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count['count'] == 1  # Not called again

        # Different arg - cache miss
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count['count'] == 2

    def test_cache_info(self):
        """Test cache info is accessible."""
        @cache_result(maxsize=4)
        def cached_func(x):
            return x

        cached_func(1)
        cached_func(1)
        cached_func(2)

        info = cached_func.cache_info()
        assert info.hits == 1
        assert info.misses == 2

    def test_cache_clear(self):
        """Test cache can be cleared."""
        @cache_result(maxsize=4)
        def cached_func(x):
            return x

        cached_func(1)
        cached_func(2)

        info_before = cached_func.cache_info()
        assert info_before.currsize == 2

        cached_func.cache_clear()

        info_after = cached_func.cache_info()
        assert info_after.currsize == 0


class TestRetryOnFailure:
    """Tests for retry decorator."""

    def test_retry_succeeds_eventually(self):
        """Test function succeeds after retries."""
        attempts = {'count': 0}

        @retry_on_failure(max_attempts=3, delay=0.01, backoff=1.0)
        def flaky_function():
            attempts['count'] += 1
            if attempts['count'] < 3:
                raise ValueError("not yet")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempts['count'] == 3

    def test_retry_exhausts_attempts(self):
        """Test function fails after max attempts."""
        @retry_on_failure(max_attempts=2, delay=0.01)
        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fails()

    def test_retry_specific_exceptions(self):
        """Test retry only catches specified exceptions."""
        @retry_on_failure(
            max_attempts=2,
            delay=0.01,
            exceptions=(IOError,)
        )
        def raises_wrong_exception():
            raise ValueError("wrong exception")

        # Should not retry for ValueError
        with pytest.raises(ValueError, match="wrong exception"):
            raises_wrong_exception()


class TestPerformanceMonitor:
    """Tests for performance monitor context manager."""

    def test_monitor_basic(self):
        """Test basic monitoring functionality."""
        with PerformanceMonitor("test_operation") as monitor:
            time.sleep(0.01)

        assert monitor.elapsed >= 0.01
        assert monitor.elapsed < 1.0  # Should be quick

    def test_monitor_with_items(self):
        """Test monitoring with item count."""
        with PerformanceMonitor("process_items", item_count=100) as monitor:
            time.sleep(0.01)

        assert monitor.item_count == 100
        assert monitor.throughput > 0
        # At least 100 items / 1 second = 100 items/sec
        # (but likely much more since we only sleep 0.01s)

    def test_monitor_with_exception(self):
        """Test monitor handles exceptions."""
        # Instantiate monitor outside pytest.raises so we can access it after
        monitor = PerformanceMonitor("failing_operation")
        with pytest.raises(ValueError):
            with monitor:
                raise ValueError("test error")

        # Should still have timing info
        assert monitor.elapsed > 0


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_cached_and_timed(self):
        """Test combining caching and timing decorators."""
        @timing_decorator
        @cache_result(maxsize=8)
        def expensive_operation(x):
            time.sleep(0.01)
            return x * 2

        # First call - slow
        start = time.perf_counter()
        result1 = expensive_operation(5)
        elapsed1 = time.perf_counter() - start

        # Second call - fast (cached)
        start = time.perf_counter()
        result2 = expensive_operation(5)
        elapsed2 = time.perf_counter() - start

        assert result1 == result2 == 10
        assert elapsed2 < elapsed1 / 2  # Much faster due to cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
