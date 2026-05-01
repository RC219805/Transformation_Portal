"""Tests for metrics.timing — TimingContext, merge_timings, compute_overhead."""

from __future__ import annotations

import time

import pytest

from transformation_portal.metrics.timing import (
    TimingContext,
    compute_overhead,
    merge_timings,
    timing_context,
)

pytestmark = pytest.mark.unit


class TestTimingContext:
    def test_elapsed_sec_positive_after_block(self):
        """elapsed_sec is non-negative after exiting the context."""
        with TimingContext("phase") as ctx:
            pass
        assert ctx.elapsed_sec >= 0.0

    def test_elapsed_sec_accumulates_into_dict(self):
        """When timings_dict is provided, phase_name key is populated."""
        d = {}
        with TimingContext("load", timings_dict=d):
            pass
        assert "load" in d
        assert d["load"] >= 0.0

    def test_multiple_phases_in_same_dict(self):
        """Multiple contexts accumulate into the same dict."""
        d = {}
        with TimingContext("a", timings_dict=d):
            pass
        with TimingContext("b", timings_dict=d):
            pass
        assert "a" in d and "b" in d

    def test_no_dict_no_error(self):
        """timings_dict=None works without AttributeError."""
        with TimingContext("x") as ctx:
            pass
        assert ctx.elapsed_sec >= 0.0

    def test_device_none_no_sync_attempted(self):
        """device=None skips _sync_device without error."""
        with TimingContext("phase", device=None) as ctx:
            pass
        assert ctx.elapsed_sec >= 0.0

    def test_disable_sync_env_var_skips_sync(self, monkeypatch):
        """TP_DISABLE_DEVICE_SYNC=1 prevents device sync."""
        monkeypatch.setenv("TP_DISABLE_DEVICE_SYNC", "1")
        with TimingContext("phase", device="cuda") as ctx:
            pass
        assert ctx.elapsed_sec >= 0.0

    def test_elapsed_sec_reflects_actual_duration(self):
        """elapsed_sec is at least as long as a known sleep."""
        with TimingContext("slow") as ctx:
            time.sleep(0.01)
        assert ctx.elapsed_sec >= 0.005  # generous lower bound


class TestTimingContextManagerFunction:
    def test_yields_timing_context(self):
        """timing_context() yields a TimingContext instance."""
        with timing_context("inference") as t:
            assert isinstance(t, TimingContext)

    def test_elapsed_available_after_block(self):
        """elapsed_sec is non-negative after the context block exits."""
        with timing_context("inference") as t:
            pass
        assert t.elapsed_sec >= 0.0

    def test_accumulates_into_dict(self):
        """timing_context with dict parameter stores result."""
        d = {}
        with timing_context("phase", d):
            pass
        assert "phase" in d


class TestMergeTimings:
    def test_two_disjoint_dicts_merged(self):
        """Two dicts with different keys are combined."""
        result = merge_timings({"a": 1.0}, {"b": 2.0})
        assert result == {"a": 1.0, "b": 2.0}

    def test_duplicate_keys_summed(self):
        """Matching keys are summed, not overwritten."""
        result = merge_timings({"a": 1.0}, {"a": 2.0})
        assert result["a"] == pytest.approx(3.0)

    def test_empty_dicts_return_empty(self):
        """merge_timings() with no args or empty dicts → {}."""
        assert not merge_timings()
        assert not merge_timings({}, {})

    def test_three_dicts_merged(self):
        """Three dicts are all summed together."""
        result = merge_timings({"x": 1.0}, {"x": 2.0}, {"x": 3.0})
        assert result["x"] == pytest.approx(6.0)

    def test_original_dicts_not_mutated(self):
        """Input dicts are not modified."""
        d1 = {"a": 1.0}
        d2 = {"b": 2.0}
        merge_timings(d1, d2)
        assert d1 == {"a": 1.0}
        assert d2 == {"b": 2.0}


class TestComputeOverhead:
    def test_overhead_positive_when_phases_sum_less_than_total(self):
        """Overhead = total - phase_sum when positive."""
        timings = {"total": 10.0, "a": 4.0, "b": 4.0}
        assert compute_overhead(timings) == pytest.approx(2.0)

    def test_overhead_can_be_negative(self):
        """Overhead is negative when phase sum exceeds total."""
        timings = {"total": 5.0, "a": 3.0, "b": 4.0}
        assert compute_overhead(timings) == pytest.approx(-2.0)

    def test_missing_total_raises_value_error(self):
        """No 'total' key raises ValueError."""
        with pytest.raises(ValueError, match="total"):
            compute_overhead({"a": 1.0, "b": 2.0})

    def test_overhead_with_only_total(self):
        """Only 'total' key → overhead equals total."""
        assert compute_overhead({"total": 5.0}) == pytest.approx(5.0)
