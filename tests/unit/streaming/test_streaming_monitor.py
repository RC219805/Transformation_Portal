"""Unit tests for streaming.streaming.RealTimeMonitor statistics."""

from __future__ import annotations

import pytest

from transformation_portal.streaming.streaming import RealTimeMonitor

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def test_items_processed_counts_every_record_beyond_the_window() -> None:
    # Finding L-09: items_processed used to saturate at window_size.
    monitor = RealTimeMonitor(window_size=100)
    for _ in range(1000):
        monitor.record(0.01)

    stats = monitor.get_stats()
    assert stats["items_processed"] == 1000
    assert len(monitor.timestamps) == 100
    assert len(monitor.processing_times) == 100


def test_items_processed_below_window_matches_record_count() -> None:
    monitor = RealTimeMonitor(window_size=10)
    for _ in range(3):
        monitor.record()

    assert monitor.get_stats()["items_processed"] == 3


def test_untimed_records_do_not_evict_timed_samples() -> None:
    monitor = RealTimeMonitor(window_size=3)
    monitor.record(0.5)
    for _ in range(5):
        monitor.record()

    assert monitor.processing_times == [0.5]
    assert monitor.avg_processing_time == pytest.approx(0.5)
    assert monitor.get_stats()["items_processed"] == 6
