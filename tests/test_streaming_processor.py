#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for streaming processing utilities."""

import pytest

from transformation_portal.streaming.streaming import (
    StreamingProcessor,
    stream_results,
    batch_stream,
    RealTimeMonitor,
)


class TestStreamingProcessor:
    """Tests for StreamingProcessor class."""

    def test_sequential_processing(self):
        """Test sequential processing (batch_size=1)."""
        def double(x):
            return x * 2

        processor = StreamingProcessor(process_func=double, batch_size=1)
        items = [1, 2, 3, 4, 5]

        results = list(processor.stream(iter(items)))

        assert results == [2, 4, 6, 8, 10]

    def test_batch_processing(self):
        """Test batch processing with threads."""
        def square(x):
            return x ** 2

        processor = StreamingProcessor(process_func=square, batch_size=2)
        items = [1, 2, 3, 4]

        results = list(processor.stream(iter(items)))

        # Results should be squares, possibly out of order due to threading
        assert sorted(results) == [1, 4, 9, 16]

    def test_batch_processing_with_remainder(self):
        """Test batch processing when items don't divide evenly."""
        def triple(x):
            return x * 3

        processor = StreamingProcessor(process_func=triple, batch_size=3)
        items = [1, 2, 3, 4, 5]  # 5 items with batch size 3

        results = list(processor.stream(iter(items)))

        assert sorted(results) == [3, 6, 9, 12, 15]

    def test_custom_max_workers(self):
        """Test custom max_workers setting."""
        def identity(x):
            return x

        processor = StreamingProcessor(
            process_func=identity,
            batch_size=4,
            max_workers=2
        )

        assert processor.max_workers == 2

    def test_default_max_workers(self):
        """Test that max_workers defaults to batch_size."""
        processor = StreamingProcessor(
            process_func=lambda x: x,
            batch_size=8
        )

        assert processor.max_workers == 8

    def test_empty_stream(self):
        """Test streaming with no items."""
        processor = StreamingProcessor(process_func=lambda x: x, batch_size=2)
        results = list(processor.stream(iter([])))

        assert results == []


class TestStreamResults:
    """Tests for stream_results function."""

    def test_stream_without_callback(self):
        """Test streaming without callback."""
        def add_one(x):
            return x + 1

        items = [1, 2, 3]
        results = list(stream_results(items, add_one))

        assert results == [2, 3, 4]

    def test_stream_with_callback(self):
        """Test streaming with callback."""
        def multiply_by_two(x):
            return x * 2

        callback_results = []

        def callback(result):
            callback_results.append(result)

        items = [1, 2, 3]
        results = list(stream_results(items, multiply_by_two, callback))

        assert results == [2, 4, 6]
        assert callback_results == [2, 4, 6]

    def test_empty_items(self):
        """Test with empty items list."""
        results = list(stream_results([], lambda x: x))
        assert results == []


class TestBatchStream:
    """Tests for batch_stream function."""

    def test_exact_batches(self):
        """Test streaming when items divide evenly into batches."""
        items = range(10)
        batches = list(batch_stream(iter(items), batch_size=5))

        assert len(batches) == 2
        assert batches[0] == [0, 1, 2, 3, 4]
        assert batches[1] == [5, 6, 7, 8, 9]

    def test_batches_with_remainder(self):
        """Test streaming when items don't divide evenly."""
        items = range(11)
        batches = list(batch_stream(iter(items), batch_size=5))

        assert len(batches) == 3
        assert batches[0] == [0, 1, 2, 3, 4]
        assert batches[1] == [5, 6, 7, 8, 9]
        assert batches[2] == [10]

    def test_single_batch(self):
        """Test when all items fit in one batch."""
        items = [1, 2, 3]
        batches = list(batch_stream(iter(items), batch_size=10))

        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_empty_stream(self):
        """Test with empty iterator."""
        batches = list(batch_stream(iter([]), batch_size=5))
        assert batches == []

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        items = [1, 2, 3]
        batches = list(batch_stream(iter(items), batch_size=1))

        assert len(batches) == 3
        assert batches == [[1], [2], [3]]


class TestRealTimeMonitor:
    """Tests for RealTimeMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = RealTimeMonitor(window_size=50)

        assert monitor.window_size == 50
        assert len(monitor.timestamps) == 0
        assert len(monitor.processing_times) == 0

    def test_record_without_processing_time(self):
        """Test recording events without processing time."""
        monitor = RealTimeMonitor()

        monitor.record()
        monitor.record()

        assert len(monitor.timestamps) == 2
        assert len(monitor.processing_times) == 0

    def test_record_with_processing_time(self):
        """Test recording events with processing time."""
        monitor = RealTimeMonitor()

        monitor.record(processing_time=0.1)
        monitor.record(processing_time=0.2)

        assert len(monitor.timestamps) == 2
        assert len(monitor.processing_times) == 2

    def test_window_size_limit(self):
        """Test that monitor respects window size limit."""
        monitor = RealTimeMonitor(window_size=3)

        for i in range(5):
            monitor.record(processing_time=0.1)

        assert len(monitor.timestamps) == 3
        assert len(monitor.processing_times) == 3

    def test_throughput_with_no_data(self):
        """Test throughput with no recorded events."""
        monitor = RealTimeMonitor()
        assert monitor.throughput == 0.0

    def test_throughput_with_one_event(self):
        """Test throughput with single event."""
        monitor = RealTimeMonitor()
        monitor.record()
        assert monitor.throughput == 0.0

    def test_throughput_calculation(self):
        """Test throughput calculation with multiple events."""
        import time

        monitor = RealTimeMonitor()

        monitor.record()
        time.sleep(0.1)
        monitor.record()
        time.sleep(0.1)
        monitor.record()

        # Should have positive throughput
        assert monitor.throughput > 0

    def test_avg_processing_time_empty(self):
        """Test average processing time with no data."""
        monitor = RealTimeMonitor()
        assert monitor.avg_processing_time == 0.0

    def test_avg_processing_time_calculation(self):
        """Test average processing time calculation."""
        monitor = RealTimeMonitor()

        monitor.record(processing_time=0.1)
        monitor.record(processing_time=0.2)
        monitor.record(processing_time=0.3)

        expected_avg = (0.1 + 0.2 + 0.3) / 3
        assert abs(monitor.avg_processing_time - expected_avg) < 0.001

    def test_total_elapsed(self):
        """Test total elapsed time tracking."""
        import time

        monitor = RealTimeMonitor()
        time.sleep(0.1)

        elapsed = monitor.total_elapsed
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Sanity check

    def test_get_stats(self):
        """Test getting statistics summary."""
        monitor = RealTimeMonitor()

        monitor.record(processing_time=0.1)
        monitor.record(processing_time=0.2)

        stats = monitor.get_stats()

        assert 'throughput' in stats
        assert 'avg_processing_time' in stats
        assert 'total_elapsed' in stats
        assert 'items_processed' in stats
        assert stats['items_processed'] == 2

    def test_throughput_zero_time_span(self):
        """Test throughput when time span is zero (edge case)."""
        monitor = RealTimeMonitor()

        # Manually set timestamps to same value
        monitor.timestamps = [100.0, 100.0]

        assert monitor.throughput == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
