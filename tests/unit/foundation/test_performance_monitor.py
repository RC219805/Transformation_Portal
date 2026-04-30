"""Unit tests for foundation.performance_monitor.

Covers MetricsCollector recording/querying, OperationProfile statistics,
PerformanceMonitor decorator and context-manager profiling, benchmark(),
enable/disable guards, and the _std helper — all on CPU with no real GPU.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

CPU_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# MetricType enum
# ---------------------------------------------------------------------------


class TestMetricTypeEnum:
    def test_all_expected_types_exist(self):
        from transformation_portal.foundation.performance_monitor import MetricType

        values = {m.value for m in MetricType}
        assert "latency" in values
        assert "throughput" in values
        assert "memory" in values
        assert "gpu_utilization" in values
        assert "bandwidth" in values


# ---------------------------------------------------------------------------
# OperationProfile
# ---------------------------------------------------------------------------


class TestOperationProfile:
    def test_initial_state(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("test_op")
        assert p.total_calls == 0
        assert p.avg_time_seconds == 0.0

    def test_update_increments_calls(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("op")
        p.update(0.1)
        p.update(0.3)
        assert p.total_calls == 2

    def test_update_tracks_min_max(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("op")
        p.update(0.5)
        p.update(0.1)
        p.update(0.9)
        assert p.min_time_seconds == pytest.approx(0.1)
        assert p.max_time_seconds == pytest.approx(0.9)

    def test_update_computes_average(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("op")
        p.update(0.2)
        p.update(0.4)
        assert p.avg_time_seconds == pytest.approx(0.3)

    def test_to_dict_keys(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("render")
        p.update(0.1, memory_mb=10.0)
        d = p.to_dict()
        for key in ("operation", "calls", "total_time_s", "avg_time_ms", "min_time_ms", "max_time_ms", "avg_memory_mb"):
            assert key in d

    def test_to_dict_avg_time_ms_is_millis(self):
        from transformation_portal.foundation.performance_monitor import OperationProfile

        p = OperationProfile("op")
        p.update(1.0)  # 1 second
        assert p.to_dict()["avg_time_ms"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_record_metric_stores_entry(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("test", MetricType.LATENCY, 42.0, "ms")
        assert len(col.metrics) == 1
        assert col.metrics[0].value == pytest.approx(42.0)

    def test_record_metric_enforces_max(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector(max_metrics=5)
        for i in range(10):
            col.record_metric(f"m{i}", MetricType.LATENCY, float(i), "ms")
        assert len(col.metrics) <= 5

    def test_record_operation_creates_profile(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector

        col = MetricsCollector()
        col.record_operation("depth_inference", 0.25)
        assert "depth_inference" in col.operation_profiles
        assert col.operation_profiles["depth_inference"].total_calls == 1

    def test_record_operation_twice_accumulates(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector

        col = MetricsCollector()
        col.record_operation("op", 0.1)
        col.record_operation("op", 0.3)
        assert col.operation_profiles["op"].total_calls == 2
        assert col.operation_profiles["op"].avg_time_seconds == pytest.approx(0.2)

    def test_get_operation_profile_returns_none_for_unknown(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector

        col = MetricsCollector()
        assert col.get_operation_profile("nonexistent") is None

    def test_get_metrics_filter_by_type(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("lat", MetricType.LATENCY, 10.0, "ms")
        col.record_metric("mem", MetricType.MEMORY, 200.0, "MB")
        latency_metrics = col.get_metrics(metric_type=MetricType.LATENCY)
        assert all(m.metric_type == MetricType.LATENCY for m in latency_metrics)
        assert len(latency_metrics) == 1

    def test_get_metrics_filter_by_name(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("depth_latency", MetricType.LATENCY, 10.0, "ms")
        col.record_metric("upscale_latency", MetricType.LATENCY, 20.0, "ms")
        results = col.get_metrics(name_filter="depth")
        assert len(results) == 1

    def test_get_metrics_limit(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        for i in range(10):
            col.record_metric(f"m{i}", MetricType.LATENCY, float(i), "ms")
        assert len(col.get_metrics(limit=3)) == 3

    def test_get_summary_empty(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector

        col = MetricsCollector()
        summary = col.get_summary()
        assert summary == {"status": "no_metrics"}

    def test_get_summary_has_by_type(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("x", MetricType.LATENCY, 5.0, "ms")
        summary = col.get_summary()
        assert "by_type" in summary
        assert "latency" in summary["by_type"]

    def test_get_summary_statistics_correct(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("a", MetricType.LATENCY, 10.0, "ms")
        col.record_metric("b", MetricType.LATENCY, 20.0, "ms")
        stats = col.get_summary()["by_type"]["latency"]
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(20.0)
        assert stats["avg"] == pytest.approx(15.0)

    def test_clear_removes_all_data(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector, MetricType

        col = MetricsCollector()
        col.record_metric("x", MetricType.LATENCY, 1.0, "ms")
        col.record_operation("op", 0.1)
        col.clear()
        assert len(col.metrics) == 0
        assert len(col.operation_profiles) == 0

    def test_get_all_profiles_returns_list_of_dicts(self):
        from transformation_portal.foundation.performance_monitor import MetricsCollector

        col = MetricsCollector()
        col.record_operation("op1", 0.1)
        col.record_operation("op2", 0.2)
        profiles = col.get_all_profiles()
        assert isinstance(profiles, list)
        assert all(isinstance(p, dict) for p in profiles)


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    def test_init_sets_cpu_device(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        assert monitor.device == CPU_DEVICE

    def test_initially_enabled(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        assert monitor._enabled is True

    def test_enable_disable_toggle(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        monitor.disable()
        assert monitor._enabled is False
        monitor.enable()
        assert monitor._enabled is True

    def test_repr_contains_device(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        assert "cpu" in repr(monitor)

    def test_profile_operation_decorator_records_metric(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)

        @monitor.profile_operation("test_op")
        def noop():
            return 42

        noop()
        profile = monitor.collector.get_operation_profile("test_op")
        assert profile is not None
        assert profile.total_calls == 1

    def test_profile_operation_preserves_return_value(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)

        @monitor.profile_operation("identity")
        def identity(x):
            return x * 2

        assert identity(21) == 42

    def test_profile_operation_skipped_when_disabled(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        monitor.disable()

        @monitor.profile_operation("no_record")
        def noop():
            pass

        noop()
        assert monitor.collector.get_operation_profile("no_record") is None

    def test_profile_context_records_metric(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        with monitor.profile_context("ctx_op"):
            _ = sum(range(100))

        profile = monitor.collector.get_operation_profile("ctx_op")
        assert profile is not None
        assert profile.total_calls == 1

    def test_profile_context_skipped_when_disabled(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        monitor.disable()
        with monitor.profile_context("disabled_ctx"):
            pass

        assert monitor.collector.get_operation_profile("disabled_ctx") is None

    def test_reset_clears_collector(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)

        @monitor.profile_operation("op")
        def noop():
            pass

        noop()
        monitor.reset()
        assert monitor.collector.get_operation_profile("op") is None

    def test_get_summary_no_ops_message(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        summary = monitor.get_summary()
        assert "No operations profiled" in summary

    def test_get_summary_with_ops_contains_header(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)

        @monitor.profile_operation("render")
        def noop():
            pass

        noop()
        summary = monitor.get_summary()
        assert "PERFORMANCE PROFILE SUMMARY" in summary

    def test_export_metrics_writes_json(self, tmp_path):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)

        @monitor.profile_operation("export_op")
        def noop():
            pass

        noop()
        outfile = tmp_path / "metrics.json"
        monitor.export_metrics(str(outfile))
        assert outfile.exists()
        import json

        data = json.loads(outfile.read_text())
        assert "profiles" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_returns_expected_keys(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        result = monitor.benchmark(lambda: None, num_iterations=5, warmup_iterations=1)
        for key in ("iterations", "avg_time_ms", "min_time_ms", "max_time_ms", "throughput_per_sec"):
            assert key in result

    def test_benchmark_iterations_count(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        result = monitor.benchmark(lambda: None, num_iterations=10, warmup_iterations=2)
        assert result["iterations"] == 10

    def test_benchmark_throughput_positive(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        result = monitor.benchmark(lambda: sum(range(100)), num_iterations=5, warmup_iterations=1)
        assert result["throughput_per_sec"] > 0


# ---------------------------------------------------------------------------
# _std helper
# ---------------------------------------------------------------------------


class TestStdHelper:
    def test_std_single_value_returns_zero(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        assert monitor._std([5.0]) == 0.0

    def test_std_empty_returns_zero(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        assert monitor._std([]) == 0.0

    def test_std_known_values(self):
        from transformation_portal.foundation.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(device=CPU_DEVICE)
        # [2, 4, 4, 4, 5, 5, 7, 9] → population std = 2.0
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert monitor._std(values) == pytest.approx(2.0)
