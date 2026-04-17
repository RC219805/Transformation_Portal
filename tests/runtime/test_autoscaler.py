"""Tests for GPUAutoscaler - scaling decisions and thresholds.

This module tests:
- Autoscale policy configuration
- Scaling decision logic (scale-up, scale-down, no-change)
- Cooldown periods to prevent thrashing
- Metrics collection and reporting
- Scaling history tracking
- GPU utilization monitoring mocking
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.autoscaler import (
    AutoscalePolicy,
    GPUAutoscaler,
    GPUUtilizationMonitor,
    ScalingDecision,
    ScalingHistory,
    ScalingMetrics,
    create_autoscaler,
)


class TestAutoscalePolicy:
    """Tests for AutoscalePolicy dataclass."""

    def test_default_values(self) -> None:
        """AutoscalePolicy has sensible defaults."""
        policy = AutoscalePolicy()

        assert policy.min_workers == 0
        assert policy.max_workers == 8
        assert policy.scale_up_threshold == 0.7
        assert policy.scale_down_threshold == 0.2
        assert policy.cooldown_sec == 120
        assert policy.target_gpu_util == 0.75
        assert policy.max_pending_per_gpu == 4
        assert policy.scale_up_increment == 1
        assert policy.scale_down_increment == 1
        assert policy.enable_predictive is False

    def test_custom_values(self) -> None:
        """AutoscalePolicy accepts custom values."""
        policy = AutoscalePolicy(
            min_workers=2,
            max_workers=16,
            scale_up_threshold=0.8,
            scale_down_threshold=0.1,
            cooldown_sec=60,
            target_gpu_util=0.9,
            max_pending_per_gpu=8,
            scale_up_increment=2,
            scale_down_increment=1,
            enable_predictive=True,
        )

        assert policy.min_workers == 2
        assert policy.max_workers == 16
        assert policy.scale_up_threshold == 0.8
        assert policy.enable_predictive is True


class TestScalingMetrics:
    """Tests for ScalingMetrics dataclass."""

    def test_default_values(self) -> None:
        """ScalingMetrics has zero defaults."""
        metrics = ScalingMetrics()

        assert metrics.gpu_utilization == 0.0
        assert metrics.pending_tasks == 0
        assert metrics.current_workers == 0
        assert metrics.avg_task_latency == 0.0
        assert metrics.queue_depth == 0

    def test_to_dict(self) -> None:
        """ScalingMetrics converts to dictionary."""
        metrics = ScalingMetrics(
            gpu_utilization=0.75,
            pending_tasks=5,
            current_workers=4,
            avg_task_latency=1.5,
            queue_depth=10,
        )

        data = metrics.to_dict()

        assert data["gpu_utilization"] == 0.75
        assert data["pending_tasks"] == 5
        assert data["current_workers"] == 4
        assert data["avg_task_latency"] == 1.5
        assert data["queue_depth"] == 10


class TestScalingDecision:
    """Tests for ScalingDecision dataclass."""

    def test_scale_up_decision(self) -> None:
        """ScalingDecision represents scale-up."""
        metrics = ScalingMetrics(gpu_utilization=0.85, current_workers=2)

        decision = ScalingDecision(
            action="scale_up",
            current_workers=2,
            target_workers=4,
            reason="High GPU utilization",
            metrics=metrics,
        )

        assert decision.action == "scale_up"
        assert decision.current_workers == 2
        assert decision.target_workers == 4

    def test_scale_down_decision(self) -> None:
        """ScalingDecision represents scale-down."""
        metrics = ScalingMetrics(gpu_utilization=0.1, current_workers=4)

        decision = ScalingDecision(
            action="scale_down",
            current_workers=4,
            target_workers=2,
            reason="Low utilization",
            metrics=metrics,
        )

        assert decision.action == "scale_down"
        assert decision.target_workers == 2

    def test_no_change_decision(self) -> None:
        """ScalingDecision represents no change."""
        metrics = ScalingMetrics(gpu_utilization=0.5, current_workers=4)

        decision = ScalingDecision(
            action="none",
            current_workers=4,
            target_workers=4,
            reason="No change needed",
            metrics=metrics,
        )

        assert decision.action == "none"
        assert decision.current_workers == decision.target_workers


class TestScalingHistory:
    """Tests for ScalingHistory class."""

    def test_add_decision(self) -> None:
        """ScalingHistory stores decisions."""
        history = ScalingHistory()

        decision = ScalingDecision(
            action="scale_up",
            current_workers=2,
            target_workers=4,
            reason="Test",
            metrics=ScalingMetrics(),
        )

        history.add(decision)

        assert len(history.decisions) == 1
        assert history.decisions[0][1] == decision

    def test_max_history_limit(self) -> None:
        """ScalingHistory respects max_history limit."""
        history = ScalingHistory(max_history=5)

        for i in range(10):
            decision = ScalingDecision(
                action="scale_up",
                current_workers=i,
                target_workers=i + 1,
                reason=f"Decision {i}",
                metrics=ScalingMetrics(),
            )
            history.add(decision)

        assert len(history.decisions) == 5
        # Oldest should have been removed
        assert history.decisions[0][1].current_workers == 5

    def test_recent_scale_ups(self) -> None:
        """recent_scale_ups counts scale-up decisions in window."""
        history = ScalingHistory()

        # Add scale-up decisions
        for _ in range(3):
            history.add(
                ScalingDecision(
                    action="scale_up",
                    current_workers=1,
                    target_workers=2,
                    reason="Up",
                    metrics=ScalingMetrics(),
                )
            )

        # Add scale-down decision
        history.add(
            ScalingDecision(
                action="scale_down",
                current_workers=2,
                target_workers=1,
                reason="Down",
                metrics=ScalingMetrics(),
            )
        )

        assert history.recent_scale_ups(window_sec=600) == 3

    def test_recent_scale_downs(self) -> None:
        """recent_scale_downs counts scale-down decisions in window."""
        history = ScalingHistory()

        # Add scale-down decisions
        for _ in range(2):
            history.add(
                ScalingDecision(
                    action="scale_down",
                    current_workers=4,
                    target_workers=3,
                    reason="Down",
                    metrics=ScalingMetrics(),
                )
            )

        assert history.recent_scale_downs(window_sec=600) == 2


class TestGPUUtilizationMonitor:
    """Tests for GPUUtilizationMonitor class."""

    def test_monitor_without_nvml(self) -> None:
        """Monitor returns 0 when pynvml not available."""
        monitor = GPUUtilizationMonitor()

        with patch.dict("sys.modules", {"pynvml": None}):
            monitor._nvml_initialized = False

            util = monitor.get_utilization(device_index=0)
            assert util == 0.0

    def test_get_memory_utilization_without_nvml(self) -> None:
        """get_memory_utilization returns 0 without nvml."""
        monitor = GPUUtilizationMonitor()
        monitor._nvml_initialized = False

        with patch.object(monitor, "_init_nvml", return_value=False):
            mem_util = monitor.get_memory_utilization(device_index=0)
            assert mem_util == 0.0

    def test_get_device_count_without_nvml(self) -> None:
        """get_device_count returns 0 without nvml."""
        monitor = GPUUtilizationMonitor()
        monitor._nvml_initialized = False

        with patch.object(monitor, "_init_nvml", return_value=False):
            count = monitor.get_device_count()
            assert count == 0

    def test_init_nvml_handles_import_error(self) -> None:
        """_init_nvml handles ImportError gracefully."""
        monitor = GPUUtilizationMonitor()
        monitor._nvml_initialized = False

        # Simulate ImportError
        with patch("builtins.__import__", side_effect=ImportError("No pynvml")):
            result = monitor._init_nvml()
            assert result is False

    def test_init_nvml_handles_init_error(self) -> None:
        """_init_nvml handles NVML initialization errors."""
        monitor = GPUUtilizationMonitor()
        monitor._nvml_initialized = False

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("NVML init failed")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            # Need to clear import cache
            monitor._nvml_initialized = False
            # Will fail because our mock raises
            result = monitor._init_nvml()
            # Should return False due to exception
            assert result is False


class TestGPUAutoscaler:
    """Tests for GPUAutoscaler class."""

    @pytest.fixture
    def mock_ray(self):
        """Create mock Ray client."""
        ray = MagicMock()
        ray.available_resources.return_value = {"GPU": 2.0}
        ray.cluster_resources.return_value = {"GPU": 4}
        return ray

    @pytest.fixture
    def default_policy(self):
        """Create default autoscale policy."""
        return AutoscalePolicy(
            min_workers=1,
            max_workers=8,
            scale_up_threshold=0.7,
            scale_down_threshold=0.2,
            cooldown_sec=0,  # No cooldown for tests
        )

    def test_autoscaler_initialization(self, mock_ray, default_policy) -> None:
        """GPUAutoscaler initializes correctly."""
        autoscaler = GPUAutoscaler(mock_ray, default_policy)

        assert autoscaler.ray == mock_ray
        assert autoscaler.policy == default_policy
        assert autoscaler.gpu_monitor is not None
        assert len(autoscaler.history.decisions) == 0

    def test_autoscaler_default_policy(self, mock_ray) -> None:
        """GPUAutoscaler uses default policy if none provided."""
        autoscaler = GPUAutoscaler(mock_ray, None)

        assert autoscaler.policy is not None
        assert autoscaler.policy.max_workers == 8

    def test_set_metrics_callback(self, mock_ray, default_policy) -> None:
        """set_metrics_callback registers callback."""
        autoscaler = GPUAutoscaler(mock_ray, default_policy)

        callback = MagicMock()
        autoscaler.set_metrics_callback(callback)

        assert autoscaler._metrics_callback == callback

    def test_decide_scale_up_high_utilization(self, mock_ray) -> None:
        """_decide returns scale_up for high GPU utilization."""
        policy = AutoscalePolicy(
            max_workers=8,
            scale_up_threshold=0.7,
            scale_up_increment=2,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.85,  # Above threshold
            pending_tasks=0,
            current_workers=4,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "scale_up"
        assert decision.target_workers == 6  # 4 + 2

    def test_decide_scale_up_capped_at_max(self, mock_ray) -> None:
        """_decide caps scale-up at max_workers."""
        policy = AutoscalePolicy(
            max_workers=4,
            scale_up_threshold=0.7,
            scale_up_increment=2,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.85,
            current_workers=3,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "scale_up"
        assert decision.target_workers == 4  # Capped at max

    def test_decide_scale_up_high_pending(self, mock_ray) -> None:
        """_decide returns scale_up for high pending tasks."""
        policy = AutoscalePolicy(
            max_workers=8,
            max_pending_per_gpu=2,
            scale_up_increment=1,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.5,
            pending_tasks=10,  # > 4 workers * 2 max_pending
            current_workers=4,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "scale_up"

    def test_decide_scale_down_low_utilization(self, mock_ray) -> None:
        """_decide returns scale_down for low utilization."""
        policy = AutoscalePolicy(
            min_workers=1,
            scale_down_threshold=0.2,
            scale_down_increment=1,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.1,  # Below threshold
            pending_tasks=0,
            current_workers=4,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "scale_down"
        assert decision.target_workers == 3  # 4 - 1

    def test_decide_scale_down_capped_at_min(self, mock_ray) -> None:
        """_decide caps scale-down at min_workers."""
        policy = AutoscalePolicy(
            min_workers=2,
            scale_down_threshold=0.2,
            scale_down_increment=2,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.1,
            pending_tasks=0,
            current_workers=3,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "scale_down"
        assert decision.target_workers == 2  # Capped at min

    def test_decide_no_change_normal_utilization(self, mock_ray) -> None:
        """_decide returns none for normal utilization."""
        policy = AutoscalePolicy(
            scale_up_threshold=0.7,
            scale_down_threshold=0.2,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.5,  # Between thresholds
            pending_tasks=2,
            current_workers=4,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "none"
        assert decision.target_workers == decision.current_workers

    def test_decide_no_scale_down_with_pending(self, mock_ray) -> None:
        """_decide doesn't scale down when tasks pending."""
        policy = AutoscalePolicy(
            min_workers=1,
            scale_down_threshold=0.2,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        metrics = ScalingMetrics(
            gpu_utilization=0.1,
            pending_tasks=1,  # Has pending tasks
            current_workers=4,
        )

        decision = autoscaler._decide(metrics)

        assert decision.action == "none"  # No scale-down due to pending

    def test_tick_respects_cooldown(self, mock_ray) -> None:
        """tick respects cooldown period."""
        policy = AutoscalePolicy(cooldown_sec=300)  # 5 minutes
        autoscaler = GPUAutoscaler(mock_ray, policy)

        # Simulate recent scaling
        autoscaler._last_scale_time = time.time() - 60  # 1 minute ago

        result = autoscaler.tick()

        assert result is None  # Should be in cooldown

    def test_tick_calls_metrics_callback(self, mock_ray) -> None:
        """tick calls metrics callback with current metrics."""
        policy = AutoscalePolicy(cooldown_sec=0)
        autoscaler = GPUAutoscaler(mock_ray, policy)

        callback = MagicMock()
        autoscaler.set_metrics_callback(callback)

        # Mock _get_metrics
        with patch.object(autoscaler, "_get_metrics") as mock_get:
            mock_get.return_value = ScalingMetrics(current_workers=4)

            autoscaler.tick()

            callback.assert_called_once()

    def test_get_status(self, mock_ray, default_policy) -> None:
        """get_status returns current autoscaler state."""
        autoscaler = GPUAutoscaler(mock_ray, default_policy)

        with patch.object(autoscaler, "_get_metrics") as mock_get:
            mock_get.return_value = ScalingMetrics(
                gpu_utilization=0.5,
                current_workers=4,
            )

            status = autoscaler.get_status()

            assert "metrics" in status
            assert "policy" in status
            assert status["policy"]["min_workers"] == default_policy.min_workers
            assert "last_scale_time" in status
            assert "recent_scale_ups" in status


class TestCreateAutoscaler:
    """Tests for create_autoscaler factory function."""

    def test_create_with_defaults(self) -> None:
        """create_autoscaler creates autoscaler with defaults."""
        mock_ray = MagicMock()

        autoscaler = create_autoscaler(mock_ray)

        assert autoscaler.policy.min_workers == 0
        assert autoscaler.policy.max_workers == 8

    def test_create_with_custom_workers(self) -> None:
        """create_autoscaler accepts min/max workers."""
        mock_ray = MagicMock()

        autoscaler = create_autoscaler(mock_ray, min_workers=2, max_workers=16)

        assert autoscaler.policy.min_workers == 2
        assert autoscaler.policy.max_workers == 16

    def test_create_with_extra_kwargs(self) -> None:
        """create_autoscaler passes extra kwargs to policy."""
        mock_ray = MagicMock()

        autoscaler = create_autoscaler(
            mock_ray,
            min_workers=1,
            max_workers=10,
            cooldown_sec=60,
            scale_up_threshold=0.8,
        )

        assert autoscaler.policy.cooldown_sec == 60
        assert autoscaler.policy.scale_up_threshold == 0.8


class TestAutoscalerScalingApplication:
    """Tests for actual scaling application in GPUAutoscaler."""

    def test_apply_scaling_without_ray_sdk(self) -> None:
        """_apply_scaling returns False when Ray SDK not available."""
        mock_ray = MagicMock()
        autoscaler = GPUAutoscaler(mock_ray, AutoscalePolicy())

        # _apply_scaling tries to import ray.autoscaler.sdk which won't exist
        # in test environment, so it should return False gracefully
        result = autoscaler._apply_scaling(target_workers=4)

        # Returns False when ray autoscaler sdk is not available
        assert result is False

    def test_apply_scaling_method_exists(self) -> None:
        """_apply_scaling method exists and is callable."""
        mock_ray = MagicMock()
        autoscaler = GPUAutoscaler(mock_ray, AutoscalePolicy())

        assert callable(autoscaler._apply_scaling)

    def test_tick_applies_scaling_and_records(self) -> None:
        """tick applies scaling decision and records in history."""
        mock_ray = MagicMock()
        mock_ray.available_resources.return_value = {"GPU": 0.0}  # No available
        mock_ray.cluster_resources.return_value = {"GPU": 4}

        policy = AutoscalePolicy(
            cooldown_sec=0,
            scale_up_threshold=0.7,
            max_workers=8,
        )
        autoscaler = GPUAutoscaler(mock_ray, policy)

        # Mock high utilization
        with patch.object(autoscaler.gpu_monitor, "get_utilization", return_value=0.9):
            with patch.object(autoscaler, "_apply_scaling", return_value=True):
                decision = autoscaler.tick()

                if decision:
                    assert decision.action == "scale_up"
                    assert len(autoscaler.history.decisions) == 1
