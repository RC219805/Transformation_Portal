"""Tests for rl.cost_reward — pure arithmetic reward computation."""

from __future__ import annotations

import math

import pytest

from transformation_portal.rl.cost_reward import (
    BALANCED_WEIGHTS,
    COST_SENSITIVE_WEIGHTS,
    EFFICIENCY_FIRST_WEIGHTS,
    QUALITY_FIRST_WEIGHTS,
    CostWeights,
    RewardBreakdown,
    SystemStats,
    SystemStatsCollector,
    compute_cost_reward,
    compute_cost_reward_detailed,
)

pytestmark = pytest.mark.unit


def _stats(latency_sec=0.0, vram_bytes=0, scale_events=0, iteration_count=0) -> SystemStats:
    return SystemStats(
        latency_sec=latency_sec,
        vram_bytes=vram_bytes,
        scale_events=scale_events,
        iteration_count=iteration_count,
    )


def _metrics(score=1.0, psnr=40.0) -> dict:
    return {"score": score, "psnr": psnr}


class TestSystemStats:
    def test_vram_gb_converts_bytes_correctly(self):
        """vram_gb = vram_bytes / 1024^3."""
        stats = _stats(vram_bytes=1024**3)
        assert stats.vram_gb == pytest.approx(1.0)

    def test_vram_gb_zero_when_no_vram(self):
        """vram_bytes=0 → vram_gb=0.0."""
        assert _stats().vram_gb == 0.0

    def test_vram_gb_large_value(self):
        """8 GB expressed correctly."""
        stats = _stats(vram_bytes=8 * 1024**3)
        assert stats.vram_gb == pytest.approx(8.0)

    def test_to_dict_includes_vram_gb(self):
        """to_dict() exposes computed vram_gb field."""
        d = _stats(vram_bytes=2 * 1024**3).to_dict()
        assert "vram_gb" in d
        assert d["vram_gb"] == pytest.approx(2.0)

    def test_to_dict_all_fields_present(self):
        """to_dict() includes all expected keys."""
        d = _stats().to_dict()
        for key in ("latency_sec", "vram_bytes", "vram_gb", "scale_events", "gpu_util", "cpu_util", "iteration_count"):
            assert key in d


class TestComputeCostReward:
    def test_zero_costs_reward_equals_quality(self):
        """With all costs zero, reward equals the quality component."""
        w = CostWeights(quality=1.0, latency=0.0, vram=0.0, scaling=0.0, iteration=0.0)
        reward = compute_cost_reward(_metrics(score=0.8), _stats(), weights=w)
        assert reward == pytest.approx(0.8)

    def test_latency_penalty_reduces_reward(self):
        """High latency lowers reward."""
        base = compute_cost_reward(_metrics(score=1.0), _stats(latency_sec=0.0))
        penalised = compute_cost_reward(_metrics(score=1.0), _stats(latency_sec=10.0))
        assert penalised < base

    def test_vram_penalty_reduces_reward(self):
        """High VRAM usage lowers reward."""
        base = compute_cost_reward(_metrics(score=1.0), _stats(vram_bytes=0))
        penalised = compute_cost_reward(_metrics(score=1.0), _stats(vram_bytes=10 * 1024**3))
        assert penalised < base

    def test_scale_events_penalty_reduces_reward(self):
        """Scale events lower reward."""
        base = compute_cost_reward(_metrics(score=1.0), _stats(scale_events=0))
        penalised = compute_cost_reward(_metrics(score=1.0), _stats(scale_events=5))
        assert penalised < base

    def test_iteration_penalty_reduces_reward(self):
        """High iteration count lowers reward."""
        base = compute_cost_reward(_metrics(score=1.0), _stats(iteration_count=0))
        penalised = compute_cost_reward(_metrics(score=1.0), _stats(iteration_count=1000))
        assert penalised < base

    def test_baseline_improvement_positive_reward(self):
        """Improvement over baseline contributes positively."""
        reward = compute_cost_reward(_metrics(score=0.8), _stats(), baseline_score=0.5)
        assert reward > 0.0

    def test_no_improvement_penalised_by_costs(self):
        """When score == baseline, only cost penalties remain."""
        w = CostWeights(quality=1.0, latency=0.1, vram=0.0, scaling=0.0, iteration=0.0)
        reward = compute_cost_reward(_metrics(score=0.5), _stats(latency_sec=5.0), weights=w, baseline_score=0.5)
        assert reward < 0.0

    def test_default_weights_used_when_none(self):
        """weights=None uses CostWeights defaults without error."""
        reward = compute_cost_reward(_metrics(score=0.7), _stats(), weights=None)
        assert math.isfinite(reward)

    def test_exact_formula(self):
        """Reward matches the exact formula: quality - latency - vram - scaling - iteration."""
        w = CostWeights(quality=1.0, latency=0.05, vram=0.1, scaling=0.2, iteration=0.01)
        stats = SystemStats(latency_sec=2.0, vram_bytes=2 * 1024**3, scale_events=1, iteration_count=10)
        expected = (
            1.0 * (0.8 - 0.0)  # quality
            - 0.05 * 2.0  # latency
            - 0.1 * 2.0  # vram
            - 0.2 * 1  # scaling
            - 0.01 * 10  # iteration
        )
        reward = compute_cost_reward({"score": 0.8}, stats, weights=w, baseline_score=0.0)
        assert reward == pytest.approx(expected)

    def test_missing_score_key_defaults_to_zero(self):
        """If 'score' key is absent, defaults to 0.0."""
        reward = compute_cost_reward({}, _stats(), baseline_score=0.0)
        assert math.isfinite(reward)


class TestComputeCostRewardDetailed:
    def test_returns_reward_breakdown(self):
        """Returns a RewardBreakdown instance."""
        result = compute_cost_reward_detailed(_metrics(), _stats())
        assert isinstance(result, RewardBreakdown)

    def test_total_matches_simple_reward(self):
        """detailed.total matches compute_cost_reward()."""
        m = _metrics(score=0.75)
        s = _stats(latency_sec=1.5, vram_bytes=1024**3, scale_events=2)
        simple = compute_cost_reward(m, s)
        detailed = compute_cost_reward_detailed(m, s)
        assert detailed.total == pytest.approx(simple)

    def test_breakdown_components_sum_to_total(self):
        """quality - penalties sum to total."""
        bd = compute_cost_reward_detailed(_metrics(score=0.9), _stats(latency_sec=1.0, scale_events=1))
        expected = bd.quality_component - bd.latency_penalty - bd.vram_penalty - bd.scaling_penalty - bd.iteration_penalty
        assert bd.total == pytest.approx(expected)

    def test_to_dict_has_all_keys(self):
        """to_dict() includes all six breakdown fields."""
        d = compute_cost_reward_detailed(_metrics(), _stats()).to_dict()
        for key in ("total", "quality_component", "latency_penalty", "vram_penalty", "scaling_penalty", "iteration_penalty"):
            assert key in d


class TestWeightPresets:
    @pytest.mark.parametrize(
        "preset",
        [
            QUALITY_FIRST_WEIGHTS,
            BALANCED_WEIGHTS,
            EFFICIENCY_FIRST_WEIGHTS,
            COST_SENSITIVE_WEIGHTS,
        ],
    )
    def test_preset_is_cost_weights_instance(self, preset):
        """Each preset is a CostWeights instance."""
        assert isinstance(preset, CostWeights)

    def test_quality_first_has_low_latency_penalty(self):
        """QUALITY_FIRST_WEIGHTS penalises latency less than COST_SENSITIVE_WEIGHTS."""
        assert QUALITY_FIRST_WEIGHTS.latency < COST_SENSITIVE_WEIGHTS.latency

    def test_efficiency_first_has_higher_latency_weight_than_quality(self):
        """EFFICIENCY_FIRST_WEIGHTS prioritises speed over quality."""
        assert EFFICIENCY_FIRST_WEIGHTS.latency > QUALITY_FIRST_WEIGHTS.latency

    def test_all_preset_fields_positive(self):
        """All weight fields are positive (no negative penalties)."""
        for preset in [QUALITY_FIRST_WEIGHTS, BALANCED_WEIGHTS, EFFICIENCY_FIRST_WEIGHTS, COST_SENSITIVE_WEIGHTS]:
            for field in ("quality", "latency", "vram", "scaling", "iteration"):
                assert getattr(preset, field) >= 0.0


class TestSystemStatsCollector:
    def test_start_and_stop_returns_stats(self):
        """start/stop cycle returns a SystemStats."""
        collector = SystemStatsCollector()
        collector.start()
        stats = collector.stop()
        assert isinstance(stats, SystemStats)

    def test_latency_is_positive_after_stop(self):
        """Latency > 0 after start/stop."""
        collector = SystemStatsCollector()
        collector.start()
        stats = collector.stop()
        assert stats.latency_sec >= 0.0

    def test_record_scale_event_increments(self):
        """record_scale_event increments scale_events."""
        collector = SystemStatsCollector()
        collector.start()
        collector.record_scale_event()
        collector.record_scale_event()
        stats = collector.stop()
        assert stats.scale_events == 2

    def test_record_iteration_increments(self):
        """record_iteration increments iteration_count."""
        collector = SystemStatsCollector()
        collector.start()
        collector.record_iteration()
        stats = collector.stop()
        assert stats.iteration_count == 1

    def test_snapshot_does_not_stop_collection(self):
        """snapshot() returns stats without stopping; subsequent stop also works."""
        collector = SystemStatsCollector()
        collector.start()
        snap = collector.snapshot()
        assert isinstance(snap, SystemStats)
        stats = collector.stop()
        assert isinstance(stats, SystemStats)
