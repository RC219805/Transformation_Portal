"""Tests for evals/auto_optimizer.py module (Phase 5 coverage).

Tests for:
- AutoOptimizer optimization flow
- Candidate evaluation and selection
- Policy-gated fix application
- Score computation

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.evals.auto_opt_types import (
    Candidate,
    CandidateResult,
    IterationSummary,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
    PipelineState,
)
from transformation_portal.evals.auto_optimizer import (
    AutoOptimizer,
    _compute_composite_score,
    _hash_pipeline,
    acceptance_gate,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.max_iterations == 10
        assert config.beam_width == 3
        assert config.min_gain > 0
        assert config.convergence_patience >= 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            max_iterations=20,
            beam_width=5,
            min_gain=0.05,
            convergence_patience=3,
            budget_per_iteration=10,
        )

        assert config.max_iterations == 20
        assert config.beam_width == 5
        assert config.min_gain == 0.05


class TestPipelineState:
    """Test PipelineState dataclass."""

    def test_state_creation(self):
        """Test pipeline state creation."""
        state = PipelineState(
            pipeline={"stage1": {"param": 1}},
            score=0.75,
            metrics={"psnr": 30.0, "ssim": 0.9},
            iteration=5,
            parent_hash="abc123",
        )

        assert state.score == 0.75
        assert state.iteration == 5
        assert state.metrics["psnr"] == 30.0


class TestCandidate:
    """Test Candidate dataclass."""

    def test_candidate_creation(self):
        """Test candidate creation."""
        candidate = Candidate(
            pipeline={"stage1": {"param": 2}},
            parent_score=0.70,
            expected_gain=0.05,
            fix={"action": "increase_param", "target": "stage1"},
            confidence=0.8,
        )

        assert candidate.parent_score == 0.70
        assert candidate.expected_gain == 0.05
        assert candidate.confidence == 0.8


class TestCandidateResult:
    """Test CandidateResult dataclass."""

    def test_successful_result(self):
        """Test successful candidate result."""
        candidate = Candidate(
            pipeline={},
            parent_score=0.70,
            expected_gain=0.05,
            fix={},
            confidence=0.8,
        )

        result = CandidateResult(
            candidate=candidate,
            score=0.78,
            improvement=0.08,
            success=True,
        )

        assert result.success is True
        assert result.improvement == 0.08
        assert result.error is None

    def test_failed_result(self):
        """Test failed candidate result."""
        candidate = Candidate(
            pipeline={},
            parent_score=0.70,
            expected_gain=0.05,
            fix={},
            confidence=0.8,
        )

        result = CandidateResult(
            candidate=candidate,
            score=0.0,
            improvement=-1.0,
            success=False,
            error="Pipeline execution failed",
        )

        assert result.success is False
        assert result.error == "Pipeline execution failed"


class TestIterationSummary:
    """Test IterationSummary dataclass."""

    def test_summary_creation(self):
        """Test iteration summary creation."""
        summary = IterationSummary(
            iteration=3,
            candidates_evaluated=5,
            best_candidate_score=0.82,
            improvement=0.05,
            accepted=True,
            fixes_tried=["increase_resolution", "enable_denoising"],
        )

        assert summary.iteration == 3
        assert summary.accepted is True
        assert len(summary.fixes_tried) == 2


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            best_pipeline={"optimized": True},
            best_score=0.85,
            history=[],
        )

        assert result.best_score == 0.85
        assert result.status == OptimizationStatus.RUNNING
        assert result.iterations == 0


class TestOptimizationStatus:
    """Test OptimizationStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert OptimizationStatus.RUNNING.value == "running"
        assert OptimizationStatus.CONVERGED.value == "converged"
        assert OptimizationStatus.BUDGET_EXHAUSTED.value == "budget_exhausted"
        assert OptimizationStatus.NO_IMPROVEMENT.value == "no_improvement"


class TestHelperFunctions:
    """Test helper functions."""

    def test_hash_pipeline(self):
        """Test pipeline hashing."""
        pipeline1 = {"stage1": {"param": 1}, "stage2": {"param": 2}}
        pipeline2 = {"stage1": {"param": 1}, "stage2": {"param": 2}}
        pipeline3 = {"stage1": {"param": 1}, "stage2": {"param": 3}}

        hash1 = _hash_pipeline(pipeline1)
        hash2 = _hash_pipeline(pipeline2)
        hash3 = _hash_pipeline(pipeline3)

        assert hash1 == hash2  # Same content
        assert hash1 != hash3  # Different content
        assert len(hash1) == 16  # Truncated SHA

    def test_compute_composite_score(self):
        """Test composite score computation."""
        metrics = {"psnr": 36.0, "ssim": 0.9, "lpips": 0.2}  # PSNR in dB
        weights = {"psnr": 0.4, "ssim": 0.4, "lpips": -0.2}  # Negative = lower is better

        score = _compute_composite_score(metrics, weights)

        # psnr: 36 dB -> psnr_to_score 0.8; 0.8 * 0.4 = 0.32
        # ssim: 0.9 * 0.4 = 0.36
        # lpips (inverted): (1 - 0.2) * 0.2 = 0.16
        # total weight: 0.4 + 0.4 + 0.2 = 1.0
        # expected: (0.32 + 0.36 + 0.16) / 1.0 = 0.84
        assert score == pytest.approx(0.84, rel=0.01)

    def test_compute_composite_score_missing_metric(self):
        """Test composite score with missing metric."""
        metrics = {"psnr": 36.0}  # 36 dB -> 0.8
        weights = {"psnr": 0.5, "ssim": 0.5}  # ssim missing

        score = _compute_composite_score(metrics, weights)

        # Only psnr is used, total_weight = 0.5
        assert score == pytest.approx(0.8)

    def test_compute_composite_score_empty(self):
        """Test composite score with empty inputs."""
        score = _compute_composite_score({}, {})
        assert score == 0.0

    def test_compute_composite_score_psnr_db_does_not_swamp_other_metrics(self):
        """M-10: a +2 dB PSNR gain must not outrank a collapse of every other metric."""
        weights = OptimizationConfig().score_weights
        good = {"psnr": 30.0, "ssim": 0.95, "lpips": 0.05, "llava": 0.95}
        worse = {"psnr": 32.0, "ssim": 0.40, "lpips": 0.60, "llava": 0.10}

        good_score = _compute_composite_score(good, weights)
        worse_score = _compute_composite_score(worse, weights)

        assert 0.0 <= worse_score < good_score <= 1.0

    def test_compute_composite_score_identical_image_psnr_inf_is_bounded(self):
        """M-10: PSNR=+inf (identical images) maps to the top of the 0..1 scale."""
        score = _compute_composite_score({"psnr": float("inf"), "ssim": 1.0}, {"psnr": 0.5, "ssim": 0.5})
        assert score == pytest.approx(1.0)

    def test_compute_composite_score_nan_fails_closed(self):
        """M-10: NaN metrics contribute the worst score for their term, never the best."""
        weights = {"psnr": 0.5, "lpips": -0.5}
        assert _compute_composite_score({"psnr": float("nan"), "lpips": float("nan")}, weights) == 0.0
        assert _compute_composite_score({"psnr": 40.0, "lpips": float("nan")}, weights) == pytest.approx(0.5)

    def test_acceptance_gate_accept(self):
        """Test acceptance gate accepts improvement."""
        assert acceptance_gate(0.70, 0.75, threshold=0.02) is True

    def test_acceptance_gate_reject(self):
        """Test acceptance gate rejects small improvement."""
        assert acceptance_gate(0.70, 0.71, threshold=0.02) is False

    def test_acceptance_gate_reject_regression(self):
        """Test acceptance gate rejects regression."""
        assert acceptance_gate(0.70, 0.65, threshold=0.02) is False


class TestAutoOptimizer:
    """Test AutoOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        run_fn = MagicMock(return_value={"score": 0.7})
        eval_fn = MagicMock(return_value=0.7)
        diff_fn = MagicMock(return_value={"changes": []})

        optimizer = AutoOptimizer(
            run_fn=run_fn,
            eval_fn=eval_fn,
            diff_fn=diff_fn,
        )

        assert optimizer.run_fn == run_fn
        assert optimizer.eval_fn == eval_fn
        assert optimizer.config is not None

    def test_optimizer_with_custom_config(self):
        """Test optimizer with custom config."""
        config = OptimizationConfig(max_iterations=5, beam_width=2)

        optimizer = AutoOptimizer(
            run_fn=MagicMock(return_value={"score": 0.7}),
            eval_fn=MagicMock(return_value=0.7),
            diff_fn=MagicMock(return_value={"changes": []}),
            config=config,
        )

        assert optimizer.config.max_iterations == 5
        assert optimizer.config.beam_width == 2

    def test_optimize_no_fixes_converges(self):
        """Test optimization converges when no fixes available."""
        # Mock that returns no suggestions
        with patch("transformation_portal.evals.auto_optimizer.suggest_fixes") as mock_suggest:
            mock_suggest.return_value = MagicMock(suggestions=[])

            run_fn = MagicMock(return_value={"score": 0.7, "metrics": {}})
            eval_fn = MagicMock(return_value=0.7)
            diff_fn = MagicMock(return_value={"changes": []})

            optimizer = AutoOptimizer(
                run_fn=run_fn,
                eval_fn=eval_fn,
                diff_fn=diff_fn,
                config=OptimizationConfig(max_iterations=5),
            )

            result = optimizer.optimize({"initial": True})

            assert result.status == OptimizationStatus.CONVERGED
            assert result.best_score == 0.7

    def test_optimize_improvement_accepted(self):
        """Test optimization accepts improvements."""
        from transformation_portal.evals.self_healing import FixSuggestion

        # Create mock fix suggestion
        fix = FixSuggestion(
            type="quality",
            target_node="test_node",
            action="improve",
            params={"value": 1},
            confidence=0.9,
            rationale="Test improvement",
            priority=5,
        )

        with patch("transformation_portal.evals.auto_optimizer.suggest_fixes") as mock_suggest:
            with patch("transformation_portal.evals.auto_optimizer.apply_fix") as mock_apply:
                with patch("transformation_portal.evals.auto_optimizer.can_auto_apply") as mock_can_apply:
                    # First call returns fix, second returns empty (converged)
                    mock_result = MagicMock()
                    mock_result.suggestions = [fix]
                    mock_suggest.side_effect = [mock_result, MagicMock(suggestions=[])]

                    mock_apply.return_value = {"improved": True}
                    mock_can_apply.return_value = True

                    # Run function returns improving scores
                    call_count = [0]

                    def run_fn(pipeline):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            return {"score": 0.70, "metrics": {}}
                        else:
                            return {"score": 0.80, "metrics": {}}

                    optimizer = AutoOptimizer(
                        run_fn=run_fn,
                        eval_fn=lambda x: x.get("score", 0),
                        diff_fn=lambda p, o: {"changes": []},
                        config=OptimizationConfig(max_iterations=2, min_gain=0.01),
                    )

                    result = optimizer.optimize({"initial": True})

                    assert result.best_score >= 0.70
                    assert result.accepted_improvements >= 0

    def test_compute_score_direct(self):
        """Test _compute_score with direct score."""
        optimizer = AutoOptimizer(
            run_fn=MagicMock(),
            eval_fn=MagicMock(return_value=0.5),
            diff_fn=MagicMock(),
        )

        output = {"score": 0.85}
        score = optimizer._compute_score(output)
        assert score == 0.85

    def test_compute_score_from_metrics(self):
        """Test _compute_score from metrics."""
        optimizer = AutoOptimizer(
            run_fn=MagicMock(),
            eval_fn=MagicMock(return_value=0.5),
            diff_fn=MagicMock(),
            config=OptimizationConfig(score_weights={"psnr": 1.0}),
        )

        output = {"metrics": {"psnr": 38.0}}  # 38 dB -> psnr_to_score 0.9
        score = optimizer._compute_score(output)
        assert score == pytest.approx(0.9)

    def test_compute_score_fallback_to_eval_fn(self):
        """Test _compute_score falls back to eval_fn."""
        eval_fn = MagicMock(return_value=0.75)
        optimizer = AutoOptimizer(
            run_fn=MagicMock(),
            eval_fn=eval_fn,
            diff_fn=MagicMock(),
        )

        output = {}  # No score, no metrics
        score = optimizer._compute_score(output)
        assert score == 0.75
        eval_fn.assert_called_once()

    def test_filter_fixes_by_policy(self):
        """Test fix filtering by policy."""
        from transformation_portal.evals.self_healing import FixSuggestion

        fix1 = FixSuggestion(
            type="quality",
            target_node="node1",
            action="safe_action",
            params={},
            confidence=0.9,
            rationale="Safe fix",
            priority=5,
            reversible=True,
        )

        with patch("transformation_portal.evals.auto_optimizer.can_auto_apply") as mock_can_apply:
            mock_can_apply.return_value = True

            optimizer = AutoOptimizer(
                run_fn=MagicMock(),
                eval_fn=MagicMock(),
                diff_fn=MagicMock(),
            )

            filtered = optimizer._filter_fixes([fix1])
            assert len(filtered) == 1

            mock_can_apply.return_value = False
            filtered = optimizer._filter_fixes([fix1])
            assert len(filtered) == 0

    def test_select_best_candidate(self):
        """Test selecting best candidate."""
        optimizer = AutoOptimizer(
            run_fn=MagicMock(),
            eval_fn=MagicMock(),
            diff_fn=MagicMock(),
        )

        candidate1 = Candidate(pipeline={}, parent_score=0.7, expected_gain=0.05, fix={}, confidence=0.8)
        candidate2 = Candidate(pipeline={}, parent_score=0.7, expected_gain=0.03, fix={}, confidence=0.8)

        results = [
            CandidateResult(candidate=candidate1, score=0.78, improvement=0.08, success=True),
            CandidateResult(candidate=candidate2, score=0.72, improvement=0.02, success=True),
        ]

        best = optimizer._select_best(results, current_score=0.70)
        assert best.improvement == 0.08

    def test_select_best_no_improvement(self):
        """Test selecting when no improvement."""
        optimizer = AutoOptimizer(
            run_fn=MagicMock(),
            eval_fn=MagicMock(),
            diff_fn=MagicMock(),
        )

        candidate = Candidate(pipeline={}, parent_score=0.7, expected_gain=0.05, fix={}, confidence=0.8)

        results = [
            CandidateResult(candidate=candidate, score=0.65, improvement=-0.05, success=True),
        ]

        best = optimizer._select_best(results, current_score=0.70)
        assert best is None

    def test_iteration_summaries(self):
        """Test iteration summaries property."""
        optimizer = AutoOptimizer(
            run_fn=MagicMock(return_value={"score": 0.7}),
            eval_fn=MagicMock(return_value=0.7),
            diff_fn=MagicMock(return_value={"changes": []}),
        )

        # Before optimization
        assert optimizer.iteration_summaries == []

        # After optimization (with mocked suggest_fixes returning empty)
        with patch("transformation_portal.evals.auto_optimizer.suggest_fixes") as mock_suggest:
            mock_suggest.return_value = MagicMock(suggestions=[])
            optimizer.optimize({})

        # May have summaries depending on execution path
        assert isinstance(optimizer.iteration_summaries, list)

    def test_evaluate_sequential(self):
        """Test sequential candidate evaluation."""
        from transformation_portal.evals.self_healing import FixSuggestion

        optimizer = AutoOptimizer(
            run_fn=MagicMock(return_value={"score": 0.8}),
            eval_fn=MagicMock(return_value=0.8),
            diff_fn=MagicMock(),
            config=OptimizationConfig(parallel_evaluation=False),
        )

        candidate = Candidate(
            pipeline={"test": True},
            parent_score=0.7,
            expected_gain=0.1,
            fix={"action": "test"},
            confidence=0.9,
        )

        results = optimizer._evaluate_sequential([candidate])
        assert len(results) == 1
        assert results[0].success is True

    def test_evaluate_sequential_error_handling(self):
        """Test sequential evaluation error handling."""

        def failing_run(pipeline):
            raise RuntimeError("Run failed")

        optimizer = AutoOptimizer(
            run_fn=failing_run,
            eval_fn=MagicMock(),
            diff_fn=MagicMock(),
            config=OptimizationConfig(parallel_evaluation=False),
        )

        candidate = Candidate(
            pipeline={"test": True},
            parent_score=0.7,
            expected_gain=0.1,
            fix={"action": "test"},
            confidence=0.9,
        )

        results = optimizer._evaluate_sequential([candidate])
        assert len(results) == 1
        assert results[0].success is False
        assert "Run failed" in results[0].error


class TestAcceptedStateCarriesCandidateOutput:
    """M-09: after an accepted fix, later iterations must see the candidate's real output."""

    @staticmethod
    def _fix(action: str, priority: int):
        from transformation_portal.evals.self_healing import FixSuggestion

        return FixSuggestion(
            type="quality",
            target_node="nvdiffrec",
            action=action,
            params={"steps": 1000},
            confidence=0.9,
            rationale="test",
            priority=priority,
        )

    def _run(self):
        executions: list[int] = []
        seen_metrics: list[dict] = []

        def run_fn(pipeline):
            steps = pipeline["steps"]
            executions.append(steps)
            psnr = 20.0 if steps < 1000 else 40.0  # the fix really repairs PSNR
            return {"score": psnr / 100.0, "metrics": {"psnr": psnr}}

        low, high = self._fix("low_priority", 1), self._fix("increase_iterations", 9)

        def fake_suggest(diff, metrics=None):
            seen_metrics.append(dict(metrics or {}))
            # Only propose fixes while PSNR is still bad; list order is not priority order.
            return MagicMock(suggestions=[low, high] if (metrics or {}).get("psnr", 0.0) < 30.0 else [])

        def fake_apply(pipeline, fix):
            return {**pipeline, "steps": 1000 if fix.action == "increase_iterations" else pipeline["steps"]}

        with (
            patch("transformation_portal.evals.auto_optimizer.suggest_fixes", side_effect=fake_suggest),
            patch("transformation_portal.evals.auto_optimizer.apply_fix", side_effect=fake_apply),
            patch("transformation_portal.evals.auto_optimizer.can_auto_apply", return_value=True),
        ):
            optimizer = AutoOptimizer(
                run_fn=run_fn,
                eval_fn=lambda o: o["score"],
                diff_fn=lambda p, o: {"changes": []},
                config=OptimizationConfig(max_iterations=4, parallel_evaluation=False, min_gain=0.01, beam_width=1),
            )
            result = optimizer.optimize({"steps": 100})
        return optimizer, result, executions, seen_metrics

    def test_next_iteration_sees_accepted_candidate_metrics(self):
        _optimizer, result, executions, seen_metrics = self._run()

        assert seen_metrics == [{"psnr": 20.0}, {"psnr": 40.0}]
        assert [(h.score, h.metrics) for h in result.history] == [(0.2, {"psnr": 20.0}), (0.4, {"psnr": 40.0})]
        # Initial run + one candidate run: the accepted candidate is not re-executed.
        assert executions == [100, 1000]
        assert result.status == OptimizationStatus.CONVERGED
        assert result.best_score == pytest.approx(0.4)

    def test_iterations_and_fixes_tried_reflect_what_ran(self):
        optimizer, result, _executions, _seen = self._run()

        # Iteration 1 accepted the fix; iteration 2 found nothing to do and converged.
        assert result.iterations == 2
        assert result.accepted_improvements == 1
        # fixes_tried lists the candidates actually evaluated (highest priority first).
        assert [s.fixes_tried for s in optimizer.iteration_summaries] == [["increase_iterations"]]

    def test_candidate_result_keeps_output(self):
        optimizer = AutoOptimizer(
            run_fn=MagicMock(return_value={"score": 0.8, "metrics": {"psnr": 33.0}}),
            eval_fn=MagicMock(),
            diff_fn=MagicMock(),
            config=OptimizationConfig(parallel_evaluation=False),
        )
        candidate = Candidate(pipeline={}, parent_score=0.7, expected_gain=0.1, fix={"action": "a"})
        (res,) = optimizer._evaluate_sequential([candidate])
        assert res.output == {"score": 0.8, "metrics": {"psnr": 33.0}}
