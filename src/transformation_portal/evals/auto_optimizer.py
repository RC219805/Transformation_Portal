"""Autonomous pipeline optimizer: Multi-iteration self-improving optimization.

This module provides the core optimization engine that:
1. Runs pipelines and evaluates outputs
2. Computes semantic diffs to identify issues
3. Generates fix candidates
4. Evaluates candidates (optionally in parallel)
5. Selects best improvements
6. Iterates until convergence or budget exhaustion

Supports:
- Greedy / beam search strategies
- Policy-gated fix application
- Multi-objective scoring
- Distributed evaluation (Ray)
- Experiment tracking integration
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from transformation_portal.evals.auto_opt_types import (
    Candidate,
    CandidateResult,
    IterationSummary,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
    PipelineState,
)
from transformation_portal.evals.self_heal_policy import (
    SelfHealPolicy,
    can_auto_apply,
)
from transformation_portal.evals.self_healing import suggest_fixes
from transformation_portal.execution_graph.patcher import apply_fix

if TYPE_CHECKING:
    from transformation_portal.evals.self_healing import FixSuggestion

logger = logging.getLogger(__name__)


# Type aliases
PipelineRunner = Callable[[dict[str, Any]], dict[str, Any]]
Evaluator = Callable[[dict[str, Any]], float]
DiffGenerator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _hash_pipeline(pipeline: dict[str, Any]) -> str:
    """Compute hash of pipeline for lineage tracking."""
    return hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest()[:16]


def _compute_composite_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted composite score from metrics.

    Args:
        metrics: Dictionary of metric values
        weights: Dictionary of metric weights (can be negative)

    Returns:
        Weighted composite score
    """
    score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        if metric in metrics:
            # Normalize LPIPS (lower is better, so we invert)
            value = metrics[metric]
            if weight < 0:
                # Negative weight means lower is better
                value = 1.0 - min(value, 1.0)
                weight = abs(weight)

            score += value * weight
            total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0


class AutoOptimizer:
    """Autonomous pipeline optimizer with beam search and policy gating.

    Example:
        >>> optimizer = AutoOptimizer(
        ...     run_fn=lambda p: executor.run(p),
        ...     eval_fn=lambda r: r["score"],
        ...     diff_fn=lambda p, r: semantic_diff(p, r),
        ... )
        >>> result = optimizer.optimize(initial_pipeline)
        >>> print(f"Improved score: {result.best_score:.3f}")
    """

    def __init__(
        self,
        *,
        run_fn: PipelineRunner,
        eval_fn: Evaluator,
        diff_fn: DiffGenerator,
        config: OptimizationConfig | None = None,
        policy: SelfHealPolicy | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            run_fn: Function to run pipeline and return results
            eval_fn: Function to evaluate results and return score
            diff_fn: Function to generate semantic diff from outputs
            config: Optimization configuration
            policy: Policy for fix gating
        """
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.diff_fn = diff_fn
        self.config = config or OptimizationConfig()
        self.policy = policy or SelfHealPolicy()

        # State tracking
        self._iteration_summaries: list[IterationSummary] = []

    def optimize(self, pipeline: dict[str, Any]) -> OptimizationResult:
        """Run multi-iteration optimization on a pipeline.

        Args:
            pipeline: Initial pipeline configuration

        Returns:
            OptimizationResult with best pipeline and history
        """
        logger.info("Starting autonomous optimization")
        logger.info("Config: max_iters=%d, beam_width=%d", self.config.max_iterations, self.config.beam_width)

        history: list[PipelineState] = []
        self._iteration_summaries = []

        # Initial run and evaluation
        logger.info("Running initial pipeline...")
        base_output = self.run_fn(pipeline)
        base_metrics = base_output.get("metrics", {})
        base_score = self._compute_score(base_output)

        current = PipelineState(
            pipeline=pipeline,
            score=base_score,
            metrics=base_metrics,
            iteration=0,
            parent_hash="",
        )
        history.append(current)

        logger.info("Initial score: %.4f", base_score)

        result = OptimizationResult(
            best_pipeline=pipeline,
            best_score=base_score,
            history=history,
        )

        # Convergence tracking
        low_improvement_count = 0

        # Main optimization loop
        for i in range(self.config.max_iterations):
            logger.info("=== Iteration %d/%d ===", i + 1, self.config.max_iterations)

            # Generate diff and fix suggestions
            diff = self.diff_fn(current.pipeline, base_output)
            suggestions = suggest_fixes(diff, current.metrics)

            # Filter by policy
            applicable_fixes = self._filter_fixes(suggestions.suggestions)

            if not applicable_fixes:
                logger.info("No applicable fixes found, stopping")
                result.status = OptimizationStatus.CONVERGED
                break

            # Generate candidates
            candidates = self._generate_candidates(current, applicable_fixes)

            if not candidates:
                logger.info("No candidates generated, stopping")
                result.status = OptimizationStatus.NO_IMPROVEMENT
                break

            # Evaluate candidates
            candidate_results = self._evaluate_candidates(candidates)
            result.total_candidates += len(candidate_results)

            # Select best
            best_result = self._select_best(candidate_results, current.score)

            # Create iteration summary
            summary = IterationSummary(
                iteration=i + 1,
                candidates_evaluated=len(candidate_results),
                best_candidate_score=best_result.score if best_result else current.score,
                improvement=best_result.improvement if best_result else 0.0,
                accepted=best_result is not None and best_result.improvement >= self.config.min_gain,
                fixes_tried=[f.action for f in applicable_fixes[: self.config.beam_width]],
            )
            self._iteration_summaries.append(summary)

            # Check acceptance
            if best_result is None or best_result.improvement < self.config.min_gain:
                logger.info(
                    "Best improvement %.4f below threshold %.4f",
                    best_result.improvement if best_result else 0.0,
                    self.config.min_gain,
                )
                low_improvement_count += 1

                if low_improvement_count >= self.config.convergence_patience:
                    logger.info("Convergence patience exhausted, stopping")
                    result.status = OptimizationStatus.CONVERGED
                    break
                continue

            # Accept improvement
            low_improvement_count = 0
            result.accepted_improvements += 1

            current = PipelineState(
                pipeline=best_result.candidate.pipeline,
                score=best_result.score,
                metrics=base_output.get("metrics", {}),  # Updated in next iteration
                iteration=i + 1,
                parent_hash=_hash_pipeline(current.pipeline),
            )
            history.append(current)

            result.best_pipeline = current.pipeline
            result.best_score = current.score

            logger.info(
                "Accepted improvement: %.4f -> %.4f (gain: %.4f)",
                best_result.candidate.parent_score,
                best_result.score,
                best_result.improvement,
            )

            # Update for next iteration
            base_output = self.run_fn(current.pipeline)

        result.iterations = len(history) - 1  # Subtract initial state
        result.history = history

        if result.status == OptimizationStatus.RUNNING:
            result.status = OptimizationStatus.BUDGET_EXHAUSTED

        logger.info(
            "Optimization complete: %d iterations, score %.4f -> %.4f",
            result.iterations,
            history[0].score,
            result.best_score,
        )

        return result

    def _compute_score(self, output: dict[str, Any]) -> float:
        """Compute score from pipeline output."""
        # Try direct score first
        if "score" in output:
            return float(output["score"])

        # Try composite from metrics
        metrics = output.get("metrics", {})
        if metrics:
            return _compute_composite_score(metrics, self.config.score_weights)

        # Fallback to eval_fn
        return self.eval_fn(output)

    def _filter_fixes(self, fixes: list["FixSuggestion"]) -> list["FixSuggestion"]:
        """Filter fixes by policy."""
        return [f for f in fixes if can_auto_apply(f, self.policy)]

    def _generate_candidates(
        self,
        current: PipelineState,
        fixes: list["FixSuggestion"],
    ) -> list[Candidate]:
        """Generate candidate pipelines from fixes."""
        candidates = []

        # Sort by expected gain (confidence * priority)
        sorted_fixes = sorted(fixes, key=lambda f: f.confidence * f.priority, reverse=True)

        for fix in sorted_fixes[: self.config.budget_per_iteration]:
            try:
                patched = apply_fix(current.pipeline, fix)
                candidates.append(
                    Candidate(
                        pipeline=patched,
                        parent_score=current.score,
                        expected_gain=fix.confidence,
                        fix=fix.to_dict(),
                        confidence=fix.confidence,
                    )
                )
            except Exception as e:
                logger.warning("Failed to generate candidate for %s: %s", fix.action, e)

        return candidates

    def _evaluate_candidates(self, candidates: list[Candidate]) -> list[CandidateResult]:
        """Evaluate candidates and return results."""
        results = []

        if self.config.parallel_evaluation:
            results = self._evaluate_parallel(candidates)
        else:
            results = self._evaluate_sequential(candidates)

        return results

    def _evaluate_sequential(self, candidates: list[Candidate]) -> list[CandidateResult]:
        """Evaluate candidates sequentially."""
        results = []

        for candidate in candidates[: self.config.beam_width]:
            try:
                output = self.run_fn(candidate.pipeline)
                score = self._compute_score(output)
                improvement = score - candidate.parent_score

                results.append(
                    CandidateResult(
                        candidate=candidate,
                        score=score,
                        improvement=improvement,
                        success=True,
                    )
                )

                logger.debug(
                    "Candidate %s: score=%.4f, improvement=%.4f",
                    candidate.fix.get("action"),
                    score,
                    improvement,
                )

            except Exception as e:
                logger.warning("Failed to evaluate candidate: %s", e)
                results.append(
                    CandidateResult(
                        candidate=candidate,
                        score=0.0,
                        improvement=-1.0,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def _evaluate_parallel(self, candidates: list[Candidate]) -> list[CandidateResult]:
        """Evaluate candidates in parallel using Ray if available."""
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            @ray.remote
            def eval_candidate(run_fn, pipeline):
                output = run_fn(pipeline)
                return output

            # Submit tasks
            futures = [eval_candidate.remote(self.run_fn, c.pipeline) for c in candidates[: self.config.beam_width]]

            # Gather results
            outputs = ray.get(futures)

            results = []
            for candidate, output in zip(candidates, outputs):
                score = self._compute_score(output)
                improvement = score - candidate.parent_score
                results.append(
                    CandidateResult(
                        candidate=candidate,
                        score=score,
                        improvement=improvement,
                        success=True,
                    )
                )

            return results

        except ImportError:
            logger.info("Ray not available, falling back to sequential evaluation")
            return self._evaluate_sequential(candidates)
        except Exception as e:
            logger.warning("Parallel evaluation failed: %s, falling back to sequential", e)
            return self._evaluate_sequential(candidates)

    def _select_best(
        self,
        results: list[CandidateResult],
        current_score: float,
    ) -> CandidateResult | None:
        """Select best candidate that improves on current score."""
        successful = [r for r in results if r.success and r.improvement > 0]

        if not successful:
            return None

        return max(successful, key=lambda r: r.improvement)

    @property
    def iteration_summaries(self) -> list[IterationSummary]:
        """Get summaries of all iterations."""
        return self._iteration_summaries


def acceptance_gate(
    old_score: float,
    new_score: float,
    threshold: float = 0.02,
) -> bool:
    """Check if improvement passes acceptance threshold.

    Args:
        old_score: Previous best score
        new_score: New candidate score
        threshold: Minimum improvement required

    Returns:
        True if improvement is accepted
    """
    return (new_score - old_score) > threshold


def log_iteration_to_experiment(
    experiment_id: int,
    state: PipelineState,
    create_run_fn: Callable,
) -> None:
    """Log iteration to experiment tracking database.

    Args:
        experiment_id: Experiment ID in database
        state: Pipeline state to log
        create_run_fn: Function to create run in database
    """
    create_run_fn(
        experiment_id=experiment_id,
        config=state.pipeline,
        metrics={
            "score": state.score,
            "iteration": state.iteration,
            **state.metrics,
        },
    )
