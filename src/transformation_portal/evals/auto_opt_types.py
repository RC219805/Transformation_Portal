"""Autonomous pipeline optimization: Core data types.

This module defines the data structures used by the auto-optimizer
for multi-iteration, self-improving pipeline optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationStatus(Enum):
    """Status of optimization process."""

    RUNNING = "running"
    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_IMPROVEMENT = "no_improvement"
    ERROR = "error"


@dataclass(frozen=True)
class PipelineState:
    """Snapshot of pipeline configuration with evaluation results.

    Attributes:
        pipeline: Pipeline configuration dict
        score: Evaluation score (higher is better)
        metrics: Detailed evaluation metrics
        iteration: Iteration number when this state was created
        parent_hash: Hash of parent state (for lineage)
    """

    pipeline: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    iteration: int = 0
    parent_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "metrics": self.metrics,
            "iteration": self.iteration,
            "parent_hash": self.parent_hash,
        }


@dataclass(frozen=True)
class Candidate:
    """A candidate pipeline configuration to evaluate.

    Attributes:
        pipeline: Modified pipeline configuration
        parent_score: Score of parent pipeline
        expected_gain: Expected improvement from this change
        fix: The fix that generated this candidate
        confidence: Confidence in this candidate
    """

    pipeline: dict[str, Any]
    parent_score: float
    expected_gain: float
    fix: dict[str, Any]
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parent_score": self.parent_score,
            "expected_gain": self.expected_gain,
            "fix": self.fix,
            "confidence": self.confidence,
        }


@dataclass
class CandidateResult:
    """Result of evaluating a candidate.

    Attributes:
        candidate: The candidate that was evaluated
        score: Actual score achieved
        improvement: Actual improvement over parent
        success: Whether evaluation succeeded
        error: Error message if failed
    """

    candidate: Candidate
    score: float
    improvement: float
    success: bool = True
    error: str | None = None


@dataclass
class OptimizationResult:
    """Result of autonomous optimization process.

    Attributes:
        best_pipeline: Best pipeline configuration found
        best_score: Best score achieved
        history: History of pipeline states
        iterations: Number of iterations run
        status: Final status
        total_candidates: Total candidates evaluated
        accepted_improvements: Number of accepted improvements
    """

    best_pipeline: dict[str, Any]
    best_score: float
    history: list[PipelineState] = field(default_factory=list)
    iterations: int = 0
    status: OptimizationStatus = OptimizationStatus.RUNNING
    total_candidates: int = 0
    accepted_improvements: int = 0

    @property
    def improvement_rate(self) -> float:
        """Rate of successful improvements."""
        if self.total_candidates == 0:
            return 0.0
        return self.accepted_improvements / self.total_candidates

    @property
    def total_improvement(self) -> float:
        """Total score improvement from start to end."""
        if not self.history:
            return 0.0
        return self.best_score - self.history[0].score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_score": self.best_score,
            "iterations": self.iterations,
            "status": self.status.value,
            "total_candidates": self.total_candidates,
            "accepted_improvements": self.accepted_improvements,
            "improvement_rate": self.improvement_rate,
            "total_improvement": self.total_improvement,
            "history": [s.to_dict() for s in self.history],
        }


@dataclass
class OptimizationConfig:
    """Configuration for autonomous optimizer.

    Attributes:
        max_iterations: Maximum optimization iterations
        beam_width: Number of top candidates to evaluate per iteration
        min_gain: Minimum improvement to accept a candidate
        convergence_threshold: Stop if improvement is below this for N iterations
        convergence_patience: Number of iterations of low improvement before stopping
        budget_per_iteration: Max candidates to evaluate per iteration
        parallel_evaluation: Whether to evaluate candidates in parallel
        score_weights: Weights for multi-objective optimization
    """

    max_iterations: int = 10
    beam_width: int = 3
    min_gain: float = 0.01
    convergence_threshold: float = 0.005
    convergence_patience: int = 2
    budget_per_iteration: int = 10
    parallel_evaluation: bool = True
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "psnr": 0.3,
            "ssim": 0.2,
            "lpips": -0.2,  # negative because lower is better
            "llava": 0.3,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_iterations": self.max_iterations,
            "beam_width": self.beam_width,
            "min_gain": self.min_gain,
            "convergence_threshold": self.convergence_threshold,
            "convergence_patience": self.convergence_patience,
            "budget_per_iteration": self.budget_per_iteration,
            "parallel_evaluation": self.parallel_evaluation,
            "score_weights": self.score_weights,
        }


@dataclass
class IterationSummary:
    """Summary of a single optimization iteration.

    Attributes:
        iteration: Iteration number
        candidates_evaluated: Number of candidates evaluated
        best_candidate_score: Best candidate score this iteration
        improvement: Improvement over previous iteration
        accepted: Whether an improvement was accepted
        fixes_tried: Fixes that were tried this iteration
    """

    iteration: int
    candidates_evaluated: int
    best_candidate_score: float
    improvement: float
    accepted: bool
    fixes_tried: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "candidates_evaluated": self.candidates_evaluated,
            "best_candidate_score": self.best_candidate_score,
            "improvement": self.improvement,
            "accepted": self.accepted,
            "fixes_tried": self.fixes_tried,
        }
