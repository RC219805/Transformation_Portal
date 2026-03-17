"""Self-healing pipeline: Validation loop runner.

This module orchestrates the self-healing process:
1. Analyze outputs via semantic diff
2. Generate fix suggestions
3. Apply fixes (with policy checks)
4. Re-run pipeline
5. Validate improvements
6. Accept or reject fixes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from transformation_portal.evals.self_heal_policy import (
    PolicyReport,
    SelfHealPolicy,
    evaluate_all,
    filter_auto_applicable,
)
from transformation_portal.evals.self_healing import FixSuggestion, suggest_fixes
from transformation_portal.execution_graph.patcher import PatchSet, apply_fixes

if TYPE_CHECKING:
    from transformation_portal.evals.semantic_diff import SemanticDiffResult
    from transformation_portal.evals.vision_language.llava_backend import (
        LlavaQualityBackend,
    )

logger = logging.getLogger(__name__)


@dataclass
class HealingCandidate:
    """A candidate fix with its validation result."""

    fix: FixSuggestion
    patched_pipeline: dict[str, Any]
    result_score: float
    improvement: float  # Score delta vs original
    validated: bool


@dataclass
class HealingResult:
    """Result of self-healing process."""

    original_score: float
    candidates: list[HealingCandidate] = field(default_factory=list)
    best_candidate: HealingCandidate | None = None
    policy_report: PolicyReport | None = None
    patch_set: PatchSet | None = None
    accepted: bool = False

    @property
    def improved(self) -> bool:
        """Whether the best candidate improved the score."""
        if self.best_candidate is None:
            return False
        return self.best_candidate.improvement > 0

    @property
    def improvement_percent(self) -> float:
        """Percentage improvement from best candidate."""
        if self.best_candidate is None or self.original_score == 0:
            return 0.0
        return (self.best_candidate.improvement / self.original_score) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_score": self.original_score,
            "candidates": [
                {
                    "fix": c.fix.to_dict(),
                    "result_score": c.result_score,
                    "improvement": c.improvement,
                    "validated": c.validated,
                }
                for c in self.candidates
            ],
            "best_candidate": (
                {
                    "fix": self.best_candidate.fix.to_dict(),
                    "result_score": self.best_candidate.result_score,
                    "improvement": self.best_candidate.improvement,
                }
                if self.best_candidate
                else None
            ),
            "improved": self.improved,
            "improvement_percent": self.improvement_percent,
            "accepted": self.accepted,
        }


# Type alias for pipeline runner function
PipelineRunner = Callable[[dict[str, Any]], dict[str, Any]]


def self_heal(
    *,
    pipeline: dict[str, Any],
    llava: "LlavaQualityBackend",
    image_a: Path,
    image_b: Path,
    metrics: dict[str, float],
    run_fn: PipelineRunner,
    policy: SelfHealPolicy | None = None,
    max_candidates: int = 5,
    score_key: str = "score",
) -> HealingResult:
    """Run self-healing process on a pipeline.

    Args:
        pipeline: Pipeline configuration to heal
        llava: LLaVA backend for semantic diff
        image_a: Reference image path
        image_b: Output image path (to compare)
        metrics: Current evaluation metrics
        run_fn: Function to run pipeline and return results
        policy: Self-healing policy (defaults to standard policy)
        max_candidates: Maximum candidates to evaluate
        score_key: Key in run results for score

    Returns:
        HealingResult with candidates and best fix

    Example:
        >>> result = self_heal(
        ...     pipeline=my_pipeline,
        ...     llava=llava_backend,
        ...     image_a=ref_image,
        ...     image_b=output_image,
        ...     metrics={"psnr": 25.0},
        ...     run_fn=lambda p: executor.run(p),
        ... )
        >>> if result.improved:
        ...     apply_best_fix(result)
    """
    from transformation_portal.evals.semantic_diff import semantic_diff

    policy = policy or SelfHealPolicy()

    # Step 1: Compute semantic diff
    logger.info("Running semantic diff analysis...")
    diff_result = semantic_diff(backend=llava, image_a=image_a, image_b=image_b)

    # Step 2: Generate fix suggestions
    logger.info("Generating fix suggestions...")
    suggestions = suggest_fixes(diff_result.structured, metrics)

    # Step 3: Evaluate against policy
    logger.info("Evaluating fixes against policy...")
    policy_report = evaluate_all(suggestions.suggestions, policy)

    # Step 4: Filter to auto-applicable fixes
    applicable_fixes = filter_auto_applicable(
        suggestions.suggestions,
        policy,
        max_fixes=max_candidates,
    )

    logger.info(
        "Found %d applicable fixes out of %d suggestions",
        len(applicable_fixes),
        len(suggestions.suggestions),
    )

    # Get baseline score
    original_result = run_fn(pipeline)
    original_score = original_result.get(score_key, 0.0)

    result = HealingResult(
        original_score=original_score,
        policy_report=policy_report,
    )

    # Step 5: Evaluate each fix
    for fix in applicable_fixes:
        logger.info("Evaluating fix: %s on %s", fix.action, fix.target_node)

        try:
            # Apply fix
            patched, patch_set = apply_fixes(pipeline, [fix])
            result.patch_set = patch_set

            # Run patched pipeline
            new_result = run_fn(patched)
            new_score = new_result.get(score_key, 0.0)

            improvement = new_score - original_score

            candidate = HealingCandidate(
                fix=fix,
                patched_pipeline=patched,
                result_score=new_score,
                improvement=improvement,
                validated=True,
            )

            result.candidates.append(candidate)

            logger.info(
                "Fix %s: score %.3f -> %.3f (improvement: %.3f)",
                fix.action,
                original_score,
                new_score,
                improvement,
            )

        except Exception as e:
            logger.error("Failed to evaluate fix %s: %s", fix.action, e)
            result.candidates.append(
                HealingCandidate(
                    fix=fix,
                    patched_pipeline={},
                    result_score=0.0,
                    improvement=-1.0,
                    validated=False,
                )
            )

    # Step 6: Select best candidate
    valid_candidates = [c for c in result.candidates if c.validated and c.improvement > 0]

    if valid_candidates:
        result.best_candidate = max(valid_candidates, key=lambda c: c.improvement)
        logger.info(
            "Best fix: %s with improvement %.3f",
            result.best_candidate.fix.action,
            result.best_candidate.improvement,
        )

    return result


def self_heal_iterative(
    *,
    pipeline: dict[str, Any],
    llava: "LlavaQualityBackend",
    image_a: Path,
    run_fn: PipelineRunner,
    get_output_fn: Callable[[dict[str, Any]], Path],
    max_iterations: int = 3,
    min_improvement: float = 0.01,
    policy: SelfHealPolicy | None = None,
) -> list[HealingResult]:
    """Run iterative self-healing until convergence.

    Args:
        pipeline: Initial pipeline configuration
        llava: LLaVA backend
        image_a: Reference image
        run_fn: Pipeline runner function
        get_output_fn: Function to get output image from results
        max_iterations: Maximum healing iterations
        min_improvement: Minimum improvement to continue
        policy: Self-healing policy

    Returns:
        List of HealingResult for each iteration
    """
    results: list[HealingResult] = []
    current_pipeline = pipeline

    for i in range(max_iterations):
        logger.info("Self-healing iteration %d/%d", i + 1, max_iterations)

        # Run current pipeline
        run_result = run_fn(current_pipeline)
        output_image = get_output_fn(run_result)

        # Run self-heal
        heal_result = self_heal(
            pipeline=current_pipeline,
            llava=llava,
            image_a=image_a,
            image_b=output_image,
            metrics=run_result.get("metrics", {}),
            run_fn=run_fn,
            policy=policy,
        )

        results.append(heal_result)

        # Check for improvement
        if not heal_result.improved:
            logger.info("No improvement found, stopping iterations")
            break

        if heal_result.best_candidate is None:
            break

        if heal_result.best_candidate.improvement < min_improvement:
            logger.info(
                "Improvement %.4f below threshold %.4f, stopping",
                heal_result.best_candidate.improvement,
                min_improvement,
            )
            break

        # Accept best fix and continue
        heal_result.accepted = True
        current_pipeline = heal_result.best_candidate.patched_pipeline

        logger.info(
            "Accepted fix %s, continuing to next iteration",
            heal_result.best_candidate.fix.action,
        )

    return results


def apply_healing_result(
    pipeline: dict[str, Any],
    result: HealingResult,
) -> dict[str, Any]:
    """Apply the best fix from a healing result.

    Args:
        pipeline: Original pipeline
        result: Healing result with best candidate

    Returns:
        Patched pipeline

    Raises:
        ValueError: If no best candidate available
    """
    if result.best_candidate is None:
        raise ValueError("No best candidate to apply")

    return result.best_candidate.patched_pipeline
