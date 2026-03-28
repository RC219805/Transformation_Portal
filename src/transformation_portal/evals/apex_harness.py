"""APEX Research Ultra evaluation harness.

This module provides a multi-layer evaluation system for quality assessment:

1. Deterministic metrics (PSNR, SSIM, sharpness, etc.)
2. VLM evaluation (LLaVA-based quality assessment)
3. Aggregation and scoring
4. Policy-based gating

Design:
    The harness combines traditional image quality metrics with
    vision-language model assessment to provide comprehensive
    quality evaluation suitable for APEX Research Ultra workflows.

Example:
    >>> from transformation_portal.evals.vision_language import LlavaQualityBackend
    >>>
    >>> harness = ApexEvaluationHarness(
    ...     llava_backend=backend,
    ...     metric_fns=[sharpness_metric, contrast_metric],
    ...     threshold=0.75,
    ... )
    >>>
    >>> result = harness.evaluate(image_paths=[Path("output.png")])
    >>> if result.passes:
    ...     print(f"Quality check passed (score: {result.score:.2f})")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalMetricResult:
    """Result from a single metric evaluation.

    Attributes:
        name: Metric name
        score: Metric score (0.0-1.0)
        metadata: Additional metric-specific data
    """

    name: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from APEX evaluation harness.

    Attributes:
        score: Combined evaluation score (0.0-1.0)
        passes: Whether the evaluation passed the threshold
        metric_scores: Individual metric scores
        vlm_score: VLM-based score
        vlm_issues: Issues detected by VLM
        details: Full evaluation details
    """

    score: float
    passes: bool
    metric_scores: dict[str, float] = field(default_factory=dict)
    vlm_score: float = 0.0
    vlm_issues: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


MetricFn = Callable[[list[Path]], float]
PromptSpecBuilder = Callable[[Optional[dict[str, Any]]], Any]


class ApexEvaluationHarness:
    """Multi-layer evaluation harness for APEX Research Ultra.

    Combines deterministic metrics with VLM-based quality assessment
    for comprehensive quality evaluation.

    Evaluation flow:
        1. Run deterministic metrics (PSNR, SSIM, sharpness, etc.)
        2. Run VLM evaluation (LLaVA quality assessment)
        3. Combine scores with configurable weights
        4. Apply policy gate (pass/fail threshold)

    Example:
        >>> def sharpness_metric(images: list[Path]) -> float:
        ...     # Compute sharpness score
        ...     return 0.85
        >>>
        >>> harness = ApexEvaluationHarness(
        ...     llava_backend=llava_backend,
        ...     metric_fns=[sharpness_metric],
        ...     threshold=0.70,
        ...     metric_weight=0.4,  # 40% metric, 60% VLM
        ... )
        >>>
        >>> result = harness.evaluate(image_paths=[Path("render.png")])
        >>> print(f"Score: {result.score:.2f}, Passes: {result.passes}")
    """

    def __init__(
        self,
        *,
        llava_backend: Optional[Any] = None,
        metric_fns: Optional[list[MetricFn]] = None,
        prompt_spec_builder: Optional[PromptSpecBuilder] = None,
        threshold: float = 0.70,
        metric_weight: float = 0.5,
        fail_on_vlm_error: bool = False,
    ) -> None:
        """Initialize evaluation harness.

        Args:
            llava_backend: LLaVA quality backend (optional)
            metric_fns: List of metric functions
            prompt_spec_builder: Optional builder for dimension-specific VLM prompts
            threshold: Pass/fail threshold (0.0-1.0)
            metric_weight: Weight for metric scores vs VLM (0.0-1.0)
            fail_on_vlm_error: If True, fail evaluation on VLM errors
        """
        self.llava_backend = llava_backend
        self.metric_fns = metric_fns or []
        self.prompt_spec_builder = prompt_spec_builder
        self.threshold = threshold
        self.metric_weight = metric_weight
        self.vlm_weight = 1.0 - metric_weight
        self.fail_on_vlm_error = fail_on_vlm_error

    def evaluate(
        self,
        *,
        image_paths: list[Path],
        context: Optional[dict[str, Any]] = None,
    ) -> EvalResult:
        """Run full evaluation pipeline.

        Args:
            image_paths: Images to evaluate
            context: Optional context for VLM prompts

        Returns:
            EvalResult with combined score and details
        """
        logger.info("Starting APEX evaluation on %d images", len(image_paths))

        # Run deterministic metrics
        metric_results = self._run_metrics(image_paths)
        metric_avg = self._aggregate_metrics(metric_results)

        # Run VLM evaluation
        vlm_result = self._run_vlm(image_paths, context)

        # Combine scores
        combined_score = self._combine_scores(metric_avg, vlm_result)

        # Build result
        result = EvalResult(
            score=combined_score,
            passes=combined_score >= self.threshold,
            metric_scores={r.name: r.score for r in metric_results},
            vlm_score=vlm_result.get("score", 0.0),
            vlm_issues=vlm_result.get("issues", []),
            details={
                "metric_average": metric_avg,
                "vlm_raw": vlm_result,
                "weights": {
                    "metric": self.metric_weight,
                    "vlm": self.vlm_weight,
                },
                "threshold": self.threshold,
            },
        )

        logger.info(
            "APEX evaluation complete: score=%.2f, passes=%s",
            result.score,
            result.passes,
        )

        return result

    def _run_metrics(
        self,
        image_paths: list[Path],
    ) -> list[EvalMetricResult]:
        """Run all deterministic metrics.

        Args:
            image_paths: Images to evaluate

        Returns:
            List of metric results
        """
        results = []

        for fn in self.metric_fns:
            try:
                score = fn(image_paths)
                # Clamp to valid range
                score = max(0.0, min(1.0, float(score)))

                results.append(
                    EvalMetricResult(
                        name=fn.__name__,
                        score=score,
                    )
                )
                logger.debug("Metric %s: %.3f", fn.__name__, score)

            except Exception as exc:
                logger.warning("Metric %s failed: %s", fn.__name__, exc)
                results.append(
                    EvalMetricResult(
                        name=fn.__name__,
                        score=0.0,
                        metadata={"error": str(exc)},
                    )
                )

        return results

    def _aggregate_metrics(
        self,
        results: list[EvalMetricResult],
    ) -> float:
        """Aggregate metric results into single score.

        Args:
            results: List of metric results

        Returns:
            Aggregated score (average)
        """
        if not results:
            return 0.0

        total = sum(r.score for r in results)
        return total / len(results)

    def _run_vlm(
        self,
        image_paths: list[Path],
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run VLM-based evaluation.

        Args:
            image_paths: Images to evaluate
            context: Optional context for prompts

        Returns:
            VLM result dictionary
        """
        if self.llava_backend is None:
            logger.debug("No LLaVA backend configured, skipping VLM evaluation")
            return {"score": 0.0, "issues": [], "skipped": True}

        try:
            prompt_spec = None
            if self.prompt_spec_builder is not None:
                prompt_spec = self.prompt_spec_builder(context)

            result = self.llava_backend.evaluate_images(
                image_paths=image_paths,
                prompt_spec=prompt_spec,
                context=context,
            )

            return {
                "score": result.summary_score,
                "passes": result.passes_basic_quality,
                "issues": [
                    {
                        "issue_type": i.issue_type,
                        "severity": i.severity,
                        "evidence": i.evidence,
                    }
                    for i in result.issues
                ],
                "raw_text": result.raw_text,
                "model_key": result.model_key,
            }

        except Exception as exc:
            logger.error("VLM evaluation failed: %s", exc)
            if self.fail_on_vlm_error:
                return {"score": 0.0, "issues": [], "error": str(exc)}
            return {"score": 0.0, "issues": [], "error": str(exc), "skipped": True}

    def _combine_scores(
        self,
        metric_avg: float,
        vlm_result: dict[str, Any],
    ) -> float:
        """Combine metric and VLM scores.

        Args:
            metric_avg: Average metric score
            vlm_result: VLM evaluation result

        Returns:
            Combined score
        """
        vlm_score = vlm_result.get("score", 0.0)
        vlm_skipped = vlm_result.get("skipped", False)

        # If VLM was skipped, use metrics only
        if vlm_skipped and self.metric_fns:
            return metric_avg

        # If no metrics, use VLM only
        if not self.metric_fns:
            return vlm_score

        # Weighted combination
        return (metric_avg * self.metric_weight) + (vlm_score * self.vlm_weight)


# ============================================================================
# Built-in Metrics
# ============================================================================


def sharpness_metric(image_paths: list[Path]) -> float:
    """Compute sharpness score using Laplacian variance.

    Higher variance indicates sharper images.
    Normalized to 0-1 range.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not available for sharpness metric")
        return 0.0

    scores = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

        # Normalize (empirical thresholds)
        # < 100: blurry, > 500: sharp
        normalized = min(1.0, laplacian_var / 500.0)
        scores.append(normalized)

    return float(np.mean(scores)) if scores else 0.0


def contrast_metric(image_paths: list[Path]) -> float:
    """Compute contrast score using standard deviation.

    Higher std indicates more contrast.
    Normalized to 0-1 range.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not available for contrast metric")
        return 0.0

    scores = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Standard deviation as contrast measure
        std = float(np.std(img))

        # Normalize (empirical: std of 50-80 is good)
        normalized = min(1.0, std / 80.0)
        scores.append(normalized)

    return float(np.mean(scores)) if scores else 0.0


def brightness_metric(image_paths: list[Path]) -> float:
    """Compute brightness score.

    Penalizes too dark or too bright images.
    Optimal mean around 128 (middle gray).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not available for brightness metric")
        return 0.0

    scores = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        mean_val = float(np.mean(img))

        # Score based on distance from optimal (128)
        # Perfect at 128, drops off toward 0 and 255
        distance = abs(mean_val - 128)
        normalized = 1.0 - (distance / 128.0)
        scores.append(max(0.0, normalized))

    return float(np.mean(scores)) if scores else 0.0
