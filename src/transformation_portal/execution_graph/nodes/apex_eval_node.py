"""APEX evaluation DAG node.

This node wraps the APEX evaluation harness for use in DAG-based
pipeline orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformation_portal.evals.apex_harness import ApexEvaluationHarness

from transformation_portal.execution_graph.nodes.base import DAGNode, NodeResult

logger = logging.getLogger(__name__)


class ApexEvalNode(DAGNode):
    """DAG node for APEX Research Ultra evaluation.

    Wraps ApexEvaluationHarness for use in execution graphs.

    Example:
        >>> from transformation_portal.evals.apex_harness import ApexEvaluationHarness
        >>> harness = ApexEvaluationHarness(
        ...     llava_backend=backend,
        ...     metric_fns=[sharpness_metric],
        ... )
        >>> node = ApexEvalNode(harness)
        >>> result = node.run(image_paths=[Path("output.png")])
        >>> print(f"Passes: {result.outputs['passes']}")
    """

    def __init__(
        self,
        harness: "ApexEvaluationHarness",  # type: ignore
    ) -> None:
        """Initialize node with harness.

        Args:
            harness: ApexEvaluationHarness instance
        """
        self.harness = harness

    def validate_inputs(
        self,
        **inputs: Any,
    ) -> Optional[str]:
        """Validate node inputs.

        Required inputs:
            image_paths: List of Path objects to images
        """
        image_paths = inputs.get("image_paths")
        if not image_paths:
            return "image_paths is required and must be non-empty"

        if not isinstance(image_paths, list):
            return "image_paths must be a list"

        return None

    def run(
        self,
        *,
        image_paths: list[Path],
        context: Optional[dict[str, Any]] = None,
    ) -> NodeResult:
        """Execute APEX evaluation.

        Args:
            image_paths: List of image paths to evaluate
            context: Optional context for VLM prompts

        Returns:
            NodeResult with evaluation outputs
        """
        # Validate inputs
        error = self.validate_inputs(image_paths=image_paths, context=context)
        if error:
            return NodeResult(error=error)

        try:
            result = self.harness.evaluate(
                image_paths=image_paths,
                context=context,
            )

            return NodeResult(
                outputs={
                    "score": result.score,
                    "passes": result.passes,
                    "metric_scores": result.metric_scores,
                    "vlm_score": result.vlm_score,
                    "vlm_issues": result.vlm_issues,
                },
                metadata={
                    "num_images": len(image_paths),
                    "threshold": self.harness.threshold,
                },
            )

        except Exception as exc:
            logger.exception("APEX evaluation node failed")
            return NodeResult(error=str(exc))
