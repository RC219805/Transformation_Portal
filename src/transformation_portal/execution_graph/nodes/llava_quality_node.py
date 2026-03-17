"""LLaVA quality validation DAG node.

This node wraps the LLaVA quality backend for use in DAG-based
pipeline orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformation_portal.evals.vision_language.llava_backend import LlavaQualityBackend

from transformation_portal.execution_graph.nodes.base import DAGNode, NodeResult

logger = logging.getLogger(__name__)


class LlavaQualityNode(DAGNode):
    """DAG node for LLaVA-based visual quality assessment.

    Wraps LlavaQualityBackend for use in execution graphs.

    Example:
        >>> from transformation_portal.evals.vision_language import LlavaQualityBackend
        >>> backend = LlavaQualityBackend(
        ...     model_key="llava",
        ...     manifest_payload={"repo_id": "...", "revision": "..."},
        ... )
        >>> node = LlavaQualityNode(backend)
        >>> result = node.run(
        ...     image_paths=[Path("image.png")],
        ...     context={"stage": "reconstruction"},
        ... )
        >>> print(f"Quality score: {result.outputs['quality_score']}")
    """

    def __init__(
        self,
        backend: "LlavaQualityBackend",  # type: ignore
    ) -> None:
        """Initialize node with backend.

        Args:
            backend: LlavaQualityBackend instance
        """
        self.backend = backend

    def validate_inputs(
        self,
        **inputs: Any,
    ) -> Optional[str]:
        """Validate node inputs.

        Required inputs:
            image_paths: List of Path objects to images

        Optional inputs:
            context: Additional context dict for prompts
        """
        image_paths = inputs.get("image_paths")
        if not image_paths:
            return "image_paths is required and must be non-empty"

        if not isinstance(image_paths, list):
            return "image_paths must be a list"

        for path in image_paths:
            if not isinstance(path, Path):
                return f"image_paths must contain Path objects, got {type(path)}"
            if not path.exists():
                return f"Image path does not exist: {path}"

        return None

    def run(
        self,
        *,
        image_paths: list[Path],
        context: Optional[dict[str, Any]] = None,
    ) -> NodeResult:
        """Execute quality assessment.

        Args:
            image_paths: List of image paths to evaluate
            context: Optional context for prompts

        Returns:
            NodeResult with quality assessment outputs
        """
        # Validate inputs
        error = self.validate_inputs(image_paths=image_paths, context=context)
        if error:
            return NodeResult(error=error)

        try:
            # Run evaluation
            vqa_result = self.backend.evaluate_images(
                image_paths=image_paths,
                context=context,
            )

            # Extract outputs
            outputs = {
                "quality_score": vqa_result.summary_score,
                "passes": vqa_result.passes_basic_quality,
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "evidence": issue.evidence,
                    }
                    for issue in vqa_result.issues
                ],
                "model_key": vqa_result.model_key,
            }

            # Include raw result for downstream processing
            if vqa_result.raw_text:
                outputs["raw_text"] = vqa_result.raw_text

            return NodeResult(
                outputs=outputs,
                metadata={
                    "num_images": len(image_paths),
                    "num_issues": len(vqa_result.issues),
                },
            )

        except Exception as exc:
            logger.exception("LLaVA quality node failed")
            return NodeResult(error=str(exc))
