"""APEX + LLaVA Integration for quality validation.

This module provides factory functions to create ApexEvaluationHarness
instances with properly configured LLaVA backends.

Example:
    >>> from transformation_portal.evals.apex_llava_integration import (
    ...     create_apex_harness_with_llava,
    ...     ApexLlavaConfig,
    ... )
    >>>
    >>> config = ApexLlavaConfig(
    ...     model_tier="quality_validation_primary",
    ...     quality_dimension="architectural",
    ...     threshold=0.75,
    ... )
    >>> harness = create_apex_harness_with_llava(config)
    >>> result = harness.evaluate(image_paths=[Path("render.png")])

Quality Dimensions:
    - "segmentation": Mask/reconstruction quality assessment
    - "architectural": Real estate/ArchViz image quality
    - "depth": Depth map quality assessment
    - "material": PBR texture quality assessment

Model Tiers:
    - "ci_smoke": LLaVA 0.5B (fastest, for CI/contracts)
    - "quality_validation_primary": LLaVA v1.6 7B (balanced)
    - "quality_max": LLaVA 1.5 13B (highest quality)
    - "legacy_fallback": LLaVA 1.5 7B (backward compatibility)

See Also:
    - docs/api/MACHINE_MODE_CONTRACT.md
    - docs/apex/ingest_contract.md
    - ADR-026 §5 (LLaVA Quality Validation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from transformation_portal.evals.apex_harness import (
    ApexEvaluationHarness,
    MetricFn,
    brightness_metric,
    contrast_metric,
    sharpness_metric,
)

logger = logging.getLogger(__name__)

# Model tier to manifest key mapping
_MODEL_TIER_MANIFEST_KEYS = {
    "ci_smoke": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "quality_validation_primary": "llava-hf/llava-v1.6-mistral-7b-hf",
    "quality_max": "llava-hf/llava-1.5-13b-hf",
    "legacy_fallback": "llava-hf/llava-1.5-7b-hf",
}

# Quality dimension to prompt builder mapping
QualityDimension = Literal["segmentation", "architectural", "depth", "material"]

# Default thresholds by quality dimension (ADR-026 §5)
_DEFAULT_THRESHOLDS = {
    "segmentation": 0.75,
    "architectural": 0.70,
    "depth": 0.75,
    "material": 0.70,
}

# Default metric weights by quality dimension
_DEFAULT_METRIC_WEIGHTS = {
    "segmentation": 0.4,  # 40% metrics, 60% VLM
    "architectural": 0.5,  # 50% metrics, 50% VLM
    "depth": 0.5,  # 50% metrics, 50% VLM
    "material": 0.4,  # 40% metrics, 60% VLM
}


@dataclass
class ApexLlavaConfig:
    """Configuration for APEX + LLaVA integration.

    Attributes:
        model_tier: Model tier to use (ci_smoke, quality_validation_primary, etc.)
        quality_dimension: Quality dimension for evaluation prompts
        threshold: Pass/fail threshold (0.0-1.0)
        metric_weight: Weight for deterministic metrics vs VLM (0.0-1.0)
        include_standard_metrics: Whether to include sharpness/contrast/brightness
        additional_metrics: Additional metric functions to include
        device_map: Device mapping for model loading
        torch_dtype: Torch dtype for model weights
        fail_on_vlm_error: If True, fail evaluation on VLM errors
        cache_dir: Optional HuggingFace cache directory
    """

    model_tier: Literal["ci_smoke", "quality_validation_primary", "quality_max", "legacy_fallback"] = (
        "quality_validation_primary"
    )
    quality_dimension: QualityDimension = "architectural"
    threshold: Optional[float] = None
    metric_weight: Optional[float] = None
    include_standard_metrics: bool = True
    additional_metrics: list[MetricFn] = field(default_factory=list)
    device_map: Optional[str] = "auto"
    torch_dtype: Any = "auto"
    fail_on_vlm_error: bool = False
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Set defaults based on quality dimension."""
        if self.threshold is None:
            self.threshold = _DEFAULT_THRESHOLDS.get(self.quality_dimension, 0.70)
        if self.metric_weight is None:
            self.metric_weight = _DEFAULT_METRIC_WEIGHTS.get(self.quality_dimension, 0.5)


class ApexLlavaIntegrationError(RuntimeError):
    """Raised for APEX + LLaVA integration failures."""


def _get_prompt_builder_for_dimension(
    quality_dimension: QualityDimension,
):
    """Get the prompt builder function for a quality dimension.

    Args:
        quality_dimension: Quality dimension identifier

    Returns:
        Prompt builder function
    """
    from transformation_portal.evals.vision_language import (
        build_architectural_quality_prompt,
        build_depth_quality_prompt,
        build_segmentation_quality_prompt,
    )

    builders = {
        "segmentation": build_segmentation_quality_prompt,
        "architectural": build_architectural_quality_prompt,
        "depth": build_depth_quality_prompt,
        "material": build_material_quality_prompt,
    }
    return builders.get(quality_dimension, build_architectural_quality_prompt)


def build_material_quality_prompt(
    context: Optional[dict[str, Any]] = None,
):
    """Build a prompt for PBR material quality assessment.

    Args:
        context: Optional additional context to include in the prompt

    Returns:
        LlavaPromptSpec configured for material quality evaluation
    """
    from transformation_portal.evals.vision_language import LlavaPromptSpec

    context_suffix = ""
    if context:
        context_suffix = f"\nAdditional context:\n{context}\n"

    return LlavaPromptSpec(
        name="material_pbr_quality",
        system_text=(
            "You are an expert PBR material quality assessor for architectural visualization. "
            "You must return only valid JSON, with no surrounding markdown."
        ),
        user_text=(
            "Evaluate this PBR material/texture for quality issues.\n"
            "Check specifically for:\n"
            "1. albedo color accuracy and consistency\n"
            "2. normal map artifacts or incorrect directionality\n"
            "3. roughness/metallic value plausibility\n"
            "4. visible tiling or repetition patterns\n"
            "5. seams or discontinuities at texture boundaries\n"
            "6. inconsistent scale between texture channels\n\n"
            "Return only valid JSON with this schema:\n"
            "{\n"
            '  "passes_basic_quality": boolean,\n'
            '  "summary_score": number,\n'
            '  "issues": [\n'
            "    {\n"
            '      "issue_type": string,\n'
            '      "severity": "low|medium|high",\n'
            '      "evidence": string\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"{context_suffix}"
        ),
    )


def _load_model_manifest_entry(model_tier: str) -> dict[str, Any]:
    """Load model manifest entry for the given tier.

    Args:
        model_tier: Model tier identifier

    Returns:
        Manifest payload dictionary

    Raises:
        ApexLlavaIntegrationError: If manifest not found or invalid
    """
    import yaml

    manifest_key = _MODEL_TIER_MANIFEST_KEYS.get(model_tier)
    if not manifest_key:
        raise ApexLlavaIntegrationError(f"Unknown model tier: {model_tier}")

    # Try to load from model_lock_manifest.yaml
    manifest_path = Path(__file__).parents[3] / "config" / "model_lock_manifest.yaml"

    if not manifest_path.exists():
        # Fallback: try relative to working directory
        manifest_path = Path("config/model_lock_manifest.yaml")

    if not manifest_path.exists():
        raise ApexLlavaIntegrationError(f"Model lock manifest not found at {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    repositories = manifest.get("repositories", {})
    if manifest_key not in repositories:
        raise ApexLlavaIntegrationError(f"Model key '{manifest_key}' not found in manifest")

    entry = repositories[manifest_key]

    return {
        "repo_id": manifest_key,
        "revision": entry.get("revision"),
        "owner": entry.get("owner"),
        "tier": entry.get("tier"),
        "license": entry.get("license"),
    }


def create_llava_backend(
    config: ApexLlavaConfig,
) -> Any:
    """Create a LLaVA backend from configuration.

    Args:
        config: APEX + LLaVA configuration

    Returns:
        LlavaQualityBackend instance

    Raises:
        ApexLlavaIntegrationError: If backend creation fails
    """
    from transformation_portal.evals.vision_language import LlavaQualityBackend

    logger.info("Creating LLaVA backend for tier: %s", config.model_tier)

    try:
        manifest_payload = _load_model_manifest_entry(config.model_tier)
    except ApexLlavaIntegrationError:
        raise
    except Exception as exc:
        raise ApexLlavaIntegrationError(f"Failed to load manifest: {exc}") from exc

    model_key = f"apex_{config.quality_dimension}_{config.model_tier}"

    backend = LlavaQualityBackend(
        model_key=model_key,
        manifest_payload=manifest_payload,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
    )

    return backend


def create_apex_harness_with_llava(
    config: Optional[ApexLlavaConfig] = None,
    *,
    preload_model: bool = False,
) -> ApexEvaluationHarness:
    """Create an ApexEvaluationHarness with LLaVA backend integration.

    This is the main entry point for APEX + LLaVA integration. It creates
    a fully configured harness with:
    - LLaVA backend for VLM-based quality assessment
    - Deterministic metrics (sharpness, contrast, brightness)
    - Quality dimension-specific prompts
    - Configurable thresholds and weights

    Args:
        config: Configuration object (uses defaults if None)
        preload_model: If True, load the model immediately

    Returns:
        Configured ApexEvaluationHarness

    Example:
        >>> config = ApexLlavaConfig(
        ...     model_tier="quality_validation_primary",
        ...     quality_dimension="architectural",
        ... )
        >>> harness = create_apex_harness_with_llava(config)
        >>> result = harness.evaluate(image_paths=[Path("image.png")])
    """
    if config is None:
        config = ApexLlavaConfig()

    logger.info(
        "Creating APEX harness with LLaVA: tier=%s, dimension=%s",
        config.model_tier,
        config.quality_dimension,
    )

    # Create LLaVA backend
    llava_backend = create_llava_backend(config)

    # Optionally preload model
    if preload_model:
        logger.info("Preloading LLaVA model...")
        llava_backend.load()

    # Build metric list
    metrics: list[MetricFn] = []
    if config.include_standard_metrics:
        metrics.extend([sharpness_metric, contrast_metric, brightness_metric])
    metrics.extend(config.additional_metrics)

    # Create harness
    harness = ApexEvaluationHarness(
        llava_backend=llava_backend,
        metric_fns=metrics,
        threshold=config.threshold,
        metric_weight=config.metric_weight,
        fail_on_vlm_error=config.fail_on_vlm_error,
    )

    logger.info(
        "APEX harness created: threshold=%.2f, metric_weight=%.2f, num_metrics=%d",
        config.threshold,
        config.metric_weight,
        len(metrics),
    )

    return harness


def create_apex_harness_without_llava(
    *,
    threshold: float = 0.70,
    include_standard_metrics: bool = True,
    additional_metrics: Optional[list[MetricFn]] = None,
) -> ApexEvaluationHarness:
    """Create an ApexEvaluationHarness with metrics only (no LLaVA).

    Useful for environments where LLaVA is not available or when
    only deterministic metrics are needed.

    Args:
        threshold: Pass/fail threshold (0.0-1.0)
        include_standard_metrics: Whether to include sharpness/contrast/brightness
        additional_metrics: Additional metric functions to include

    Returns:
        Configured ApexEvaluationHarness (metrics only)

    Example:
        >>> harness = create_apex_harness_without_llava(threshold=0.75)
        >>> result = harness.evaluate(image_paths=[Path("image.png")])
    """
    metrics: list[MetricFn] = []
    if include_standard_metrics:
        metrics.extend([sharpness_metric, contrast_metric, brightness_metric])
    if additional_metrics:
        metrics.extend(additional_metrics)

    return ApexEvaluationHarness(
        llava_backend=None,
        metric_fns=metrics,
        threshold=threshold,
        metric_weight=1.0,  # All weight to metrics
        fail_on_vlm_error=False,
    )


# Convenience aliases for common configurations
def create_ci_smoke_harness() -> ApexEvaluationHarness:
    """Create a lightweight APEX harness for CI/smoke testing.

    Uses LLaVA 0.5B model for fast validation.
    """
    config = ApexLlavaConfig(
        model_tier="ci_smoke",
        quality_dimension="architectural",
        threshold=0.60,  # Lower threshold for CI
    )
    return create_apex_harness_with_llava(config)


def create_production_harness(
    quality_dimension: QualityDimension = "architectural",
) -> ApexEvaluationHarness:
    """Create a production-grade APEX harness.

    Uses LLaVA v1.6 7B model for balanced quality/speed.

    Args:
        quality_dimension: Quality dimension for prompts
    """
    config = ApexLlavaConfig(
        model_tier="quality_validation_primary",
        quality_dimension=quality_dimension,
    )
    return create_apex_harness_with_llava(config)


def create_quality_max_harness(
    quality_dimension: QualityDimension = "architectural",
) -> ApexEvaluationHarness:
    """Create a maximum quality APEX harness.

    Uses LLaVA 1.5 13B model for highest quality assessment.

    Args:
        quality_dimension: Quality dimension for prompts
    """
    config = ApexLlavaConfig(
        model_tier="quality_max",
        quality_dimension=quality_dimension,
        threshold=0.75,  # Higher threshold for quality-max
    )
    return create_apex_harness_with_llava(config)
