"""Depth Ensemble Backend — Multi-Model Variance-Weighted Fusion.

Implements APEX Research Ultra (ADR-026) multi-model depth ensemble with:
- Variance-weighted adaptive fusion (Depth Pro + DA3 + DepthCrafter)
- Per-pixel confidence estimation
- Metric depth scaling and alignment
- Graceful fallback to single-model if ensemble unavailable

Architecture:
- Follows DepthBackend protocol (ADR-019)
- Research-only license tier (multi-layer enforcement)
- Outputs enhanced DepthResult with variance and per-model contributions

See ADR-026 Section 4.1 for implementation specifications.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model in the ensemble.

    Attributes:
        name: Backend identifier (e.g., "depth_pro", "da3").
        weight: Weight for this model in the ensemble (0.0-1.0).
        checkpoint: Path to checkpoint or HuggingFace model ID.
        device: Device for this model (cpu, mps, cuda).
        enabled: Whether this model is enabled.
    """

    name: str
    weight: float
    checkpoint: Optional[str] = None
    device: str = "auto"
    enabled: bool = True


@dataclass
class EnsembleDepthResult(DepthResult):
    """Extended DepthResult for ensemble with variance and per-model data.

    Attributes:
        All DepthResult fields, plus:
        variance_map: Per-pixel variance across models (H, W).
        per_model_depths: Dict mapping model name to depth map.
        per_model_weights: Dict mapping model name to effective weight used.
        fusion_method: Fusion algorithm used ("variance_weighted").
        model_agreement: Overall model agreement score (0.0-1.0).
    """

    variance_map: Optional[np.ndarray] = None
    per_model_depths: Dict[str, np.ndarray] = field(default_factory=dict)
    per_model_weights: Dict[str, float] = field(default_factory=dict)
    fusion_method: str = "variance_weighted"
    model_agreement: float = 0.0


class DepthEnsembleBackend:
    """Multi-model depth ensemble backend with variance-weighted fusion.

    Combines predictions from multiple depth models using adaptive per-pixel
    variance weighting. Models with low local variance get higher weight.

    Supports:
    - Depth Pro (metric depth, focal length)
    - DA3 1.1 Nested Giant Large (detail preservation)
    - DepthCrafter (temporal consistency for video)

    Attributes:
        name: Backend identifier ("ensemble").
        license_type: RESEARCH_ONLY (requires non_commercial_ok=True).
        requires_checkpoint: True (at least one model needs checkpoint).

    Example:
        >>> config = EnhanceConfig(
        ...     non_commercial_ok=True,
        ...     accept_research_tools_license=True,
        ...     spatial_ai_linear_ingest=True,
        ... )
        >>> ensemble = DepthEnsembleBackend(config)
        >>> result = ensemble.compute(image)
        >>> print(f"Variance: {result.variance_map.mean():.4f}")
    """

    # Backend protocol attributes
    name = "ensemble"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = True  # At least one model requires checkpoint

    def __init__(
        self,
        config: Optional["EnhanceConfig"] = None,
        models: Optional[List[ModelConfig]] = None,
        fusion_method: str = "variance_weighted",
        max_variance_threshold: float = 0.15,
    ):
        """Initialize depth ensemble backend.

        Args:
            config: EnhanceConfig for license validation and device settings.
            models: List of ModelConfig for ensemble. If None, uses default 3-model config.
            fusion_method: Fusion algorithm ("variance_weighted").
            max_variance_threshold: Max acceptable variance (>threshold flags warning).
        """
        self._config = config
        self._fusion_method = fusion_method
        self._max_variance_threshold = max_variance_threshold

        # Initialize models
        if models is None:
            self._models = self._get_default_models(config)
        else:
            self._models = models

        # Lazy-loaded backends
        self._backends: Dict[str, any] = {}

        # Validate ensemble configuration
        self._validate_ensemble()

    def _get_default_models(self, config: Optional["EnhanceConfig"]) -> List[ModelConfig]:
        """Get default 3-model ensemble configuration from ADR-026.

        Returns:
            List of ModelConfig for Depth Pro + DA3 + DepthCrafter.
        """
        return [
            ModelConfig(
                name="depth_pro",
                weight=0.5,  # Primary model
                checkpoint=getattr(config, "depth_pro_checkpoint_path", None) if config else None,
                device="auto",
            ),
            ModelConfig(
                name="da3",
                weight=0.3,  # Secondary (detail preservation)
                checkpoint=None,  # Auto-download from HF
                device="auto",
            ),
            # DepthCrafter: Stub for now (Phase 2)
            ModelConfig(
                name="depthcrafter_stub",
                weight=0.2,  # Tertiary (temporal consistency)
                checkpoint=None,
                device="auto",
                enabled=False,  # Disabled until Phase 2 implementation
            ),
        ]

    def _validate_ensemble(self) -> None:
        """Validate ensemble configuration.

        Raises:
            ValueError: If ensemble has <2 enabled models.
            RuntimeError: If total weight != 1.0.
        """
        enabled_models = [m for m in self._models if m.enabled]
        if len(enabled_models) < 2:
            logger.warning(
                f"Ensemble has <2 enabled models ({len(enabled_models)}). " "Consider using single-model backend instead."
            )

        # Validate weights sum to 1.0
        total_weight = sum(m.weight for m in enabled_models)
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Model weights sum to {total_weight}, not 1.0. " "Normalizing weights automatically.")
            # Normalize weights
            for model in enabled_models:
                model.weight /= total_weight

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> EnsembleDepthResult:
        """Estimate depth using multi-model ensemble.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3).
            device: Optional device override (cpu, mps, cuda).

        Returns:
            EnsembleDepthResult with depth map, variance, and per-model data.

        Raises:
            RuntimeError: If inference fails.
        """
        logger.info(f"Running depth ensemble with {len([m for m in self._models if m.enabled])} models")

        # Get per-model predictions
        model_results = self._run_models(image, device)

        # Fuse predictions with variance weighting
        fused_result = self._fuse_predictions(model_results, image)

        # Add ensemble-specific metadata
        fused_result.metadata["ensemble"] = {
            "models": [m.name for m in self._models if m.enabled],
            "fusion_method": self._fusion_method,
            "model_agreement": fused_result.model_agreement,
            "variance_threshold": self._max_variance_threshold,
        }

        # Quality gate: warn if high variance
        if fused_result.variance_map.mean() > self._max_variance_threshold:
            logger.warning(
                f"High inter-model variance ({fused_result.variance_map.mean():.3f} "
                f"> threshold {self._max_variance_threshold}). "
                "Review depth map manually for quality."
            )
            fused_result.warnings.append(f"High variance: {fused_result.variance_map.mean():.3f}")

        return fused_result

    def _run_models(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str],
    ) -> Dict[str, DepthResult]:
        """Run all enabled models and collect results.

        Args:
            image: Input image.
            device: Device override.

        Returns:
            Dict mapping model name to DepthResult.
        """
        results = {}
        enabled_models = [m for m in self._models if m.enabled]

        for model_config in enabled_models:
            try:
                # Get or create backend
                backend = self._get_backend(model_config)

                # Run inference
                logger.debug(f"Running model: {model_config.name}")
                result = backend.compute(image, device=device or model_config.device)
                results[model_config.name] = result

            except Exception as e:
                logger.error(f"Model {model_config.name} failed: {e}. " "Excluding from ensemble.")
                # Continue with remaining models

        if not results:
            raise RuntimeError("All ensemble models failed. Cannot compute depth.")

        return results

    def _get_backend(self, model_config: ModelConfig):
        """Get or create backend for model.

        Args:
            model_config: Model configuration.

        Returns:
            DepthBackend instance.

        Raises:
            ValueError: If backend not found.
        """
        # Return cached backend if available
        if model_config.name in self._backends:
            return self._backends[model_config.name]

        # Create new backend
        from .registry import get_registry

        registry = get_registry()

        # Special handling for stubs
        if model_config.name.endswith("_stub"):
            logger.debug(f"Using synthetic stub for {model_config.name}")
            backend = registry.get_backend("synthetic", self._config)
        else:
            backend = registry.get_backend(model_config.name, self._config)

        # Cache backend
        self._backends[model_config.name] = backend
        return backend

    def _fuse_predictions(
        self,
        model_results: Dict[str, DepthResult],
        original_image: Union[Image.Image, np.ndarray],
    ) -> EnsembleDepthResult:
        """Fuse multi-model predictions with variance weighting.

        Algorithm (ADR-026):
        1. Align all depth maps to metric scale (using Depth Pro as reference)
        2. Compute per-pixel variance across models
        3. Weight by inverse variance (low variance = high confidence)
        4. Fuse depth maps with adaptive weights

        Args:
            model_results: Dict of model results.
            original_image: Original input image.

        Returns:
            EnsembleDepthResult with fused depth and metadata.
        """
        # Convert image to numpy if needed
        if isinstance(original_image, Image.Image):
            img_array = np.array(original_image)
        else:
            img_array = original_image

        # Step 1: Align depth maps to metric scale
        aligned_depths = self._align_depth_maps(model_results)

        # Step 2: Compute per-pixel statistics
        # depth_stack: (N, H, W)
        names = list(aligned_depths.keys())
        depth_stack = np.stack([aligned_depths[n] for n in names], axis=0).astype(np.float32)
        mean_map = np.mean(depth_stack, axis=0)  # (H, W)
        variance_map = np.var(depth_stack, axis=0)  # (H, W)

        # Step 3: Compute per-model confidence maps (ACTUALLY adaptive)
        #
        # Key idea:
        # - A single "inv_variance" map applied to every model cancels algebraically in the fusion ratio.
        # - We need *per-model* per-pixel confidences that downweight outliers.
        #
        # We compute a normalized squared deviation (z^2) and convert it to a confidence:
        #   z2_i = (d_i - mean)^2 / (var + eps)
        #   conf_i = exp(-0.5 * z2_i)
        #
        # This yields:
        #   fused = Σ(d_i * w_i * conf_i) / Σ(w_i * conf_i)
        #
        epsilon = 1e-6
        denom = variance_map + epsilon
        z2 = (depth_stack - mean_map[None, :, :]) ** 2 / denom[None, :, :]
        conf = np.exp(-0.5 * z2).astype(np.float32)  # (N, H, W)

        # Get model weights from config
        model_weights = {m.name: m.weight for m in self._models if m.enabled and m.name in aligned_depths}

        # Normalize model weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

        # Build base weight tensor aligned to the same model order
        base_w = np.array([model_weights.get(n, 0.0) for n in names], dtype=np.float32)[:, None, None]  # (N,1,1)

        # Effective per-pixel weights
        w_eff = base_w * conf  # (N,H,W)
        w_sum = np.sum(w_eff, axis=0)  # (H,W)
        w_sum = np.maximum(w_sum, epsilon)

        # Fuse
        fused_depth = np.sum(w_eff * depth_stack, axis=0) / w_sum  # (H,W)

        # Compute model agreement metric (0.0-1.0, higher is better)
        # Agreement = 1 / (1 + mean_variance)
        mean_variance = variance_map.mean()
        model_agreement = 1.0 / (1.0 + mean_variance)

        # Select primary model for metadata (Depth Pro if available)
        primary_result = model_results.get("depth_pro", next(iter(model_results.values())))

        # Store a compact "effective weight" summary per model (scalar), for observability.
        # This avoids huge per-pixel maps in the result while still showing who contributed.
        per_model_effective_weight = {names[i]: float(np.mean(w_eff[i])) for i in range(len(names))}

        # Build ensemble result
        return EnsembleDepthResult(
            depth_map=fused_depth,
            original_image=img_array,
            metadata={
                "backend": "ensemble",
                "models_used": list(aligned_depths.keys()),
                "mean_variance": float(mean_variance),
            },
            depth_units="meters" if primary_result.is_metric else "relative",
            focal_length_px=primary_result.focal_length_px,
            field_of_view_deg=primary_result.field_of_view_deg,
            backend_id="ensemble",
            device=primary_result.device,
            dtype="float32",
            input_size=primary_result.input_size,
            # Ensemble-specific fields
            variance_map=variance_map,
            per_model_depths=aligned_depths,
            per_model_weights=per_model_effective_weight,
            fusion_method=self._fusion_method,
            model_agreement=model_agreement,
        )

    def _align_depth_maps(self, model_results: Dict[str, DepthResult]) -> Dict[str, np.ndarray]:
        """Align all depth maps to common metric scale.

        Uses Depth Pro as reference if available (metric depth).
        Otherwise, normalizes all to relative depth [0, 1].

        Args:
            model_results: Dict of model results.

        Returns:
            Dict mapping model name to aligned depth map.
        """
        aligned = {}

        # Check if we have metric depth (Depth Pro)
        has_metric = any(r.is_metric for r in model_results.values())

        if has_metric:
            # Use metric depth scale
            for name, result in model_results.items():
                if result.is_metric:
                    # Already metric
                    aligned[name] = result.depth_map
                else:
                    # Convert relative to metric (approximate)
                    # Note: This is a heuristic. Ideally, we'd have camera intrinsics.
                    logger.warning(f"Model {name} outputs relative depth. " "Scaling to metric is approximate.")
                    # Simple scaling: assume depth range 0-10 meters
                    aligned[name] = result.depth_map * 10.0
        else:
            # All relative depth - normalize to [0, 1]
            for name, result in model_results.items():
                depth = result.depth_map.astype(np.float32)
                # Robust normalization
                vmin = np.percentile(depth, 1)
                vmax = np.percentile(depth, 99)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                aligned[name] = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)

        return aligned

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for ensemble.

        Incorporates:
        - Image content hash
        - Enabled model names and weights
        - Fusion method

        Args:
            image: Input image.

        Returns:
            Cache key string.
        """
        # Convert image to numpy for hashing
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Hash image content
        hasher = hashlib.sha256()
        hasher.update(img_array.tobytes())

        # Add ensemble config
        enabled_models = [m for m in self._models if m.enabled]
        config_str = (
            f"ensemble_"
            f"{'_'.join(m.name for m in enabled_models)}_"
            f"{'_'.join(str(m.weight) for m in enabled_models)}_"
            f"{self._fusion_method}"
        )
        hasher.update(config_str.encode())

        return hasher.hexdigest()

    def ensure_available(self) -> None:
        """Ensure ensemble dependencies are available.

        Raises:
            ImportError: If required packages missing.
            RuntimeError: If no models can be loaded.
        """
        enabled_models = [m for m in self._models if m.enabled]

        available_count = 0
        for model_config in enabled_models:
            try:
                # Try to get backend (will raise if unavailable)
                self._get_backend(model_config)
                available_count += 1
            except Exception as e:
                logger.warning(f"Model {model_config.name} unavailable: {e}")

        if available_count < 2:
            raise RuntimeError(
                f"Ensemble requires ≥2 models, but only {available_count} available. " "Cannot initialize ensemble."
            )

        logger.info(f"Ensemble initialized with {available_count} models")

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import modules for ensemble.

        Ensemble requires at least torch + transformers (for DA3).
        Depth Pro is optional (graceful degradation).

        Returns:
            List of required module names.
        """
        return ["transformers"]  # DA3 minimum
