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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ...lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...core.execution_plan import BackendModelIntent
    from ...lux_depth_v3.config import EnhanceConfig
    from ...lux_depth_v3.execution_lifecycle import BackendCandidateAuthority

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
    model_contract: Optional["BackendModelIntent"] = None


@dataclass
class TemporalPostFilterConfig:
    """Configuration for post-fusion temporal filter (ADR-026 §2.2).

    Applied to the fused ensemble depth to provide temporal consistency
    when DepthCrafter is unavailable or running in synthetic fallback mode.

    Attributes:
        mode: Filter mode ("ema" for exponential
            moving average, "off" to disable).
        alpha: EMA smoothing factor (0.0–1.0). Lower = stronger smoothing.
    """

    mode: str = "off"
    alpha: float = 0.3


class TemporalPostFilter:
    """Post-fusion EMA temporal filter for ensemble depth (ADR-026 §2.2).

    Provides a cheap temporal stabilizer on the fused ensemble depth output
    for video workflows. This is a fallback when DepthCrafter's native
    temporal prior is unavailable.

    Implements StatefulBackend protocol for orchestrator lifecycle management.
    """

    def __init__(self, config: Optional[TemporalPostFilterConfig] = None):
        self._config = config or TemporalPostFilterConfig()
        self._ema_state: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        """Whether filtering is active."""
        return self._config.mode == "ema"

    @property
    def mode(self) -> str:
        """Current temporal filter mode."""
        return self._config.mode

    @property
    def alpha(self) -> float:
        """Current EMA alpha."""
        return self._config.alpha

    def get_config(self) -> TemporalPostFilterConfig:
        """Return a copy of the active filter configuration."""
        return TemporalPostFilterConfig(
            mode=self._config.mode,
            alpha=self._config.alpha,
        )

    def has_state(self) -> bool:
        """Whether EMA state is initialized."""
        return self._ema_state is not None

    def apply(self, depth_map: np.ndarray) -> np.ndarray:
        """Apply temporal filter to a fused depth map.

        Args:
            depth_map: Fused depth (H, W), float32.

        Returns:
            Temporally-smoothed depth map (H, W), float32.
        """
        if not self.enabled:
            return depth_map

        depth = depth_map.astype(np.float32)

        if self._ema_state is None or self._ema_state.shape != depth.shape:
            self._ema_state = depth.copy()
            return depth

        alpha = self._config.alpha
        self._ema_state = alpha * depth + (1.0 - alpha) * self._ema_state
        return self._ema_state.copy()

    def reset_state(self, sequence_id: Optional[str] = None) -> None:
        """Reset temporal state (StatefulBackend protocol).

        Args:
            sequence_id: Optional identifier for the new sequence.
        """
        self._ema_state = None
        logger.debug(
            "TemporalPostFilter state reset (sequence_id=%s)",
            sequence_id,
        )


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
        temporal_post_filter: Optional[TemporalPostFilterConfig] = None,
        *,
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ):
        """Initialize depth ensemble backend.

        Args:
            config: EnhanceConfig for license validation and device settings.
            models: List of ModelConfig for ensemble.
                If None, uses default 3-model config.
            fusion_method: Fusion algorithm
                ("variance_weighted").
            max_variance_threshold: Max acceptable
                variance (>threshold flags warning).
            temporal_post_filter: Optional post-fusion
                temporal filter config (ADR-026 §2.2).
        """
        if (candidate_authority is None) != (canonical_plan_bytes is None):
            raise ValueError("candidate_authority and canonical_plan_bytes must be provided together")
        if candidate_authority is not None:
            if config is None:
                raise ValueError("Canonical ensemble authority requires its projected runtime config")
            if (
                candidate_authority.backend_id != self.name
                or candidate_authority.candidate.backend_id != self.name
                or candidate_authority.constituent_backend_id is not None
            ):
                raise ValueError("Canonical ensemble authority does not select the top-level ensemble candidate")
            if candidate_authority.model_contract is not None:
                raise ValueError("Top-level ensemble authority must not collapse to one constituent")
            if models is not None:
                raise ValueError("Canonical ensemble authority cannot be mixed with caller-supplied models")
            if type(canonical_plan_bytes) is not bytes or not canonical_plan_bytes:
                raise ValueError("Canonical ensemble authority requires non-empty immutable plan bytes")

        self._config = config
        self._candidate_authority = candidate_authority
        self._canonical_plan_bytes = canonical_plan_bytes
        self._canonical_plan = None
        if candidate_authority is not None:
            from ...lux_depth_v3.execution_lifecycle import (
                backend_candidate_authority,
                consume_lux_worker_execution_plan,
            )

            if type(canonical_plan_bytes) is not bytes:  # pragma: no cover - narrowed above
                raise ValueError("Canonical ensemble authority requires immutable plan bytes")
            self._canonical_plan = consume_lux_worker_execution_plan(canonical_plan_bytes)
            reselected = backend_candidate_authority(
                self._canonical_plan,
                candidate_authority.candidate_id,
            )
            if reselected != candidate_authority:
                raise ValueError("Canonical ensemble authority does not match its exact plan bytes")
        if candidate_authority is not None:
            fusion_method = str(getattr(config, "ensemble_fusion_method", fusion_method))
            max_variance_threshold = float(getattr(config, "ensemble_max_variance_threshold", max_variance_threshold))
            temporal_post_filter = TemporalPostFilterConfig(
                mode=str(getattr(config, "ensemble_temporal_filter_mode", "off")),
                alpha=float(getattr(config, "ensemble_temporal_filter_alpha", 0.3)),
            )
        self._fusion_method = fusion_method
        self._max_variance_threshold = max_variance_threshold
        self._temporal_post_filter = TemporalPostFilter(temporal_post_filter)

        # Initialize models
        if candidate_authority is not None:
            self._models = self._models_from_authority(candidate_authority)
        elif models is None:
            self._models = self._get_default_models(config)
        else:
            self._models = models

        # Lazy-loaded backends
        self._backends: Dict[str, Any] = {}

        # Validate ensemble configuration
        self._validate_ensemble()

    @staticmethod
    def _models_from_authority(authority: "BackendCandidateAuthority") -> List[ModelConfig]:
        """Project the exact ordered ensemble contracts without defaults."""

        models: List[ModelConfig] = []
        seen: set[str] = set()
        for contract in authority.candidate.model_contracts:
            if contract.backend_id in seen:
                raise ValueError(f"Canonical ensemble repeats constituent {contract.backend_id!r}")
            seen.add(contract.backend_id)
            if contract.weight is None:
                raise ValueError(f"Canonical ensemble constituent {contract.backend_id!r} lacks a weight")
            models.append(
                ModelConfig(
                    name=contract.backend_id,
                    weight=float(contract.weight),
                    checkpoint=contract.artifact_path,
                    device=contract.device,
                    enabled=contract.enabled,
                    model_contract=contract,
                )
            )
        return models

    def _get_default_models(
        self,
        config: Optional["EnhanceConfig"],
    ) -> List[ModelConfig]:
        """Get default 3-model ensemble configuration from ADR-026.

        Returns:
            List of ModelConfig for Depth Pro + DA3 + DepthCrafter.
        """
        return [
            ModelConfig(
                name="depth_pro",
                weight=0.5,  # Primary model
                checkpoint=(
                    getattr(
                        config,
                        "depth_pro_checkpoint_path",
                        None,
                    )
                    if config
                    else None
                ),
                device="auto",
            ),
            ModelConfig(
                name="da3",
                weight=0.3,  # Secondary (detail preservation)
                checkpoint=None,  # Auto-download from HF
                device="auto",
            ),
            # DepthCrafter: temporal consistency (Phase 2)
            ModelConfig(
                name="depthcrafter",
                weight=0.2,  # Tertiary (temporal consistency)
                checkpoint=None,
                device="auto",
                enabled=False,  # Disabled until checkpoint available
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
                f"Ensemble has <2 enabled models "
                f"({len(enabled_models)}). "
                "Consider using single-model backend "
                "instead."
            )

        # Validate weights sum to 1.0
        total_weight = sum(m.weight for m in enabled_models)
        if abs(total_weight - 1.0) > 1e-6:
            if self._candidate_authority is not None:
                raise ValueError(
                    f"Canonical ensemble weights must sum to 1.0 without runtime normalization (got {total_weight})"
                )
            logger.warning(
                "Model weights sum to %s, not 1.0. " "Normalizing weights automatically.",
                total_weight,
            )
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
        enabled_models = [m for m in self._models if m.enabled]
        if self._candidate_authority is not None and device is not None:
            requested = str(device).strip().lower()
            mismatches = [model.name for model in enabled_models if model.device not in {"auto", requested}]
            if mismatches:
                raise ValueError(
                    "Ensemble device override disagrees with carried constituent authority for " + ", ".join(mismatches)
                )
        enabled_count = len(enabled_models)
        logger.info(
            "Running depth ensemble with %d models",
            enabled_count,
        )

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
        if self._candidate_authority is not None:
            fused_result.metadata["execution_authority"] = {
                "plan_fingerprint_sha256": self._candidate_authority.plan_fingerprint_sha256,
                "candidate_id": self._candidate_authority.candidate_id,
                "model_backend_id": None,
                "executed_backend_id": self.name,
            }
            fused_result.metadata["ensemble"]["constituents"] = [
                {
                    "backend_id": model.name,
                    "weight": model.weight,
                    "device": model.device,
                    "enabled": model.enabled,
                }
                for model in self._models
            ]

        # Quality gate: warn if high variance
        if fused_result.variance_map is not None and fused_result.variance_map.mean() > self._max_variance_threshold:
            var_mean = fused_result.variance_map.mean()
            logger.warning(
                "High inter-model variance " "(%.3f > threshold %s). " "Review depth map manually " "for quality.",
                var_mean,
                self._max_variance_threshold,
            )
            fused_result.warnings.append(f"High variance: {var_mean:.3f}")

        # ADR-026 §2.2: Apply optional post-fusion temporal filter (video only)
        if self._temporal_post_filter.enabled:
            fused_result.depth_map = self._temporal_post_filter.apply(
                fused_result.depth_map,
            )
            fused_result.metadata["temporal_post_filter"] = {
                "mode": self._temporal_post_filter.mode,
                "alpha": self._temporal_post_filter.alpha,
            }

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
                model_device = device
                if model_device is None and model_config.device != "auto":
                    model_device = model_config.device
                result = backend.compute(image, device=model_device)
                results[model_config.name] = result

            except Exception as e:
                if self._candidate_authority is not None:
                    raise LuxExecutionPlanAuthorityError(
                        f"Canonical ensemble constituent {model_config.name!r} failed; " "exact planned membership is required"
                    ) from e
                logger.error(
                    "Model %s failed: %s. " "Excluding from ensemble.",
                    model_config.name,
                    e,
                )
                # Continue with remaining models

        if not results:
            raise RuntimeError("All ensemble models failed. " "Cannot compute depth.")
        if self._candidate_authority is not None:
            planned = tuple(model.name for model in enabled_models)
            executed = tuple(results)
            if executed != planned:
                raise LuxExecutionPlanAuthorityError(
                    f"Canonical ensemble executed constituents {executed!r}; expected exact planned order {planned!r}"
                )

        return results

    def _get_backend(
        self,
        model_config: ModelConfig,
    ) -> Any:
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
        if self._candidate_authority is not None:
            contract = model_config.model_contract
            if contract is None:
                raise ValueError(f"Canonical ensemble constituent {model_config.name!r} lacks its exact contract")
            if self._canonical_plan is None:
                raise RuntimeError("Canonical ensemble plan is unavailable")
            from ...lux_depth_v3.execution_lifecycle import backend_candidate_authority

            child_authority = backend_candidate_authority(
                self._canonical_plan,
                self._candidate_authority.candidate_id,
                model_backend_id=contract.backend_id,
            )
            if child_authority.model_contract != contract:
                raise ValueError("Canonical ensemble constituent changed during exact selection")
            backend = registry.get_backend(
                model_config.name,
                self._config,
                candidate_authority=child_authority,
                canonical_plan_bytes=self._canonical_plan_bytes,
            )
        elif model_config.name.endswith("_stub"):
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
        depth_stack = np.stack(
            [aligned_depths[n] for n in names],
            axis=0,
        ).astype(np.float32)
        mean_map = np.mean(depth_stack, axis=0)  # (H, W)
        variance_map = np.var(depth_stack, axis=0)  # (H, W)

        # Step 3: Compute per-model confidence maps (ACTUALLY adaptive)
        #
        # Key idea:
        # - A single "inv_variance" map applied to
        #   every model cancels algebraically in the
        #   fusion ratio.
        # - We need *per-model* per-pixel confidences
        #   that downweight outliers.
        #
        # We compute a normalized squared deviation
        # (z^2) and convert it to a confidence:
        #   z2_i = (d_i - mean)^2 / (var + eps)
        #   conf_i = exp(-0.5 * z2_i)
        #
        # This yields:
        #   fused = Σ(d_i * w_i * conf_i) / Σ(w_i * conf_i)
        #
        # ADR-026 §2.1: Synthetic/fallback models get confidence=0 so they
        # cannot poison the ensemble fusion.
        #
        epsilon = 1e-6
        denom = variance_map + epsilon
        z2 = (depth_stack - mean_map[None, :, :]) ** 2 / denom[None, :, :]
        conf = np.exp(-0.5 * z2).astype(np.float32)  # (N, H, W)

        # ADR-026 §2.1: Zero out confidence for synthetic/fallback models.
        # A synthetic depth signal must not distort variance-weighted fusion.
        for i, name in enumerate(names):
            result_meta = model_results[name].metadata
            is_synthetic = result_meta.get("synthetic") or result_meta.get("fallback_mode")
            if is_synthetic:
                logger.info(
                    "Model '%s' is in synthetic/fallback" " mode; setting ensemble " "confidence to 0.",
                    name,
                )
                conf[i] = 0.0

        # Get model weights from config
        model_weights = {m.name: m.weight for m in self._models if m.enabled and m.name in aligned_depths}

        # Normalize model weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight for k, v in model_weights.items()}

        # Build base weight tensor aligned to the same model order
        base_w = np.array(
            [model_weights.get(n, 0.0) for n in names],
            dtype=np.float32,
        )[
            :, None, None
        ]  # (N,1,1)

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
        primary_result = model_results.get(
            "depth_pro",
            next(iter(model_results.values())),
        )

        # Store a compact "effective weight" summary
        # per model (scalar), for observability. This
        # avoids huge per-pixel maps in the result
        # while still showing who contributed.
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

    def _align_depth_maps(
        self,
        model_results: Dict[str, DepthResult],
    ) -> Dict[str, np.ndarray]:
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
                    # Note: This is a heuristic.
                    # Ideally, we'd have camera
                    # intrinsics.
                    logger.warning(
                        "Model %s outputs relative " "depth. Scaling to metric " "is approximate.",
                        name,
                    )
                    # Simple scaling: assume depth range 0-10 meters
                    aligned[name] = result.depth_map * 10.0
        else:
            # All relative depth - normalize to [0, 1]
            for name, result in model_results.items():
                # Use float64 for normalization to avoid precision loss
                # when vmin is large (float32 epsilon > 1e-6 at ≥50).
                depth = result.depth_map.astype(np.float64)
                # Robust normalization
                vmin = np.percentile(depth, 1)
                vmax = np.percentile(depth, 99)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                aligned[name] = np.clip(
                    (depth - vmin) / (vmax - vmin),
                    0.0,
                    1.0,
                ).astype(np.float32)

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
                if self._candidate_authority is not None:
                    raise LuxExecutionPlanAuthorityError(
                        f"Canonical ensemble constituent {model_config.name!r} is unavailable; "
                        "exact planned membership is required"
                    ) from e
                logger.warning(f"Model {model_config.name} unavailable: {e}")

        if self._candidate_authority is not None and available_count != len(enabled_models):
            raise LuxExecutionPlanAuthorityError("Canonical ensemble availability did not preserve exact planned membership")
        if available_count < 2:
            raise RuntimeError(
                f"Ensemble requires ≥2 models, but " f"only {available_count} available. " "Cannot initialize ensemble."
            )

        logger.info(f"Ensemble initialized with {available_count} models")

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return additional required import modules for ensemble.

        Per the DepthBackend protocol, torch is
        assumed and should not be listed here.
        The ensemble additionally requires
        ``transformers`` (for DA3). Depth Pro and
        DepthCrafter remain optional and will degrade
        gracefully if unavailable.

        Returns:
            List of required module names.
        """
        return ["transformers"]  # DA3 minimum

    def reset_state(self, sequence_id: Optional[str] = None) -> None:
        """Reset temporal state for a new sequence (StatefulBackend protocol).

        Resets the post-fusion temporal filter and delegates to any stateful
        sub-backends (e.g., DepthCrafter). Called by the orchestrator at
        sequence boundaries (ADR-026 §2.3).

        Args:
            sequence_id: Optional identifier for the new sequence.
        """
        # Reset post-fusion temporal filter
        self._temporal_post_filter.reset_state(sequence_id)

        # Reset stateful sub-backends
        for backend in self._backends.values():
            reset = getattr(backend, "reset_state", None)
            if reset is not None:
                reset(sequence_id)
                continue
            reset_temporal = getattr(
                backend,
                "reset_temporal_state",
                None,
            )
            if reset_temporal is not None:
                reset_temporal()

        logger.debug(
            "Ensemble state reset (sequence_id=%s)",
            sequence_id,
        )
