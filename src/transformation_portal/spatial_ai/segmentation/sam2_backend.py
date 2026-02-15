"""SAM2 backend for segmentation (Phase 2.1).

This module wraps Meta's Segment Anything Model 2 (SAM2) for:
- Automatic mask generation (full image)
- Prompted segmentation (points/bboxes)
- Video temporal tracking

Architecture (ADR-027):
- HuggingFace model loading with revision pinning
- GPU/CPU device selection
- Batched inference for efficiency
- Contract-driven input/output

Model Variants:
- sam2-hiera-base-plus: Faster, good quality
- sam2-hiera-large: Slower, best quality (research tier)

Example:
    >>> backend = SAM2Backend(model_size="large", device="cuda")
    >>> result = backend.segment(
    ...     image=linear_rgb,  # (H, W, 3) float32
    ...     gamma=1.0,
    ...     mode="auto"
    ... )
    >>> print(f"Found {len(result.masks)} segments")

License: Apache 2.0 (commercial OK, no tier restrictions)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult

logger = logging.getLogger(__name__)


class SAM2Backend:
    """SAM2 segmentation backend with HuggingFace integration.

    Attributes:
        model_size: Model variant ("base" or "large").
        device: Compute device ("cuda", "cpu", "mps").
        revision: HuggingFace model revision (commit SHA or placeholder).
    """

    SUPPORTED_MODELS = {
        "base": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        revision: Optional[str] = None,
    ):
        """Initialize SAM2 backend.

        Args:
            model_size: Model variant ("base" or "large").
            device: Compute device.
            revision: HuggingFace commit SHA (None = use latest).
                For experimental presets: "NEEDS_VERIFICATION_SAM2_..."
                For stable presets: Must be 40-char commit SHA.

        Raises:
            ValueError: If model_size invalid.
            ImportError: If SAM2 dependencies missing.
        """
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"Invalid model_size '{model_size}', " f"must be one of {list(self.SUPPORTED_MODELS.keys())}")

        self.model_size = model_size
        self.device = device
        self.revision = revision or "NEEDS_VERIFICATION_SAM2_BASE_20260211"

        self._model = None
        self._processor = None

        logger.info(f"SAM2Backend initialized: model={model_size}, " f"device={device}, revision={self.revision[:12]}...")

    def _load_model(self):
        """Lazy load SAM2 model and processor.

        Raises:
            ImportError: If transformers/torch missing.
            ValueError: If revision is an unverified placeholder.
            RuntimeError: If model download fails.
        """
        if self._model is not None:
            return  # Already loaded

        # ADR-027: reject unverified revision placeholders at load time
        if self.revision.startswith("NEEDS_VERIFICATION"):
            raise ValueError(
                f"SAM2 model revision '{self.revision}' is an unverified placeholder. "
                "Supply a pinned HuggingFace commit SHA for production use. "
                "See ADR-027 for revision pinning policy."
            )

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError("SAM2 requires transformers and torch. " "Install with: pip install transformers torch") from e

        model_id = self.SUPPORTED_MODELS[self.model_size]

        logger.info(f"Loading SAM2 model: {model_id} @ {self.revision[:12]}...")

        try:
            # Load processor (handles image preprocessing)
            self._processor = AutoProcessor.from_pretrained(  # nosec B615
                model_id,
                revision=self.revision,
            )

            # Load model
            self._model = AutoModel.from_pretrained(  # nosec B615
                model_id,
                revision=self.revision,
            )

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            else:
                self._model = self._model.to("cpu")
                if self.device != "cpu":
                    logger.warning(f"Device '{self.device}' unavailable, using CPU")

            self._model.eval()  # Inference mode
            logger.info("SAM2 model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model: {e}") from e

    def segment(
        self,
        seg_input: SegmentationInput,
    ) -> SegmentationResult:
        """Segment image with SAM2.

        Args:
            seg_input: Validated segmentation input contract.

        Returns:
            SegmentationResult with masks, scores, and metadata.

        Raises:
            ValueError: If input contract violated.
            RuntimeError: If inference fails.
        """
        # Contract is already validated in SegmentationInput.__post_init__

        # Lazy load model
        self._load_model()

        # Execute segmentation based on mode
        if seg_input.mode == "auto":
            return self._segment_auto(seg_input)
        elif seg_input.mode in ["points", "bbox"]:
            return self._segment_prompted(seg_input)
        elif seg_input.mode == "video":
            return self._segment_video(seg_input)
        else:
            raise ValueError(f"Unsupported mode: {seg_input.mode}")

    def _extract_sam2_predictions(self, model_output) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract masks, IoU scores, and stability scores from SAM2 output.

        SAM2's mask decoder outputs:
        - pred_masks: (N, H, W) boolean masks
        - iou_predictions: (N,) float32 scores in [0, 1]
        - stability_scores: (N,) float32 scores in [0, 1]

        This method provides defensive extraction with fallback to 1.0
        if attributes are missing (e.g., in stub implementations).

        Args:
            model_output: SAM2 model output object

        Returns:
            Tuple of (masks, iou_scores, stability_scores)
            - masks: np.ndarray of shape (N, H, W), dtype bool
            - iou_scores: np.ndarray of shape (N,), dtype float32
            - stability_scores: np.ndarray of shape (N,), dtype float32

        Note:
            If SAM2 attributes are missing (stub backend), falls back
            to 1.0 for all scores to maintain backward compatibility.
        """
        # Extract masks (always present) with torch.Tensor handling
        masks = model_output.pred_masks  # (N, H, W) bool or torch.Tensor

        # Handle torch.Tensor masks (SAM2 returns CUDA/MPS tensors)
        if hasattr(masks, "detach"):  # torch.Tensor
            masks = masks.detach().cpu().numpy().astype(bool)
        else:
            masks = np.asarray(masks, dtype=bool)

        n_masks = len(masks)

        # Extract IoU scores (defensive) with torch.Tensor handling
        if hasattr(model_output, "iou_predictions") and model_output.iou_predictions is not None:
            iou_preds = model_output.iou_predictions
            if hasattr(iou_preds, "detach"):  # torch.Tensor
                iou_scores = iou_preds.detach().cpu().numpy().astype(np.float32)
            else:
                iou_scores = np.asarray(iou_preds, dtype=np.float32)
        else:
            # Fallback for stub backends
            iou_scores = np.ones(n_masks, dtype=np.float32)

        # Extract stability scores (defensive) with torch.Tensor handling
        if hasattr(model_output, "stability_scores") and model_output.stability_scores is not None:
            stab_scores = model_output.stability_scores
            if hasattr(stab_scores, "detach"):  # torch.Tensor
                stability_scores = stab_scores.detach().cpu().numpy().astype(np.float32)
            else:
                stability_scores = np.asarray(stab_scores, dtype=np.float32)
        else:
            # Fallback for stub backends
            stability_scores = np.ones(n_masks, dtype=np.float32)

        return masks, iou_scores, stability_scores

    def _segment_auto(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Automatic mask generation (entire image).

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult with all detected masks.

        Raises:
            NotImplementedError: SAM2 auto mode not yet integrated.
        """
        # SAM2 auto mode requires integration with the official automatic mask generator
        # The transformers AutoModel API does not expose this functionality directly
        #
        # To implement:
        # 1. Use SAM2's native automatic mask generator API
        # 2. Or implement custom grid-based prompting with the transformers model
        # 3. Ensure output format matches SegmentationResult contract
        # 4. Wrap inference in try-finally with _cleanup_inference_state() (A6)
        #
        # Example pattern for future implementation:
        #     inference_state = None
        #     try:
        #         inference_state = self._model.init_state(image)
        #         model_output = self._model.predict(...)
        #
        #         # Phase C.2: Extract real SAM2 scores (not placeholders)
        #         masks, iou_scores, stability_scores = self._extract_sam2_predictions(model_output)
        #
        #         # Build metadata with real stability scores
        #         metadata_list = []
        #         for i, mask in enumerate(masks):
        #             # ... compute area, bbox ...
        #             metadata = MaskMetadata(
        #                 area=area,
        #                 bbox=bbox,
        #                 stability_score=stability_scores[i],  # Real SAM2 confidence
        #             )
        #             metadata_list.append(metadata)
        #
        #         return SegmentationResult(
        #             masks=masks,
        #             scores=iou_scores,  # Real SAM2 IoU predictions
        #             metadata=metadata_list,
        #         )
        #     finally:
        #         self._cleanup_inference_state(inference_state)
        #
        # For Phase 2.1 scaffolding, this remains unimplemented to prevent runtime crashes
        # on untested placeholder code.

        raise NotImplementedError(
            "SAM2 automatic mask generation not yet integrated with official mask generator. "
            "The transformers AutoModel does not guarantee 'pred_masks' or 'iou_scores' attributes. "
            "Use prompted segmentation (mode='points' or mode='bbox') or integrate SAM2's native "
            "automatic mask generator API for production use."
        )

    def _segment_prompted(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Prompted segmentation (points/bboxes).

        Args:
            seg_input: Validated segmentation input with prompts.

        Returns:
            SegmentationResult with prompted masks.

        Note:
            When implementing, use try-finally pattern with _cleanup_inference_state() (A6).
        """
        # TODO: Implement prompted segmentation with memory cleanup
        # Pattern:
        #     inference_state = None
        #     try:
        #         inference_state = self._model.init_state(image)
        #         model_output = self._model.predict(prompts=...)
        #
        #         # Phase C.2: Extract real SAM2 scores (not placeholders)
        #         masks, iou_scores, stability_scores = self._extract_sam2_predictions(model_output)
        #
        #         # Build metadata with real stability scores
        #         metadata_list = []
        #         for i, mask in enumerate(masks):
        #             # ... compute area, bbox ...
        #             metadata = MaskMetadata(
        #                 area=area,
        #                 bbox=bbox,
        #                 stability_score=stability_scores[i],  # Real SAM2 confidence
        #             )
        #             metadata_list.append(metadata)
        #
        #         return SegmentationResult(
        #             masks=masks,
        #             scores=iou_scores,  # Real SAM2 IoU predictions
        #             metadata=metadata_list,
        #         )
        #     finally:
        #         self._cleanup_inference_state(inference_state)
        raise NotImplementedError("Prompted segmentation not yet implemented")

    def _segment_video(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Video temporal tracking.

        Args:
            seg_input: Validated segmentation input with prev_masks.

        Returns:
            SegmentationResult with temporally-tracked masks.

        Note:
            When implementing, use try-finally pattern with _cleanup_inference_state() (A6).
            This is CRITICAL for video mode to prevent VRAM accumulation across frames.
        """
        # TODO: Implement video tracking with memory cleanup
        # Pattern:
        #     inference_state = None
        #     try:
        #         inference_state = self._model.init_state(image)
        #         # Propagate masks from previous frame
        #         model_output = self._model.track(prev_masks=...)
        #
        #         # Phase C.2: Extract real SAM2 scores (not placeholders)
        #         masks, iou_scores, stability_scores = self._extract_sam2_predictions(model_output)
        #
        #         # Build metadata with real stability scores
        #         metadata_list = []
        #         for i, mask in enumerate(masks):
        #             # ... compute area, bbox ...
        #             metadata = MaskMetadata(
        #                 area=area,
        #                 bbox=bbox,
        #                 stability_score=stability_scores[i],  # Real SAM2 confidence
        #             )
        #             metadata_list.append(metadata)
        #
        #         return SegmentationResult(
        #             masks=masks,
        #             scores=iou_scores,  # Real SAM2 IoU predictions
        #             metadata=metadata_list,
        #             temporal_ids=...,  # Frame-to-frame tracking IDs
        #         )
        #     finally:
        #         self._cleanup_inference_state(inference_state)  # CRITICAL for video!
        raise NotImplementedError("Video tracking not yet implemented")

    def _cleanup_inference_state(self, inference_state: object) -> None:
        """Clean up SAM2 inference state to prevent memory leaks.

        SAM2's memory bank retains CUDA tensors across frames in video mode.
        Explicit cleanup prevents VRAM accumulation during batch processing.

        This method is defensive and should never raise exceptions.

        Args:
            inference_state: SAM2 inference state object to clean up.

        Note:
            Called in finally block to guarantee cleanup even on errors.
        """
        if inference_state is None:
            return

        try:
            import gc

            import torch

            # Device-agnostic synchronization before cleanup
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have synchronize(), but we can still proceed with cleanup
                pass

            # Reset state if the method exists (defensive check)
            if hasattr(inference_state, "reset_state"):
                inference_state.reset_state()

            # Delete reference
            del inference_state

            # Force garbage collection
            gc.collect()

            # Empty device cache (device-specific)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            # Defensive: log but don't raise, cleanup should never crash
            logger.warning(f"Error during SAM2 inference state cleanup: {e}")

    def _linear_to_srgb(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB uint8 for SAM2 preprocessing.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB [0, ∞).

        Returns:
            (H, W, 3) uint8 sRGB [0, 255].
        """
        # Clip HDR values (SAM2 expects [0, 1])
        linear_clipped = np.clip(linear_rgb, 0, 1)

        # Apply sRGB gamma (approximation: gamma 2.2)
        srgb = np.power(linear_clipped, 1.0 / 2.2)

        # Convert to uint8
        srgb_uint8 = (srgb * 255).astype(np.uint8)

        return srgb_uint8
