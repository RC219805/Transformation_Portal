"""Material classification using CLIP (Phase 2.1).

This module provides optional semantic labeling of segmented regions
using OpenAI's CLIP model for zero-shot classification.

Common materials in architectural visualization:
- Wood (oak, walnut, maple, teak)
- Stone (marble, granite, limestone, travertine)
- Metal (steel, brass, copper, aluminum)
- Glass (clear, frosted, tinted)
- Fabric (linen, velvet, silk, cotton)
- Concrete
- Leather
- Paint/plaster

Architecture (ADR-027):
- Optional dependency (graceful fallback if CLIP unavailable)
- Zero-shot classification (no fine-tuning required)
- Confidence thresholding (low confidence = unlabeled)

Example:
    >>> classifier = MaterialClassifier(device="cuda")
    >>> if classifier.is_available():
    ...     labels = classifier.classify_masks(image, masks)
    ...     print(labels)  # ["wood", "marble", "glass", ...]
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from transformation_portal.core.security.model_lock import resolve_model_lock_revision

logger = logging.getLogger(__name__)


class MaterialClassifier:
    """CLIP-based material classifier for segmented regions.

    Attributes:
        device: Compute device ("cuda", "cpu", "mps").
        confidence_threshold: Minimum confidence to assign label.
        material_classes: List of material class names.
    """

    DEFAULT_MATERIAL_CLASSES = [
        "wood floor",
        "wood panel",
        "marble surface",
        "granite stone",
        "limestone wall",
        "travertine tile",
        "brushed steel",
        "polished brass",
        "copper metal",
        "aluminum frame",
        "clear glass",
        "frosted glass",
        "tinted glass",
        "linen fabric",
        "velvet upholstery",
        "silk curtain",
        "cotton textile",
        "concrete wall",
        "concrete floor",
        "leather furniture",
        "painted wall",
        "plaster ceiling",
        "ceramic tile",
        "porcelain surface",
    ]
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        material_classes: Optional[List[str]] = None,
        *,
        model_revision: Optional[str] = None,
        strict_model_lock: Optional[bool] = None,
        strict: bool = False,
    ):
        """Initialize material classifier.

        Args:
            device: Compute device.
            confidence_threshold: Minimum confidence [0, 1] to assign label.
            material_classes: Custom material classes (or None for defaults).
            model_revision: Optional immutable revision for CLIP model assets.
            strict_model_lock: Enforce pinned revisions for remote model loads.
                If None, uses ``TP_STRICT_MODEL_LOCK`` environment variable.
            strict: If True, model load/inference failures are raised instead
                of returning unlabeled material results.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.material_classes = material_classes or self.DEFAULT_MATERIAL_CLASSES
        self.strict = bool(strict)
        self.strict_model_lock = strict_model_lock
        self.model_revision = resolve_model_lock_revision(
            self.CLIP_MODEL_ID,
            model_revision,
            strict=self.strict_model_lock,
            context="MaterialClassifier",
        )

        self._model = None
        self._processor = None
        self._available = None  # Lazy check
        self._last_timing_ms = {}

        logger.info(
            f"MaterialClassifier initialized: "
            f"device={device}, threshold={confidence_threshold}, "
            f"classes={len(self.material_classes)}"
        )

    def is_available(self) -> bool:
        """Check if CLIP is available.

        Returns:
            True if CLIP can be imported and loaded.
        """
        if self._available is not None:
            return self._available

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._available = True
            logger.info("CLIP is available")
        except ImportError:
            self._available = False
            logger.warning("CLIP not available (transformers or torch missing)")

        return self._available

    def _load_model(self):
        """Lazy load CLIP model and processor.

        Raises:
            ImportError: If CLIP dependencies missing.
            RuntimeError: If model loading fails.
        """
        if self._model is not None:
            return  # Already loaded

        if not self.is_available():
            raise ImportError("CLIP not available. Install with: pip install transformers torch")

        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model...")

        try:
            # Use OpenAI's CLIP ViT-B/32 (good balance of speed/quality)
            self._processor = CLIPProcessor.from_pretrained(  # nosec B615
                self.CLIP_MODEL_ID,
                revision=self.model_revision,
            )
            self._model = CLIPModel.from_pretrained(  # nosec B615
                self.CLIP_MODEL_ID,
                revision=self.model_revision,
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

            self._model.eval()
            logger.info("CLIP model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}") from e

    def classify_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
    ) -> List[Tuple[Optional[str], Optional[float]]]:
        """Classify material for each masked region.

        Args:
            image: RGB image (H, W, 3) uint8 or float32.
            masks: Boolean masks (N, H, W).

        Returns:
            List of (label, confidence) tuples (N items).
            Label is None if confidence below threshold.
        """
        self._last_timing_ms = {}
        if not self.is_available():
            if self.strict:
                raise RuntimeError(
                    "Material classification is enabled in strict mode, but CLIP is unavailable. "
                    "Install transformers and torch, or disable strict material classification."
                )
            logger.warning("CLIP not available, returning unlabeled")
            return [(None, None) for _ in range(len(masks))]

        # Empty masks never require model inference; keep this path torch-free.
        if len(masks) == 0 or not np.any(masks):
            return [(None, None) for _ in range(len(masks))]

        try:
            self._load_model()
            import torch
        except Exception as exc:
            if self.strict:
                raise
            logger.warning("CLIP material classification unavailable; returning unlabeled masks: %s", exc)
            return [(None, None) for _ in range(len(masks))]

        results: List[Tuple[Optional[str], Optional[float]]] = [(None, None) for _ in range(len(masks))]
        extracted_regions = []
        extracted_indices = []

        t_extract = time.perf_counter()
        for idx, mask in enumerate(masks):
            # Extract masked region
            masked_image = self._extract_masked_region(image, mask)

            if masked_image is None:
                # Empty mask or extraction failed
                continue

            extracted_regions.append(masked_image)
            extracted_indices.append(idx)
        self._last_timing_ms["extract_regions"] = round((time.perf_counter() - t_extract) * 1000.0, 3)

        if not extracted_regions:
            return results

        try:
            t_clip = time.perf_counter()
            # Prepare all crops in one processor call. This keeps text/material
            # ordering stable while avoiding per-mask tokenizer/model overhead.
            inputs = self._processor(
                text=self.material_classes,
                images=extracted_regions,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Run CLIP for the whole mask batch
            with torch.no_grad():
                outputs = self._model(**inputs)

            probs_by_image = outputs.logits_per_image.softmax(dim=1)
            for row_idx, mask_idx in enumerate(extracted_indices):
                probs = probs_by_image[row_idx]
                best_idx = probs.argmax().item()
                best_prob = probs[best_idx].item()
                if best_prob >= self.confidence_threshold:
                    results[mask_idx] = (self.material_classes[best_idx], best_prob)
            self._last_timing_ms["clip_batch"] = round((time.perf_counter() - t_clip) * 1000.0, 3)
            self._last_timing_ms["batch_size"] = len(extracted_regions)
        except Exception as exc:
            if self.strict:
                raise
            logger.warning("CLIP material classification failed for mask batch; leaving masks unlabeled: %s", exc)

        return results

    def _extract_masked_region(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract masked region from image.

        Args:
            image: RGB image (H, W, 3) uint8 or float32.
            mask: Boolean mask (H, W).

        Returns:
            Cropped RGB image (H', W', 3) uint8, or None if empty.
        """
        if mask.sum() == 0:
            return None  # Empty mask

        # Find bounding box
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        # Crop image
        cropped = image[y1:y2, x1:x2].copy()

        # Convert to uint8 if needed
        if cropped.dtype != np.uint8:
            if cropped.dtype == np.float32 or cropped.dtype == np.float64:
                # Assume [0, 1] range for float
                cropped = np.clip(cropped, 0, 1)
                cropped = (cropped * 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image dtype: {cropped.dtype}")

        # Apply mask within crop (zero out non-masked pixels)
        mask_crop = mask[y1:y2, x1:x2]
        cropped[~mask_crop] = 0

        return cropped
