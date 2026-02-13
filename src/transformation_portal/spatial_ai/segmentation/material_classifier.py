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
from typing import List, Optional, Tuple

import numpy as np

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

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        material_classes: Optional[List[str]] = None,
    ):
        """Initialize material classifier.

        Args:
            device: Compute device.
            confidence_threshold: Minimum confidence [0, 1] to assign label.
            material_classes: Custom material classes (or None for defaults).
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.material_classes = material_classes or self.DEFAULT_MATERIAL_CLASSES

        self._model = None
        self._processor = None
        self._available = None  # Lazy check

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
            model_id = "openai/clip-vit-base-patch32"

            self._processor = CLIPProcessor.from_pretrained(model_id)  # nosec B615 - public model; revision pinning tracked in ADR-027
            self._model = CLIPModel.from_pretrained(model_id)  # nosec B615 - public model; revision pinning tracked in ADR-027

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
        if not self.is_available():
            logger.warning("CLIP not available, returning unlabeled")
            return [(None, None) for _ in range(len(masks))]

        self._load_model()

        import torch

        results = []

        for mask in masks:
            # Extract masked region
            masked_image = self._extract_masked_region(image, mask)

            if masked_image is None:
                # Empty mask or extraction failed
                results.append((None, None))
                continue

            # Prepare inputs for CLIP
            inputs = self._processor(
                text=self.material_classes,
                images=masked_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Run CLIP
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Compute similarities (logits)
            logits_per_image = outputs.logits_per_image  # (1, N_classes)
            probs = logits_per_image.softmax(dim=1)[0]  # (N_classes,)

            # Get top prediction
            best_idx = probs.argmax().item()
            best_prob = probs[best_idx].item()

            # Check confidence threshold
            if best_prob >= self.confidence_threshold:
                label = self.material_classes[best_idx]
                results.append((label, best_prob))
            else:
                results.append((None, None))

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
