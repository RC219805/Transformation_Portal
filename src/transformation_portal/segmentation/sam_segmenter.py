"""SAM (Segment Anything Model) integration for universal segmentation.

SAM provides:
- Universal segmentation without category-specific training
- 50ms inference time after image embedding
- Trained on 1.1B masks across 11M images
- Zero-shot generalization to architectural imagery

For luxury real estate:
- Segment architectural elements (walls, fixtures, materials)
- Enable region-specific enhancement
- Maintain semantic boundaries during processing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from segment_anything import (
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning(
        "SAM not available. Install with: "
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    )


logger = logging.getLogger(__name__)


class SAMSegmenter:
    """Segment Anything Model for universal image segmentation.

    Provides zero-shot segmentation for architectural imagery without
    requiring domain-specific training.

    Attributes:
        model_type: SAM model variant (vit_h, vit_l, vit_b)
        device: Computation device
        predictor: SAM predictor for prompted segmentation
        mask_generator: Automatic mask generation

    Example:
        >>> segmenter = SAMSegmenter(model_type="vit_h")
        >>> masks = segmenter.segment_automatic("luxury_kitchen.jpg")
        >>> print(f"Found {len(masks)} segments")

        >>> # Prompted segmentation
        >>> point_coords = [[500, 300]]  # Point at marble countertop
        >>> mask = segmenter.segment_from_points("luxury_kitchen.jpg", point_coords)
    """

    # Model checkpoints (download from SAM releases)
    MODEL_CHECKPOINTS = {
        "vit_h": "sam_vit_h_4b8939.pth",  # Highest quality, 2.4GB
        "vit_l": "sam_vit_l_0b3195.pth",  # Large, 1.2GB
        "vit_b": "sam_vit_b_01ec64.pth",  # Base, 375MB
    }

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ):
        """Initialize SAM segmenter.

        Args:
            model_type: Model variant (vit_h/vit_l/vit_b)
            checkpoint_path: Path to model checkpoint
            device: Computation device (auto-detected if None)
            points_per_side: Grid points for automatic segmentation
            pred_iou_thresh: Predicted IoU threshold for mask quality
            stability_score_thresh: Stability threshold for mask quality
            min_mask_region_area: Minimum mask area in pixels

        Raises:
            ImportError: If SAM not installed
            FileNotFoundError: If checkpoint not found
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM required. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.model_type = model_type
        self.device = device or self._detect_device()

        # Locate checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint(model_type)

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )

        logger.info(f"Initializing SAM ({model_type}) on {self.device}")

        # Load model
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(device=self.device)
        self.sam.eval()

        # Create predictor for prompted segmentation
        self.predictor = SamPredictor(self.sam)

        # Create automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

        logger.info("SAM initialized successfully")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _find_checkpoint(self, model_type: str) -> Path:
        """Find SAM checkpoint in common locations."""
        checkpoint_name = self.MODEL_CHECKPOINTS[model_type]

        search_paths = [
            Path.home() / ".cache" / "sam" / checkpoint_name,
            Path("checkpoints") / checkpoint_name,
            Path("models") / checkpoint_name,
            Path(checkpoint_name),
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Not found - provide helpful error
        raise FileNotFoundError(
            f"SAM checkpoint '{checkpoint_name}' not found.\n"
            f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
            f"Place in: {search_paths[0]}"
        )

    def segment_automatic(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        filter_by_area: bool = True,
        min_area: int = 500,
        max_masks: Optional[int] = None
    ) -> List[Dict]:
        """Automatic segmentation without prompts.

        Generates masks for all objects/regions in the image.

        Args:
            image: Input image
            filter_by_area: Filter out small masks
            min_area: Minimum mask area if filtering
            max_masks: Maximum number of masks to return (None = all)

        Returns:
            List of mask dictionaries with keys:
                - segmentation: bool array (H, W)
                - area: mask area in pixels
                - bbox: bounding box [x, y, w, h]
                - predicted_iou: quality score
                - stability_score: mask stability score
        """
        # Load and convert image
        image_np = self._load_image_rgb(image)

        # Generate masks
        logger.info("Generating automatic masks...")
        masks = self.mask_generator.generate(image_np)
        logger.info(f"Generated {len(masks)} masks")

        # Filter by area
        if filter_by_area:
            masks = [m for m in masks if m['area'] >= min_area]
            logger.info(f"After area filtering: {len(masks)} masks")

        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Limit number
        if max_masks is not None:
            masks = masks[:max_masks]

        return masks

    def segment_from_points(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        point_coords: List[List[int]],
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = False
    ) -> np.ndarray:
        """Segment using point prompts.

        Args:
            image: Input image
            point_coords: List of [x, y] coordinates
            point_labels: 1 for foreground point, 0 for background point
            multimask_output: Return multiple candidate masks

        Returns:
            Segmentation mask (H, W) or multiple masks (N, H, W)
        """
        # Load and convert image
        image_np = self._load_image_rgb(image)

        # Set image
        self.predictor.set_image(image_np)

        # Convert to numpy arrays
        point_coords = np.array(point_coords)
        if point_labels is None:
            point_labels = np.ones(len(point_coords))  # All foreground
        else:
            point_labels = np.array(point_labels)

        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        if multimask_output:
            # Return best mask based on score
            best_idx = np.argmax(scores)
            return masks[best_idx]
        else:
            return masks[0]

    def segment_from_box(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        box: List[int]  # [x1, y1, x2, y2]
    ) -> np.ndarray:
        """Segment using bounding box prompt.

        Args:
            image: Input image
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            Segmentation mask (H, W)
        """
        # Load and convert image
        image_np = self._load_image_rgb(image)

        # Set image
        self.predictor.set_image(image_np)

        # Predict mask
        masks, scores, logits = self.predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        return masks[0]

    def create_colored_mask_overlay(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        masks: List[Dict],
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create visualization with colored masks overlaid.

        Args:
            image: Original image
            masks: List of mask dictionaries from segment_automatic
            alpha: Overlay transparency (0-1)

        Returns:
            RGB image with colored mask overlay
        """
        image_np = self._load_image_rgb(image)
        overlay = image_np.copy()

        # Generate distinct colors
        np.random.seed(42)  # Consistent colors
        colors = np.random.randint(0, 255, size=(len(masks), 3))

        # Apply each mask with different color
        for mask_dict, color in zip(masks, colors):
            mask = mask_dict['segmentation']
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        return overlay.astype(np.uint8)

    def extract_largest_segments(
        self,
        masks: List[Dict],
        n: int = 10
    ) -> List[Dict]:
        """Extract N largest segments.

        Args:
            masks: List of mask dictionaries
            n: Number of largest segments to return

        Returns:
            List of N largest masks
        """
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        return sorted_masks[:n]

    def merge_masks(
        self,
        masks: List[np.ndarray],
        image_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Merge multiple masks into single mask.

        Args:
            masks: List of boolean masks
            image_shape: Output shape (H, W) if masks are different sizes

        Returns:
            Merged boolean mask
        """
        if image_shape is not None:
            merged = np.zeros(image_shape, dtype=bool)
        else:
            merged = np.zeros_like(masks[0], dtype=bool)

        for mask in masks:
            merged = np.logical_or(merged, mask)

        return merged

    def get_mask_statistics(self, masks: List[Dict]) -> Dict:
        """Calculate statistics for segmentation results.

        Args:
            masks: List of mask dictionaries

        Returns:
            Dictionary with statistics
        """
        if not masks:
            return {
                "num_masks": 0,
                "total_area": 0,
                "avg_area": 0,
                "median_area": 0,
                "avg_iou": 0
            }

        areas = [m['area'] for m in masks]
        ious = [m['predicted_iou'] for m in masks]

        return {
            "num_masks": len(masks),
            "total_area": sum(areas),
            "avg_area": np.mean(areas),
            "median_area": np.median(areas),
            "min_area": min(areas),
            "max_area": max(areas),
            "avg_iou": np.mean(ious),
            "avg_stability": np.mean([m['stability_score'] for m in masks])
        }

    def _load_image_rgb(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Load image as RGB numpy array.

        Args:
            image: Input in various formats

        Returns:
            RGB numpy array (H, W, 3) with dtype uint8
        """
        if isinstance(image, np.ndarray):
            # Already numpy array
            if image.ndim == 2:
                # Grayscale - convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA - convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image

        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image.convert("RGB"))

        elif isinstance(image, (str, Path)):
            # File path
            img = Image.open(image).convert("RGB")
            return np.array(img)

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __repr__(self) -> str:
        return (
            f"SAMSegmenter(model='{self.model_type}', "
            f"device='{self.device}')"
        )
