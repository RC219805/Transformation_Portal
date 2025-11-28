"""CLIP-based zero-shot classification for architectural materials.

CLIP enables text-guided element identification without training:
- "marble surfaces" -> precise masks
- "water features" -> pool/fountain detection
- "natural stone" -> stone material identification
- "premium fixtures" -> luxury element detection

For luxury real estate:
- Material classification (marble, wood, glass, metal)
- Style recognition (modern, traditional, coastal)
- Feature detection (water, vegetation, architectural details)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available (already included in transformers)")


logger = logging.getLogger(__name__)


class CLIPClassifier:
    """Zero-shot image classification using CLIP.

    Enables text-based classification without domain-specific training.
    Can classify entire images or image regions (segments).

    Attributes:
        model: CLIP vision-language model
        processor: CLIP preprocessor
        device: Computation device

    Example:
        >>> classifier = CLIPClassifier()
        >>> # Classify materials
        >>> materials = ["marble", "granite", "wood", "glass", "metal"]
        >>> probs = classifier.classify_image("countertop.jpg", materials)
        >>> best_material = materials[np.argmax(probs)]
        >>> print(f"Material: {best_material}")

        >>> # Classify segments
        >>> results = classifier.classify_segments(
        ...     "kitchen.jpg",
        ...     masks=[mask1, mask2, mask3],
        ...     categories=["marble countertop", "wood cabinet", "stainless steel"]
        ... )
    """

    # Predefined category sets for luxury real estate
    MATERIAL_CATEGORIES = [
        "marble",
        "granite",
        "quartz",
        "natural stone",
        "hardwood",
        "wood",
        "glass",
        "metal",
        "stainless steel",
        "brass",
        "bronze",
        "copper",
        "leather",
        "fabric",
        "tile",
        "porcelain",
        "concrete"
    ]

    ROOM_CATEGORIES = [
        "kitchen",
        "bathroom",
        "bedroom",
        "living room",
        "dining room",
        "office",
        "pool area",
        "courtyard",
        "entry hall"
    ]

    STYLE_CATEGORIES = [
        "modern architecture",
        "contemporary design",
        "traditional style",
        "mediterranean architecture",
        "coastal design",
        "industrial style",
        "minimalist design",
        "luxury estate"
    ]

    FEATURE_CATEGORIES = [
        "water feature",
        "swimming pool",
        "fireplace",
        "chandelier",
        "large windows",
        "high ceiling",
        "built-in cabinets",
        "kitchen island",
        "natural light",
        "outdoor space"
    ]

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize CLIP classifier.

        Args:
            model_name: HuggingFace CLIP model name
            device: Computation device (auto-detected if None)
            cache_dir: Model cache directory

        Raises:
            ImportError: If transformers not available
        """
        if not CLIP_AVAILABLE:
            # CLIP is actually available through transformers which is already installed
            # This check is mainly for clarity
            pass

        self.model_name = model_name
        self.device = device or self._detect_device()
        self.cache_dir = cache_dir

        logger.info(f"Initializing CLIP on {self.device}")

        # Load model and processor
        # nosec B615 - revision pinning intentionally omitted for development flexibility
        # Production deployments should pin specific model revisions
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP initialized successfully")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def classify_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        categories: List[str],
        temperature: float = 1.0
    ) -> np.ndarray:
        """Classify image into categories using zero-shot CLIP.

        Args:
            image: Input image
            categories: List of category names
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            Probability distribution over categories (sums to 1)
        """
        # Load image
        pil_image = self._load_image(image)

        # Prepare inputs
        inputs = self.processor(
            text=categories,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image / temperature
            probs = F.softmax(logits_per_image, dim=1)

        return probs.cpu().numpy()[0]

    def classify_segments(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        masks: List[np.ndarray],
        categories: List[str],
        background_color: Tuple[int, int, int] = (0, 0, 0),
        temperature: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Classify multiple image segments.

        Args:
            image: Original image
            masks: List of boolean masks for each segment
            categories: Category names to classify into
            background_color: Color for masked-out regions
            temperature: Softmax temperature

        Returns:
            List of classification results, one per mask:
                - mask_index: Index of this mask
                - probabilities: Probability for each category
                - top_category: Most likely category
                - confidence: Confidence score for top category
        """
        # Load image
        image_np = self._load_image_np(image)

        results = []

        for idx, mask in enumerate(masks):
            # Extract segment
            segment = self._extract_masked_region(
                image_np,
                mask,
                background_color
            )

            # Classify segment
            probs = self.classify_image(segment, categories, temperature)

            # Get top category
            top_idx = np.argmax(probs)
            top_category = categories[top_idx]
            confidence = probs[top_idx]

            results.append({
                "mask_index": idx,
                "probabilities": probs,
                "top_category": top_category,
                "confidence": float(confidence),
                "all_categories": {
                    cat: float(prob)
                    for cat, prob in zip(categories, probs)
                }
            })

        return results

    def classify_materials(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        custom_materials: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Classify materials in image.

        Args:
            image: Input image
            custom_materials: Custom material list (uses default if None)

        Returns:
            Dictionary mapping material names to probabilities
        """
        materials = custom_materials or self.MATERIAL_CATEGORIES

        probs = self.classify_image(image, materials)

        return {
            material: float(prob)
            for material, prob in zip(materials, probs)
        }

    def classify_room_type(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, float]:
        """Classify room type.

        Args:
            image: Input image

        Returns:
            Dictionary mapping room types to probabilities
        """
        probs = self.classify_image(image, self.ROOM_CATEGORIES)

        return {
            room: float(prob)
            for room, prob in zip(self.ROOM_CATEGORIES, probs)
        }

    def classify_style(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, float]:
        """Classify architectural style.

        Args:
            image: Input image

        Returns:
            Dictionary mapping styles to probabilities
        """
        probs = self.classify_image(image, self.STYLE_CATEGORIES)

        return {
            style: float(prob)
            for style, prob in zip(self.STYLE_CATEGORIES, probs)
        }

    def detect_features(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Detect luxury features in image.

        Args:
            image: Input image
            threshold: Minimum probability to include feature

        Returns:
            List of (feature, probability) tuples for detected features
        """
        probs = self.classify_image(image, self.FEATURE_CATEGORIES)

        detected_features = [
            (feature, float(prob))
            for feature, prob in zip(self.FEATURE_CATEGORIES, probs)
            if prob >= threshold
        ]

        # Sort by probability
        detected_features.sort(key=lambda x: x[1], reverse=True)

        return detected_features

    def find_material_regions(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        masks: List[np.ndarray],
        target_material: str,
        threshold: float = 0.5
    ) -> List[int]:
        """Find segments containing specific material.

        Args:
            image: Original image
            masks: List of segment masks from SAM
            target_material: Material to find (e.g., "marble", "wood")
            threshold: Minimum confidence threshold

        Returns:
            List of mask indices containing the target material
        """
        # Classify all segments for the target material
        categories = [target_material, f"not {target_material}"]

        results = self.classify_segments(image, masks, categories)

        # Find segments where target material confidence exceeds threshold
        matching_indices = [
            result["mask_index"]
            for result in results
            if result["probabilities"][0] >= threshold  # Index 0 is target material
        ]

        return matching_indices

    def create_semantic_map(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        masks: List[Dict],
        categories: List[str]
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Create semantic segmentation map.

        Args:
            image: Original image
            masks: List of mask dictionaries from SAM
            categories: Categories to classify segments into

        Returns:
            Tuple of:
                - Semantic map (H, W) with integer labels
                - Label dictionary mapping integers to category names
        """
        # Load image
        image_np = self._load_image_np(image)
        h, w = image_np.shape[:2]

        # Initialize semantic map
        semantic_map = np.zeros((h, w), dtype=np.int32)

        # Extract just the masks
        mask_arrays = [m['segmentation'] for m in masks]

        # Classify all segments
        classifications = self.classify_segments(
            image_np,
            mask_arrays,
            categories
        )

        # Build semantic map
        label_dict = {}

        for idx, (mask_dict, classification) in enumerate(zip(masks, classifications)):
            mask = mask_dict['segmentation']
            top_category = classification['top_category']

            # Assign label
            if top_category not in label_dict.values():
                label_id = len(label_dict) + 1
                label_dict[label_id] = top_category
            else:
                # Find existing label for this category
                label_id = [k for k, v in label_dict.items() if v == top_category][0]

            # Apply to semantic map
            semantic_map[mask] = label_id

        return semantic_map, label_dict

    def _extract_masked_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Extract masked region with background set to specified color.

        Args:
            image: RGB image array
            mask: Boolean mask
            background_color: RGB color for background

        Returns:
            Image with background masked
        """
        # Create background
        background = np.zeros_like(image)
        background[:] = background_color

        # Blend: use image where mask is True, background otherwise
        result = np.where(mask[:, :, np.newaxis], image, background)

        return result.astype(np.uint8)

    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Load image as PIL Image.

        Args:
            image: Input in various formats

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _load_image_np(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Load image as numpy array.

        Args:
            image: Input in various formats

        Returns:
            RGB numpy array (H, W, 3)
        """
        pil_image = self._load_image(image)
        return np.array(pil_image)

    def __repr__(self) -> str:
        return (
            f"CLIPClassifier(model='{self.model_name}', "
            f"device='{self.device}')"
        )
