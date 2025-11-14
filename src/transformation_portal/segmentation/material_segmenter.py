"""Material-aware segmentation combining SAM and CLIP.

Provides intelligent segmentation that understands:
- Material types (marble, wood, glass, metal, stone)
- Architectural elements (walls, fixtures, surfaces)
- Luxury features (water features, premium finishes)

Enables context-aware enhancement that processes different materials appropriately.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import cv2

from transformation_portal.segmentation.sam_segmenter import SAMSegmenter
from transformation_portal.segmentation.clip_classifier import CLIPClassifier


logger = logging.getLogger(__name__)


@dataclass
class MaterialSegment:
    """Segmented region with material classification.

    Attributes:
        mask: Boolean segmentation mask (H, W)
        material: Identified material type
        confidence: Classification confidence (0-1)
        area: Segment area in pixels
        bbox: Bounding box [x, y, w, h]
        centroid: Segment center point [x, y]
        properties: Additional segment properties
    """
    mask: np.ndarray
    material: str
    confidence: float
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    properties: Dict


class MaterialSegmenter:
    """Combined SAM + CLIP for material-aware segmentation.

    Workflow:
    1. SAM generates universal segments
    2. CLIP classifies each segment by material
    3. Results enable region-specific enhancement

    Example:
        >>> segmenter = MaterialSegmenter()
        >>> segments = segmenter.segment_materials("luxury_kitchen.jpg")
        >>>
        >>> # Find marble surfaces
        >>> marble_segments = [s for s in segments if s.material == "marble"]
        >>> print(f"Found {len(marble_segments)} marble regions")
        >>>
        >>> # Get enhancement recommendations
        >>> recommendations = segmenter.get_enhancement_recommendations(segments)
    """

    # Material-specific enhancement strategies
    MATERIAL_ENHANCEMENT = {
        "marble": {
            "clarity": 0.8,
            "saturation": 0.9,
            "sharpness": 0.7,
            "preserve_veining": True,
            "enhance_reflection": True
        },
        "granite": {
            "clarity": 0.75,
            "saturation": 1.0,
            "sharpness": 0.8,
            "preserve_texture": True
        },
        "wood": {
            "clarity": 0.7,
            "saturation": 1.05,
            "warmth": 1.1,
            "preserve_grain": True
        },
        "glass": {
            "clarity": 1.0,
            "preserve_transparency": True,
            "enhance_reflection": True,
            "edge_softness": 0.3
        },
        "metal": {
            "clarity": 0.85,
            "saturation": 0.95,
            "enhance_specular": True,
            "preserve_reflection": True
        },
        "stainless steel": {
            "clarity": 0.9,
            "saturation": 0.85,
            "enhance_specular": True,
            "preserve_reflection": True
        },
        "water": {
            "clarity": 0.8,
            "saturation": 1.15,
            "enhance_reflection": True,
            "color_enhance": "blue_shift"
        },
        "natural stone": {
            "clarity": 0.75,
            "saturation": 1.0,
            "preserve_texture": True,
            "warmth": 1.05
        }
    }

    def __init__(
        self,
        sam_segmenter: Optional[SAMSegmenter] = None,
        clip_classifier: Optional[CLIPClassifier] = None,
        **kwargs
    ):
        """Initialize material segmenter.

        Args:
            sam_segmenter: Existing SAM segmenter (creates new if None)
            clip_classifier: Existing CLIP classifier (creates new if None)
            **kwargs: Arguments passed to SAM/CLIP if creating new instances
        """
        # Initialize SAM
        if sam_segmenter is not None:
            self.sam = sam_segmenter
        else:
            logger.info("Creating SAM segmenter...")
            self.sam = SAMSegmenter(**kwargs)

        # Initialize CLIP
        if clip_classifier is not None:
            self.clip = clip_classifier
        else:
            logger.info("Creating CLIP classifier...")
            self.clip = CLIPClassifier(**kwargs)

        logger.info("MaterialSegmenter initialized")

    def segment_materials(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        materials: Optional[List[str]] = None,
        min_segment_area: int = 500,
        max_segments: int = 50,
        confidence_threshold: float = 0.3
    ) -> List[MaterialSegment]:
        """Segment image by materials.

        Args:
            image: Input image
            materials: Material categories (uses default if None)
            min_segment_area: Minimum segment size in pixels
            max_segments: Maximum number of segments to process
            confidence_threshold: Minimum classification confidence

        Returns:
            List of MaterialSegment objects
        """
        # Use default materials if none provided
        if materials is None:
            materials = self.clip.MATERIAL_CATEGORIES

        logger.info(f"Segmenting image into materials: {materials}")

        # Step 1: Generate segments with SAM
        logger.info("Generating segments with SAM...")
        sam_masks = self.sam.segment_automatic(
            image,
            min_area=min_segment_area,
            max_masks=max_segments
        )
        logger.info(f"Generated {len(sam_masks)} segments")

        # Step 2: Classify segments with CLIP
        logger.info("Classifying segments with CLIP...")
        mask_arrays = [m['segmentation'] for m in sam_masks]

        classifications = self.clip.classify_segments(
            image,
            mask_arrays,
            materials
        )

        # Step 3: Combine into MaterialSegments
        material_segments = []

        for sam_mask, classification in zip(sam_masks, classifications):
            # Only include if confidence exceeds threshold
            if classification['confidence'] < confidence_threshold:
                continue

            # Calculate centroid
            mask = sam_mask['segmentation']
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            else:
                centroid = (0, 0)

            segment = MaterialSegment(
                mask=mask,
                material=classification['top_category'],
                confidence=classification['confidence'],
                area=sam_mask['area'],
                bbox=tuple(sam_mask['bbox']),
                centroid=centroid,
                properties={
                    'predicted_iou': sam_mask['predicted_iou'],
                    'stability_score': sam_mask['stability_score'],
                    'all_material_probs': classification['all_categories']
                }
            )

            material_segments.append(segment)

        logger.info(
            f"Created {len(material_segments)} material segments "
            f"(filtered by confidence >= {confidence_threshold})"
        )

        return material_segments

    def get_material_masks(
        self,
        segments: List[MaterialSegment],
        material: str
    ) -> List[np.ndarray]:
        """Get all masks for a specific material.

        Args:
            segments: List of material segments
            material: Material name to filter by

        Returns:
            List of boolean masks for specified material
        """
        return [
            seg.mask
            for seg in segments
            if seg.material.lower() == material.lower()
        ]

    def create_material_map(
        self,
        image_shape: Tuple[int, int],
        segments: List[MaterialSegment]
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Create material segmentation map.

        Args:
            image_shape: Output shape (H, W)
            segments: Material segments

        Returns:
            Tuple of:
                - Material map (H, W) with integer material IDs
                - Label dictionary mapping IDs to material names
        """
        # Create empty map
        material_map = np.zeros(image_shape, dtype=np.int32)

        # Build material ID mapping
        unique_materials = sorted(set(seg.material for seg in segments))
        material_to_id = {mat: idx + 1 for idx, mat in enumerate(unique_materials)}

        # Fill map
        for segment in segments:
            material_id = material_to_id[segment.material]
            material_map[segment.mask] = material_id

        # Reverse mapping for output
        id_to_material = {v: k for k, v in material_to_id.items()}

        return material_map, id_to_material

    def get_enhancement_recommendations(
        self,
        segments: List[MaterialSegment]
    ) -> Dict[str, any]:
        """Get region-specific enhancement recommendations.

        Args:
            segments: Material segments

        Returns:
            Dictionary with enhancement strategies per material
        """
        recommendations = {
            "materials_detected": [],
            "region_enhancements": {},
            "overall_strategy": {}
        }

        # Analyze detected materials
        material_areas = {}
        for segment in segments:
            mat = segment.material
            if mat not in material_areas:
                material_areas[mat] = 0
            material_areas[mat] += segment.area

        # Sort by area coverage
        sorted_materials = sorted(
            material_areas.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recommendations["materials_detected"] = [
            {
                "material": mat,
                "total_area": area,
                "num_regions": len([s for s in segments if s.material == mat])
            }
            for mat, area in sorted_materials
        ]

        # Get enhancement strategies for each material
        for material, _ in sorted_materials:
            if material in self.MATERIAL_ENHANCEMENT:
                recommendations["region_enhancements"][material] = \
                    self.MATERIAL_ENHANCEMENT[material]

        # Determine overall strategy based on dominant materials
        if sorted_materials:
            dominant_material = sorted_materials[0][0]
            recommendations["overall_strategy"] = {
                "dominant_material": dominant_material,
                "suggested_processing": self.MATERIAL_ENHANCEMENT.get(
                    dominant_material,
                    {}
                ),
                "preserve_material_boundaries": True
            }

        return recommendations

    def visualize_materials(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        segments: List[MaterialSegment],
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create visualization with colored material regions.

        Args:
            image: Original image
            segments: Material segments
            alpha: Overlay transparency

        Returns:
            RGB visualization image
        """
        # Load image
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            pil_image = Image.open(image) if isinstance(image, (str, Path)) else image
            image_np = np.array(pil_image)

        overlay = image_np.copy().astype(np.float32)

        # Assign consistent color to each material
        unique_materials = sorted(set(seg.material for seg in segments))
        np.random.seed(42)  # Consistent colors
        material_colors = {
            mat: np.random.randint(0, 255, size=3)
            for mat in unique_materials
        }

        # Apply colored masks
        for segment in segments:
            color = material_colors[segment.material]
            mask = segment.mask

            # Blend
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        return overlay.astype(np.uint8)

    def create_material_labels(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        segments: List[MaterialSegment],
        font_scale: float = 0.5,
        thickness: int = 1
    ) -> np.ndarray:
        """Create visualization with material labels.

        Args:
            image: Original image
            segments: Material segments
            font_scale: Text size
            thickness: Text thickness

        Returns:
            Image with material labels
        """
        # Load image
        if isinstance(image, np.ndarray):
            image_np = image.copy()
        else:
            pil_image = Image.open(image) if isinstance(image, (str, Path)) else image
            image_np = np.array(pil_image).copy()

        # Add labels at centroids
        for segment in segments:
            x, y = segment.centroid
            label = f"{segment.material} ({segment.confidence:.2f})"

            # Draw text with background
            cv2.putText(
                image_np,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness + 1,
                cv2.LINE_AA
            )
            cv2.putText(
                image_np,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return image_np

    def get_statistics(
        self,
        segments: List[MaterialSegment]
    ) -> Dict:
        """Calculate material segmentation statistics.

        Args:
            segments: Material segments

        Returns:
            Statistics dictionary
        """
        if not segments:
            return {
                "num_segments": 0,
                "num_materials": 0,
                "materials": []
            }

        # Count by material
        material_counts = {}
        material_areas = {}
        material_confidences = {}

        for segment in segments:
            mat = segment.material

            if mat not in material_counts:
                material_counts[mat] = 0
                material_areas[mat] = 0
                material_confidences[mat] = []

            material_counts[mat] += 1
            material_areas[mat] += segment.area
            material_confidences[mat].append(segment.confidence)

        # Build statistics
        materials = []
        for mat in material_counts:
            materials.append({
                "material": mat,
                "num_segments": material_counts[mat],
                "total_area": material_areas[mat],
                "avg_confidence": np.mean(material_confidences[mat]),
                "min_confidence": np.min(material_confidences[mat]),
                "max_confidence": np.max(material_confidences[mat])
            })

        # Sort by total area
        materials.sort(key=lambda x: x['total_area'], reverse=True)

        return {
            "num_segments": len(segments),
            "num_materials": len(material_counts),
            "materials": materials,
            "total_area": sum(seg.area for seg in segments),
            "avg_confidence": np.mean([seg.confidence for seg in segments]),
            "avg_segment_area": np.mean([seg.area for seg in segments])
        }

    def __repr__(self) -> str:
        return (
            f"MaterialSegmenter(\n"
            f"  SAM: {self.sam}\n"
            f"  CLIP: {self.clip}\n"
            f")"
        )
