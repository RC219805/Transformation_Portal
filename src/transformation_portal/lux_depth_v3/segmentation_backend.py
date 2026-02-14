"""Material segmentation backend for Materials V3.

This module provides material segmentation functionality for the Materials V3 pipeline.

Architecture:
- Protocol-based design (SegmentationBackend Protocol)
- Stub backend (default, production-safe, returns empty masks)
- EfficientSAM backend (opt-in, requires ML dependencies)
- Fail-safe fallback: missing weights → stub backend with warning
- Lazy loading: models loaded only on first inference

Backends:
1. StubBackend (default):
   - Returns empty masks
   - Zero dependencies
   - Production-safe default
   - No GPU required

2. EfficientSAMBackend (opt-in via config):
   - Lightweight Segment Anything Model variant
   - License: MIT (commercial use allowed)
   - Model size: ~50MB
   - Performance: Works on CPU, optimized for MPS/CUDA
   - Material detection: Heuristic-based labeling (v1)

Configuration:
- enable_material_segmentation: Enable/disable segmentation
- material_segmentation_backend: "stub" (default) or "efficientsam"
- strict_backend: If True, raise on missing weights instead of falling back

For usage examples, see docs/materials_v3_quick_reference.md
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import EnhanceConfig
from .protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo

logger = logging.getLogger(__name__)

# Lazy imports for ML dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import torchvision

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    torchvision = None  # type: ignore

# V2 model dependencies (optional)
try:
    from efficientsam.cached_sam_model import CachedSamModel
    from efficientsam.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator
    from efficientsam.sam_model_zoo import create_efficientvit_sam_model

    EFFICIENTVIT_AVAILABLE = True
except ImportError:
    EFFICIENTVIT_AVAILABLE = False
    create_efficientvit_sam_model = None  # type: ignore
    CachedSamModel = None  # type: ignore
    EfficientViTSamAutomaticMaskGenerator = None  # type: ignore

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    open_clip = None  # type: ignore


# =============================================================================
# Stub Backend (Default, Production-Safe)
# =============================================================================


class StubBackend:
    """Stub segmentation backend that returns empty masks.

    This is the default backend to avoid heavy ML dependencies.
    It's production-safe and will never fail, but provides no segmentation.
    """

    @property
    def info(self) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name="Stub Segmentation Backend",
            model_id="stub",
            requires_gpu=False,
            requires_weights=False,
            approximate_memory_mb=0,
            description="Stub backend that returns empty masks (production-safe default)",
        )

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """No-op load for stub backend."""
        logger.debug("StubBackend.load() called - no model to load")

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Return empty masks dict."""
        logger.debug("StubBackend.segment() returning empty masks")
        return {}


# =============================================================================
# EfficientSAM Backend (Opt-In, ML-Powered)
# =============================================================================


class EfficientSAMBackend:
    """EfficientSAM-based material segmentation backend.

    Uses a lightweight Segment Anything Model variant for automatic
    material detection in architectural images.

    Architecture:
    - Model: EfficientSAM (CVPR 2024, MIT license)
    - Material labeling: Heuristic-based (v1 implementation)
    - Device support: CPU, MPS (Apple Silicon), CUDA
    - Lazy loading: Model loaded on first inference
    - Caching: Model instance cached after first load

    Performance (1024×1024, Apple M4):
    - CPU: ~1.5s
    - MPS: ~400ms
    - CUDA: ~300ms (estimated)

    Memory: ~50MB model + ~200MB inference overhead
    """

    def __init__(self):
        self._model = None
        self._device = None
        self._model_loaded = False
        self._use_real_model = False  # Track whether real model or heuristics
        self._mask_generator = None  # Automatic mask generator for SAM
        # Cache CLIP model to avoid repeated loading
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    @property
    def info(self) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name="EfficientSAM",
            model_id="yunyangx/efficientvit-sam",
            requires_gpu=False,  # Works on CPU, but much slower
            requires_weights=True,
            approximate_memory_mb=50,
            description="Lightweight Segment Anything Model for material detection (MIT license)",
        )

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """Load EfficientSAM model with device selection.

        Args:
            device: Target device ("auto", "cpu", "mps", "cuda")
            weights_path: Optional local path to model weights (not yet supported)

        Raises:
            RuntimeError: If torch not available or model loading fails
        """
        if self._model_loaded:
            logger.debug("EfficientSAM model already loaded, skipping")
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch torchvision\n"
                "Or disable EfficientSAM backend in config."
            )

        if not TORCHVISION_AVAILABLE:
            raise RuntimeError(
                "torchvision not available. Install with: pip install torchvision\n"
                "Or disable EfficientSAM backend in config."
            )

        # Resolve device
        self._device = self._resolve_device(device)
        logger.info(f"Loading EfficientSAM backend on device: {self._device}")

        try:
            # V2: Load actual EfficientSAM model if dependencies available
            if EFFICIENTVIT_AVAILABLE:
                logger.info("EfficientVIT available - loading real EfficientSAM model")
                self._model = self._load_efficientvit_model(weights_path)
                self._use_real_model = True
            else:
                # V1: Fall back to heuristic-only mode
                logger.warning(
                    "EfficientVIT not available - falling back to heuristic-based segmentation (v1). "
                    "For real model inference, install: pip install efficientvit"
                )
                self._model = self._create_placeholder_model()
                self._use_real_model = False

            self._model_loaded = True
            model_type = "real EfficientSAM" if self._use_real_model else "heuristic fallback"
            logger.info(f"EfficientSAM backend loaded successfully ({model_type}) on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load EfficientSAM backend: {e}")
            raise RuntimeError(f"EfficientSAM backend loading failed: {e}") from e

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run material segmentation on an image.

        Args:
            image: Input RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict mapping material names to binary masks (H, W), float32 [0.0-1.0]
            V2: Real EfficientSAM + CLIP classification (if dependencies available)
            V1: Heuristic-based segmentation (fallback)

        Raises:
            RuntimeError: If model not loaded
            ValueError: If image format invalid
        """
        if not self._model_loaded:
            raise RuntimeError("EfficientSAM model not loaded. Call .load() first or enable lazy loading in config.")

        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

        # V2: Real EfficientSAM inference if available
        if self._use_real_model and EFFICIENTVIT_AVAILABLE:
            try:
                masks = self._real_model_inference(image)
                logger.debug(f"EfficientSAM (v2) segmented {len(masks)} materials: {list(masks.keys())}")
                return masks
            except Exception as e:
                logger.warning(f"Real model inference failed, falling back to heuristics: {e}")
                # Fall through to heuristics

        # V1: Heuristic-based segmentation (fallback or primary for v1)
        masks = self._heuristic_segmentation(image)
        logger.debug(f"EfficientSAM (heuristic) segmented {len(masks)} materials: {list(masks.keys())}")
        logger.debug(f"Heuristic segmentation detected {len(masks)} materials")
        return masks

    def _resolve_device(self, device: str) -> str:
        """Resolve device string for PyTorch.

        Follows same pattern as depth backends in inference.py.
        """
        device_lower = device.lower()

        # Explicit device override
        if device_lower == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device_lower == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device_lower == "cpu":
            return "cpu"

        # Auto-detect (prefer MPS on Apple Silicon, then CUDA, then CPU)
        if device_lower == "auto" or device_lower not in ["cuda", "mps", "cpu"]:
            if torch.backends.mps.is_available():
                logger.info("Auto-detected MPS (Apple Silicon) for segmentation")
                return "mps"
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA for segmentation")
                return "cuda"
            logger.info("Using CPU for segmentation (no GPU detected)")
            return "cpu"

        return "cpu"

    def _load_efficientvit_model(self, weights_path: Optional[Path] = None):
        """Load real EfficientSAM model and automatic mask generator (v2).

        Args:
            weights_path: Optional local path to weights (not used with CachedSamModel)

        Returns:
            Loaded EfficientSAM model on target device

        Raises:
            RuntimeError: If model loading fails

        Note:
            MPS has float64 compatibility issues with the automatic mask generator.
            We fall back to CPU when MPS is requested to ensure stability.
        """
        if not EFFICIENTVIT_AVAILABLE:
            raise RuntimeError("efficientsam not available. Install with: pip install efficientsam")

        logger.info("Loading EfficientSAM model (l0 variant, ~50MB)...")

        try:
            # Determine cache directory
            cache_dir = Path.home() / ".cache" / "transformation_portal" / "segmentation"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # MPS has float64 issues with automatic mask generator - use CPU instead
            device_for_sam = self._device
            if device_for_sam == "mps":
                logger.warning(
                    "MPS has float64 compatibility issues with EfficientSAM automatic mask generator. "
                    "Falling back to CPU for stability. Performance impact: ~2-3x slower."
                )
                device_for_sam = "cpu"

            # Use CachedSamModel helper which handles downloads and caching
            logger.info(f"Loading model on device: {device_for_sam}")
            cached_model = CachedSamModel(model_name="efficientvit-sam-l0", device=device_for_sam, checkpoint_dir=cache_dir)

            # Get the predictor and extract the model
            predictor = cached_model()
            model = predictor.model

            logger.info(f"EfficientSAM model loaded successfully on {device_for_sam}")

            # Create automatic mask generator for inference
            logger.info("Initializing automatic mask generator...")
            self._mask_generator = EfficientViTSamAutomaticMaskGenerator(
                model=model,
                points_per_side=32,  # 32x32 grid = 1024 points (balanced quality/speed)
                points_per_batch=64,  # Process 64 points at a time
                pred_iou_thresh=0.7,  # Filter masks with IoU < 0.7
                stability_score_thresh=0.85,  # Filter unstable masks
                box_nms_thresh=0.7,  # IoU threshold for duplicate removal
                min_mask_region_area=500,  # Filter tiny masks (< 500px)
            )
            logger.info("Automatic mask generator initialized")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load EfficientSAM model: {e}") from e

    def _real_model_inference(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run real EfficientSAM + CLIP inference (v2).

        Three-step pipeline:
        1. Generate segment proposals with EfficientSAM
        2. Classify segments by material type (CLIP or heuristics)
        3. Aggregate into confidence-weighted masks

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict[material_name, (mask, confidence)] where:
            - mask: (H, W) float32 [0.0-1.0]
            - confidence: CLIP similarity or heuristic score [0.0-1.0]
        """
        # Step 1: Run EfficientSAM to generate segment proposals
        segments = self._run_sam_inference(image)

        # Step 2: Classify segments by material type
        # V2.0: Use CLIP if available, otherwise fall back to heuristics
        if OPEN_CLIP_AVAILABLE:
            classified_masks = self._classify_segments_with_clip(image, segments)
        else:
            logger.debug("CLIP not available, using heuristic classification")
            classified_masks = self._classify_segments_heuristic(image, segments)

        # Step 3: Aggregate masks (for now, just return the classified masks)
        # Future: Add confidence scoring, top-K selection, mask merging
        return classified_masks

    def _run_sam_inference(self, image: np.ndarray) -> list:
        """Generate segment proposals with EfficientSAM automatic mask generator.

        Uses EfficientViTSamAutomaticMaskGenerator with grid-based point prompting
        to generate high-quality segment proposals for CLIP classification.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]

        Returns:
            List of segment dictionaries with keys:
                - 'segmentation': Binary mask (H, W) bool np.ndarray
                - 'bbox': Bounding box in XYWH format [x, y, w, h]
                - 'area': Mask area in pixels
                - 'predicted_iou': Model's quality prediction [0.0-1.0]
                - 'stability_score': Mask stability metric [0.0-1.0]

        Note:
            V2.0: Fully implemented using EfficientViTSamAutomaticMaskGenerator!
            The automatic mask generator runs inference over a 32x32 grid of points
            (1024 total), applies IoU-based deduplication, and filters by quality.

            Performance (on CPU, 512x512 image):
            - Grid size 32x32: ~1-2 seconds
            - Generates 5-30 high-quality masks depending on image content

            MPS compatibility: The automatic mask generator has float64 issues on MPS.
            We automatically fall back to CPU during model loading for stability.
        """
        if self._mask_generator is None:
            logger.warning(
                "SAM automatic mask generator not initialized. "
                "This should not happen if model loaded correctly. "
                "Falling back to empty segments."
            )
            return []

        try:
            import time

            start = time.time()

            logger.debug(f"Running SAM automatic mask generation on {image.shape[:2]} image...")

            # Generate masks using the automatic mask generator
            # This internally:
            # 1. Creates a 32x32 grid of point prompts (1024 points)
            # 2. Runs SAM inference on each point (batched by 64)
            # 3. Applies IoU-based NMS for deduplication (threshold 0.7)
            # 4. Filters by predicted_iou (>0.7) and stability_score (>0.85)
            # 5. Removes small masks (<500px area)
            masks = self._mask_generator.generate(image)

            elapsed = time.time() - start
            logger.info(
                f"SAM generated {len(masks)} high-quality segments in {elapsed:.2f}s " f"({len(masks)/elapsed:.1f} masks/sec)"
            )

            # Log quality statistics
            if len(masks) > 0:
                avg_iou = sum(m.get("predicted_iou", 0) for m in masks) / len(masks)
                avg_stability = sum(m.get("stability_score", 0) for m in masks) / len(masks)
                logger.debug(f"SAM mask quality: avg_iou={avg_iou:.3f}, " f"avg_stability={avg_stability:.3f}")

            return masks

        except Exception as e:
            logger.error(f"SAM inference failed: {e}", exc_info=True)
            logger.warning("Falling back to empty segments (CLIP will classify heuristic masks)")
            return []

    def _classify_segments_with_clip(self, image: np.ndarray, segments: list) -> Dict[str, Tuple[np.ndarray, float]]:
        """Classify segments using CLIP zero-shot classification.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            segments: List of segment dictionaries from SAM (may be empty)

        Returns:
            Dict mapping material names to (mask, confidence) tuples:
            - mask: Aggregated binary mask (H, W) float32 [0.0-1.0]
            - confidence: Average CLIP similarity score [0.0-1.0]

        Note:
            If segments list is empty (which happens when SAM doesn't run),
            we first generate heuristic segments and then classify them with CLIP.
            This provides better material detection than pure heuristics alone.
        """
        # If no SAM segments, generate heuristic segments first
        if not segments:
            logger.debug("No SAM segments provided, generating heuristic segments for CLIP classification")
            heuristic_results = self._heuristic_segmentation(image)

            # Convert heuristic masks to segment format for CLIP
            # Note: heuristic_results is Dict[str, Tuple[np.ndarray, float]]
            segments = []
            for material_name, (mask, _) in heuristic_results.items():  # Unpack tuple
                # Find all connected components in this material mask
                from scipy import ndimage

                labeled, num_features = ndimage.label(mask > 0.5)

                for region_id in range(1, num_features + 1):
                    region_mask = labeled == region_id
                    area = region_mask.sum()

                    if area < 500:  # Skip small regions
                        continue

                    # Compute bounding box
                    rows, cols = np.where(region_mask)
                    if len(rows) == 0:
                        continue

                    x1, x2 = cols.min(), cols.max()
                    y1, y2 = rows.min(), rows.max()
                    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

                    segments.append(
                        {
                            "segmentation": region_mask,
                            "bbox": bbox,
                            "area": int(area),
                            "heuristic_label": material_name,  # Keep for comparison
                        }
                    )

        if not segments:
            logger.debug("No segments to classify, using pure heuristics")
            return self._heuristic_segmentation(image)

        # Allow disabling CLIP for faster tests
        import os

        if os.getenv("SKIP_CLIP_INFERENCE", "").lower() in ("1", "true", "yes"):
            logger.debug("SKIP_CLIP_INFERENCE set, using heuristics instead of CLIP")
            return self._heuristic_segmentation(image)

        try:
            import open_clip
            import torch
            from PIL import Image

            # Load CLIP model (cached at instance level to avoid repeated loading)
            if self._clip_model is None:
                logger.debug("Loading CLIP model for segment classification...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32",
                    pretrained="openai",
                    device=self._device,
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")

                # Cache for future calls
                self._clip_model = model
                self._clip_preprocess = preprocess
                self._clip_tokenizer = tokenizer
            else:
                logger.debug("Using cached CLIP model")
                model = self._clip_model
                preprocess = self._clip_preprocess
                tokenizer = self._clip_tokenizer

            # Define material text prompts
            material_prompts = {
                "glass": "a photo of glass windows and reflective surfaces",
                "water": "a photo of water, pools, and blue reflective surfaces",
                "foliage": "a photo of plants, trees, and green foliage",
                "stone": "a photo of stone, concrete, and gray surfaces",
            }

            # Tokenize text prompts
            text_tokens = tokenizer(list(material_prompts.values())).to(self._device)

            with torch.no_grad():
                # Encode text prompts
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Initialize material masks with tracking for confidence scores
                h, w = image.shape[:2]
                material_data = {
                    name: {"mask": np.zeros((h, w), dtype=np.float32), "scores": [], "areas": []} for name in material_prompts
                }

                # Convert image to PIL for preprocessing
                pil_image = Image.fromarray(image)

                # Classify each segment
                for seg_idx, segment in enumerate(segments):
                    mask = segment["segmentation"]
                    bbox = segment["bbox"]  # [x, y, w, h]

                    # Extract segment region with some padding
                    x, y, w_box, h_box = [int(v) for v in bbox]
                    x1 = max(0, x - 10)
                    y1 = max(0, y - 10)
                    x2 = min(w, x + w_box + 10)
                    y2 = min(h, y + h_box + 10)

                    # Crop and preprocess region
                    region = pil_image.crop((x1, y1, x2, y2))
                    region_tensor = preprocess(region).unsqueeze(0).to(self._device)

                    # Encode image region
                    image_features = model.encode_image(region_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Compute similarity scores
                    similarities = (image_features @ text_features.T).squeeze(0)

                    # Assign to best matching material
                    best_idx = similarities.argmax().item()
                    best_material = list(material_prompts.keys())[best_idx]
                    best_score = similarities[best_idx].item()

                    # Only add if confidence is reasonable (> 0.2)
                    if best_score > 0.2:
                        # Track confidence score and segment area for weighted averaging
                        material_data[best_material]["scores"].append(best_score)
                        material_data[best_material]["areas"].append(segment["area"])

                        # Add segment mask to material (using max to handle overlaps)
                        material_data[best_material]["mask"] = np.maximum(
                            material_data[best_material]["mask"], mask.astype(np.float32)
                        )

                        heuristic_label = segment.get("heuristic_label", "unknown")
                        match_str = "✓" if heuristic_label == best_material else f"({heuristic_label}→{best_material})"
                        logger.debug(
                            f"Segment {seg_idx}: {best_material} {match_str} "
                            f"score={best_score:.3f}, area={segment['area']}px"
                        )

                # Compute aggregate confidence per material (area-weighted average)
                material_masks = {}
                for name, data in material_data.items():
                    mask = data["mask"]
                    scores = data["scores"]
                    areas = data["areas"]

                    # Only include materials with sufficient coverage
                    if mask.sum() > 500:
                        if scores:
                            # Area-weighted average of CLIP scores for this material
                            total_area = sum(areas)
                            weighted_conf = sum(s * a for s, a in zip(scores, areas)) / total_area
                            material_masks[name] = (mask, float(weighted_conf))
                        else:
                            # No scores (shouldn't happen, but handle gracefully)
                            material_masks[name] = (mask, 0.5)

                logger.info(
                    f"CLIP classified {len(segments)} segments into {len(material_masks)} materials: "
                    f"{', '.join(f'{m} ({c:.0%})' for m, (_, c) in material_masks.items())}"
                )
                return material_masks

        except Exception as e:
            logger.warning(f"CLIP classification failed: {e}, falling back to heuristics")
            import traceback

            logger.debug(f"CLIP error traceback: {traceback.format_exc()}")
            return self._heuristic_segmentation(image)

    def _classify_segments_heuristic(self, image: np.ndarray, segments: list) -> Dict[str, Tuple[np.ndarray, float]]:
        """Classify segments using color/texture heuristics.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            segments: List of segment masks

        Returns:
            Dict mapping material names to (mask, confidence) tuples with confidence=0.5
        """
        # Fall back to existing heuristic method
        return self._heuristic_segmentation(image)

    def _create_placeholder_model(self):
        """Create a placeholder model for v1 heuristic-only mode.

        Used when EfficientVIT dependencies not available.
        """

        # Simple placeholder that demonstrates the pattern
        class PlaceholderModel:
            def __init__(self, device):
                self.device = device

            def eval(self):
                return self

            def to(self, device):
                self.device = device
                return self

        return PlaceholderModel(self._device)

    def _heuristic_segmentation(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Heuristic-based material segmentation (v1 placeholder).

        This is a simplified implementation to demonstrate integration.
        Future versions will use real EfficientSAM + CLIP classification.

        Materials detected:
        - sky: Top-of-frame regions with smooth gradients (Phase B)
        - glass: High brightness regions with blue tint
        - water: Blue-dominant regions
        - foliage: Green-dominant regions
        - stone: Gray/neutral regions with texture

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            Dict of (mask, confidence) tuples:
            - mask: Material mask (H, W), float32 [0.0-1.0]
            - confidence: Fixed at 0.5 to indicate heuristic classification
                         (sky uses bootstrap confidence score)
        """
        masks = {}

        # Convert to float for analysis
        img_float = image.astype(np.float32) / 255.0

        # Sky detection (Phase B): Use bootstrap heuristic
        # Note: This is integrated here for v1, but in future versions
        # sky detection will use SAM2 refinement with these bootstrap seeds
        try:
            from .bootstrap.sky_seed import detect_sky_seed

            # Get config values from self._config if available, else use defaults
            cfg = getattr(self, "_config", None)
            top_region_fraction = getattr(cfg, "sky_top_region_fraction", 0.5) if cfg else 0.5
            gradient_threshold = getattr(cfg, "sky_gradient_threshold", 0.05) if cfg else 0.05
            brightness_threshold = getattr(cfg, "sky_brightness_threshold", 0.4) if cfg else 0.4

            # Create minimal config object for bootstrap
            sky_config = SimpleNamespace(
                sky_top_region_fraction=top_region_fraction,
                sky_gradient_threshold=gradient_threshold,
                sky_brightness_threshold=brightness_threshold,
            )

            sky_result = detect_sky_seed(image, sky_config)
            if sky_result["confidence"] > 0.1:  # Only include if confident
                masks["sky"] = (sky_result["coarse_mask"], sky_result["confidence"])
        except Exception as e:
            logger.debug(f"Sky bootstrap failed (non-critical): {e}")

        # Glass detection: High brightness + blue tint
        brightness = img_float.mean(axis=2)
        blue_tint = (img_float[..., 2] > img_float[..., 0]) & (img_float[..., 2] > img_float[..., 1])
        glass_mask = (brightness > 0.6) & blue_tint
        if glass_mask.sum() > 500:  # Min coverage threshold
            masks["glass"] = (glass_mask.astype(np.float32), 0.5)

        # Water detection: Blue-dominant regions
        blue_dominant = (img_float[..., 2] > img_float[..., 0] + 0.1) & (img_float[..., 2] > img_float[..., 1] + 0.1)
        water_mask = blue_dominant & (brightness > 0.2) & (brightness < 0.8)
        if water_mask.sum() > 500:
            masks["water"] = (water_mask.astype(np.float32), 0.5)

        # Foliage detection: Green-dominant regions
        green_dominant = (img_float[..., 1] > img_float[..., 0] + 0.1) & (img_float[..., 1] > img_float[..., 2] + 0.05)
        foliage_mask = green_dominant & (brightness > 0.2)
        if foliage_mask.sum() > 500:
            masks["foliage"] = (foliage_mask.astype(np.float32), 0.5)

        # Stone detection: Gray/neutral regions (low color saturation)
        rgb_std = img_float.std(axis=2)
        stone_mask = (rgb_std < 0.15) & (brightness > 0.3) & (brightness < 0.7)
        if stone_mask.sum() > 500:
            masks["stone"] = (stone_mask.astype(np.float32), 0.5)

        return masks

    def _bootstrap_sky(self, image: np.ndarray, config: Any) -> Dict[str, Any]:
        """Bootstrap sky detection using heuristics (Phase B).

        Uses spatial and intensity priors to detect sky regions, which are
        amorphous "stuff" materials difficult for standard object detection.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            config: Configuration object with sky_* attributes

        Returns:
            Dict with coarse_mask, confidence, bbox, and prompt points

        Note:
            This method delegates to the sky_seed module, which provides
            heuristic-based detection optimized for sky regions.
        """
        from .bootstrap.sky_seed import detect_sky_seed

        return detect_sky_seed(image, config)


# =============================================================================
# Backend Factory and Public API
# =============================================================================


@lru_cache(maxsize=2)  # Cache both stub and efficientsam instances
def _get_backend_instance(
    backend_name: str,
    device: str = "auto",
    strict: bool = False,
) -> SegmentationBackend:
    """Get or create a cached backend instance.

    Args:
        backend_name: "stub" or "efficientsam"
        device: Device for backend (only used for efficientsam)
        strict: If True, raise on errors instead of falling back

    Returns:
        SegmentationBackend instance

    Raises:
        ValueError: If backend_name is unknown
        RuntimeError: If strict=True and backend fails to load
    """
    if backend_name == "stub":
        backend = StubBackend()
        backend.load()  # No-op for stub
        return backend

    elif backend_name == "efficientsam":
        backend = EfficientSAMBackend()
        # Lazy load will happen on first segment() call if needed
        # But we can pre-load here for better error handling
        try:
            backend.load(device=device)
        except RuntimeError as e:
            if strict:
                # In strict mode, propagate the error
                raise RuntimeError(f"Failed to load {backend_name} backend: {e}") from e

            # Non-strict mode: log warning and fall back to stub
            logger.warning(
                f"Failed to load EfficientSAM backend: {e}\n"
                f"This is expected if torch is not installed or weights are missing.\n"
                f"Falling back to stub backend."
            )
            # Return stub instead
            return _get_backend_instance("stub", device="cpu", strict=False)
        return backend

    else:
        raise ValueError(f"Unknown segmentation backend: {backend_name}\n" f"Valid options: 'stub', 'efficientsam'")


def segment_materials(
    image: np.ndarray,
    config: EnhanceConfig,
) -> Dict[str, np.ndarray]:
    """Segment image into material masks.

    This is the main entry point for material segmentation in Materials V3.

    Backends:
    - stub (default): Returns empty masks, production-safe
    - efficientsam (opt-in): ML-powered segmentation

    Args:
        image: Input image as numpy array (H, W, 3) in RGB, uint8 [0-255]
        config: EnhanceConfig instance with segmentation settings
            - enable_material_segmentation: Enable/disable segmentation
            - material_segmentation_backend: Backend to use ("stub" or "efficientsam")
            - strict_backend: If True, raise on errors instead of falling back

    Returns:
        Dict mapping material names to binary masks (H, W) with values 0.0-1.0
        Example: {"glass": mask1, "water": mask2, ...}

        For stub backend, returns empty dict.
        For real backends, returns detected materials only.

    Raises:
        RuntimeError: If strict_backend=True and backend fails to load
        ValueError: If image format is invalid
    """
    # Check if segmentation is enabled
    enable_segmentation = getattr(config, "enable_material_segmentation", False)

    if not enable_segmentation:
        logger.debug("Material segmentation disabled in config")
        return {}

    # Get backend selection
    backend_name = getattr(config, "material_segmentation_backend", "stub")
    strict_backend = getattr(config, "strict_backend", False)

    # Get device for backend (if applicable)
    device = getattr(config, "depth_device", "cpu")  # Reuse depth_device setting

    try:
        # Get or create backend instance (cached)
        backend = _get_backend_instance(backend_name, device=device, strict=strict_backend)

        # Run segmentation
        results = backend.segment(image)

        # Extract masks from (mask, confidence) tuples for backward compatibility
        # The public API returns Dict[str, np.ndarray] while backends return Dict[str, Tuple[np.ndarray, float]]
        masks = {material: mask for material, (mask, confidence) in results.items()}

        logger.debug(
            f"Segmentation completed using {backend.info.name}: " f"{len(masks)} materials detected: {list(masks.keys())}"
        )

        return masks

    except Exception as e:
        if strict_backend:
            logger.error(f"Segmentation failed with strict_backend=True: {e}")
            raise RuntimeError(f"Material segmentation failed: {e}") from e

        # Fail-safe: Return empty masks on error
        logger.warning(
            f"Material segmentation failed, returning empty masks: {e}\n"
            f"This is safe - Materials V3 will continue without segmentation.\n"
            f"To debug, set strict_backend=True in config."
        )
        return {}
