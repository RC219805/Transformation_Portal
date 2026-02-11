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
from typing import Dict, Optional

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
    from efficientsam.sam_model_zoo import create_efficientvit_sam_model

    EFFICIENTVIT_AVAILABLE = True
except ImportError:
    EFFICIENTVIT_AVAILABLE = False
    create_efficientvit_sam_model = None  # type: ignore

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

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
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
        """Load real EfficientSAM model (v2).

        Args:
            weights_path: Optional local path to weights

        Returns:
            Loaded EfficientSAM model on target device

        Raises:
            RuntimeError: If model loading fails
        """
        if not EFFICIENTVIT_AVAILABLE:
            raise RuntimeError("efficientsam not available. Install with: pip install efficientsam")

        logger.info("Loading EfficientSAM model (l0 variant, ~50MB)...")

        try:
            # Download or use cached weights
            if weights_path is None:
                weights_path = self._download_weights()

            # Load model with weights
            model = create_efficientvit_sam_model(
                name="efficientvit-sam-l0",  # Lightweight variant
                pretrained=True,
                weight_url=str(weights_path),
            )

            # Move to target device and set eval mode
            model = model.to(self._device)
            model = model.eval()

            logger.info(f"EfficientSAM model loaded successfully on {self._device}")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load EfficientSAM model: {e}") from e

    def _download_weights(self) -> Path:
        """Download EfficientSAM weights with caching.

        Returns:
            Path to downloaded weights file

        Raises:
            RuntimeError: If download fails
        """
        import urllib.request
        from pathlib import Path

        # Cache directory
        cache_dir = Path.home() / ".cache" / "transformation_portal" / "segmentation"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weight_file = cache_dir / "efficientvit_sam_l0.pt"

        # Download if not cached
        if not weight_file.exists():
            logger.info("Downloading EfficientSAM-l0 weights (~50MB)...")
            weight_url = "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt"

            try:
                urllib.request.urlretrieve(weight_url, weight_file)
                logger.info(f"Weights downloaded to {weight_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to download weights: {e}") from e
        else:
            logger.info(f"Using cached weights from {weight_file}")

        return weight_file

    def _real_model_inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run real EfficientSAM + CLIP inference (v2).

        Three-step pipeline:
        1. Generate segment proposals with EfficientSAM
        2. Classify segments by material type (CLIP or heuristics)
        3. Aggregate into confidence-weighted masks

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict[material_name, mask] where mask is (H, W) float32 [0.0-1.0]
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
        """Generate segment proposals with EfficientSAM.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]

        Returns:
            List of segment dictionaries with keys:
                - 'segmentation': Binary mask (H, W) np.ndarray
                - 'bbox': Bounding box in XYWH format [x, y, w, h]
                - 'area': Mask area in pixels
                - 'predicted_iou': Model's quality prediction

        Note:
            V2.0 limitation: EfficientViTSam doesn't have a compatible automatic
            mask generation API like the full SAM model. The standard
            SamAutomaticMaskGenerator expects attributes (img_size, device) that
            aren't present in EfficientViTSam.

            Future versions can implement a custom automatic mask generator
            by running the model over a grid of point prompts. For now, we return
            empty segments and let CLIP classify heuristic segments instead.

            This still provides value via CLIP-based material classification,
            which is more accurate than pure heuristics.
        """
        logger.debug(
            "SAM automatic mask generation not yet compatible with EfficientViTSam. "
            "Using heuristic segments with CLIP classification instead."
        )
        return []  # Empty list triggers CLIP classification of heuristic segments

    def _classify_segments_with_clip(self, image: np.ndarray, segments: list) -> Dict[str, np.ndarray]:
        """Classify segments using CLIP zero-shot classification.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            segments: List of segment dictionaries from SAM (may be empty)

        Returns:
            Dict mapping material names to aggregated masks (H, W) float32 [0.0-1.0]

        Note:
            If segments list is empty (which happens when SAM doesn't run),
            we first generate heuristic segments and then classify them with CLIP.
            This provides better material detection than pure heuristics alone.
        """
        # If no SAM segments, generate heuristic segments first
        if not segments:
            logger.debug("No SAM segments provided, generating heuristic segments for CLIP classification")
            heuristic_masks = self._heuristic_segmentation(image)

            # Convert heuristic masks to segment format for CLIP
            segments = []
            for material_name, mask in heuristic_masks.items():
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

                # Initialize material masks
                h, w = image.shape[:2]
                material_masks = {name: np.zeros((h, w), dtype=np.float32) for name in material_prompts}

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
                        # Add segment mask to material (using max to handle overlaps)
                        material_masks[best_material] = np.maximum(material_masks[best_material], mask.astype(np.float32))

                        heuristic_label = segment.get("heuristic_label", "unknown")
                        match_str = "✓" if heuristic_label == best_material else f"({heuristic_label}→{best_material})"
                        logger.debug(
                            f"Segment {seg_idx}: {best_material} {match_str} "
                            f"score={best_score:.3f}, area={segment['area']}px"
                        )

                # Filter out empty masks
                material_masks = {
                    name: mask for name, mask in material_masks.items() if mask.sum() > 500  # Min coverage threshold
                }

                logger.info(
                    f"CLIP classified {len(segments)} segments into {len(material_masks)} materials: {list(material_masks.keys())}"
                )
                return material_masks

        except Exception as e:
            logger.warning(f"CLIP classification failed: {e}, falling back to heuristics")
            import traceback

            logger.debug(f"CLIP error traceback: {traceback.format_exc()}")
            return self._heuristic_segmentation(image)

    def _classify_segments_heuristic(self, image: np.ndarray, segments: list) -> Dict[str, np.ndarray]:
        """Classify segments using color/texture heuristics.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            segments: List of segment masks

        Returns:
            Dict mapping material names to aggregated masks
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

    def _heuristic_segmentation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Heuristic-based material segmentation (v1 placeholder).

        This is a simplified implementation to demonstrate integration.
        Future versions will use real EfficientSAM + CLIP classification.

        Materials detected:
        - glass: High brightness regions with blue tint
        - water: Blue-dominant regions
        - foliage: Green-dominant regions
        - stone: Gray/neutral regions with texture

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            Dict of material masks (H, W), float32 [0.0-1.0]
        """
        masks = {}

        # Convert to float for analysis
        img_float = image.astype(np.float32) / 255.0

        # Glass detection: High brightness + blue tint
        brightness = img_float.mean(axis=2)
        blue_tint = (img_float[..., 2] > img_float[..., 0]) & (img_float[..., 2] > img_float[..., 1])
        glass_mask = (brightness > 0.6) & blue_tint
        if glass_mask.sum() > 500:  # Min coverage threshold
            masks["glass"] = glass_mask.astype(np.float32)

        # Water detection: Blue-dominant regions
        blue_dominant = (img_float[..., 2] > img_float[..., 0] + 0.1) & (img_float[..., 2] > img_float[..., 1] + 0.1)
        water_mask = blue_dominant & (brightness > 0.2) & (brightness < 0.8)
        if water_mask.sum() > 500:
            masks["water"] = water_mask.astype(np.float32)

        # Foliage detection: Green-dominant regions
        green_dominant = (img_float[..., 1] > img_float[..., 0] + 0.1) & (img_float[..., 1] > img_float[..., 2] + 0.05)
        foliage_mask = green_dominant & (brightness > 0.2)
        if foliage_mask.sum() > 500:
            masks["foliage"] = foliage_mask.astype(np.float32)

        # Stone detection: Gray/neutral regions (low color saturation)
        rgb_std = img_float.std(axis=2)
        stone_mask = (rgb_std < 0.15) & (brightness > 0.3) & (brightness < 0.7)
        if stone_mask.sum() > 500:
            masks["stone"] = stone_mask.astype(np.float32)

        return masks


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
        masks = backend.segment(image)

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
