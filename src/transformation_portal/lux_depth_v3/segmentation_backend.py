"""Material segmentation backend for Materials V3.

This module provides material segmentation functionality for the Materials V3 pipeline.

Architecture:
- Protocol-based design (SegmentationBackend Protocol)
- Stub backend (default, production-safe, returns empty masks)
- EfficientSAM backend (opt-in, requires ML dependencies)
- SAM2 backend (opt-in, uses spatial_ai SAM2 integration)
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

3. SAM2SegmentationBackend (opt-in via config):
   - Uses spatial_ai SAM2 backend for mask proposals
   - License: Apache 2.0 (commercial use allowed)
   - Model size: base (~400MB) or large (~850MB) checkpoints
   - Material detection: metadata labels when available, else CLIP/heuristic fallback

Configuration:
- enable_material_segmentation: Enable/disable segmentation
- material_segmentation_backend: "stub" (default), "efficientsam", or "sam2"
- strict_backend: If True, raise on missing weights instead of falling back

For usage examples, see docs/materials_v3_quick_reference.md
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EnhanceConfig
from .protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo

logger = logging.getLogger(__name__)

_LAST_SEGMENTATION_RUNTIME_METADATA: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_LAST_SEGMENTATION_RUNTIME_METADATA",
    default=None,
)

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

try:
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput as SpatialSegmentationInput
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend as SpatialSAM2Backend

    SPATIAL_SAM2_AVAILABLE = True
except ImportError:
    SPATIAL_SAM2_AVAILABLE = False
    SpatialSAM2Backend = None  # type: ignore
    SpatialSegmentationInput = None  # type: ignore


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

    _CLIP_MODEL_NAME = "ViT-B-32"
    _CLIP_PRETRAINED_TAG = "openai"

    def __init__(
        self,
        sky_top_region_fraction: float = 0.5,
        sky_gradient_threshold: float = 0.05,
        sky_brightness_threshold: float = 0.4,
    ):
        self._model = None
        self._device = None
        self._model_loaded = False
        self._use_real_model = False  # Track whether real model or heuristics
        self._mask_generator = None  # Automatic mask generator for SAM
        # Cache CLIP model to avoid repeated loading
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_runtime_metadata: Optional[Dict[str, Any]] = None
        self._sky_top_region_fraction = float(sky_top_region_fraction)
        self._sky_gradient_threshold = float(sky_gradient_threshold)
        self._sky_brightness_threshold = float(sky_brightness_threshold)

    @staticmethod
    def _hf_offline_mode_enabled() -> bool:
        """Return True when HuggingFace/Transformers offline flags are enabled."""
        return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    @classmethod
    def _resolve_cached_clip_checkpoint_path(
        cls,
        model_name: str = _CLIP_MODEL_NAME,
        pretrained_tag: str = _CLIP_PRETRAINED_TAG,
    ) -> Optional[str]:
        """Resolve a local cached OpenCLIP checkpoint path without network access."""
        if not OPEN_CLIP_AVAILABLE:
            return None

        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:
            return None

        try:
            cfg = open_clip.get_pretrained_cfg(model_name, pretrained_tag)
        except Exception:
            return None
        if not cfg:
            return None

        hf_hub_ref = str(cfg.get("hf_hub", "")).strip()
        if not hf_hub_ref:
            return None

        repo_id, explicit_filename = os.path.split(hf_hub_ref)
        repo_id = repo_id.rstrip("/")
        candidates = []
        if explicit_filename:
            candidates.append(explicit_filename)
            if explicit_filename.endswith(".bin"):
                candidates.append(explicit_filename[:-4] + ".safetensors")
        else:
            # OpenCLIP defaults to binary filename; prefer safetensors first.
            candidates.extend(["open_clip_model.safetensors", "open_clip_pytorch_model.bin"])

        for filename in dict.fromkeys(candidates):
            try:
                cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            except Exception:
                continue
            if isinstance(cached_path, str) and Path(cached_path).is_file():
                return cached_path
        return None

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 for a local file path using streaming reads."""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_clip_runtime_metadata(self, weights_source: str, weights_path: Optional[str]) -> None:
        """Capture machine-parseable CLIP runtime provenance."""
        metadata: Dict[str, Any] = {
            "offline_mode": self._hf_offline_mode_enabled(),
            "weights_source": weights_source,
            "weights_path": weights_path,
            "weights_sha256": None,
        }
        if weights_path:
            try:
                metadata["weights_sha256"] = self._compute_sha256(Path(weights_path))
            except Exception as exc:
                logger.debug("Failed to hash CLIP weights file for provenance (%s): %s", weights_path, exc)
        self._clip_runtime_metadata = metadata

    def get_runtime_metadata(self) -> Optional[Dict[str, Any]]:
        """Expose runtime metadata for governance/attestation reports."""
        if self._clip_runtime_metadata is None:
            return None
        return {"clip_runtime": dict(self._clip_runtime_metadata)}

    def _load_clip_runtime(self):
        """Load CLIP model/tokenizer with cache-first offline-safe resolution."""
        if self._clip_model is not None:
            return self._clip_model, self._clip_preprocess, self._clip_tokenizer

        if not OPEN_CLIP_AVAILABLE:
            raise RuntimeError("open_clip is unavailable")

        pretrained_source: str = self._CLIP_PRETRAINED_TAG
        cached_path = self._resolve_cached_clip_checkpoint_path(
            model_name=self._CLIP_MODEL_NAME,
            pretrained_tag=self._CLIP_PRETRAINED_TAG,
        )
        if cached_path is not None:
            pretrained_source = cached_path
            logger.info("Using cached CLIP checkpoint (offline-safe): %s", Path(cached_path).name)
            self._record_clip_runtime_metadata(weights_source="cache_path", weights_path=cached_path)
        elif self._hf_offline_mode_enabled():
            raise RuntimeError(
                "HF offline mode is enabled but cached CLIP checkpoint was not found for "
                f"{self._CLIP_MODEL_NAME}/{self._CLIP_PRETRAINED_TAG}"
            )
        else:
            self._record_clip_runtime_metadata(weights_source="tag_resolution", weights_path=None)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._CLIP_MODEL_NAME,
            pretrained=pretrained_source,
            device=self._device,
        )
        tokenizer = open_clip.get_tokenizer(self._CLIP_MODEL_NAME)

        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer
        return model, preprocess, tokenizer

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
                    "EfficientVIT not available - falling back to heuristic-based segmentation (v1) "
                    "with fixed confidence scores (0.5). For real model inference, install: pip install efficientvit"
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
        if os.getenv("SKIP_CLIP_INFERENCE", "").lower() in ("1", "true", "yes"):
            logger.debug("SKIP_CLIP_INFERENCE set, using heuristics instead of CLIP")
            return self._heuristic_segmentation(image)

        try:
            import torch
            from PIL import Image

            if self._clip_model is None:
                logger.debug("Loading CLIP model for segment classification...")
            else:
                logger.debug("Using cached CLIP model")
            model, preprocess, tokenizer = self._load_clip_runtime()

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

            # Create minimal config object for bootstrap
            sky_config = SimpleNamespace(
                sky_top_region_fraction=self._sky_top_region_fraction,
                sky_gradient_threshold=self._sky_gradient_threshold,
                sky_brightness_threshold=self._sky_brightness_threshold,
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
# SAM2 Backend (Opt-In, spatial_ai Integration)
# =============================================================================


class SAM2SegmentationBackend(EfficientSAMBackend):
    """SAM2-based material segmentation backend.

    This backend reuses the existing CLIP/heuristic material classification flow
    from EfficientSAMBackend, but sources instance masks from the spatial_ai SAM2
    backend.
    """

    _LABEL_ALIASES = {
        "sky": "sky",
        "cloud": "sky",
        "glass": "glass",
        "window": "glass",
        "water": "water",
        "pool": "water",
        "ocean": "water",
        "sea": "water",
        "foliage": "foliage",
        "plant": "foliage",
        "tree": "foliage",
        "leaf": "foliage",
        "stone": "stone",
        "marble": "stone",
        "granite": "stone",
        "limestone": "stone",
        "travertine": "stone",
        "concrete": "stone",
        "wood": "wood",
        "metal": "metal",
        "fabric": "fabric",
        "stucco": "stucco",
        "plaster": "stucco",
    }

    def __init__(
        self,
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        enable_material_classification: bool = False,
        material_confidence_threshold: float = 0.3,
        sky_top_region_fraction: float = 0.5,
        sky_gradient_threshold: float = 0.05,
        sky_brightness_threshold: float = 0.4,
    ):
        super().__init__(
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        self._model_size = model_size
        self._checkpoint_path = checkpoint_path
        self._enable_material_classification = enable_material_classification
        self._material_confidence_threshold = material_confidence_threshold
        self._sam2_backend = None

    @property
    def info(self) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name="SAM2",
            model_id=f"facebook/sam2-hiera-{self._model_size}",
            requires_gpu=False,
            requires_weights=True,
            approximate_memory_mb=850 if self._model_size == "large" else 400,
            description="SAM2 segmentation backend via spatial_ai wrapper",
        )

    @classmethod
    def _canonicalize_material_label(cls, label: Optional[str]) -> Optional[str]:
        """Map free-form labels to the Materials V3 taxonomy keys."""
        if not label:
            return None

        norm = label.strip().lower()
        for token, canonical in cls._LABEL_ALIASES.items():
            if token in norm:
                return canonical
        return None

    @staticmethod
    def _merge_material_result(
        accumulator: Dict[str, Tuple[np.ndarray, float, int]],
        material: str,
        mask: np.ndarray,
        confidence: float,
    ) -> None:
        """Merge another (mask, confidence) contribution into a material bucket."""
        mask_f32 = mask.astype(np.float32, copy=False)
        area = int(np.count_nonzero(mask_f32 > 0.5))
        if area <= 0:
            return

        previous = accumulator.get(material)
        if previous is None:
            accumulator[material] = (mask_f32, float(np.clip(confidence, 0.0, 1.0)), area)
            return

        prev_mask, prev_conf, prev_area = previous
        merged_mask = np.maximum(prev_mask, mask_f32)
        total_area = prev_area + area
        if total_area <= 0:
            merged_conf = 0.0
        else:
            merged_conf = (prev_conf * prev_area + float(np.clip(confidence, 0.0, 1.0)) * area) / total_area
        accumulator[material] = (merged_mask, merged_conf, total_area)

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """Load SAM2 backend from spatial_ai module."""
        if self._model_loaded:
            logger.debug("SAM2 backend already loaded, skipping")
            return

        if not SPATIAL_SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 backend unavailable. Install spatial AI segmentation deps "
                "(sam2 + torch + torchvision), or choose --segmentation-backend efficientsam."
            )

        if self._model_size not in {"base", "large"}:
            raise RuntimeError(f"Invalid sam2 model size '{self._model_size}'. Expected 'base' or 'large'.")

        resolved_device = self._resolve_device(device)
        checkpoint_override: Optional[str] = None
        if weights_path is not None:
            checkpoint_override = str(weights_path)
        elif self._checkpoint_path:
            checkpoint_override = str(self._checkpoint_path)

        try:
            self._sam2_backend = SpatialSAM2Backend(
                model_size=self._model_size,
                device=resolved_device,
                checkpoint_path=checkpoint_override,
                enable_material_classification=self._enable_material_classification,
                material_confidence_threshold=self._material_confidence_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"SAM2 backend loading failed: {exc}") from exc

        self._device = getattr(self._sam2_backend, "device", resolved_device)
        self._model = self._sam2_backend
        self._model_loaded = True
        self._use_real_model = True
        logger.info("SAM2 backend loaded successfully (model=%s, device=%s)", self._model_size, self._device)

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run SAM2 segmentation and map masks to material outputs."""
        if not self._model_loaded or self._sam2_backend is None:
            raise RuntimeError("SAM2 model not loaded. Call .load() first.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

        image_linear = image.astype(np.float32) / 255.0

        try:
            seg_input = SpatialSegmentationInput(
                image=image_linear,
                gamma=1.0,
                mode="auto",
            )
            seg_result = self._sam2_backend.segment(seg_input)
        except Exception as exc:
            raise RuntimeError(f"SAM2 inference failed: {exc}") from exc

        if seg_result.masks.shape[0] == 0:
            logger.debug("SAM2 produced no masks; falling back to heuristic material segmentation")
            return self._heuristic_segmentation(image)

        # First preference: use SAM2/CLIP labels when available.
        material_buckets: Dict[str, Tuple[np.ndarray, float, int]] = {}
        segments: List[Dict[str, Any]] = []
        masks = np.asarray(seg_result.masks)
        scores = np.asarray(seg_result.scores, dtype=np.float32)

        for idx in range(masks.shape[0]):
            raw_mask = masks[idx]
            mask_2d = np.asarray(raw_mask).squeeze()
            if mask_2d.ndim != 2:
                logger.debug("Skipping SAM2 mask with unexpected shape: %s", raw_mask.shape)
                continue

            mask_bool = mask_2d.astype(bool, copy=False)
            area = int(mask_bool.sum())
            if area <= 0:
                continue

            metadata = seg_result.metadata[idx] if idx < len(seg_result.metadata) else None
            if metadata is not None:
                x, y, w_box, h_box = metadata.bbox
                bbox = [int(x), int(y), int(w_box), int(h_box)]
            else:
                rows, cols = np.where(mask_bool)
                bbox = [
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max() - cols.min() + 1),
                    int(rows.max() - rows.min() + 1),
                ]

            segments.append(
                {
                    "segmentation": mask_bool,
                    "bbox": bbox,
                    "area": area,
                    "predicted_iou": float(scores[idx]) if idx < len(scores) else 0.5,
                }
            )

            label = self._canonicalize_material_label(getattr(metadata, "material_label", None))
            if label:
                confidence = getattr(metadata, "material_confidence", None)
                if confidence is None:
                    confidence = float(scores[idx]) if idx < len(scores) else 0.5
                self._merge_material_result(
                    material_buckets,
                    label,
                    mask_bool.astype(np.float32, copy=False),
                    float(confidence),
                )

        if material_buckets:
            logger.debug(
                "SAM2 classified %d masks via metadata labels: %s",
                len(segments),
                list(material_buckets.keys()),
            )
            return {k: (v[0], float(v[1])) for k, v in material_buckets.items()}

        # No explicit labels: reuse existing CLIP/heuristic material labeling.
        classified = self._classify_segments_with_clip(image, segments)
        if classified:
            return classified
        return self._heuristic_segmentation(image)


# =============================================================================
# Backend Factory and Public API
# =============================================================================


# Keep this small: SAM2 instances are heavyweight and multiple cached variants can exhaust memory.
@lru_cache(maxsize=2)  # Cache backend instances by backend + device + model options
def _get_backend_instance(
    backend_name: str,
    device: str = "auto",
    strict: bool = False,
    sam2_model_size: str = "base",
    sam2_checkpoint_path: Optional[str] = None,
    sky_top_region_fraction: float = 0.5,
    sky_gradient_threshold: float = 0.05,
    sky_brightness_threshold: float = 0.4,
) -> SegmentationBackend:
    """Get or create a cached backend instance.

    Args:
        backend_name: "stub", "efficientsam", or "sam2"
        device: Device for backend (used by model backends)
        strict: If True, raise on errors instead of falling back
        sam2_model_size: SAM2 checkpoint family ("base" or "large")
        sam2_checkpoint_path: Optional SAM2 checkpoint override

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

    if backend_name == "efficientsam":
        backend = EfficientSAMBackend(
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
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

    if backend_name == "sam2":
        backend = SAM2SegmentationBackend(
            model_size=sam2_model_size,
            checkpoint_path=sam2_checkpoint_path,
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        try:
            backend.load(device=device)
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"Failed to load {backend_name} backend: {e}") from e
            logger.warning(
                "Failed to load SAM2 backend: %s\n"
                "This is expected if checkpoint/dependencies are missing.\n"
                "Falling back to stub backend.",
                e,
            )
            return _get_backend_instance("stub", device="cpu", strict=False)
        return backend

    raise ValueError(f"Unknown segmentation backend: {backend_name}\n" f"Valid options: 'stub', 'efficientsam', 'sam2'")


def segment_materials(
    image: np.ndarray,
    config: EnhanceConfig,
) -> Dict[str, np.ndarray]:
    """Segment image into material masks.

    This is the main entry point for material segmentation in Materials V3.

    Backends:
    - stub (default): Returns empty masks, production-safe
    - efficientsam (opt-in): ML-powered segmentation
    - sam2 (opt-in): SAM2 mask proposals + CLIP/heuristic material labeling

    Args:
        image: Input image as numpy array (H, W, 3) in RGB, uint8 [0-255]
        config: EnhanceConfig instance with segmentation settings
            - enable_material_segmentation: Enable/disable segmentation
            - material_segmentation_backend: Backend to use ("stub", "efficientsam", or "sam2")
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
    _LAST_SEGMENTATION_RUNTIME_METADATA.set(None)

    # Check if segmentation is enabled
    enable_segmentation = getattr(config, "enable_material_segmentation", False)

    if not enable_segmentation:
        logger.debug("Material segmentation disabled in config")
        return {}

    # Get backend selection
    backend_name = getattr(config, "material_segmentation_backend", "stub")
    strict_backend = getattr(config, "strict_backend", False)
    sam2_model_size = str(getattr(config, "sam2_model_size", "base")).lower()
    sam2_checkpoint_path = getattr(config, "sam2_checkpoint_path", None)
    sky_top_region_fraction = float(getattr(config, "sky_top_region_fraction", 0.5))
    sky_gradient_threshold = float(getattr(config, "sky_gradient_threshold", 0.05))
    sky_brightness_threshold = float(getattr(config, "sky_brightness_threshold", 0.4))

    # Get device for backend (if applicable)
    device = getattr(config, "depth_device", "cpu")  # Reuse depth_device setting

    try:
        # Get or create backend instance (cached)
        backend = _get_backend_instance(
            backend_name,
            device=device,
            strict=strict_backend,
            sam2_model_size=sam2_model_size,
            sam2_checkpoint_path=sam2_checkpoint_path,
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )

        # Run segmentation
        results = backend.segment(image)
        if hasattr(backend, "get_runtime_metadata"):
            try:
                runtime_metadata = backend.get_runtime_metadata()
            except Exception as exc:
                logger.debug("Failed to query segmentation runtime metadata: %s", exc)
                runtime_metadata = None
            if isinstance(runtime_metadata, dict):
                _LAST_SEGMENTATION_RUNTIME_METADATA.set(dict(runtime_metadata))

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


def get_last_segmentation_runtime_metadata() -> Optional[Dict[str, Any]]:
    """Return last segmentation runtime metadata captured by segment_materials()."""
    metadata = _LAST_SEGMENTATION_RUNTIME_METADATA.get()
    if metadata is None:
        return None
    return dict(metadata)
