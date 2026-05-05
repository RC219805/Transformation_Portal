"""EfficientSAM material segmentation backend."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..material_confidence_contract import (
    CLIP_SOFTMAX_MARGIN_SCORE_TYPE,
    HEURISTIC_MATERIAL_SCORE_TYPE,
    MATERIALS_V3_CALIBRATION_VERSION,
    MISSING_CLIP_SCORE_FALLBACK_TYPE,
)
from ..protocols.segmentation_backend import SegmentationBackendInfo
from ._cache import _softmax_probabilities, _tensor_values_1d

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._model: Any = None
        self._device: Optional[str] = None
        self._model_loaded = False
        self._use_real_model = False  # Track whether real model or heuristics
        self._mask_generator: Any = None  # Automatic mask generator for SAM
        # Cache CLIP model to avoid repeated loading
        self._clip_model: Any = None
        self._clip_preprocess: Any = None
        self._clip_tokenizer: Any = None
        self._clip_runtime_metadata: Optional[Dict[str, Any]] = None
        self._clip_text_features: Any = None
        self._clip_text_prompt_signature: Optional[Tuple[str, ...]] = None
        self._clip_classification_timing_ms: Dict[str, float] = {}
        self._material_confidence_evidence: Dict[str, Dict[str, Any]] = {}
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
        except ImportError:
            # huggingface_hub not installed - cannot check cache
            return None

        try:
            cfg = open_clip.get_pretrained_cfg(model_name, pretrained_tag)
        except (KeyError, ValueError, AttributeError):
            # Config not found or malformed - skip cache resolution
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
            except (OSError, ValueError):
                # Cache check failed for this candidate - try next
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
        metadata: Dict[str, Any] = {}
        if self._clip_runtime_metadata is not None:
            metadata["clip_runtime"] = dict(self._clip_runtime_metadata)
        if self._clip_classification_timing_ms:
            metadata["clip_classification"] = {"timing_ms": dict(self._clip_classification_timing_ms)}
        if self._material_confidence_evidence:
            metadata["material_confidence_evidence"] = {
                str(material): dict(values) for material, values in self._material_confidence_evidence.items()
            }
        if not metadata:
            return None
        return metadata

    def _load_clip_runtime(self) -> Tuple[Any, Any, Any]:
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

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run material segmentation on an image.

        Args:
            image: Input RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict mapping material names to ``(mask, confidence)`` tuples:
            - mask: Binary mask (H, W), float32 [0.0-1.0]
            - confidence: Material confidence score [0.0-1.0]
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

    def _load_efficientvit_model(self, weights_path: Optional[Path] = None) -> Any:
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
            - confidence: calibrated CLIP probability or heuristic score [0.0-1.0]
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
            - confidence: area-weighted calibrated CLIP softmax probability [0.0-1.0]

        Note:
            If segments list is empty (which happens when SAM doesn't run),
            we first generate heuristic segments and then classify them with CLIP.
            Raw CLIP similarity is retained as evidence metadata, not as
            authoritative Materials V3 confidence.
        """
        self._clip_classification_timing_ms = {}
        self._material_confidence_evidence = {}
        t_total = time.perf_counter()

        # If no SAM segments, generate heuristic segments first
        if not segments:
            t_heuristic = time.perf_counter()
            logger.debug("No SAM segments provided, generating heuristic segments for CLIP classification")
            heuristic_results = self._heuristic_segmentation(image)
            self._clip_classification_timing_ms["heuristic_segments"] = round(
                (time.perf_counter() - t_heuristic) * 1000.0,
                3,
            )

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

            with torch.no_grad():
                prompt_signature = tuple(material_prompts.values())
                if self._clip_text_features is None or self._clip_text_prompt_signature != prompt_signature:
                    t_text = time.perf_counter()
                    text_tokens = tokenizer(list(material_prompts.values())).to(self._device)
                    text_features = model.encode_text(text_tokens)
                    self._clip_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    self._clip_text_prompt_signature = prompt_signature
                    self._clip_classification_timing_ms["text_encode"] = round(
                        (time.perf_counter() - t_text) * 1000.0,
                        3,
                    )
                text_features = self._clip_text_features

                # Initialize material masks with tracking for confidence scores
                h, w = image.shape[:2]
                material_data: Dict[str, Dict[str, Any]] = {
                    name: {
                        "mask": np.zeros((h, w), dtype=np.float32),
                        "scores": [],
                        "raw_similarities": [],
                        "softmax_probabilities": [],
                        "top2_margins": [],
                        "areas": [],
                    }
                    for name in material_prompts
                }

                # Convert image to PIL for preprocessing
                pil_image = Image.fromarray(image)

                t_crop = time.perf_counter()
                region_tensors = []
                region_segments: list[tuple[int, Dict[str, Any]]] = []
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
                    region_tensors.append(region_tensor)
                    region_segments.append((seg_idx, segment))
                self._clip_classification_timing_ms["crop_preprocess"] = round(
                    (time.perf_counter() - t_crop) * 1000.0,
                    3,
                )

                if not region_tensors:
                    return self._heuristic_segmentation(image)

                t_image = time.perf_counter()
                image_batch = torch.cat(region_tensors, dim=0)
                image_features = model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity_batch = image_features @ text_features.T
                self._clip_classification_timing_ms["image_encode"] = round(
                    (time.perf_counter() - t_image) * 1000.0,
                    3,
                )
                self._clip_classification_timing_ms["batch_size"] = float(len(region_tensors))

                # Classify each segment in stable input order
                for row_idx, (seg_idx, segment) in enumerate(region_segments):
                    mask = segment["segmentation"]
                    similarities = similarity_batch[row_idx]
                    similarity_values = _tensor_values_1d(similarities)
                    probabilities = _softmax_probabilities(similarity_values)
                    if similarity_values.size == 0:
                        continue
                    best_idx = int(np.argmax(similarity_values))
                    best_material = list(material_prompts.keys())[best_idx]
                    raw_similarity = float(similarity_values[best_idx])
                    softmax_probability = float(probabilities[best_idx]) if probabilities.size else 0.0
                    if probabilities.size >= 2:
                        top2 = np.partition(probabilities, -2)[-2:]
                        top1 = float(max(top2[0], top2[1]))
                        top2_second = float(min(top2[0], top2[1]))
                        top2_margin = top1 - top2_second
                    else:
                        top2_margin = float(softmax_probability)

                    # Only add if calibrated probability is reasonable (> 0.2)
                    if softmax_probability > 0.2:
                        # Track calibrated confidence, raw evidence, and segment area for weighted averaging.
                        material_data[best_material]["scores"].append(softmax_probability)
                        material_data[best_material]["raw_similarities"].append(raw_similarity)
                        material_data[best_material]["softmax_probabilities"].append(softmax_probability)
                        material_data[best_material]["top2_margins"].append(top2_margin)
                        material_data[best_material]["areas"].append(segment["area"])

                        # Add segment mask to material (using max to handle overlaps)
                        material_data[best_material]["mask"] = np.maximum(
                            material_data[best_material]["mask"], mask.astype(np.float32)
                        )

                        heuristic_label = segment.get("heuristic_label", "unknown")
                        match_str = "✓" if heuristic_label == best_material else f"({heuristic_label}→{best_material})"
                        logger.debug(
                            f"Segment {seg_idx}: {best_material} {match_str} "
                            f"raw_similarity={raw_similarity:.3f}, "
                            f"softmax_probability={softmax_probability:.3f}, "
                            f"top2_margin={top2_margin:.3f}, area={segment['area']}px"
                        )

                # Compute aggregate calibrated confidence per material (area-weighted average).
                material_masks: Dict[str, Tuple[np.ndarray, float]] = {}
                material_evidence: Dict[str, Dict[str, Any]] = {}
                for name, data in material_data.items():
                    mask = data["mask"]
                    scores = data["scores"]
                    areas = data["areas"]

                    # Only include materials with sufficient coverage
                    if mask.sum() > 500:
                        if scores:
                            # Area-weighted average of calibrated CLIP probabilities for this material.
                            total_area = sum(areas)
                            weighted_conf = sum(s * a for s, a in zip(scores, areas)) / total_area
                            raw_scores = data["raw_similarities"]
                            probs = data["softmax_probabilities"]
                            margins = data["top2_margins"]
                            weighted_raw = sum(s * a for s, a in zip(raw_scores, areas)) / total_area
                            weighted_prob = sum(s * a for s, a in zip(probs, areas)) / total_area
                            weighted_margin = sum(s * a for s, a in zip(margins, areas)) / total_area
                            material_masks[name] = (mask, float(weighted_conf))
                            material_evidence[name] = {
                                "material_confidence": float(weighted_conf),
                                "confidence_score_type": CLIP_SOFTMAX_MARGIN_SCORE_TYPE,
                                "raw_clip_similarity": float(weighted_raw),
                                "clip_softmax_probability": float(weighted_prob),
                                "clip_top2_margin": float(weighted_margin),
                                "calibration_version": MATERIALS_V3_CALIBRATION_VERSION,
                            }
                        else:
                            # No scores (shouldn't happen, but handle gracefully)
                            material_masks[name] = (mask, 0.5)
                            material_evidence[name] = {
                                "material_confidence": 0.5,
                                "confidence_score_type": MISSING_CLIP_SCORE_FALLBACK_TYPE,
                                "raw_clip_similarity": None,
                                "clip_softmax_probability": None,
                                "clip_top2_margin": None,
                                "calibration_version": None,
                            }

                logger.info(
                    f"CLIP classified {len(segments)} segments into {len(material_masks)} materials: "
                    f"{', '.join(f'{m} ({c:.0%})' for m, (_, c) in material_masks.items())}"
                )
                self._material_confidence_evidence = material_evidence
                self._clip_classification_timing_ms["total"] = round((time.perf_counter() - t_total) * 1000.0, 3)
                return material_masks

        except Exception as e:
            self._clip_classification_timing_ms["total"] = round((time.perf_counter() - t_total) * 1000.0, 3)
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

    def _create_placeholder_model(self) -> Any:
        """Create a placeholder model for v1 heuristic-only mode.

        Used when EfficientVIT dependencies not available.
        """

        # Simple placeholder that demonstrates the pattern
        class PlaceholderModel:
            def __init__(self, device: Optional[str]) -> None:
                self.device = device

            def eval(self) -> "PlaceholderModel":
                return self

            def to(self, device: str) -> "PlaceholderModel":
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
            from ..bootstrap.sky_seed import detect_sky_seed

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

        self._material_confidence_evidence = {
            material: {
                "material_confidence": float(confidence),
                "confidence_score_type": HEURISTIC_MATERIAL_SCORE_TYPE,
                "raw_clip_similarity": None,
                "clip_softmax_probability": None,
                "clip_top2_margin": None,
                "calibration_version": None,
            }
            for material, (_, confidence) in masks.items()
        }
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
        from ..bootstrap.sky_seed import detect_sky_seed

        return detect_sky_seed(image, config)
