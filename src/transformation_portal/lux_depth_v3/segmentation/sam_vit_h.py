"""SAM ViT-H research material segmentation backend."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..protocols.segmentation_backend import SegmentationBackendInfo

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from segment_anything import SamAutomaticMaskGenerator as _SamAutomaticMaskGenerator
    from segment_anything import sam_model_registry as _sam_model_registry

    SAM_AVAILABLE = True
    _SAM_IMPORT_ERROR: str = ""
    sam_model_registry = _sam_model_registry
    SamAutomaticMaskGenerator = _SamAutomaticMaskGenerator
except ImportError as _e:
    SAM_AVAILABLE = False
    _SAM_IMPORT_ERROR = str(_e)
    sam_model_registry = None  # type: ignore
    SamAutomaticMaskGenerator = None  # type: ignore


class SAMVitHBackend:
    """SAM ViT-H segmentation backend for APEX Research tier.

    Wraps Meta's Segment Anything Model (ViT-H variant) for research-grade
    universal segmentation. Available via the apex_research preset and
    directly via CLI (``--segmentation-backend sam_vit_h``).

    - Model: SAM ViT-H (Apache 2.0, Meta AI)
    - Checkpoint: sam_vit_h_4b8939.pth (~2.4 GB)
    - Device: MPS > CUDA > CPU auto-detection
    - Lazy loading: model loaded only on first .load() call
    - Fail-safe: missing checkpoint falls back to stub (non-strict mode)
    """

    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    CHECKPOINT_FILENAME = "sam_vit_h_4b8939.pth"
    # SHA-256 of the official SAM ViT-H checkpoint from Meta AI.
    # None means validation is skipped by default.  Operators can supply a
    # hash via EnhanceConfig.sam_vit_h_expected_sha256 to enable integrity
    # checking without modifying this class constant.
    EXPECTED_SHA256: Optional[str] = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        confidence_threshold: float = 0.85,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._points_per_side = points_per_side
        self._pred_iou_thresh = pred_iou_thresh
        self._confidence_threshold = confidence_threshold
        self._model_loaded: bool = False
        self._device: Optional[str] = None
        self._mask_generator: Any = None
        self._runtime_metadata: Optional[Dict[str, Any]] = None

    @property
    def info(self) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name="SAM ViT-H",
            model_id="facebook/sam-vit-huge",
            requires_gpu=False,
            requires_weights=True,
            approximate_memory_mb=2400,
            description="Segment Anything Model ViT-H — research-grade universal segmentation (Apache 2.0)",
        )

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
        expected_sha256: Optional[str] = None,
    ) -> None:
        """Load SAM ViT-H model.

        Args:
            device: Target device ("auto", "cpu", "mps", "cuda")
            weights_path: Optional checkpoint path override
            expected_sha256: Optional SHA-256 hex digest to validate the checkpoint;
                overrides the class-level EXPECTED_SHA256 constant.

        Raises:
            RuntimeError: If segment_anything or torch not installed
            FileNotFoundError: If checkpoint not found at any search path
            RuntimeError: If SHA-256 validation fails
        """
        if self._model_loaded:
            return

        if not SAM_AVAILABLE:
            raise RuntimeError(
                f"segment_anything is not available: {_SAM_IMPORT_ERROR}\n"
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        resolved_device = self._resolve_device(device)
        checkpoint = self._resolve_checkpoint(weights_path)

        effective_sha256 = expected_sha256 or self.EXPECTED_SHA256
        if effective_sha256:
            self._validate_checkpoint_sha256(checkpoint, effective_sha256)

        logger.info("Loading SAM ViT-H on %s (checkpoint: %s)", resolved_device, checkpoint)
        try:
            model = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
            model.to(device=resolved_device)
            model.eval()
            self._mask_generator = SamAutomaticMaskGenerator(
                model=model,
                points_per_side=self._points_per_side,
                pred_iou_thresh=self._pred_iou_thresh,
            )
        except Exception as exc:
            raise RuntimeError(f"SAM ViT-H loading failed: {exc}") from exc

        self._device = resolved_device
        self._model_loaded = True
        self._runtime_metadata = {
            "backend": "sam_vit_h",
            "model_id": "facebook/sam-vit-huge",
            "device": resolved_device,
            "checkpoint": str(checkpoint),
            "points_per_side": self._points_per_side,
            "pred_iou_thresh": self._pred_iou_thresh,
            "confidence_threshold": self._confidence_threshold,
        }
        logger.info("SAM ViT-H loaded successfully on %s", resolved_device)

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run SAM ViT-H automatic mask generation and heuristic material labeling.

        Args:
            image: Input RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict mapping material names to (mask, confidence) tuples:
            - mask: (H, W) float32 [0.0-1.0]
            - confidence: predicted_iou of the best matching mask [0.0-1.0]

        Raises:
            RuntimeError: If model not loaded or inference fails
            ValueError: If image format is invalid
        """
        if not self._model_loaded or self._mask_generator is None:
            raise RuntimeError("SAM ViT-H model not loaded. Call .load() first.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

        try:
            raw_masks = self._mask_generator.generate(image)
        except Exception as exc:
            raise RuntimeError(f"SAM ViT-H inference failed: {exc}") from exc

        return self._masks_to_material_dict(image, raw_masks)

    def _resolve_device(self, device: str) -> str:
        """Resolve device: MPS > CUDA > CPU (matches EfficientSAMBackend ordering)."""
        device_lower = device.lower()
        if device_lower == "cpu":
            return "cpu"
        if not TORCH_AVAILABLE:
            return "cpu"
        # Use module-level torch (already imported at module load time)
        if device_lower == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device_lower == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device_lower == "auto" or device_lower not in ["cuda", "mps", "cpu"]:
            if torch.backends.mps.is_available():
                logger.info("Auto-detected MPS (Apple Silicon) for SAM ViT-H")
                return "mps"
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA for SAM ViT-H")
                return "cuda"
            logger.info("Using CPU for SAM ViT-H (no GPU detected)")
            return "cpu"
        return "cpu"

    def _resolve_checkpoint(self, weights_path: Optional[Path]) -> Path:
        """Find the SAM ViT-H checkpoint file.

        Search order: explicit weights_path arg → self._checkpoint_path →
        ~/.cache/sam/ → checkpoints/ → current directory.
        """
        if weights_path is not None:
            p = Path(weights_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"SAM ViT-H checkpoint not found at specified path: {p}\n" f"Download from: {self.CHECKPOINT_URL}"
                )
            return p
        if self._checkpoint_path:
            p = Path(self._checkpoint_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"SAM ViT-H checkpoint not found at configured path: {p}\n" f"Download from: {self.CHECKPOINT_URL}"
                )
            return p
        search_paths = [
            Path.home() / ".cache" / "sam" / self.CHECKPOINT_FILENAME,
            Path("checkpoints") / self.CHECKPOINT_FILENAME,
            Path("models") / self.CHECKPOINT_FILENAME,
            Path(self.CHECKPOINT_FILENAME),
        ]
        for p in search_paths:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"SAM ViT-H checkpoint '{self.CHECKPOINT_FILENAME}' not found.\n"
            f"Download (~2.4GB) from:\n"
            f"  {self.CHECKPOINT_URL}\n"
            f"Place at: {search_paths[0]}"
        )

    @staticmethod
    def _validate_checkpoint_sha256(checkpoint: Path, expected: str) -> None:
        """Validate checkpoint SHA-256 matches expected hash.

        Raises:
            RuntimeError: On hash mismatch (checkpoint may be corrupted)
        """
        digest = hashlib.sha256()
        with checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {checkpoint}:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"Checkpoint may be corrupted. Re-download from:\n"
                f"  {SAMVitHBackend.CHECKPOINT_URL}"
            )

    def _masks_to_material_dict(
        self,
        image: np.ndarray,
        raw_masks: List[Dict],
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Convert SAM mask list to material dict using per-mask heuristic labeling.

        Applies the same material taxonomy as EfficientSAMBackend._heuristic_segmentation()
        but on a per-mask basis: the mean color of each masked region determines the label.
        Masks below confidence_threshold (predicted_iou) are discarded.
        Masks for the same material are merged via np.maximum; the best predicted_iou
        for each material is reported as the material's confidence score.

        Material taxonomy:
        - glass:   high brightness (>0.6) + blue tint
        - water:   blue dominant (>0.1 over other channels), medium brightness
        - foliage: green dominant (>0.1 over red, >0.05 over blue)
        - stone:   low color saturation (rgb_std < 0.15), medium brightness
        """
        h, w = image.shape[:2]
        img_float = image.astype(np.float32) / 255.0

        material_masks: Dict[str, np.ndarray] = {}
        material_confidences: Dict[str, float] = {}

        for mask_dict in raw_masks:
            iou = float(mask_dict.get("predicted_iou", 0.0))
            if iou < self._confidence_threshold:
                continue

            seg: np.ndarray = mask_dict["segmentation"]
            if not isinstance(seg, np.ndarray) or seg.shape != (h, w):
                continue
            area = int(seg.sum())
            if area < 500:
                continue

            # Mean color in masked region
            masked = img_float[seg]
            if masked.shape[0] == 0:
                continue
            mean_r, mean_g, mean_b = float(masked[:, 0].mean()), float(masked[:, 1].mean()), float(masked[:, 2].mean())
            brightness = (mean_r + mean_g + mean_b) / 3.0

            label = self._heuristic_label(mean_r, mean_g, mean_b, brightness)
            if label is None:
                continue

            float_mask = seg.astype(np.float32)
            if label in material_masks:
                material_masks[label] = np.maximum(material_masks[label], float_mask)
                material_confidences[label] = max(material_confidences[label], iou)
            else:
                material_masks[label] = float_mask
                material_confidences[label] = iou

        return {mat: (material_masks[mat], material_confidences[mat]) for mat in material_masks}

    @staticmethod
    def _heuristic_label(r: float, g: float, b: float, brightness: float) -> Optional[str]:
        """Assign a material label based on mean RGB values of the masked region."""
        # glass: high brightness + blue tint
        if brightness > 0.6 and b > r and b > g:
            return "glass"
        # water: blue dominant, medium brightness
        if b > r + 0.1 and b > g + 0.1 and 0.2 < brightness < 0.8:
            return "water"
        # foliage: green dominant
        if g > r + 0.1 and g > b + 0.05:
            return "foliage"
        # stone: low color saturation, medium brightness
        rgb_std = float(np.std([r, g, b]))
        if rgb_std < 0.15 and 0.3 < brightness < 0.7:
            return "stone"
        return None

    def get_runtime_metadata(self) -> Optional[Dict[str, Any]]:
        return self._runtime_metadata
