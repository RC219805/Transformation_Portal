# lux_depth_v2/backends/efficientsam_backend.py
"""
EfficientSAM backend integration (V3 skeleton)

This module provides a thin wrapper around an EfficientSAM ONNX model.
It is intentionally conservative and defensive:

- Does NOT assume the model is available.
- Fails loudly and clearly if onnxruntime or the model cannot be loaded.
- Keeps the public API stable so higher-level fusion code can be built
  and tested (with mocks) before the real model is wired.

Stage 1 Scope:
- Define prompt dataclasses (points/boxes).
- Implement EfficientSAMBackend with:
  - session lifecycle (lazy loading)
  - input validation & preprocessing hooks
  - a public `segment` API ready for later wiring.

Actual ONNX I/O mapping is a TODO for Stage 2 once the model path/format is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import logging

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    ort = None  # type: ignore

from .model_cache import get_model_path, check_model_available, ModelDownloadError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Types
# ---------------------------------------------------------------------------

@dataclass
class PointPrompt:
    """
    A single point prompt in normalized image coordinates.

    Attributes
    ----------
    x : float
        X coordinate in [0, 1], relative to image width.
    y : float
        Y coordinate in [0, 1], relative to image height.
    label : int
        1 for foreground, 0 for background. Some EfficientSAM variants
        use this to bias the mask toward / away from the point.
    """
    x: float
    y: float
    label: int = 1

    def as_tuple(self) -> Tuple[float, float, int]:
        return (self.x, self.y, self.label)


@dataclass
class BoxPrompt:
    """
    A single box prompt in normalized image coordinates.

    Coordinates are (x0, y0, x1, y1) in [0, 1] relative to width/height.

    This should represent a *tight-ish* bounding box around the object
    or region of interest.
    """
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


Prompt = Union[PointPrompt, BoxPrompt]


# ---------------------------------------------------------------------------
# Backend Implementation Skeleton
# ---------------------------------------------------------------------------

class EfficientSAMNotAvailable(RuntimeError):
    """Raised when EfficientSAM cannot be used (missing onnxruntime or model)."""


class EfficientSAMBackend:
    """
    Thin wrapper around an EfficientSAM ONNX model.

    Notes
    -----
    - This is a *skeleton* implementation for V3:
      - The model I/O contract (input/output tensor names and shapes)
        must be filled in once the concrete EfficientSAM ONNX model is chosen.
    - Higher-level code (Material Segmentation V3 / fusion) should:
      - Treat failures here as a reason to fall back to SegFormer-only masks.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = "efficientsam_ti_vit_s",
        device: str = "cpu",
        providers: Optional[Sequence[str]] = None,
        lazy_load: bool = True,
        auto_download: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_path : Optional[str | Path]
            Optional explicit path to the EfficientSAM ONNX file.
            If None, will use model_name to resolve from cache.
        model_name : str
            Logical model identifier (e.g. 'efficientsam_ti_vit_s').
        device : str
            Execution device hint ('cpu' for now).
        providers : Optional[Sequence[str]]
            Optional explicit ONNX Runtime providers.
        lazy_load : bool
            If True, the ONNX session is created on first use (segment()).
        auto_download : bool
            If True and model is missing, attempt download.
            Default False (offline-by-default).
        cache_dir : Optional[Path]
            Model cache directory. Default: weights/efficientsam/
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = model_name
        self.device = device
        self.providers = list(providers) if providers is not None else None
        self.auto_download = auto_download
        self.cache_dir = cache_dir

        self._session: Optional["ort.InferenceSession"] = None  # type: ignore

        if not lazy_load:
            self._ensure_session()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        """
        Returns True if onnxruntime is available AND model exists or can be downloaded.

        Stage 5B: Stricter semantics - only True if model is actually usable.
        """
        if ort is None:
            return False
        
        try:
            resolved = self._resolve_model_path()
            return resolved.exists()
        except (EfficientSAMNotAvailable, ModelDownloadError):
            return False

    def segment(
        self,
        image: np.ndarray,
        prompts: Sequence[Prompt],
    ) -> np.ndarray:
        """
        Run EfficientSAM on a single image with a set of prompts.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 RGB image as uint8 or float32.
        prompts : Sequence[Prompt]
            List of PointPrompt / BoxPrompt objects.

        Returns
        -------
        np.ndarray
            Float32 mask in [0, 1] of shape (H, W).

        Raises
        ------
        EfficientSAMNotAvailable
            If onnxruntime is missing or the model cannot be loaded.
        """
        if not self.available:
            raise EfficientSAMNotAvailable(
                "EfficientSAM is not available: onnxruntime is not installed "
                "or no model is configured."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image of shape (H, W, 3), got {image.shape}"
            )

        if len(prompts) == 0:
            raise ValueError("At least one prompt (point or box) is required.")

        session = self._ensure_session()

        # Stage 4: Full ONNX I/O implementation
        h_orig, w_orig = image.shape[:2]
        input_tensor, prompt_tensors = self._preprocess(image, prompts)

        # Prepare ONNX feed dict
        onnx_inputs = self._prepare_onnx_inputs(
            input_tensor, prompt_tensors, h_orig, w_orig
        )

        # Run inference
        try:
            outputs = session.run(self._output_names, onnx_inputs)
        except Exception as exc:
            log.error(f"EfficientSAM ONNX inference failed: {exc}")
            raise EfficientSAMNotAvailable(
                f"ONNX inference failed: {exc}"
            ) from exc

        # Parse outputs and postprocess
        mask = self._postprocess_outputs(outputs, h_orig, w_orig)

        return mask

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_session(self) -> "ort.InferenceSession":  # type: ignore
        """
        Lazily initialize and return the ONNX Runtime session.

        Raises
        ------
        EfficientSAMNotAvailable
            If onnxruntime is missing or the session cannot be created.
        """
        if self._session is not None:
            return self._session

        if ort is None:
            raise EfficientSAMNotAvailable(
                "onnxruntime is not installed; cannot initialize EfficientSAM."
            )

        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise EfficientSAMNotAvailable(
                f"EfficientSAM model not found at {model_path}. "
                "Download or configure the ONNX path before using this backend."
            )

        providers = self._resolve_providers()

        log.info(
            "Initializing EfficientSAM ONNX session: path=%s, providers=%s",
            model_path,
            providers,
        )

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise EfficientSAMNotAvailable(
                f"Failed to create EfficientSAM ONNX session: {exc}"
            ) from exc

        # Introspect model I/O for safer runtime usage
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        log.debug(
            "EfficientSAM ONNX model introspection: inputs=%s, outputs=%s",
            self._input_names,
            self._output_names,
        )

        return self._session

    def _resolve_model_path(self) -> Path:
        """
        Determine the ONNX model path to use.

        Stage 5B: Use model_cache for resolution + optional auto-download.

        Returns
        -------
        Path
            Path to ONNX model file.

        Raises
        ------
        EfficientSAMNotAvailable
            If model cannot be resolved or downloaded.
        """
        if self.model_path is not None:
            return self.model_path

        try:
            return get_model_path(
                self.model_name,
                cache_dir=self.cache_dir,
                auto_download=self.auto_download,
            )
        except ModelDownloadError as exc:
            raise EfficientSAMNotAvailable(
                f"Could not resolve model {self.model_name}: {exc}"
            ) from exc

    def _resolve_providers(self) -> Sequence[str]:
        """
        Determine ONNX Runtime providers.

        For now we default to CPU execution; later we can extend this
        to support GPU/MPS when available.
        """
        if self.providers:
            return self.providers

        # Minimal, conservative default.
        return ["CPUExecutionProvider"]

    def _prepare_onnx_inputs(
        self,
        image: np.ndarray,
        prompt_tensors: dict,
        h_orig: int,
        w_orig: int,
    ) -> dict:
        """
        Prepare ONNX Runtime feed dictionary from preprocessed inputs.

        Stage 4: Implements standard EfficientSAM ONNX I/O contract.
        Adapts to actual model tensor names via introspection.

        Expected model contract (typical EfficientSAM ONNX):
        - Image input: 'image' or 'pixel_values', shape (1, 3, H, W), float32
        - Box prompts: 'boxes', shape (1, N, 4), float32, in normalized coords
        - Optional point prompts: 'points', 'labels'

        Returns
        -------
        dict
            ONNX Runtime feed dict ready for session.run()
        """
        # Convert HxWx3 to 1x3xHxW (NCHW)
        img_nchw = np.transpose(image, (2, 0, 1))[None, :, :, :]  # (1,3,H,W)

        # Build feed dict with model-specific tensor names
        # For now, use common naming convention; can be made configurable
        feed = {}

        # Image input (check common names)
        if "image" in self._input_names:
            feed["image"] = img_nchw
        elif "pixel_values" in self._input_names:
            feed["pixel_values"] = img_nchw
        else:
            # Fallback: use first input
            feed[self._input_names[0]] = img_nchw

        # Box prompts (if present)
        if "boxes" in prompt_tensors:
            boxes = prompt_tensors["boxes"]  # (N, 4) normalized
            # Expand to (1, N, 4) for ONNX
            boxes_batch = boxes[None, :, :]

            # Convert normalized [0,1] to pixel coords if required by model
            # Standard EfficientSAM expects pixel coords, so scale:
            boxes_px = boxes_batch.copy()
            boxes_px[..., [0, 2]] *= w_orig  # x coords
            boxes_px[..., [1, 3]] *= h_orig  # y coords

            if "boxes" in self._input_names:
                feed["boxes"] = boxes_px.astype(np.float32)
            elif "box" in self._input_names:
                feed["box"] = boxes_px.astype(np.float32)

        # Point prompts (if present and supported)
        if "points" in prompt_tensors:
            points = prompt_tensors["points"]  # (N, 3) [x, y, label]
            points_batch = points[None, :, :2]  # (1, N, 2) drop label for now

            # Scale to pixel coords
            points_px = points_batch.copy()
            points_px[..., 0] *= w_orig
            points_px[..., 1] *= h_orig

            if "point_coords" in self._input_names:
                feed["point_coords"] = points_px.astype(np.float32)
            if "point_labels" in self._input_names:
                labels = points[:, 2][None, :]  # (1, N)
                feed["point_labels"] = labels.astype(np.int64)

        return feed

    def _postprocess_outputs(
        self,
        outputs: List[np.ndarray],
        h_orig: int,
        w_orig: int,
    ) -> np.ndarray:
        """
        Parse ONNX outputs and return a single HxW float32 mask.

        Stage 4: Handles typical EfficientSAM output formats.

        Expected outputs:
        - Logits or probability masks, typically shape (1, 1, H, W) or (1, H, W)

        Returns
        -------
        np.ndarray
            HxW float32 mask in [0, 1]
        """
        if len(outputs) == 0:
            raise ValueError("ONNX model returned no outputs")

        # Take first output (typically the mask)
        raw = outputs[0]

        # Handle different output shapes
        if raw.ndim == 4:  # (1, 1, H, W) or (1, C, H, W)
            mask = raw[0, 0]  # Take first batch, first channel
        elif raw.ndim == 3:  # (1, H, W)
            mask = raw[0]
        elif raw.ndim == 2:  # (H, W)
            mask = raw
        else:
            raise ValueError(f"Unexpected output shape: {raw.shape}")

        # Apply sigmoid if output is logits (values outside [0,1])
        if mask.min() < -0.1 or mask.max() > 1.1:
            mask = 1.0 / (1.0 + np.exp(-mask))

        # Resize to original dimensions if needed
        h_out, w_out = mask.shape
        if (h_out, w_out) != (h_orig, w_orig):
            # Use cv2 for efficient resize if available
            try:
                import cv2
                mask = cv2.resize(
                    mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
                )
            except ImportError:
                # Fallback to scipy
                from scipy.ndimage import zoom
                scale_y = h_orig / h_out
                scale_x = w_orig / w_out
                mask = zoom(mask, (scale_y, scale_x), order=1)

        # Clamp to [0, 1]
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

        return mask

    def _preprocess(
        self,
        image: np.ndarray,
        prompts: Sequence[Prompt],
    ) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image and prompts into tensors appropriate for the ONNX model.

        Stage 1 behavior:
        - Normalizes image to float32 in [0, 1].
        - Leaves resizing / padding to Stage 2.
        - Converts prompts into simple numpy arrays (points, boxes) in
          normalized coordinates so the later ONNX I/O mapping can plug in.

        Returns
        -------
        input_tensor : np.ndarray
            Preprocessed image tensor ready for further transformation.
        prompt_tensors : dict
            Dict containing 'points' and/or 'boxes' numpy arrays suitable
            for later feeding to the ONNX model.
        """
        if image.dtype == np.uint8:
            img = image.astype("float32") / 255.0
        else:
            img = image.astype("float32")

        h, w, _ = img.shape

        point_list: List[Tuple[float, float, int]] = []
        box_list: List[Tuple[float, float, float, float]] = []

        for p in prompts:
            if isinstance(p, PointPrompt):
                # Ensure normalized to [0,1]
                point_list.append(p.as_tuple())
            elif isinstance(p, BoxPrompt):
                box_list.append(p.as_tuple())
            else:
                raise TypeError(f"Unsupported prompt type: {type(p)}")

        prompt_tensors: dict = {}
        if point_list:
            prompt_tensors["points"] = np.asarray(point_list, dtype="float32")
        if box_list:
            prompt_tensors["boxes"] = np.asarray(box_list, dtype="float32")

        # Stage 1: we return HxWx3 float32; Stage 2 may convert to NCHW, etc.
        return img, prompt_tensors
