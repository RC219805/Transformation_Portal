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
    ) -> None:
        """
        Parameters
        ----------
        model_path : Optional[str | Path]
            Optional explicit path to the EfficientSAM ONNX file.
            If None, the backend will later derive a default location
            based on `model_name` (to be implemented in Stage 2).
        model_name : str
            Logical model identifier (e.g. 'efficientsam_ti_vit_s').
        device : str
            Execution device hint ('cpu' for now).
        providers : Optional[Sequence[str]]
            Optional explicit ONNX Runtime providers.
        lazy_load : bool
            If True, the ONNX session is created on first use (segment()).
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = model_name
        self.device = device
        self.providers = list(providers) if providers is not None else None

        self._session: Optional["ort.InferenceSession"] = None  # type: ignore

        if not lazy_load:
            self._ensure_session()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        """
        Returns True if onnxruntime is importable and a model path is known.

        This does not guarantee that the session has been initialized; it only
        checks that EfficientSAM *could* be used in this environment.
        """
        if ort is None:
            return False
        # For Stage 1 we only require that either a model_path is set or
        # we have a model_name that future logic can resolve.
        return True

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
            Float32 mask in [0, 1] of shape (H, W). In later stages this may
            be extended to (N, H, W) for multiple masks; for now we return a
            single composite mask.

        Raises
        ------
        EfficientSAMNotAvailable
            If onnxruntime is missing or the model cannot be loaded.
        NotImplementedError
            Until the actual ONNX I/O mapping is implemented in Stage 2.
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

        # Stage 1: We only implement validation + preprocessing hooks.
        # The actual ONNX input / output mapping is a Stage 2 task.
        input_tensor, prompt_tensors = self._preprocess(image, prompts)

        # TODO (Stage 2): Implement real ONNX model execution and output parsing.
        raise NotImplementedError(
            "EfficientSAMBackend.segment is a skeleton. "
            "ONNX I/O wiring must be implemented in Stage 2."
        )

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

        return self._session

    def _resolve_model_path(self) -> Path:
        """
        Determine the ONNX model path to use.

        For Stage 1 we keep this simple:
        - If `model_path` was provided, use it.
        - Else, default to `weights/efficientsam/{model_name}.onnx`.

        This is easy to override later based on how you actually store models.
        """
        if self.model_path is not None:
            return self.model_path

        default_root = Path("weights") / "efficientsam"
        return default_root / f"{self.model_name}.onnx"

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
