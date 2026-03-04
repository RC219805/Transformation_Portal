"""SegmentationBackend Protocol.

Unified interface for material segmentation backends.

This module defines the protocol (interface) that all segmentation backends
must implement for Materials V3. This enables:

- Hot-swappable backends (stub, efficientsam, sam2, future models)
- Fail-safe defaults (stub backend returns empty masks)
- Consistent device handling (MPS/CUDA/CPU)
- Lazy loading and model caching

Protocol Version: 2.0.0
Compatible with: Materials V3

Example
-------

.. code-block:: python

    from transformation_portal.lux_depth_v3.protocols import (
        SegmentationBackend,
    )

    class MySegmentationBackend(SegmentationBackend):
        def load(self, device: str = "auto") -> None:
            ...

        def segment(
            self, image: np.ndarray,
        ) -> Dict[str, Tuple[np.ndarray, float]]:
            # Returns material names mapped to
            # (mask, confidence) tuples
            return {
                "glass": (mask_array, 0.87),
                "water": (mask_array, 0.64),
            }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Protocol version for compatibility checks
SEGMENTATION_BACKEND_PROTOCOL_VERSION = "2.0.0"


@dataclass(frozen=True)
class SegmentationBackendInfo:
    """Metadata describing a segmentation backend.

    Attributes:
        name: Human-readable backend name
        model_id: Model identifier (e.g., HuggingFace ID or "stub")
        requires_gpu: Whether backend requires GPU for reasonable performance
        requires_weights: Whether backend requires downloaded model weights
        approximate_memory_mb: Approximate memory usage in MB
        description: Human-readable description
    """

    name: str
    model_id: str
    requires_gpu: bool = False
    requires_weights: bool = True
    approximate_memory_mb: Optional[float] = None
    description: str = ""


@runtime_checkable
class SegmentationBackend(Protocol):
    """Protocol defining the interface for material segmentation backends.

    All segmentation backends must implement this protocol to be used with
    Materials V3. The protocol enforces:

    1. **load()**: Lazy model loading with device selection
    2. **segment()**: Image segmentation returning material masks
    3. **info**: Backend metadata for configuration

    Example Implementation
    ----------------------
    ::

        class EfficientSAMBackend:
            @property
            def info(self) -> SegmentationBackendInfo:
                return SegmentationBackendInfo(
                    name="EfficientSAM",
                    model_id="yunyangx/efficientvit-sam",
                    requires_gpu=False,
                    approximate_memory_mb=50,
                )

            def load(self, device: str = "auto") -> None:
                # Load model weights...

            def segment(
                self, image: np.ndarray,
            ) -> Dict[str, Tuple[np.ndarray, float]]:
                # Return material masks with confidence
                return {
                    "glass": (mask_array, 0.87),  # 87% confidence
                    "water": (mask_array, 0.64),  # 64% confidence
                }
    """

    @property
    def info(self) -> SegmentationBackendInfo:
        """Return backend metadata.

        This property provides configuration information including:
        - GPU requirements
        - Memory usage
        - Model identifier
        """
        return None  # type: ignore[return-value]

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
    ) -> None:
        """Load model weights and prepare for inference.

        This method must be called before segment(). It handles:
        - Device selection (auto, cpu, mps, cuda)
        - Weight loading (local path or download)
        - Model initialization and caching

        Args:
            device: Target device ("auto", "cpu", "mps", "cuda")
            weights_path: Optional local path to model weights

        Raises:
            FileNotFoundError: If weights_path specified but not found
            RuntimeError: If model loading fails
        """
        return None

    def segment(
        self,
        image: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run material segmentation on an image.

        Args:
            image: Input RGB image as numpy array (H, W, 3), uint8 [0-255]

        Returns:
            Dict mapping material names to (mask, confidence) tuples:
            - mask: Binary mask (H, W) with values 0.0-1.0
            - confidence: Classification confidence score [0.0-1.0]

            Example: {
                "glass": (mask1, 0.87),  # 87% confidence
                "water": (mask2, 0.34),  # Low confidence - might filter
                "foliage": (mask3, 0.76),
            }

            For stub backend, returns empty dict.
            For real backends, returns detected
            materials only (omits non-detected).

        Raises:
            RuntimeError: If model not loaded or inference fails
            ValueError: If image format is invalid
        """
        return None  # type: ignore[return-value]
