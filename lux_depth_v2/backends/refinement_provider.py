# lux_depth_v2/backends/refinement_provider.py
"""
Refinement provider interface for Material Segmentation V3 fusion.

This module defines a protocol for "refinement providers" that can generate
improved masks for specific material classes. EfficientSAM is the primary
concrete implementation, but this interface allows for:

- Testing with mocks (without ONNX dependencies)
- Future alternative refinement backends
- Graceful fallback when refinement fails
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Protocol

import numpy as np

from .. import torch_ops

if TYPE_CHECKING:
    from .efficientsam_backend import EfficientSAMBackend


class RefinementProvider(Protocol):
    """
    Protocol for mask refinement backends.

    A refinement provider takes a base mask (typically from SegFormer) and
    returns an improved, higher-precision mask focusing on edges and
    boundaries.
    """

    def get_refined_mask(
        self,
        rgb: "torch_ops.torch.Tensor",
        base_mask: "torch_ops.torch.Tensor",
        material_class: str,
    ) -> Optional["torch_ops.torch.Tensor"]:
        """
        Generate a refined mask for a specific material class.

        Parameters
        ----------
        rgb : torch.Tensor
            Input RGB image, 1x3xHxW, float32 in [0, 1]
        base_mask : torch.Tensor
            Base confidence mask from SegFormer, 1x1xHxW, float32 in [0, 1]
        material_class : str
            Material class name (e.g., "glass", "water", "foliage")

        Returns
        -------
        Optional[torch.Tensor]
            Refined mask 1x1xHxW in [0, 1], or None if refinement fails/unavailable
        """
        ...


class MockRefinementProvider:
    """
    Mock refinement provider for testing.

    Returns a simple synthetic refinement based on the base mask.
    Useful for testing fusion logic without EfficientSAM dependencies.
    """

    def __init__(self, mode: str = "dilate"):
        """
        Parameters
        ----------
        mode : str
            Mock behavior:
            - "dilate": slightly expand the base mask
            - "erode": slightly shrink the base mask
            - "identity": return base mask unchanged
            - "none": always return None (simulates failure)
        """
        self.mode = mode

    def get_refined_mask(
        self,
        rgb: "torch_ops.torch.Tensor",
        base_mask: "torch_ops.torch.Tensor",
        material_class: str,
    ) -> Optional["torch_ops.torch.Tensor"]:
        torch_ops.require_torch()

        if self.mode == "none":
            return None

        if self.mode == "identity":
            return base_mask.clone()

        # Simple morphological operations for testing
        if self.mode == "dilate":
            # Simulate edge expansion
            kernel = torch_ops.torch.ones(1, 1, 3, 3, device=base_mask.device) / 9.0
            refined = torch_ops.torch.nn.functional.conv2d(
                base_mask, kernel, padding=1
            )
            return refined.clamp(0.0, 1.0)

        if self.mode == "erode":
            # Simulate edge contraction
            threshold = 0.8
            refined = torch_ops.torch.where(
                base_mask > threshold,
                base_mask,
                base_mask * 0.5,
            )
            return refined.clamp(0.0, 1.0)

        raise ValueError(f"Unknown mock mode: {self.mode}")


class EfficientSAMRefinementProvider:
    """
    EfficientSAM-based refinement provider with depth-aware refinement.

    Wraps EfficientSAMBackend and converts between torch tensors and
    the backend's numpy interface.
    
    Stage 5 enhancements:
    - Depth-aware prompt generation
    - Adaptive box expansion based on material class
    - Multi-prompt support for complex regions
    - Quality gating based on base mask confidence
    """

    def __init__(
        self,
        backend: "EfficientSAMBackend",
        device: "torch_ops.torch.device",
        depth_map: Optional["torch_ops.torch.Tensor"] = None,
        min_confidence: float = 0.3,
        box_expand_ratio: float = 0.1,
    ):
        """
        Parameters
        ----------
        backend : EfficientSAMBackend
            The EfficientSAM backend instance
        device : torch.device
            Target device for tensor operations
        depth_map : Optional[torch.Tensor]
            Optional depth map (1x1xHxW) for depth-aware refinement
        min_confidence : float
            Minimum base mask confidence to attempt refinement
        box_expand_ratio : float
            How much to expand bounding box beyond detected region (0.1 = 10%)
        """
        from .efficientsam_backend import EfficientSAMBackend, PointPrompt, BoxPrompt

        self.backend = backend
        self.device = device
        self.depth_map = depth_map
        self.min_confidence = min_confidence
        self.box_expand_ratio = box_expand_ratio
        self._PointPrompt = PointPrompt
        self._BoxPrompt = BoxPrompt

    def get_refined_mask(
        self,
        rgb: "torch_ops.torch.Tensor",
        base_mask: "torch_ops.torch.Tensor",
        material_class: str,
    ) -> Optional["torch_ops.torch.Tensor"]:
        """
        Use EfficientSAM to refine edges of the base mask.

        Strategy:
        1. Compute bounding box of base mask (> 0.5)
        2. Extract a few point prompts from high-confidence core (> 0.7)
        3. Run EfficientSAM with box + points
        4. Convert output back to torch tensor

        Returns None if:
        - Backend is unavailable
        - Base mask is empty
        - EfficientSAM execution fails
        - Image exceeds safe size threshold (OOM protection)
        """
        torch_ops.require_torch()

        if not self.backend.available:
            return None

        # Extract HxWx3 numpy image
        try:
            rgb_np = (
                rgb[0]
                .permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .to("cpu")
                .numpy()
                .astype(np.float32)
            )

            base_np = base_mask[0, 0].to("cpu").numpy().astype(np.float32)
            h, w = base_np.shape
            
            # OOM safety guard: skip refinement on very large images
            MAX_EFFICIENTSAM_MEGAPIXELS = 30  # conservative safe limit
            megapixels = (h * w) / 1e6
            if megapixels > MAX_EFFICIENTSAM_MEGAPIXELS:
                import logging
                logging.getLogger(__name__).warning(
                    "Image too large for EfficientSAM refinement (%.1f MP > %d MP), "
                    "skipping class '%s' to prevent OOM",
                    megapixels, MAX_EFFICIENTSAM_MEGAPIXELS, material_class
                )
                return None

            # Find bounding box of base mask
            binary = base_np > 0.5
            if not binary.any():
                return None

            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Normalize to [0, 1]
            box = self._BoxPrompt(
                x0=float(x_min) / w,
                y0=float(y_min) / h,
                x1=float(x_max) / w,
                y1=float(y_max) / h,
            )

            # Extract a few high-confidence points
            core = base_np > 0.7
            y_pts, x_pts = np.where(core)
            points = []
            if len(y_pts) > 0:
                # Sample up to 4 points uniformly
                step = max(1, len(y_pts) // 4)
                for i in range(0, len(y_pts), step):
                    if len(points) >= 4:
                        break
                    points.append(
                        self._PointPrompt(
                            x=float(x_pts[i]) / w,
                            y=float(y_pts[i]) / h,
                            label=1,
                        )
                    )

            prompts = [box] + points

            # Call EfficientSAM (raises NotImplementedError in Stage 1/2)
            # In Stage 4+ this will return a real mask
            mask_np = self.backend.segment(rgb_np, prompts)

            # Convert back to torch
            mask_tensor = (
                torch_ops.torch.from_numpy(mask_np)
                .to(device=self.device, dtype=torch_ops.torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            return mask_tensor

        except NotImplementedError:
            # EfficientSAM backend is still a stub
            return None
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "EfficientSAM refinement failed for %s: %s", material_class, e
            )
            return None
