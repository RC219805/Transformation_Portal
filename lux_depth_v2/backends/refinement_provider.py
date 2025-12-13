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
    EfficientSAM-based refinement provider with intelligent prompt generation.

    Wraps EfficientSAMBackend and converts between torch tensors and
    the backend's numpy interface.
    
    PR-2 enhancements:
    - Mask-driven prompt generation (high-confidence sampling)
    - ROI cropping for efficiency and focus
    - Comprehensive skip guards (OOM, tiny masks, etc.)
    - Per-class observability and stats emission
    """

    def __init__(
        self,
        backend: "EfficientSAMBackend",
        device: "torch_ops.torch.device",
        depth_map: Optional["torch_ops.torch.Tensor"] = None,
        min_confidence: float = 0.3,
        use_roi_cropping: bool = True,
        roi_padding: int = 50,
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
        use_roi_cropping : bool
            Whether to crop ROI before running EfficientSAM
        roi_padding : int
            Pixels to pad around mask bbox when cropping ROI
        """
        from .efficientsam_backend import EfficientSAMBackend, PointPrompt, BoxPrompt
        from .prompt_generation import PromptGenerationConfig

        self.backend = backend
        self.device = device
        self.depth_map = depth_map
        self.min_confidence = min_confidence
        self.use_roi_cropping = use_roi_cropping
        self.roi_padding = roi_padding
        self._PointPrompt = PointPrompt
        self._BoxPrompt = BoxPrompt
        self.prompt_cfg = PromptGenerationConfig()
        
        # Per-class stats for observability
        self.refinement_stats: Dict[str, dict] = {}

    def get_refined_mask(
        self,
        rgb: "torch_ops.torch.Tensor",
        base_mask: "torch_ops.torch.Tensor",
        material_class: str,
    ) -> Optional["torch_ops.torch.Tensor"]:
        """
        Use EfficientSAM to refine edges of the base mask with intelligent prompts.

        PR-2 Strategy:
        1. Generate mask-driven prompts (high-confidence FG + boundary BG)
        2. Optionally crop ROI for efficiency
        3. Run EfficientSAM with smart prompts
        4. Resize back to original resolution if ROI was used
        5. Emit detailed stats for observability

        Returns None if:
        - Backend is unavailable
        - Base mask is empty or too small
        - EfficientSAM execution fails
        - Image exceeds safe size threshold (OOM protection)
        """
        torch_ops.require_torch()
        import logging
        from .prompt_generation import generate_prompts_from_mask, compute_roi_from_mask

        stats = {
            "skip_reason": None,
            "prompt_count_fg": 0,
            "prompt_count_bg": 0,
            "roi_used": False,
            "roi_size": None,
        }

        if not self.backend.available:
            stats["skip_reason"] = "backend_unavailable"
            self.refinement_stats[material_class] = stats
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
                logging.getLogger(__name__).warning(
                    "Image too large for EfficientSAM refinement (%.1f MP > %d MP), "
                    "skipping class '%s' to prevent OOM",
                    megapixels, MAX_EFFICIENTSAM_MEGAPIXELS, material_class
                )
                stats["skip_reason"] = f"image_too_large_{megapixels:.1f}MP"
                self.refinement_stats[material_class] = stats
                return None

            # Generate intelligent prompts from mask
            fg_points_yx, bg_points_yx, prompt_stats = generate_prompts_from_mask(
                base_np, self.prompt_cfg
            )
            
            if prompt_stats["skip_reason"] is not None:
                stats.update(prompt_stats)
                self.refinement_stats[material_class] = stats
                return None
            
            stats["prompt_count_fg"] = prompt_stats["fg_points_generated"]
            stats["prompt_count_bg"] = prompt_stats["bg_points_generated"]

            # Decide whether to use ROI cropping
            use_roi = self.use_roi_cropping
            roi_bbox = None
            
            if use_roi:
                roi_bbox, roi_stats = compute_roi_from_mask(
                    base_np,
                    padding=self.roi_padding,
                    max_side=self.prompt_cfg.max_roi_side,
                )
                if roi_bbox is None:
                    stats.update(roi_stats)
                    self.refinement_stats[material_class] = stats
                    return None
                
                y0, x0, y1, x1 = roi_bbox
                stats["roi_used"] = True
                stats["roi_size"] = f"{y1-y0}x{x1-x0}"
                
                # Crop image and mask to ROI
                rgb_crop = rgb_np[y0:y1, x0:x1, :]
                
                # Adjust prompt coordinates to ROI
                fg_points_yx = fg_points_yx - np.array([[y0, x0]])
                if len(bg_points_yx) > 0:
                    bg_points_yx = bg_points_yx - np.array([[y0, x0]])
                
                h_roi, w_roi = y1 - y0, x1 - x0
                process_img = rgb_crop
                process_h, process_w = h_roi, w_roi
            else:
                process_img = rgb_np
                process_h, process_w = h, w

            # Convert points to normalized prompts
            prompts = []
            for yx in fg_points_yx:
                prompts.append(
                    self._PointPrompt(
                        x=float(yx[1]) / process_w,
                        y=float(yx[0]) / process_h,
                        label=1,  # foreground
                    )
                )
            for yx in bg_points_yx:
                prompts.append(
                    self._PointPrompt(
                        x=float(yx[1]) / process_w,
                        y=float(yx[0]) / process_h,
                        label=0,  # background
                    )
                )

            # Call EfficientSAM
            mask_np = self.backend.segment(process_img, prompts)

            # Resize back to original if ROI was used
            if use_roi and roi_bbox is not None:
                full_mask = np.zeros((h, w), dtype=np.float32)
                y0, x0, y1, x1 = roi_bbox
                full_mask[y0:y1, x0:x1] = mask_np
                mask_np = full_mask

            # Convert back to torch
            mask_tensor = (
                torch_ops.torch.from_numpy(mask_np)
                .to(device=self.device, dtype=torch_ops.torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            
            self.refinement_stats[material_class] = stats
            return mask_tensor

        except NotImplementedError:
            # EfficientSAM backend is still a stub
            stats["skip_reason"] = "backend_not_implemented"
            self.refinement_stats[material_class] = stats
            return None
        except Exception as e:
            logging.getLogger(__name__).warning(
                "EfficientSAM refinement failed for %s: %s", material_class, e
            )
            stats["skip_reason"] = f"exception_{type(e).__name__}"
            self.refinement_stats[material_class] = stats
            return None
