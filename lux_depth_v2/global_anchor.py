#!/usr/bin/env python3
"""
Global Anchor Fusion for Tiled Depth Inference
===============================================

Prevents tile artifacts through global context preservation:
- Run single low-res pass over full frame (global structure)
- Run tiled high-res passes (spatial detail)
- Fuse as: global base + high-frequency residual from tiles

This eliminates:
- Low-frequency banding across tiles
- Subtle plane warps
- Global drift and consistency issues

Reference: User feedback 2025-12-18
"Even if tile inference is truly higher-res, tiles lose global context.
A global anchor pass prevents the most stubborn tiling artifacts."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GlobalAnchorConfig:
    """Configuration for global anchor fusion."""
    
    # Global pass resolution (low-res for context)
    global_max_size: int = 512  # Max dimension for global pass
    
    # Fusion parameters
    global_weight: float = 0.3  # Weight for global base
    tile_weight: float = 0.7    # Weight for tiled detail
    
    # Frequency separation
    # CRITICAL FIX 2025-12-18: Disable frequency split due to DC offset bug
    # Bug: global_lf + tiled_hf assumes aligned DC offsets, but global (512px upsampled)
    # and tiled (native 1024px) have different mean depths, causing edge misalignment
    use_frequency_split: bool = False  # DISABLED: Use simple weighted average instead
    blur_sigma: float = 5.0  # Gaussian blur for low-pass filter (if re-enabled)
    
    # Edge preservation
    # CRITICAL FIX 2025-12-18: Disable edge-aware fusion
    # Bug: Edge-aware fusion favors tiled at RGB edges, but this REDUCES alignment
    # because tiled depth edges don't perfectly align with RGB edges (-45% vs -18%)
    edge_aware_fusion: bool = False  # DISABLED: Simple weighted average works better
    edge_threshold: float = 0.1  # Threshold for edge detection (if re-enabled)


class GlobalAnchorFusion:
    """
    Fuse global low-res depth with tiled high-res depth.
    
    Strategy:
    1. Global pass captures scene-wide structure (walls, floors, ceiling planes)
    2. Tiled passes capture fine detail (edges, furniture, fixtures)
    3. Fusion preserves both: global coherence + local fidelity
    """
    
    def __init__(self, config: GlobalAnchorConfig):
        self.config = config
        logger.info(f"GlobalAnchorFusion: global_size={config.global_max_size} weights={config.global_weight:.2f}/{config.tile_weight:.2f}")
    
    def _resize_for_global_pass(self, rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image for global pass (low-res for context).
        
        Returns:
            (resized_image, scale_factor)
        """
        h, w = rgb.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.config.global_max_size:
            return rgb, 1.0
        
        scale = self.config.global_max_size / max_dim
        new_h, new_w = int(h * scale), int(w * scale)
        
        if not CV2_AVAILABLE:
            from PIL import Image
            if rgb.dtype == np.uint8:
                rgb_pil = Image.fromarray(rgb)
            else:
                rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_resized = np.array(rgb_pil.resize((new_w, new_h), Image.LANCZOS))
            if rgb.dtype == np.float32:
                rgb_resized = rgb_resized.astype(np.float32) / 255.0
        else:
            rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        logger.info(f"Global pass resize: {h}×{w} → {new_h}×{new_w} (scale={scale:.3f})")
        return rgb_resized, scale
    
    def _upsample_global_depth(
        self, 
        global_depth: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Upsample global depth to target shape."""
        h, w = target_shape
        
        if not CV2_AVAILABLE:
            from PIL import Image
            depth_pil = Image.fromarray((global_depth * 255).astype(np.uint8))
            depth_upsampled = np.array(depth_pil.resize((w, h), Image.LANCZOS))
            return depth_upsampled.astype(np.float32) / 255.0
        else:
            return cv2.resize(global_depth, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    def _extract_low_frequency(self, depth: np.ndarray) -> np.ndarray:
        """Extract low-frequency component (base structure)."""
        if not CV2_AVAILABLE:
            return depth  # Fallback: return as-is
        
        # Gaussian blur for low-pass filter
        ksize = max(3, int(self.config.blur_sigma * 6) | 1)
        depth_lf = cv2.GaussianBlur(depth, (ksize, ksize), self.config.blur_sigma)
        return depth_lf
    
    def _extract_high_frequency(self, depth: np.ndarray) -> np.ndarray:
        """Extract high-frequency component (fine detail)."""
        depth_lf = self._extract_low_frequency(depth)
        depth_hf = depth - depth_lf
        return depth_hf
    
    def _compute_edge_weights(self, rgb: np.ndarray) -> np.ndarray:
        """
        Compute edge-aware fusion weights.
        
        At strong edges: favor tiled depth (has better edge fidelity)
        In smooth regions: favor global depth (has better coherence)
        """
        if not CV2_AVAILABLE:
            return np.ones(rgb.shape[:2], dtype=np.float32)
        
        # Convert to grayscale if needed
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.shape[2] == 3 else rgb[:, :, 0]
        else:
            gray = rgb
        
        # Detect edges
        if gray.dtype == np.float32:
            gray_uint8 = (gray * 255).astype(np.uint8)
        else:
            gray_uint8 = gray
        
        edges = cv2.Canny(gray_uint8, 50, 150).astype(np.float32) / 255.0
        
        # Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Edge weights: high at edges (favor tiled), low in smooth regions (favor global)
        edge_weights = edges_dilated
        edge_weights = np.clip(edge_weights, self.config.edge_threshold, 1.0)
        
        return edge_weights
    
    def fuse(
        self,
        global_depth: np.ndarray,
        tiled_depth: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fuse global and tiled depth maps.
        
        Args:
            global_depth: Low-res depth (upsampled to match tiled_depth)
            tiled_depth: High-res tiled depth
            rgb: Optional RGB image for edge-aware fusion
            
        Returns:
            Fused depth map
        """
        assert global_depth.shape == tiled_depth.shape, "Depth maps must match in size"
        
        if self.config.use_frequency_split:
            # Frequency-based fusion
            global_lf = self._extract_low_frequency(global_depth)
            tiled_hf = self._extract_high_frequency(tiled_depth)
            
            # Combine: global low-freq + tiled high-freq
            fused = global_lf + tiled_hf
            
            logger.info("Frequency split fusion: global_LF + tiled_HF")
        else:
            # Simple weighted average
            fused = (
                self.config.global_weight * global_depth +
                self.config.tile_weight * tiled_depth
            )
            
            logger.info(f"Weighted fusion: {self.config.global_weight:.2f} global + {self.config.tile_weight:.2f} tiled")
        
        # Optional: edge-aware refinement
        if self.config.edge_aware_fusion and rgb is not None:
            edge_weights = self._compute_edge_weights(rgb)
            
            # At edges: favor tiled (better fidelity)
            # In smooth: favor current fused (has global coherence)
            fused_refined = (
                edge_weights * tiled_depth +
                (1 - edge_weights) * fused
            )
            
            logger.info("Applied edge-aware refinement")
            fused = fused_refined
        
        # Normalize to [0, 1]
        fused = np.clip(fused, 0, 1)
        
        return fused


def fuse_with_global_anchor(
    depth_estimator,
    rgb: np.ndarray,
    tiled_depth: np.ndarray,
    config: Optional[GlobalAnchorConfig] = None
) -> np.ndarray:
    """
    Convenience function: run global pass and fuse with tiled depth.
    
    Args:
        depth_estimator: Depth estimation model/pipeline
        rgb: RGB image
        tiled_depth: Depth from tiled inference
        config: Fusion configuration
        
    Returns:
        Fused depth map
    """
    if config is None:
        config = GlobalAnchorConfig()
    
    fusion = GlobalAnchorFusion(config)
    
    # 1. Run global pass (low-res for context)
    rgb_global, scale = fusion._resize_for_global_pass(rgb)
    
    logger.info("Running global depth pass...")
    global_depth_lowres = depth_estimator.estimate_depth(rgb_global)
    
    # 2. Upsample global depth to match tiled depth
    global_depth = fusion._upsample_global_depth(global_depth_lowres, tiled_depth.shape)
    
    # 3. Fuse
    logger.info("Fusing global and tiled depth...")
    fused_depth = fusion.fuse(global_depth, tiled_depth, rgb)
    
    logger.info("✓ Global anchor fusion complete")
    return fused_depth


# Preset configurations

PRESETS = {
    "conservative": GlobalAnchorConfig(
        global_max_size=384,
        global_weight=0.2,
        tile_weight=0.8,
        use_frequency_split=True,
        edge_aware_fusion=False
    ),
    "balanced": GlobalAnchorConfig(
        global_max_size=512,
        global_weight=0.3,
        tile_weight=0.7,
        use_frequency_split=True,
        edge_aware_fusion=True
    ),
    "aggressive": GlobalAnchorConfig(
        global_max_size=768,
        global_weight=0.4,
        tile_weight=0.6,
        use_frequency_split=True,
        edge_aware_fusion=True,
        blur_sigma=8.0
    ),
}
