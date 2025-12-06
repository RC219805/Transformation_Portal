#!/usr/bin/env python3
"""
Depth Processing Integration Module
====================================

Wrapper for depth-aware processing functionality, providing a clean interface
for the unified pipeline to utilize depth estimation and zone-based enhancements.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Configuration for depth processing."""
    model_name: str = "depth_anything_v2"
    tile_size: int = 518
    enable_zone_processing: bool = True
    foreground_boost: float = 1.2
    midground_balance: float = 1.0
    background_soften: float = 0.9
    device: str = "auto"
    
    def __post_init__(self):
        """Auto-detect device if needed."""
        if self.device == "auto":
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"


class DepthProcessor:
    """
    Depth-aware image processor.
    
    Estimates depth maps and applies zone-based enhancements that respect
    spatial relationships in the scene.
    """
    
    def __init__(self, config: DepthConfig):
        self.config = config
        self.depth_model = None
        logger.info(f"Initializing depth processor (device: {config.device})")
        
    def _load_depth_model(self):
        """Load depth estimation model (lazy loading)."""
        if self.depth_model is not None:
            return self.depth_model
        
        logger.info(f"Loading depth model: {self.config.model_name}")
        
        try:
            if self.config.model_name == "depth_anything_v2":
                # Try to use transformers pipeline
                from transformers import pipeline
                self.depth_model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Large",
                    device=self.config.device if self.config.device != "mps" else "cpu"
                )
                logger.info("✓ Depth model loaded via transformers")
            else:
                raise ValueError(f"Unsupported depth model: {self.config.model_name}")
        except Exception as e:
            logger.warning(f"Could not load depth model: {e}")
            self.depth_model = None
        
        return self.depth_model
    
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: RGB image as float32 [0, 1] or uint8
            
        Returns:
            Depth map as float32 [0, 1], or None if estimation fails
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping depth estimation")
            return None
        
        model = self._load_depth_model()
        if model is None:
            logger.warning("Depth model not available - skipping depth estimation")
            return None
        
        try:
            # Convert to PIL for model
            if image.dtype == np.float32 or image.dtype == np.float64:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image)
            
            # Estimate depth
            logger.info(f"Estimating depth for {image_pil.size} image...")
            result = model(image_pil)
            
            # Extract depth map
            if hasattr(result, 'depth'):
                depth_map = np.array(result.depth)
            elif isinstance(result, dict) and 'depth' in result:
                depth_map = np.array(result['depth'])
            elif hasattr(result, 'predicted_depth'):
                depth_map = np.array(result.predicted_depth)
            else:
                # Assume result is the depth map
                depth_map = np.array(result)
            
            # Normalize to [0, 1]
            if depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0]  # Take first channel
            
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Resize to match input if needed
            if depth_map.shape[:2] != image.shape[:2]:
                from PIL import Image as PILImage
                depth_pil = PILImage.fromarray((depth_map * 255).astype(np.uint8))
                depth_pil = depth_pil.resize((image.shape[1], image.shape[0]), PILImage.LANCZOS)
                depth_map = np.array(depth_pil).astype(np.float32) / 255.0
            
            logger.info(f"✓ Depth estimation complete: {depth_map.shape}")
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def create_zone_masks(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create foreground/midground/background masks from depth map.
        
        Args:
            depth_map: Depth map [0, 1], where 0=far, 1=near
            
        Returns:
            (foreground, midground, background) masks as float32 [0, 1]
        """
        # Define zone thresholds
        # Depth map: 0=far (background), 1=near (foreground)
        foreground_threshold = 0.66  # Top 33% = foreground
        midground_threshold = 0.33    # Middle 33% = midground
        # Bottom 33% = background
        
        foreground = np.clip(depth_map - foreground_threshold, 0, 1) / (1 - foreground_threshold)
        background = np.clip(midground_threshold - depth_map, 0, 1) / midground_threshold
        midground = 1.0 - foreground - background
        
        # Smooth transitions
        from scipy.ndimage import gaussian_filter
        foreground = gaussian_filter(foreground, sigma=5)
        background = gaussian_filter(background, sigma=5)
        midground = gaussian_filter(midground, sigma=5)
        
        # Normalize
        total = foreground + midground + background + 1e-8
        foreground = foreground / total
        midground = midground / total
        background = background / total
        
        return foreground, midground, background
    
    def apply_zone_adjustments(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply zone-based enhancements to image.
        
        Args:
            image: RGB image as float32 [0, 1]
            depth_map: Depth map as float32 [0, 1]
            
        Returns:
            Enhanced image as float32 [0, 1]
        """
        if not self.config.enable_zone_processing:
            return image
        
        logger.info("Applying zone-based adjustments...")
        
        # Create zone masks
        foreground, midground, background = self.create_zone_masks(depth_map)
        
        # Apply zone-specific enhancements
        enhanced = image.copy()
        
        # Foreground: Boost clarity and detail
        if self.config.foreground_boost != 1.0:
            # Enhance contrast slightly
            fg_adjustment = self.config.foreground_boost
            enhanced = enhanced * (1 - foreground[:, :, None]) + \
                      (enhanced ** (1/fg_adjustment)) * foreground[:, :, None]
        
        # Midground: Keep neutral
        # (no adjustment needed if balance = 1.0)
        
        # Background: Soften slightly (reduce clarity)
        if self.config.background_soften != 1.0:
            from scipy.ndimage import gaussian_filter
            bg_softened = np.stack([
                gaussian_filter(enhanced[:, :, c], sigma=1.5)
                for c in range(3)
            ], axis=-1)
            
            enhanced = enhanced * (1 - background[:, :, None]) + \
                      bg_softened * background[:, :, None]
        
        logger.info("✓ Zone adjustments applied")
        return np.clip(enhanced, 0, 1)
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full depth-aware processing pipeline.
        
        Args:
            image: Input image as float32 [0, 1]
            
        Returns:
            (enhanced_image, depth_map)
        """
        # Estimate depth
        depth_map = self.estimate_depth(image)
        
        if depth_map is None:
            logger.warning("Depth estimation failed - returning original image")
            return image, None
        
        # Apply zone-based adjustments
        enhanced = self.apply_zone_adjustments(image, depth_map)
        
        return enhanced, depth_map
    
    def save_depth_visualization(
        self,
        depth_map: np.ndarray,
        output_path: Path,
        colormap: str = "viridis"
    ):
        """Save depth map with colormap visualization."""
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(depth_map)[:, :, :3]  # RGB only
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((colored * 255).astype(np.uint8)).save(output_path)
        logger.info(f"Depth visualization saved: {output_path}")


def create_depth_processor(
    model_name: str = "depth_anything_v2",
    enable_zone_processing: bool = True,
    device: str = "auto"
) -> DepthProcessor:
    """
    Convenience function to create depth processor.
    
    Args:
        model_name: Depth model to use
        enable_zone_processing: Enable zone-based enhancements
        device: Device to use (auto, cpu, cuda, mps)
        
    Returns:
        Configured DepthProcessor
    """
    config = DepthConfig(
        model_name=model_name,
        enable_zone_processing=enable_zone_processing,
        device=device
    )
    return DepthProcessor(config)


if __name__ == "__main__":
    # Test depth processor
    logging.basicConfig(level=logging.INFO)
    
    processor = create_depth_processor()
    
    # Create test image
    test_image = np.random.rand(512, 512, 3).astype(np.float32)
    
    enhanced, depth_map = processor.process(test_image)
    
    if depth_map is not None:
        print(f"✓ Depth processing successful")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Enhanced shape: {enhanced.shape}")
    else:
        print("⚠️ Depth processing unavailable")
