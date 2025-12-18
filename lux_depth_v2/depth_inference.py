#!/usr/bin/env python3
"""
High-Resolution Tiled Depth Inference with Edge-Preserving Fusion
===================================================================

Implements the #1 critical fix: tile-based inference at native model resolution
to avoid the "low-res inference + bicubic upscale" quality bottleneck.

Key Features:
- Overlapping tile inference (1024-1536px tiles, 128-256px overlap)
- Per-tile scale/shift reconciliation for seamless blending
- Cosine/Hann window blending to avoid tile seams
- Median-based ensemble fusion (edge-preserving, not smoothing)
- Edge alignment metrics for quality validation

Reference: User feedback 2025-12-17 - "tile-based high-resolution inference (the real unlock)"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TiledInferenceConfig:
    """Configuration for high-resolution tiled depth inference."""
    
    # Tile parameters
    tile_size: int = 1024  # Size of each tile (1024-1536 recommended)
    overlap: int = 128     # Overlap between tiles (128-256 recommended)
    
    # Model parameters
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    device: str = "auto"
    bypass_image_processor: bool = True  # CRITICAL: Bypass HF's 518px resize
    
    # Fusion parameters
    fusion_mode: str = "weighted"  # weighted | median | confidence
    blend_window: str = "hann"   # hann | cosine | linear
    
    # Scale reconciliation
    reconcile_scales: bool = True  # Align tile scales before blending
    reconcile_method: str = "robust"  # robust | mean_variance | none
    
    # Quality validation
    validate_edges: bool = True
    edge_alignment_threshold: float = 0.5  # Min correlation with RGB edges
    
    # Global anchor fusion
    use_global_anchor: bool = True
    global_anchor_config: Optional['GlobalAnchorConfig'] = None
    
    # Edge snapping
    use_edge_snapping: bool = True
    edge_snap_config: Optional['EdgeSnappingConfig'] = None
    
    # Production refinement (CLAHE + guided filter + edge snap)
    use_production_refinement: bool = True
    refinement_use_clahe: bool = True
    refinement_use_edge_filter: bool = True
    refinement_use_edge_snap: bool = True


class TiledDepthEstimator:
    """
    High-resolution depth estimator using overlapping tile inference.
    
    This is the critical fix for spatial fidelity - prevents the "smooth ramps 
    and soft boundaries" caused by low-res inference + bicubic upsampling.
    """
    
    def __init__(self, config: TiledInferenceConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for tiled inference")
        
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.image_processor = None
        
        # Initialize global fusion if enabled
        self.global_fusion = None
        if self.config.use_global_anchor:
            from .global_anchor import GlobalAnchorFusion, GlobalAnchorConfig
            if self.config.global_anchor_config is None:
                self.config.global_anchor_config = GlobalAnchorConfig()
            self.global_fusion = GlobalAnchorFusion(self.config.global_anchor_config)
        
        # Initialize edge snapping if enabled
        self.edge_snapper = None
        if self.config.use_edge_snapping:
            from .edge_snapping import EdgeSnapper, EdgeSnappingConfig
            if self.config.edge_snap_config is None:
                self.config.edge_snap_config = EdgeSnappingConfig()
            self.edge_snapper = EdgeSnapper(self.config.edge_snap_config)
        
        logger.info(f"TiledDepthEstimator initialized: tile={config.tile_size} overlap={config.overlap} device={self.device}")
        logger.info(f"  Global anchor: {self.config.use_global_anchor}, Edge snapping: {self.config.use_edge_snapping}")
    
    def _setup_device(self) -> torch.device:
        """Auto-detect or validate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _load_model(self):
        """Lazy load depth model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading depth model: {self.config.model_name}")
        
        if self.config.bypass_image_processor:
            # Load model directly to bypass HF's 518px resize
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            self.image_processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ Model loaded directly (bypassing HF pipeline's 518px resize)")
        else:
            # Use HF pipeline (will resize to 518px)
            self.model = pipeline(
                "depth-estimation",
                model=self.config.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            logger.warning("⚠️  Using HF pipeline - will resize to 518px internally!")
        
        logger.info("✓ Depth model loaded")
    
    def _compute_gradient_magnitude(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude for edge detection.
        
        Used to identify stable regions (low gradient) for scale reconciliation.
        """
        from scipy import ndimage
        
        # Sobel filters for gradient computation
        gx = ndimage.sobel(depth, axis=1)
        gy = ndimage.sobel(depth, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        return grad_mag
    
    def _validate_boundary_energy(
        self, 
        depth: np.ndarray,
        tile_coords: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        """
        Validate that tile boundaries don't have excessive gradient energy.
        
        High boundary energy indicates visible seam artifacts.
        """
        h, w = depth.shape
        
        # Create boundary mask (2-pixel band at tile boundaries)
        boundary_mask = self._create_boundary_mask(h, w, band_width=2)
        
        if boundary_mask.sum() == 0:
            return
        
        # Compute gradient energy
        grad_mag = self._compute_gradient_magnitude(depth)
        
        boundary_energy = grad_mag[boundary_mask].mean()
        interior_energy = grad_mag[~boundary_mask].mean()
        
        energy_ratio = boundary_energy / max(interior_energy, 1e-8)
        
        if energy_ratio > 1.2:
            logger.warning(
                f"⚠️  Seam artifacts detected: boundary energy {energy_ratio:.2f}x interior "
                f"(boundary={boundary_energy:.4f}, interior={interior_energy:.4f})"
            )
        else:
            logger.debug(
                f"✓ Boundary energy OK: {energy_ratio:.2f}x interior "
                f"(boundary={boundary_energy:.4f}, interior={interior_energy:.4f})"
            )
    
    def _create_boundary_mask(self, h: int, w: int, band_width: int = 2) -> np.ndarray:
        """
        Create mask for tile boundary regions.
        
        Marks pixels near tile boundaries for quality validation.
        """
        mask = np.zeros((h, w), dtype=bool)
        
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        stride = tile_size - overlap
        
        # Mark vertical boundaries
        x = overlap
        while x < w:
            x_start = max(0, x - band_width)
            x_end = min(w, x + band_width)
            mask[:, x_start:x_end] = True
            x += stride
        
        # Mark horizontal boundaries
        y = overlap
        while y < h:
            y_start = max(0, y - band_width)
            y_end = min(h, y + band_width)
            mask[y_start:y_end, :] = True
            y += stride
        
        return mask
    
    def _make_blend_window(self, tile_size: int, overlap: int) -> np.ndarray:
        """Create smooth blending window for overlap regions."""
        window = np.ones((tile_size, tile_size), dtype=np.float32)
        
        if overlap == 0:
            return window
        
        # Create 1D ramp using Hann window
        if self.config.blend_window == "hann":
            ramp = np.hanning(2 * overlap)[:overlap]
        elif self.config.blend_window == "cosine":
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
        else:  # linear
            ramp = np.linspace(0, 1, overlap)
        
        # Apply ramp to edges
        window[:overlap, :] *= ramp[:, None]  # Top
        window[-overlap:, :] *= ramp[::-1, None]  # Bottom
        window[:, :overlap] *= ramp[None, :]  # Left
        window[:, -overlap:] *= ramp[::-1][None, :]  # Right
        
        return window
    
    def _reconcile_tile_scale(
        self, 
        tile_depth: np.ndarray, 
        reference_overlap: Optional[np.ndarray],
        overlap_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Reconcile tile scale/shift to match reference overlap region.
        
        Prevents discontinuities when tiles disagree on absolute depth values.
        Uses robust linear fit (RANSAC-like) to handle outliers.
        """
        if not self.config.reconcile_scales or reference_overlap is None:
            return tile_depth
        
        if overlap_mask is None or overlap_mask.sum() < 100:
            return tile_depth
        
        # Extract overlap pixels
        tile_overlap = tile_depth[overlap_mask]
        ref_overlap = reference_overlap[overlap_mask]
        
        if self.config.reconcile_method == "robust":
            # Robust linear fit (resist outliers)
            # Estimate a*tile + b = ref using Theil-Sen estimator
            try:
                from sklearn.linear_model import TheilSenRegressor
                model = TheilSenRegressor(random_state=42)
                model.fit(tile_overlap.reshape(-1, 1), ref_overlap)
                a, b = model.coef_[0], model.intercept_
            except ImportError:
                # Fallback: simple percentile matching
                a = np.percentile(ref_overlap, 75) - np.percentile(ref_overlap, 25)
                a /= max(np.percentile(tile_overlap, 75) - np.percentile(tile_overlap, 25), 1e-6)
                b = np.median(ref_overlap) - a * np.median(tile_overlap)
        
        elif self.config.reconcile_method == "mean_variance":
            # Match mean and variance
            ref_mean, ref_std = ref_overlap.mean(), ref_overlap.std()
            tile_mean, tile_std = tile_overlap.mean(), tile_overlap.std()
            a = ref_std / max(tile_std, 1e-6)
            b = ref_mean - a * tile_mean
        
        else:  # none
            return tile_depth
        
        # Apply transform
        return a * tile_depth + b
    
    def _extract_tiles(
        self, 
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int, int, int]]:
        """
        Extract overlapping tiles from image.
        
        Returns: List of (tile, y_start, y_end, x_start, x_end)
        """
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        stride = tile_size - overlap
        
        tiles = []
        
        # Calculate tile grid
        y_starts = list(range(0, h - overlap, stride))
        x_starts = list(range(0, w - overlap, stride))
        
        # Ensure we cover the entire image
        if y_starts[-1] + tile_size < h:
            y_starts.append(h - tile_size)
        if x_starts[-1] + tile_size < w:
            x_starts.append(w - tile_size)
        
        for y in y_starts:
            for x in x_starts:
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append((tile, y_start, y_end, x_start, x_end))
        
        logger.info(f"Extracted {len(tiles)} tiles from {h}x{w} image")
        return tiles
    
    def _infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """
        Run depth inference on a single tile.
        
        Key: Process at model's native resolution to maximize spatial fidelity.
        """
        from PIL import Image
        
        # Convert to PIL
        if tile_rgb.dtype == np.float32:
            tile_pil = Image.fromarray((tile_rgb * 255).astype(np.uint8))
        else:
            tile_pil = Image.fromarray(tile_rgb)
        
        if self.config.bypass_image_processor:
            # Direct model inference - NO RESIZE
            # Prepare inputs at native resolution
            inputs = self.image_processor(
                images=tile_pil, 
                return_tensors="pt",
                do_resize=False  # CRITICAL: Disable resize
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # INSTRUMENTATION: Log input tensor shape (CRITICAL for validation)
            H_in, W_in = inputs["pixel_values"].shape[-2:]
            logger.info(f"🔍 Tile inference: tile_rgb={tile_rgb.shape}, PIL={tile_pil.size}, pixel_values={H_in}×{W_in}")
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract depth from model outputs
            if hasattr(outputs, 'predicted_depth'):
                depth_tensor = outputs.predicted_depth
            elif hasattr(outputs, 'depth'):
                depth_tensor = outputs.depth
            else:
                depth_tensor = outputs[0]
            
            # INSTRUMENTATION: Log output tensor shape
            H_out, W_out = depth_tensor.shape[-2:]
            logger.info(f"🔍 Tile output: predicted_depth={H_out}×{W_out}")
            
            # Convert to numpy
            depth = depth_tensor.squeeze().cpu().numpy()
            
            # CRITICAL FIX: Resize depth to match tile size
            # Model may output slightly smaller spatial dimensions (e.g., 1016 from 1024 input)
            target_h, target_w = tile_rgb.shape[:2]
            if depth.shape != (target_h, target_w):
                import cv2
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                logger.info(f"⚠️  Resized tile depth: {depth_tensor.shape[-2:]} → {depth.shape}")
            
        else:
            # Use pipeline (will resize internally to 518px)
            result = self.model(tile_pil)
            
            # Extract depth
            if hasattr(result, "depth"):
                depth = np.array(result.depth)
            elif isinstance(result, dict) and "depth" in result:
                depth = np.array(result["depth"])
            elif hasattr(result, "predicted_depth"):
                depth = np.array(result.predicted_depth)
            else:
                depth = np.array(result)
        
        # Ensure single channel
        if depth.ndim == 3:
            depth = depth[..., 0]
        
        # Resize to match tile size if needed
        if depth.shape != tile_rgb.shape[:2]:
            from PIL import Image as PILImage
            depth = np.array(
                PILImage.fromarray(depth).resize(
                    (tile_rgb.shape[1], tile_rgb.shape[0]), 
                    PILImage.LANCZOS
                )
            )
        
        # Normalize
        depth = depth.astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        
        return depth
    
    def _blend_tiles(
        self, 
        tile_depths: List[Tuple[np.ndarray, int, int, int, int]],
        output_shape: Tuple[int, int],
        global_anchor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Blend overlapping tiles with smooth windowing and scale reconciliation.
        
        Args:
            tile_depths: List of (tile_depth, y0, y1, x0, x1) tuples
            output_shape: Target output shape (h, w)
            global_anchor: Optional global depth map for scale reconciliation
        
        Returns:
            Blended depth map with seamless tile transitions
        """
        h, w = output_shape
        
        # Scale reconciliation: match each tile to its predecessors in overlap regions
        if self.config.reconcile_scales and global_anchor is not None:
            logger.info("Applying per-tile scale reconciliation...")
            
            # Build accumulated depth map as we go (first tile sets the scale)
            h, w = output_shape
            reference_depth = np.copy(global_anchor)  # Start with global as reference
            reconciled_tiles = []
            
            for idx, (tile_depth, y0, y1, x0, x1) in enumerate(tile_depths):
                if idx == 0:
                    # First tile: reconcile to global anchor
                    reference_region = global_anchor[y0:y1, x0:x1]
                else:
                    # Subsequent tiles: reconcile to accumulated depth in overlap region
                    reference_region = reference_depth[y0:y1, x0:x1]
                
                # Ensure shapes match
                if reference_region.shape != tile_depth.shape:
                    logger.warning(f"Tile {idx}: shape mismatch, skipping reconciliation")
                    reconciled_tiles.append((tile_depth, y0, y1, x0, x1))
                    continue
                
                # Compute gradient magnitude for robust pixel selection
                grad_mag = self._compute_gradient_magnitude(reference_region)
                
                # Create overlap mask (only use the overlap regions with existing tiles)
                th, tw = tile_depth.shape
                overlap = self.config.overlap
                overlap_mask = np.zeros((th, tw), dtype=bool)
                
                # Left overlap
                if x0 > 0:
                    overlap_mask[:, :overlap] = True
                # Top overlap  
                if y0 > 0:
                    overlap_mask[:overlap, :] = True
                
                # Exclude high-gradient pixels from overlap
                stable_threshold = np.percentile(grad_mag[overlap_mask], 80) if overlap_mask.sum() > 0 else np.inf
                stable_mask = overlap_mask & (grad_mag < stable_threshold)
                
                if stable_mask.sum() < 100:
                    # Fallback: use all overlap pixels
                    logger.debug(f"Tile {idx}: insufficient stable pixels, using all overlap")
                    stable_mask = overlap_mask
                
                if stable_mask.sum() < 50:
                    # No overlap or insufficient pixels: use global anchor for full tile
                    logger.debug(f"Tile {idx}: no overlap, using global anchor")
                    stable_mask = np.ones((th, tw), dtype=bool)
                
                # Robust affine fit: a * tile + b ≈ reference
                tile_pixels = tile_depth[stable_mask].flatten()
                ref_pixels = reference_region[stable_mask].flatten()
                
                if len(tile_pixels) > 10:
                    # Percentile-based robust fit
                    tile_p25, tile_p75 = np.percentile(tile_pixels, [25, 75])
                    ref_p25, ref_p75 = np.percentile(ref_pixels, [25, 75])
                    
                    tile_iqr = max(tile_p75 - tile_p25, 1e-6)
                    ref_iqr = ref_p75 - ref_p25
                    
                    a = ref_iqr / tile_iqr
                    b = ref_p25 - a * tile_p25
                    
                    # Clamp scale to reasonable range
                    a = np.clip(a, 0.7, 1.4)  # Tighter bounds for tile-to-tile matching
                    b = np.clip(b, -0.3, 0.3)
                    
                    # Apply calibration
                    tile_depth_calibrated = a * tile_depth + b
                    tile_depth_calibrated = np.clip(tile_depth_calibrated, 0.0, 1.0)
                    logger.debug(f"Tile {idx}: scale={a:.3f}, shift={b:.3f}, stable_px={stable_mask.sum()}")
                else:
                    tile_depth_calibrated = tile_depth
                    logger.debug(f"Tile {idx}: skipped (insufficient pixels)")
                
                reconciled_tiles.append((tile_depth_calibrated, y0, y1, x0, x1))
                
                # Update reference depth with this tile (for next tile's overlap)
                if idx < len(tile_depths) - 1:  # Don't update after last tile
                    reference_depth[y0:y1, x0:x1] = tile_depth_calibrated
            
            tile_depths = reconciled_tiles
            logger.info(f"✓ Scale reconciliation complete for {len(tile_depths)} tiles")
        
        # Blending phase
        if self.config.fusion_mode == "median":
            # Median fusion - most robust, preserves discontinuities
            depth_stack = np.zeros((len(tile_depths), h, w), dtype=np.float32)
            weight_stack = np.zeros((len(tile_depths), h, w), dtype=np.float32)
            
            blend_window = self._make_blend_window(self.config.tile_size, self.config.overlap)
            
            for idx, (tile_depth, y0, y1, x0, x1) in enumerate(tile_depths):
                th, tw = tile_depth.shape
                window = blend_window[:th, :tw]
                
                depth_stack[idx, y0:y1, x0:x1] = tile_depth
                weight_stack[idx, y0:y1, x0:x1] = window
            
            # Weighted median (use weights as sampling probability)
            # Simplified: compute median where weight > 0.5
            depth_final = np.median(depth_stack, axis=0)
            
        else:  # weighted average
            depth_accum = np.zeros((h, w), dtype=np.float32)
            weight_accum = np.zeros((h, w), dtype=np.float32)
            
            blend_window = self._make_blend_window(self.config.tile_size, self.config.overlap)
            
            for tile_depth, y0, y1, x0, x1 in tile_depths:
                th, tw = tile_depth.shape
                window = blend_window[:th, :tw]
                
                depth_accum[y0:y1, x0:x1] += tile_depth * window
                weight_accum[y0:y1, x0:x1] += window
            
            # Normalize
            depth_final = depth_accum / np.maximum(weight_accum, 1e-8)
        
        # Boundary energy check (validation metric)
        if self.config.validate_edges:
            self._validate_boundary_energy(depth_final, tile_depths[0][1:5] if tile_depths else None)
        
        return depth_final
    
    def _resize_for_global_pass(self, rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image for global anchor pass (low-res inference for context).
        
        Returns:
            Tuple of (resized_rgb, scale_factor)
        """
        from PIL import Image
        
        h, w = rgb.shape[:2]
        max_size = 768  # Low-res for global context
        
        if max(h, w) <= max_size:
            return rgb, 1.0
        
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_pil = Image.fromarray(rgb_uint8)
        rgb_resized = rgb_pil.resize((new_w, new_h), Image.LANCZOS)
        
        return np.array(rgb_resized), scale
    
    def _upsample_global_depth(self, depth_lowres: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Upsample low-res global anchor depth to target shape.
        
        Uses bicubic interpolation for smooth upsampling.
        """
        from PIL import Image
        
        h, w = target_shape
        depth_pil = Image.fromarray(depth_lowres)
        depth_upsampled = depth_pil.resize((w, h), Image.BICUBIC)
        
        return np.array(depth_upsampled).astype(np.float32)
    
    def estimate_depth(self, image: np.ndarray, rgb_for_fusion: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate high-resolution depth using tiled inference.
        
        Args:
            image: RGB image as uint8 or float32
            rgb_for_fusion: Optional RGB for global anchor and edge snapping (defaults to image)
            
        Returns:
            Depth map as float32 [0, 1] with maximum spatial fidelity
        """
        self._load_model()
        
        if rgb_for_fusion is None:
            rgb_for_fusion = image
        
        h, w = image.shape[:2]
        logger.info(f"Starting tiled depth inference on {h}x{w} image...")
        
        # Step 1: Global anchor pass (REQUIRED for scale reconciliation)
        global_depth = None
        if self.config.reconcile_scales or (self.config.use_global_anchor and self.global_fusion is not None):
            logger.info("Running global anchor pass (required for scale reconciliation)...")
            rgb_global, scale = self._resize_for_global_pass(rgb_for_fusion)
            
            # Run single low-res inference
            global_depth_lowres = self._infer_single_image(rgb_global)
            
            # Upsample to target size
            global_depth = self._upsample_global_depth(global_depth_lowres, (h, w))
            logger.info(f"✓ Global anchor depth: {global_depth.shape}")
        
        # Step 2: Tiled high-res inference
        # Extract tiles
        tiles = self._extract_tiles(image)
        
        # Infer depth for each tile
        tile_depths = []
        for idx, (tile_rgb, y0, y1, x0, x1) in enumerate(tiles):
            logger.debug(f"Processing tile {idx+1}/{len(tiles)}: ({y0}:{y1}, {x0}:{x1})")
            tile_depth = self._infer_tile(tile_rgb)
            tile_depths.append((tile_depth, y0, y1, x0, x1))
        
        # Blend tiles with scale reconciliation
        logger.info(f"Blending {len(tile_depths)} tiles using {self.config.fusion_mode} fusion...")
        depth = self._blend_tiles(tile_depths, (h, w), global_anchor=global_depth)
        
        # Step 3: Fuse with global anchor (if enabled and not already used for reconciliation)
        if self.config.use_global_anchor and global_depth is not None and self.global_fusion is not None:
            logger.info("Fusing tiled depth with global anchor...")
            depth = self.global_fusion.fuse(global_depth, depth, rgb_for_fusion)
            logger.info("✓ Global anchor fusion complete")
        
        # Step 4: Edge snapping (if enabled)
        # CRITICAL FIX: Prevent double application of edge snapping
        if self.config.use_edge_snapping and self.config.use_production_refinement and self.config.refinement_use_edge_snap:
            logger.warning(
                "⚠️  Both use_edge_snapping and production refinement edge_snap are enabled. "
                "This would apply edge snapping TWICE, causing over-sharpening artifacts. "
                "Disabling standalone edge snapping in favor of production refinement."
            )
            # Disable standalone edge snapping to avoid double application
            apply_standalone_edge_snap = False
        else:
            apply_standalone_edge_snap = self.config.use_edge_snapping
        
        if apply_standalone_edge_snap and self.edge_snapper is not None:
            logger.info("Applying edge snapping...")
            depth = self.edge_snapper.snap(depth, rgb_for_fusion)
            logger.info("✓ Edge snapping complete")
        
        # Step 5: Production refinement (CLAHE + guided filter + edge snap)
        if self.config.use_production_refinement:
            logger.info("Applying production refinement (CLAHE + guided filter + edge snap)...")
            from .depth_refinement import refine_depth_production
            
            depth = refine_depth_production(
                depth,
                rgb=rgb_for_fusion,
                use_clahe=self.config.refinement_use_clahe,
                use_edge_filter=self.config.refinement_use_edge_filter,
                use_edge_snap=self.config.refinement_use_edge_snap
            )
            logger.info("✓ Production refinement complete")
        
        logger.info(f"✓ Tiled depth inference complete: {depth.shape}, unique={len(np.unique(depth))}")
        return depth
    
    def _infer_single_image(self, image: np.ndarray) -> np.ndarray:
        """Infer depth for a single image (used for global anchor pass)."""
        from PIL import Image
        
        if image.dtype == np.float32:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image)
        
        if self.config.bypass_image_processor:
            inputs = self.image_processor(
                images=image_pil, 
                return_tensors="pt",
                do_resize=True  # Allow resize for global pass (low-res is fine)
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if hasattr(outputs, 'predicted_depth'):
                depth_tensor = outputs.predicted_depth
            else:
                depth_tensor = outputs[0]
            
            depth = depth_tensor.squeeze().cpu().numpy()
        else:
            result = self.model(image_pil)
            if hasattr(result, "depth"):
                depth = np.array(result.depth)
            else:
                depth = np.array(result)
        
        if depth.ndim == 3:
            depth = depth[..., 0]
        
        depth = depth.astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        
        return depth
    
    def compute_edge_alignment(self, rgb: np.ndarray, depth: np.ndarray) -> float:
        """
        Compute edge alignment score: correlation between RGB edges and depth edges.
        
        This is the correct quality metric (not edge gradient magnitude).
        """
        import cv2
        
        # RGB edges (Canny)
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        rgb_edges = cv2.Canny(rgb_gray, 50, 150).astype(np.float32) / 255.0
        
        # Depth edges (Sobel magnitude)
        depth_uint8 = (depth * 255).astype(np.uint8)
        sobel_x = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        depth_edges = depth_edges / (depth_edges.max() + 1e-8)
        
        # Correlation
        correlation = np.corrcoef(rgb_edges.ravel(), depth_edges.ravel())[0, 1]
        
        logger.info(f"Edge alignment score: {correlation:.3f}")
        return correlation


def create_tiled_estimator(
    tile_size: int = 1024,
    overlap: int = 128,
    fusion_mode: str = "median",
    device: str = "auto",
    use_global_anchor: bool = True,
    use_edge_snapping: bool = True
) -> TiledDepthEstimator:
    """
    Convenience factory for tiled depth estimator with all enhancements enabled.
    
    Args:
        tile_size: Size of each tile (1024-1536 recommended)
        overlap: Overlap between tiles (128-256 recommended)
        fusion_mode: Tile blending mode (median | weighted)
        device: Device for inference (auto | cuda | mps | cpu)
        use_global_anchor: Enable global context preservation
        use_edge_snapping: Enable edge sharpening
        
    Returns:
        Configured TiledDepthEstimator
    """
    from .global_anchor import GlobalAnchorConfig
    from .edge_snapping import EdgeSnappingConfig
    
    config = TiledInferenceConfig(
        tile_size=tile_size,
        overlap=overlap,
        fusion_mode=fusion_mode,
        device=device,
        bypass_image_processor=True,  # CRITICAL: Always bypass 518px resize
        use_global_anchor=use_global_anchor,
        global_anchor_config=GlobalAnchorConfig() if use_global_anchor else None,
        use_edge_snapping=use_edge_snapping,
        edge_snap_config=EdgeSnappingConfig() if use_edge_snapping else None
    )
    return TiledDepthEstimator(config)
