#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tiff_enhancement_pipeline_v2.py
================================
Enhanced variant with adaptive segmentation strategy:
  - Semantic segmentation (Detectron2) for complex scenes
  - Material clustering (K-means) for aerial/simple scenes
  - Hybrid mode combining both approaches

New Features:
  --segmentation-mode [semantic|material|hybrid|auto]
  --material-clusters N (for K-means clustering)
  --material-textures PATH (JSON config for material rules)
  --aerial-mode (optimized preset for drone imagery)

Backward Compatible: Defaults to original semantic segmentation behavior.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("pipeline_v2")


@dataclass
class MaterialClusterConfig:
    """Configuration for material clustering."""
    n_clusters: int = 6
    seed: int = 42
    max_iter: int = 100
    texture_config: Optional[Path] = None
    enable_spatial_coherence: bool = False
    feature_mode: str = "rgb"  # rgb, rgb+texture, rgb+lbp


class MaterialClusterer:
    """K-means based material clustering for aerial imagery."""
    
    def __init__(self, config: MaterialClusterConfig):
        self.config = config
    
    def cluster(self, image_path: Path, output_dir: Path) -> Tuple[Path, Path]:
        """
        Cluster image into material regions.
        Returns paths to (material_labels.png, cluster_stats.json)
        """
        log.info(f"Material clustering: {image_path.name}")
        
        # Load image
        img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
        h, w, c = img.shape
        
        # Prepare features
        if self.config.feature_mode == "rgb":
            features = img.reshape(-1, c)
        else:
            # Extended feature extraction could go here
            features = img.reshape(-1, c)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.seed,
            max_iter=self.config.max_iter
        )
        labels = kmeans.fit_predict(features)
        label_map = labels.reshape(h, w)
        
        # Save label map
        base_name = image_path.stem
        labels_path = output_dir / f"{base_name}_material_labels.png"
        
        # Colorize labels for visualization
        colorized = self._colorize_labels(label_map, self.config.n_clusters)
        Image.fromarray((colorized * 255).astype(np.uint8)).save(labels_path)
        
        # Compute cluster statistics
        stats = self._compute_stats(labels, features, kmeans.cluster_centers_)
        stats_path = output_dir / f"{base_name}_cluster_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate material masks (similar to Detectron2 output format)
        self._generate_material_masks(label_map, output_dir, base_name)
        
        log.info(f"  Clustered into {self.config.n_clusters} materials")
        
        return labels_path, stats_path
    
    def _colorize_labels(self, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """Create colorized visualization of label map."""
        h, w = labels.shape
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ])
        
        colorized = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(min(n_clusters, len(colors))):
            mask = labels == i
            colorized[mask] = colors[i % len(colors)]
        
        return colorized
    
    def _compute_stats(self, labels: np.ndarray, features: np.ndarray, 
                      centroids: np.ndarray) -> Dict:
        """Compute cluster statistics."""
        stats = {
            "n_clusters": self.config.n_clusters,
            "clusters": []
        }
        
        for i in range(self.config.n_clusters):
            mask = labels == i
            cluster_features = features[mask]
            
            cluster_stat = {
                "id": int(i),
                "count": int(mask.sum()),
                "percentage": float(mask.sum() / len(labels) * 100),
                "centroid": centroids[i].tolist(),
                "std": cluster_features.std(axis=0).tolist() if len(cluster_features) > 0 else [0, 0, 0]
            }
            stats["clusters"].append(cluster_stat)
        
        return stats
    
    def _generate_material_masks(self, label_map: np.ndarray, 
                                 output_dir: Path, base_name: str):
        """Generate individual binary masks per material cluster."""
        for cluster_id in range(self.config.n_clusters):
            mask = (label_map == cluster_id).astype(np.uint8) * 255
            mask_path = output_dir / f"{base_name}_mask_material{cluster_id}.png"
            Image.fromarray(mask, mode='L').save(mask_path)


class AdaptiveSegmentationStage:
    """
    Stage 4 replacement with adaptive segmentation strategy.
    Chooses between semantic and material-based segmentation.
    """
    
    def __init__(self, config, mode: str = "semantic"):
        self.config = config
        self.mode = mode  # semantic, material, hybrid, auto
        
        if mode in ["material", "hybrid", "auto"]:
            self.material_clusterer = MaterialClusterer(
                MaterialClusterConfig(
                    n_clusters=config.material_clusters,
                    texture_config=config.material_textures
                )
            )
    
    def execute(self, enhanced_images_dir: Path, depth_maps_dir: Path, 
                output_dir: Path) -> Tuple[bool, int]:
        """Execute segmentation based on selected mode."""
        
        if self.mode == "semantic":
            return self._run_semantic_segmentation(
                enhanced_images_dir, depth_maps_dir, output_dir
            )
        elif self.mode == "material":
            return self._run_material_clustering(
                enhanced_images_dir, depth_maps_dir, output_dir
            )
        elif self.mode == "hybrid":
            return self._run_hybrid_segmentation(
                enhanced_images_dir, depth_maps_dir, output_dir
            )
        elif self.mode == "auto":
            return self._run_auto_segmentation(
                enhanced_images_dir, depth_maps_dir, output_dir
            )
    
    def _run_semantic_segmentation(self, enhanced_dir: Path, depth_dir: Path,
                                   output_dir: Path) -> Tuple[bool, int]:
        """Run Detectron2 panoptic segmentation (original behavior)."""
        log.info("Running semantic segmentation (Detectron2)...")
        
        script = Path(__file__).parent / "run_detectron2_panoptic_batch.py"
        
        cmd = [
            sys.executable, str(script),
            "--images-root", str(enhanced_dir),
            "--depths-root", str(depth_dir),
            "--mask-root", str(output_dir),
            "--device", self.config.device
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            masks = list(output_dir.glob("*_mask_*.png"))
            return True, len(masks)
        except subprocess.CalledProcessError as e:
            log.error(f"Semantic segmentation failed: {e}")
            return False, 0
    
    def _run_material_clustering(self, enhanced_dir: Path, depth_dir: Path,
                                output_dir: Path) -> Tuple[bool, int]:
        """Run K-means material clustering (fast aerial mode)."""
        log.info(f"Running material clustering (k={self.config.material_clusters})...")
        
        # Find enhanced images
        enhanced_images = list(enhanced_dir.glob("*.tif")) + \
                         list(enhanced_dir.glob("*.tiff")) + \
                         list(enhanced_dir.glob("*.jpg"))
        
        mask_count = 0
        for img_path in enhanced_images:
            try:
                labels_path, stats_path = self.material_clusterer.cluster(
                    img_path, output_dir
                )
                # Count generated masks
                base_name = img_path.stem
                masks = list(output_dir.glob(f"{base_name}_mask_material*.png"))
                mask_count += len(masks)
            except Exception as e:
                log.error(f"Clustering failed for {img_path.name}: {e}")
        
        return True, mask_count
    
    def _run_hybrid_segmentation(self, enhanced_dir: Path, depth_dir: Path,
                                output_dir: Path) -> Tuple[bool, int]:
        """Run both semantic and material segmentation."""
        log.info("Running hybrid segmentation (semantic + material)...")
        
        # Create subdirectories
        semantic_dir = output_dir / "semantic"
        material_dir = output_dir / "material"
        semantic_dir.mkdir(exist_ok=True)
        material_dir.mkdir(exist_ok=True)
        
        # Run both
        success_sem, count_sem = self._run_semantic_segmentation(
            enhanced_dir, depth_dir, semantic_dir
        )
        success_mat, count_mat = self._run_material_clustering(
            enhanced_dir, depth_dir, material_dir
        )
        
        # Copy all masks to main output directory
        for mask in semantic_dir.glob("*"):
            shutil.copy2(mask, output_dir / mask.name)
        for mask in material_dir.glob("*"):
            shutil.copy2(mask, output_dir / mask.name)
        
        return success_sem and success_mat, count_sem + count_mat
    
    def _run_auto_segmentation(self, enhanced_dir: Path, depth_dir: Path,
                              output_dir: Path) -> Tuple[bool, int]:
        """Automatically choose segmentation method based on image analysis."""
        log.info("Auto-detecting optimal segmentation method...")
        
        # Analyze first image to determine type
        enhanced_images = list(enhanced_dir.glob("*.tif")) + \
                         list(enhanced_dir.glob("*.tiff"))
        
        if not enhanced_images:
            log.error("No images found for analysis")
            return False, 0
        
        # Simple heuristic: check if image looks aerial
        img = np.array(Image.open(enhanced_images[0]).convert("RGB"))
        is_aerial = self._detect_aerial(img)
        
        if is_aerial:
            log.info("  Detected: Aerial imagery → using material clustering")
            return self._run_material_clustering(enhanced_dir, depth_dir, output_dir)
        else:
            log.info("  Detected: Ground-level imagery → using semantic segmentation")
            return self._run_semantic_segmentation(enhanced_dir, depth_dir, output_dir)
    
    def _detect_aerial(self, image: np.ndarray) -> bool:
        """Detect if image is aerial/drone photography."""
        h, w = image.shape[:2]
        
        # Check top region brightness (sky in ground photos)
        top_region = image[:h//5, :]
        top_brightness = top_region.mean()
        
        # Check for perspective (ground photos have convergence)
        # Simple proxy: variance in horizontal lines
        mid_section = image[h//3:2*h//3, :]
        row_means = mid_section.mean(axis=1)
        perspective_indicator = row_means.std()
        
        # Aerial: bright top region (likely sky) AND low perspective
        is_aerial = top_brightness > 180 and perspective_indicator < 15
        
        return is_aerial


# Integration with existing pipeline config
@dataclass 
class EnhancedPipelineConfig:
    """Extended configuration with material clustering options."""
    # ... (inherit all original fields)
    
    # New fields for material clustering
    segmentation_mode: str = "semantic"  # semantic, material, hybrid, auto
    material_clusters: int = 6
    material_textures: Optional[Path] = None
    aerial_mode: bool = False


def build_enhanced_parser() -> argparse.ArgumentParser:
    """Build enhanced CLI parser with material clustering options."""
    ap = argparse.ArgumentParser(
        description="Enhanced TIFF pipeline with adaptive segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Original arguments
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--preset", default="dramatic")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--depth-effects", nargs="+", default=["haze", "clarity"])
    
    # New segmentation options
    seg = ap.add_argument_group("Segmentation Options")
    seg.add_argument("--segmentation-mode", 
                     choices=["semantic", "material", "hybrid", "auto"],
                     default="semantic",
                     help="Segmentation strategy (semantic=Detectron2, material=K-means)")
    seg.add_argument("--material-clusters", type=int, default=6,
                     help="Number of material clusters for K-means")
    seg.add_argument("--material-textures", type=Path, default=None,
                     help="JSON config file for material texture rules")
    seg.add_argument("--aerial-mode", action="store_true",
                     help="Optimize for aerial/drone imagery (implies --segmentation-mode material)")
    
    return ap


if __name__ == "__main__":
    # This is a template/documentation file
    # Real implementation would need full integration with existing pipeline
    log.info("Enhanced pipeline v2 with adaptive segmentation")
    log.info("Usage: python tiff_enhancement_pipeline_v2.py --input-dir ... --segmentation-mode material")
