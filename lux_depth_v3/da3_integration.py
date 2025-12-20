"""
DA3 Integration for Luxury Real Estate Rendering Pipeline

This module provides a clean Python wrapper around Depth Anything 3 (DA3)
for integration into the Transformation Portal rendering pipeline.

Author: Transformation Portal Team
Date: 2025-12-19
"""

import subprocess
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Literal
from dataclasses import dataclass


@dataclass
class DA3Result:
    """Results from DA3 depth estimation."""
    success: bool
    output_dir: Path
    glb_path: Optional[Path] = None
    depth_vis_dir: Optional[Path] = None
    npz_path: Optional[Path] = None
    scene_jpg: Optional[Path] = None
    stdout: str = ""
    stderr: str = ""
    
    @property
    def depth_array(self) -> Optional[np.ndarray]:
        """Load depth array from NPZ file."""
        if self.npz_path and self.npz_path.exists():
            data = np.load(self.npz_path)
            return data.get('depth', None)
        return None
    
    @property
    def confidence_array(self) -> Optional[np.ndarray]:
        """Load confidence array from NPZ file."""
        if self.npz_path and self.npz_path.exists():
            data = np.load(self.npz_path)
            return data.get('conf', None)
        return None


class DA3DepthEstimator:
    """
    Wrapper for Depth Anything 3 in rendering pipeline.
    
    This class provides a simple Python interface to the DA3 CLI,
    handling subprocess calls, file management, and result parsing.
    
    Example:
        >>> estimator = DA3DepthEstimator()
        >>> result = estimator.process_image("render.jpg", "output/")
        >>> if result.success:
        >>>     depth = result.depth_array
        >>>     print(f"Depth shape: {depth.shape}")
    """
    
    AVAILABLE_MODELS = {
        "giant-1.1": "depth-anything/DA3-GIANT-1.1",
        "large-1.1": "depth-anything/DA3-LARGE-1.1",
        "base": "depth-anything/DA3-BASE",
        "small": "depth-anything/DA3-SMALL",
        "metric-large": "depth-anything/DA3METRIC-LARGE",
        "mono-large": "depth-anything/DA3MONO-LARGE",
        "nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    }
    
    def __init__(
        self,
        model: str = "large-1.1",
        device: str = "cpu",
        auto_cleanup: bool = True,
        verbose: bool = False
    ):
        """
        Initialize DA3 depth estimator.
        
        Args:
            model: Model name (see AVAILABLE_MODELS) or full HuggingFace path
            device: Device to use ('cpu', 'cuda', 'mps')
            auto_cleanup: Automatically clean export directories
            verbose: Print detailed output
        """
        if model in self.AVAILABLE_MODELS:
            self.model = self.AVAILABLE_MODELS[model]
        elif model.startswith("depth-anything/"):
            self.model = model
        else:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.device = device
        self.auto_cleanup = auto_cleanup
        self.verbose = verbose
        
        # Fix OpenMP duplicate library issue on Mac
        if os.environ.get('KMP_DUPLICATE_LIB_OK') != 'TRUE':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    def process_image(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        export_format: str = "glb-depth_vis-mini_npz",
        process_res: int = 504,
        **kwargs
    ) -> DA3Result:
        """
        Process single image with DA3.
        
        Args:
            input_path: Path to input image
            output_dir: Directory for output files
            export_format: Export format(s), separated by '-'
                          Options: glb, depth_vis, mini_npz, npz, feat_vis, gs_ply, gs_video
            process_res: Processing resolution
            **kwargs: Additional arguments passed to DA3 CLI
        
        Returns:
            DA3Result object with paths and status
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        if not input_path.exists():
            return DA3Result(
                success=False,
                output_dir=output_dir,
                stderr=f"Input file not found: {input_path}"
            )
        
        cmd = [
            "da3", "auto", str(input_path),
            "--export-dir", str(output_dir),
            "--export-format", export_format,
            "--model-dir", self.model,
            "--device", self.device,
            "--process-res", str(process_res),
        ]
        
        if self.auto_cleanup:
            cmd.append("--auto-cleanup")
        
        # Add any extra kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output files
        glb_path = output_dir / "scene.glb" if (output_dir / "scene.glb").exists() else None
        depth_vis_dir = output_dir / "depth_vis" if (output_dir / "depth_vis").exists() else None
        
        # NPZ can be in multiple locations depending on export format
        npz_path = None
        possible_npz_paths = [
            output_dir / "scene.npz",
            output_dir / "exports" / "mini_npz" / "results.npz",
            output_dir / "exports" / "npz" / "results.npz",
        ]
        for path in possible_npz_paths:
            if path.exists():
                npz_path = path
                break
        
        scene_jpg = output_dir / "scene.jpg" if (output_dir / "scene.jpg").exists() else None
        
        return DA3Result(
            success=result.returncode == 0,
            output_dir=output_dir,
            glb_path=glb_path,
            depth_vis_dir=depth_vis_dir,
            npz_path=npz_path,
            scene_jpg=scene_jpg,
            stdout=result.stdout,
            stderr=result.stderr
        )
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: List[str] = ["jpg", "png", "jpeg"],
        export_format: str = "glb-depth_vis-mini_npz",
        **kwargs
    ) -> DA3Result:
        """
        Batch process directory of images.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory for output files
            extensions: Image file extensions to process
            export_format: Export format(s)
            **kwargs: Additional arguments passed to DA3 CLI
        
        Returns:
            DA3Result object
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            return DA3Result(
                success=False,
                output_dir=output_dir,
                stderr=f"Input directory not found: {input_dir}"
            )
        
        cmd = [
            "da3", "images", str(input_dir),
            "--export-dir", str(output_dir),
            "--image-extensions", ",".join(extensions),
            "--export-format", export_format,
            "--model-dir", self.model,
            "--device", self.device,
        ]
        
        if self.auto_cleanup:
            cmd.append("--auto-cleanup")
        
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return DA3Result(
            success=result.returncode == 0,
            output_dir=output_dir,
            stdout=result.stdout,
            stderr=result.stderr
        )
    
    def process_video(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float = 1.0,
        export_format: str = "glb-depth_vis",
        **kwargs
    ) -> DA3Result:
        """
        Process video file, extracting frames at specified FPS.
        
        Args:
            input_path: Path to video file
            output_dir: Directory for output files
            fps: Frame extraction rate
            export_format: Export format(s)
            **kwargs: Additional arguments
        
        Returns:
            DA3Result object
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        if not input_path.exists():
            return DA3Result(
                success=False,
                output_dir=output_dir,
                stderr=f"Video file not found: {input_path}"
            )
        
        cmd = [
            "da3", "video", str(input_path),
            "--export-dir", str(output_dir),
            "--fps", str(fps),
            "--export-format", export_format,
            "--model-dir", self.model,
            "--device", self.device,
        ]
        
        if self.auto_cleanup:
            cmd.append("--auto-cleanup")
        
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return DA3Result(
            success=result.returncode == 0,
            output_dir=output_dir,
            stdout=result.stdout,
            stderr=result.stderr
        )


def convert_to_metric_depth(
    depth_array: np.ndarray,
    focal_length_px: float,
    model_type: Literal["metric", "relative"] = "metric"
) -> np.ndarray:
    """
    Convert DA3 depth output to metric depth in meters.
    
    For DA3METRIC models, the conversion formula is:
        metric_depth = focal_length_px * depth_output / 300.0
    
    For relative depth models, this returns the array unchanged
    (relative depth values).
    
    Args:
        depth_array: Depth array from DA3
        focal_length_px: Focal length in pixels (typically (fx + fy) / 2)
        model_type: 'metric' for DA3METRIC models, 'relative' for others
    
    Returns:
        Depth array in meters (for metric) or unchanged (for relative)
    """
    if model_type == "metric":
        return focal_length_px * depth_array / 300.0
    return depth_array


# Convenience function for quick usage
def estimate_depth(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    model: str = "large-1.1",
    device: str = "cpu"
) -> DA3Result:
    """
    Quick depth estimation helper function.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        model: Model name (default: 'large-1.1')
        device: Device ('cpu', 'cuda', 'mps')
    
    Returns:
        DA3Result object
    """
    estimator = DA3DepthEstimator(model=model, device=device)
    return estimator.process_image(image_path, output_dir)
