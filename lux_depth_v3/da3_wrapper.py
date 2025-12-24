"""Depth Anything 3 API wrapper.

This module provides three integration modes:
1. Official DA3 Python API (depth_anything_3.api.DepthAnything3)
2. Official DA3 CLI wrapper (da3 command)
3. Fallback placeholder (for testing when DA3 not available)

The Python API mode is recommended for most use cases as it provides full access
to all DA3 features including Gaussian Splatting, pose estimation, and feature extraction.
"""

from __future__ import annotations

import atexit
import json
import logging
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class DA3Error(RuntimeError):
    """Base exception for DA3 wrapper errors."""


class DA3NotInstalledError(DA3Error):
    """Raised when the DA3 CLI is not available on PATH."""


class DA3TimeoutError(DA3Error):
    """Raised when DA3 inference exceeds the allowed timeout."""


class DA3CommandError(DA3Error):
    """Raised when a DA3 CLI invocation fails."""


def _tail(text: str, max_chars: int = 4000) -> str:
    """Return the last max_chars of text for error messages."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[-max_chars:]


def _run_subprocess(
    cmd: List[str],
    *,
    timeout_s: Optional[int] = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command with consistent logging, timeout and output capture."""
    logger.debug("Running: %s", shlex.join(cmd))
    start = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        raise DA3TimeoutError(
            f"Command timed out after {timeout_s}s (elapsed={elapsed:.2f}s): {shlex.join(cmd)}"
        ) from e
    setattr(cp, "_tp_elapsed_s", time.perf_counter() - start)
    return cp


def check_da3_cli_available() -> bool:
    """Check if da3 CLI is available in PATH."""
    return shutil.which("da3") is not None


@dataclass
class DA3Prediction:
    """Wrapper for DA3 prediction results.
    
    Provides unified interface for depth predictions from both Python API
    and CLI modes.
    """
    
    # Core outputs
    depth: np.ndarray              # (N, H, W) or (H, W)
    conf: Optional[np.ndarray] = None     # (N, H, W) confidence maps
    
    # Camera parameters
    extrinsics: Optional[np.ndarray] = None  # (N, 4, 4) camera extrinsics
    intrinsics: Optional[np.ndarray] = None  # (N, 3, 3) camera intrinsics
    
    # Additional outputs
    processed_images: Optional[np.ndarray] = None  # (N, H, W, 3)
    aux: Optional[Dict[str, Any]] = None           # Auxiliary outputs (GS, features, etc.)
    
    def __post_init__(self):
        """Validate shapes."""
        if self.depth.ndim not in [2, 3]:
            raise ValueError(f"Depth must be 2D or 3D, got shape {self.depth.shape}")
        
        if self.conf is not None and self.conf.shape != self.depth.shape:
            raise ValueError(
                f"Confidence shape {self.conf.shape} != depth shape {self.depth.shape}"
            )


class DA3Backend:
    """Manages DA3 backend service lifecycle.

    The backend service keeps the model loaded in GPU memory, providing
    10-20x speedup for batch processing by avoiding model reload overhead.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        port: int = 8008,
        host: str = "127.0.0.1",
        log_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize backend manager.

        Args:
            model_dir: Path to DA3 model directory
            device: Device to use (cuda, mps, cpu)
            port: Port for backend service
            host: Host address for backend
            log_path: Optional path to log file for backend output (avoids PIPE backpressure)
        """
        self.model_dir = model_dir
        self.device = device
        self.port = port
        self.host = host
        self.log_path: Optional[Path] = Path(log_path).expanduser().resolve() if log_path else None
        self._log_fh = None
        self._process: Optional[subprocess.Popen] = None

        # Register cleanup on exit
        atexit.register(self.stop)

    def start(self, timeout: int = 30) -> None:
        """Start backend service if not running.

        Args:
            timeout: Seconds to wait for service to start

        Raises:
            DA3NotInstalledError: If DA3 CLI is not available
            RuntimeError: If backend fails to start
        """
        if self.is_running():
            logger.info("DA3 backend already running at %s", self.get_url())
            return

        if not Path(self.model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if not check_da3_cli_available():
            raise DA3NotInstalledError(
                "DA3 CLI not found. Install from: "
                "https://github.com/DepthAnything/Depth-Anything-V3"
            )

        # Start backend process
        cmd = [
            "da3", "backend",
            "--model-dir", self.model_dir,
            "--device", self.device,
            "--port", str(self.port),
            "--host", self.host,
        ]

        logger.info("Starting DA3 backend: %s", shlex.join(cmd))

        # Deadlock-safe backend launch: avoid PIPE backpressure
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
            stdout_target = self._log_fh
            stderr_target = subprocess.STDOUT

        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
        )

        # Wait for service to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info("DA3 backend started at %s", self.get_url())
                return
            time.sleep(0.5)

        # Timeout - kill process and raise
        self.stop()
        raise RuntimeError(f"Backend failed to start within {timeout}s")

    def stop(self) -> None:
        """Stop backend service."""
        if self._process is not None:
            logger.info("Stopping DA3 backend...")
            try:
                self._process.send_signal(signal.SIGTERM)
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                finally:
                    self._log_fh = None
            logger.info("DA3 backend stopped")

    def is_running(self) -> bool:
        """Check if backend is running and healthy."""
        try:
            response = requests.get(f"{self.get_url()}/status", timeout=1)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            return False

    def get_url(self) -> str:
        """Get backend URL."""
        return f"http://{self.host}:{self.port}"


class DA3CLI:
    """Official DA3 CLI wrapper.

    Provides Python interface to the da3 command-line tool with support
    for backend service acceleration.
    """

    def __init__(self, backend: Optional[DA3Backend] = None):
        """Initialize CLI wrapper.

        Args:
            backend: Optional backend service for acceleration

        Raises:
            DA3NotInstalledError: If DA3 CLI is not available
        """
        if not check_da3_cli_available():
            raise DA3NotInstalledError(
                "DA3 CLI not found. Install from: "
                "https://github.com/DepthAnything/Depth-Anything-V3"
            )

        self.backend = backend

    def _build_base_cmd(self, subcommand: str, **kwargs) -> List[str]:
        """Build base command with common options.

        Args:
            subcommand: DA3 subcommand (auto, image, images, video, colmap)
            **kwargs: Additional CLI arguments

        Returns:
            Command list
        """
        cmd = ["da3", subcommand]

        # Add backend URL if backend is provided
        if self.backend is not None:
            cmd.extend(["--use-backend", self.backend.get_url()])

        # Add common options from kwargs
        for key, value in kwargs.items():
            if value is not None:
                # Convert snake_case to kebab-case
                flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        return cmd

    def _run_command(
        self,
        cmd: List[str],
        capture_output: bool = True,
        timeout_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run CLI command and parse output.

        Args:
            cmd: Command list
            capture_output: Whether to capture stdout/stderr
            timeout_s: Optional timeout in seconds

        Returns:
            Result dictionary with output paths and metadata

        Raises:
            DA3CommandError: If command fails
            DA3TimeoutError: If command times out
        """
        result = _run_subprocess(cmd, timeout_s=timeout_s, capture_output=capture_output)
        runtime_s = getattr(result, "_tp_elapsed_s", None)

        if result.returncode != 0:
            raise DA3CommandError(
                f"DA3 CLI command failed (returncode={result.returncode}).\n"
                f"Command: {shlex.join(cmd)}\n"
                f"stderr (tail):\n{_tail(result.stderr)}"
            )

        # Parse output - DA3 CLI typically outputs JSON or file paths
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "runtime_s": runtime_s,
            "command": cmd,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def process_auto(
        self,
        input_path: Path,
        export_dir: Path,
        export_format: str = "mini_npz",
        timeout_s: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Auto-detect input type and process.

        Args:
            input_path: Input file or directory
            export_dir: Output directory
            export_format: Export format (mini_npz, glb, etc.)
            timeout_s: Optional timeout in seconds
            **kwargs: Additional CLI arguments

        Returns:
            Processing result
        """
        cmd = self._build_base_cmd(
            "auto",
            input_path=str(input_path),
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs
        )
        return self._run_command(cmd, timeout_s=timeout_s)

    def process_image(
        self,
        image_path: Path,
        export_dir: Path,
        export_format: str = "mini_npz",
        timeout_s: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process single image.

        Args:
            image_path: Input image path
            export_dir: Output directory
            export_format: Export format
            timeout_s: Optional timeout in seconds
            **kwargs: Additional CLI arguments

        Returns:
            Processing result
        """
        cmd = self._build_base_cmd(
            "image",
            image_path=str(image_path),
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs
        )
        return self._run_command(cmd, timeout_s=timeout_s)

    def process_images(
        self,
        images_dir: Path,
        export_dir: Path,
        export_format: str = "mini_npz",
        pattern: str = "*.jpg",
        timeout_s: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process image directory.

        Args:
            images_dir: Input directory
            export_dir: Output directory
            export_format: Export format
            pattern: File pattern
            timeout_s: Optional timeout in seconds
            **kwargs: Additional CLI arguments

        Returns:
            Processing result
        """
        cmd = self._build_base_cmd(
            "images",
            images_dir=str(images_dir),
            export_dir=str(export_dir),
            export_format=export_format,
            pattern=pattern,
            **kwargs
        )
        return self._run_command(cmd, timeout_s=timeout_s)

    def process_video(
        self,
        video_path: Path,
        export_dir: Path,
        fps: float = 1.0,
        export_format: str = "mini_npz",
        timeout_s: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process video with frame extraction.

        Args:
            video_path: Input video path
            export_dir: Output directory
            fps: Frame extraction rate
            export_format: Export format
            timeout_s: Optional timeout in seconds
            **kwargs: Additional CLI arguments

        Returns:
            Processing result
        """
        cmd = self._build_base_cmd(
            "video",
            video_path=str(video_path),
            export_dir=str(export_dir),
            fps=str(fps),
            export_format=export_format,
            **kwargs
        )
        return self._run_command(cmd, timeout_s=timeout_s)

    def process_colmap(
        self,
        colmap_dir: Path,
        export_dir: Path,
        export_format: str = "mini_npz-glb",
        timeout_s: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process COLMAP dataset.

        Args:
            colmap_dir: COLMAP dataset directory
            export_dir: Output directory
            export_format: Export format (supports hyphen-separated combinations)
            timeout_s: Optional timeout in seconds
            **kwargs: Additional CLI arguments

        Returns:
            Processing result
        """
        cmd = self._build_base_cmd(
            "colmap",
            colmap_dir=str(colmap_dir),
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs
        )
        return self._run_command(cmd, timeout_s=timeout_s)


class DepthAnything3Wrapper:
    """
    Wrapper for official DepthAnything3 Python API.
    
    Provides Pythonic interface to DA3 models with full feature support:
    - Monocular and multi-view depth estimation
    - Pose-conditioned depth estimation
    - Gaussian Splatting (3DGS)
    - Feature extraction from intermediate layers
    - Multiple export formats (NPZ, GLB, PLY, videos)
    - Ray-based pose estimation
    - Reference view selection strategies
    
    Example:
        >>> wrapper = DepthAnything3Wrapper(model_name="da3-large")
        >>> prediction = wrapper.inference(
        ...     image=["/path/to/image.jpg"],
        ...     export_dir="output",
        ...     export_format="mini_npz-glb"
        ... )
        >>> print(prediction.depth.shape)
    """
    
    # Available model names in official DA3 API
    AVAILABLE_MODELS = {
        "da3-giant": {
            "hf_id": "depth-anything/DA3-GIANT",
            "gs_capable": False,
            "description": "1.15B params, any-view with GS support",
        },
        "da3-large": {
            "hf_id": "depth-anything/DA3-LARGE",
            "gs_capable": False,
            "description": "0.35B params, recommended general use",
        },
        "da3-base": {
            "hf_id": "depth-anything/DA3-BASE",
            "gs_capable": False,
            "description": "0.12B params, balanced performance",
        },
        "da3-small": {
            "hf_id": "depth-anything/DA3-SMALL",
            "gs_capable": False,
            "description": "0.08B params, lightweight",
        },
        "da3mono-large": {
            "hf_id": "depth-anything/DA3MONO-LARGE",
            "gs_capable": False,
            "description": "0.35B params, monocular only",
        },
        "da3metric-large": {
            "hf_id": "depth-anything/DA3METRIC-LARGE",
            "gs_capable": False,
            "description": "0.35B params, metric depth + sky segmentation",
        },
        "da3nested-giant-large": {
            "hf_id": "depth-anything/DA3NESTED-GIANT-LARGE",
            "gs_capable": True,
            "description": "1.40B params, all features (any-view + metric + GS)",
        }
    }
    
    # Mapping from our ModelVariant names to DA3 API names
    VARIANT_TO_API_NAME = {
        # v1.1 models (use v1.0 API names - DA3 API doesn't distinguish versions in model names)
        "DA3NESTED-GIANT-LARGE-1.1": "da3nested-giant-large",
        "DA3-GIANT-1.1": "da3-giant",
        "DA3-LARGE-1.1": "da3-large",
        
        # v1.0 models
        "DA3NESTED-GIANT-LARGE": "da3nested-giant-large",
        "DA3-GIANT": "da3-giant",
        "DA3-LARGE": "da3-large",
        
        # Other variants
        "DA3-BASE": "da3-base",
        "DA3-SMALL": "da3-small",
        "DA3METRIC-LARGE": "da3metric-large",
        "DA3MONO-LARGE": "da3mono-large",
    }
    
    GS_CAPABLE_MODELS = ["da3-giant", "da3nested-giant-large"]
    
    def __init__(
        self,
        model_name: str = "da3-large",
        device: str = "cuda",
        commercial_use: bool = False,
        validate_license_strict: bool = False
    ):
        """Initialize DA3 wrapper.
        
        Args:
            model_name: Model variant name (da3-large, da3-giant, etc.)
            device: Device to use (cuda/cpu/mps)
            commercial_use: Whether this is commercial use
            validate_license_strict: If True, raise error on license violation
        
        Raises:
            ImportError: If official DA3 API is not installed
            RuntimeError: If strict license validation fails
        """
        self.model_name = model_name
        self.device = device
        
        # Fix OpenMP duplicate library issue before importing
        import os
        if os.environ.get('KMP_DUPLICATE_LIB_OK') != 'TRUE':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Try to import official API
        try:
            from depth_anything_3.api import DepthAnything3
            self.DepthAnything3 = DepthAnything3
            self.available = True
        except ImportError:
            self.DepthAnything3 = None
            self.available = False
            logger.warning(
                "Official DA3 API not available. "
                "Install with: pip install depth-anything-3"
            )
        
        # Initialize model if available
        if self.available:
            self.model = self._load_model()
        else:
            self.model = None
    
    def _load_model(self):
        """Load DA3 model using official API."""
        # Map our model name to DA3 API name if needed
        api_model_name = self.VARIANT_TO_API_NAME.get(self.model_name, self.model_name)
        
        logger.info(f"Loading DA3 model: {self.model_name} (API name: {api_model_name})")
        model = self.DepthAnything3(model_name=api_model_name)
        model = model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        return model
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cuda"
    ) -> "DepthAnything3Wrapper":
        """
        Load model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., "depth-anything/DA3-GIANT")
            device: Device to use
        
        Returns:
            Initialized wrapper
        """
        # Extract model name from ID
        model_name = model_id.split("/")[-1].lower()
        # Normalize to expected format
        if not model_name.startswith("da3"):
            model_name = f"da3-{model_name}"
        return cls(model_name=model_name, device=device)
    
    def inference(
        self,
        # Input parameters
        image: Optional[List[Union[np.ndarray, Image.Image, str, Path]]] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        
        # Pose alignment parameters
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        
        # Rendering parameters (for gs_video)
        render_exts: Optional[np.ndarray] = None,
        render_ixts: Optional[np.ndarray] = None,
        render_hw: Optional[Tuple[int, int]] = None,
        
        # Processing parameters
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        
        # Export parameters
        export_dir: Optional[Union[str, Path]] = None,
        export_format: str = "mini_npz",
        export_feat_layers: Optional[List[int]] = None,
        
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        
        # Feature visualization parameters
        feat_vis_fps: int = 15,
        
        # Additional export kwargs
        export_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DA3Prediction:
        """
        Run depth inference with full DA3 capabilities.
        
        Args:
            image: List of images (arrays, PIL Images, or paths)
            extrinsics: Camera extrinsics (N, 4, 4) for pose-conditioned inference
            intrinsics: Camera intrinsics (N, 3, 3) for pose-conditioned inference
            align_to_input_ext_scale: Align predicted poses to input scale
            infer_gs: Enable Gaussian Splatting branch (requires GS-capable model)
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Reference view selection strategy
                - "first": Use first view as reference
                - "middle": Use middle view as reference
                - "saddle_balanced": Balanced saddle point strategy
                - "saddle_sim_range": Saddle point with similarity range
            render_exts: Rendering extrinsics for gs_video (M, 4, 4)
            render_ixts: Rendering intrinsics for gs_video (M, 3, 3)
            render_hw: Rendering resolution for gs_video (height, width)
            process_res: Processing resolution (affects quality/speed tradeoff)
            process_res_method: Resize method ("upper_bound_resize" or "lower_bound_resize")
            export_dir: Export directory path (creates if doesn't exist)
            export_format: Export format(s) separated by "-" (e.g., "mini_npz-glb-gs_ply")
                - "mini_npz": Minimal NPZ (depth + conf)
                - "full_npz": Full NPZ (depth + conf + poses + images)
                - "glb": GLTF binary 3D mesh
                - "gs_ply": Gaussian Splatting PLY
                - "gs_video": Gaussian Splatting video
                - "depth_vis": Depth visualization video
                - "feat_vis": Feature visualization video
            export_feat_layers: Layers to export features from (e.g., [0, 3, 6, 9])
            conf_thresh_percentile: GLB confidence threshold percentile (0-100)
            num_max_points: GLB max points for point cloud
            show_cameras: GLB show camera frustums in visualization
            feat_vis_fps: Feature visualization video FPS
            export_kwargs: Additional export arguments per format
        
        Returns:
            DA3Prediction object with depth, confidence, poses, and auxiliary data
        
        Raises:
            RuntimeError: If DA3 API is not available
            ValueError: If invalid parameters are provided
        
        Example:
            >>> # Basic monocular depth
            >>> pred = wrapper.inference(
            ...     image=["image.jpg"],
            ...     export_dir="output"
            ... )
            
            >>> # Multi-view with poses
            >>> pred = wrapper.inference(
            ...     image=["img1.jpg", "img2.jpg", "img3.jpg"],
            ...     extrinsics=camera_extrinsics,  # (3, 4, 4)
            ...     intrinsics=camera_intrinsics,  # (3, 3, 3)
            ...     export_format="mini_npz-glb"
            ... )
            
            >>> # Gaussian Splatting workflow
            >>> pred = wrapper.inference(
            ...     image=image_list,
            ...     infer_gs=True,
            ...     export_format="gs_ply-gs_video",
            ...     render_exts=render_poses,
            ...     render_hw=(1080, 1920)
            ... )
        """
        if not self.available:
            raise RuntimeError(
                "DA3 API not available. "
                "Install with: pip install depth-anything-3"
            )
        
        # Validate inputs and track original dimensions
        image_prepared, original_sizes = self._prepare_images_with_sizes(image)
        
        # Validate GS requirements
        if infer_gs and self.model_name not in self.GS_CAPABLE_MODELS:
            raise ValueError(
                f"Gaussian Splatting requires {', '.join(self.GS_CAPABLE_MODELS)}, "
                f"but got {self.model_name}"
            )
        
        # Validate reference view strategy
        valid_strategies = ["first", "middle", "saddle_balanced", "saddle_sim_range"]
        if ref_view_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid ref_view_strategy: {ref_view_strategy}. "
                f"Must be one of {valid_strategies}"
            )
        
        # Call official API
        logger.info(f"Running DA3 inference on {len(image_prepared)} images")
        prediction = self.model.inference(
            image=image_prepared,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            render_exts=render_exts,
            render_ixts=render_ixts,
            render_hw=render_hw,
            process_res=process_res,
            process_res_method=process_res_method,
            export_dir=str(export_dir) if export_dir else None,
            export_format=export_format,
            export_feat_layers=export_feat_layers,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
            export_kwargs=export_kwargs or {}
        )
        
        # Upsample depth maps to original resolutions
        depth_upsampled = self._upsample_depth_to_native(
            prediction.depth,
            original_sizes
        )
        
        # Upsample confidence maps if present
        conf_upsampled = None
        if hasattr(prediction, 'conf') and prediction.conf is not None:
            conf_upsampled = self._upsample_depth_to_native(
                prediction.conf,
                original_sizes
            )
        
        # Wrap in our dataclass
        return DA3Prediction(
            depth=depth_upsampled,
            conf=conf_upsampled,
            extrinsics=getattr(prediction, 'extrinsics', None),
            intrinsics=getattr(prediction, 'intrinsics', None),
            processed_images=getattr(prediction, 'processed_images', None),
            aux=getattr(prediction, 'aux', None)
        )
    
    def _prepare_images_with_sizes(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]]
    ) -> Tuple[List[Union[np.ndarray, Image.Image, str]], List[Tuple[int, int]]]:
        """Convert Path objects and ImageInput to formats for API compatibility,
        and track original image sizes.
        
        Returns:
            Tuple of (prepared_images, original_sizes)
            where original_sizes is a list of (height, width) tuples
        """
        # Import here to avoid circular dependency
        from lux_depth_v3.input_manager import ImageInput
        
        prepared = []
        sizes = []
        
        for img in images:
            if isinstance(img, ImageInput):
                # Handle ImageInput objects
                if img.path is not None:
                    # Load to get size
                    pil_img = Image.open(img.path)
                    sizes.append((pil_img.height, pil_img.width))
                    prepared.append(str(img.path))
                elif img.array is not None:
                    h, w = img.array.shape[:2]
                    sizes.append((h, w))
                    prepared.append(img.array)
                else:
                    raise ValueError("ImageInput has neither path nor array")
            elif isinstance(img, Path):
                pil_img = Image.open(img)
                sizes.append((pil_img.height, pil_img.width))
                prepared.append(str(img))
            elif isinstance(img, str):
                pil_img = Image.open(img)
                sizes.append((pil_img.height, pil_img.width))
                prepared.append(img)
            elif isinstance(img, Image.Image):
                sizes.append((img.height, img.width))
                prepared.append(img)
            elif isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                sizes.append((h, w))
                prepared.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        return prepared, sizes
    
    def _upsample_depth_to_native(
        self,
        depth: np.ndarray,
        original_sizes: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Upsample depth maps to native resolution using bicubic interpolation.
        
        Args:
            depth: Depth array, shape (N, H_low, W_low) or (H_low, W_low)
            original_sizes: List of (height, width) tuples for each image
        
        Returns:
            Upsampled depth array, shape (N, H_orig, W_orig) or (H_orig, W_orig)
        """
        import cv2
        
        # Handle single image case
        is_batched = depth.ndim == 3
        if not is_batched:
            depth = depth[np.newaxis, ...]  # Add batch dimension
        
        if len(original_sizes) != depth.shape[0]:
            raise ValueError(
                f"Mismatch: {len(original_sizes)} original sizes but "
                f"{depth.shape[0]} depth maps"
            )
        
        upsampled = []
        for i, (h_orig, w_orig) in enumerate(original_sizes):
            depth_map = depth[i]  # (H_low, W_low)
            h_low, w_low = depth_map.shape
            
            # Skip upsampling if already at native resolution
            if (h_low, w_low) == (h_orig, w_orig):
                upsampled.append(depth_map)
                continue
            
            # Upsample using bicubic interpolation
            depth_upsampled = cv2.resize(
                depth_map,
                (w_orig, h_orig),
                interpolation=cv2.INTER_CUBIC
            )
            upsampled.append(depth_upsampled)
            
            logger.debug(
                f"Upsampled depth {i}: {depth_map.shape} -> {depth_upsampled.shape}"
            )
        
        # Stack back to array
        result = np.stack(upsampled, axis=0)
        
        # Remove batch dimension if input was single image
        if not is_batched:
            result = result[0]
        
        return result
    
    def _prepare_images(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]]
    ) -> List[Union[np.ndarray, Image.Image, str]]:
        """Convert Path objects and ImageInput to formats for API compatibility.
        
        DEPRECATED: Use _prepare_images_with_sizes() instead for proper upsampling.
        """
        # Import here to avoid circular dependency
        from lux_depth_v3.input_manager import ImageInput
        
        result = []
        for img in images:
            if isinstance(img, ImageInput):
                # Handle ImageInput objects
                if img.path is not None:
                    result.append(str(img.path))
                elif img.array is not None:
                    result.append(img.array)
                else:
                    raise ValueError("ImageInput has neither path nor array")
            elif isinstance(img, Path):
                result.append(str(img))
            else:
                result.append(img)
        return result


class DepthAnything3(nn.Module):
    """Placeholder for Depth Anything 3 model.
    
    This will be replaced with the official implementation when available.
    For now, it provides a compatible interface for testing.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.dtype = dtype
        
        # Placeholder: very simple network that preserves dimensions
        # In reality, this would be the full DA3 architecture
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        self.to(device=device, dtype=dtype)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
    ) -> DepthAnything3:
        """Load pretrained DA3 model.
        
        Args:
            model_name: Model variant name
            device: Device to load model on
            dtype: Model precision
            cache_dir: Cache directory for model weights

        Returns:
            Loaded model
        """
        logger.warning("Loading placeholder DA3 model: %s", model_name)
        logger.warning("DA3 placeholder active - install official API for production use")

        model = cls(model_name, device, dtype)

        # In production, this would download and load pretrained weights
        # For now, we initialize with random weights

        return model
    
    def inference(
        self,
        images: torch.Tensor,
        mode: str = "monocular",
        poses: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run depth inference.
        
        Args:
            images: Input images (B, C, H, W)
            mode: Inference mode ("monocular" or "multi_view")
            poses: Camera poses (B, 4, 4) for multi-view
        
        Returns:
            Dictionary with depth predictions
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Forward pass through placeholder network
        x = self.conv1(images)
        x = torch.relu(x)
        depth = self.conv2(x)
        depth = self.sigmoid(depth)
        
        result = {
            "depth": depth,
        }
        
        # Multi-view mode adds point cloud
        if mode == "multi_view" and poses is not None:
            # Placeholder: in reality, this would use poses to create 3D reconstruction
            batch_size = images.shape[0]
            result["depths"] = [depth[i] for i in range(batch_size)]
            result["point_cloud"] = None  # Would be computed from depth + poses
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (for compatibility)."""
        x = self.conv1(x)
        x = torch.relu(x)
        depth = self.conv2(x)
        depth = self.sigmoid(depth)
        return depth
