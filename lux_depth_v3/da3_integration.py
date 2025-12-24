"""
DA3 Integration for Luxury Real Estate Rendering Pipeline

This module provides a clean Python wrapper around Depth Anything 3 (DA3)
for integration into the Transformation Portal rendering pipeline.

Author: Transformation Portal Team
Date: 2025-12-19
"""

import hashlib
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class DA3Error(RuntimeError):
    """Base exception for DA3 integration errors."""


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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(metadata_path: Path, payload: Dict[str, object]) -> None:
    """Write run metadata atomically to a JSON file."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(metadata_path)


def _run_da3(
    cmd: List[str],
    *,
    timeout_s: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run DA3 CLI robustly with captured output and optional timeout."""
    start = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        raise DA3TimeoutError(
            f"DA3 timed out after {timeout_s}s (elapsed={elapsed:.2f}s). "
            f"Command: {shlex.join(cmd)}"
        ) from e
    setattr(cp, "_tp_elapsed_s", time.perf_counter() - start)
    return cp


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
    returncode: Optional[int] = None
    command: Optional[List[str]] = None
    runtime_s: Optional[float] = None
    metadata_path: Optional[Path] = None

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

    def raise_for_status(self) -> "DA3Result":
        """Raise a typed error if the DA3 invocation failed."""
        if self.success:
            return self
        cmd = shlex.join(self.command) if self.command else "<unknown>"
        raise DA3CommandError(
            f"DA3 command failed (returncode={self.returncode}). Command: {cmd}\n"
            f"stderr (tail):\n{_tail(self.stderr)}"
        )


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

        Raises:
            DA3NotInstalledError: If DA3 CLI is not available on PATH
            ValueError: If model name is not recognized
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

        # Validate DA3 CLI availability early (production-safe)
        if shutil.which("da3") is None:
            raise DA3NotInstalledError(
                "DA3 CLI not found on PATH. Install from: "
                "https://github.com/DepthAnything/Depth-Anything-V3"
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
        timeout_s: Optional[int] = None,
        write_metadata: bool = True,
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
            timeout_s: Optional timeout in seconds for the DA3 CLI command
            write_metadata: Whether to write run metadata to a JSON file
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
            logger.info("Running DA3: %s", shlex.join(cmd))

        result = _run_da3(cmd, timeout_s=timeout_s)
        runtime_s = getattr(result, "_tp_elapsed_s", None)
        returncode = result.returncode

        metadata_path = None
        if write_metadata:
            try:
                in_path = Path(input_path)
                metadata_path = output_dir / "da3_run_metadata.json"
                payload: Dict[str, object] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "device": self.device,
                    "command": cmd,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "input_path": str(in_path),
                    "input_sha256": _sha256_file(in_path) if in_path.exists() else None,
                    "returncode": returncode,
                    "runtime_s": runtime_s,
                }
                _write_run_metadata(metadata_path, payload)
            except Exception:
                logger.exception("Failed to write DA3 run metadata")

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
            stderr=result.stderr,
            returncode=returncode,
            command=cmd,
            runtime_s=runtime_s,
            metadata_path=metadata_path,
        )
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: List[str] = ["jpg", "png", "jpeg"],
        export_format: str = "glb-depth_vis-mini_npz",
        timeout_s: Optional[int] = None,
        write_metadata: bool = True,
        **kwargs
    ) -> DA3Result:
        """
        Batch process directory of images.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for output files
            extensions: Image file extensions to process
            export_format: Export format(s)
            timeout_s: Optional timeout in seconds for the DA3 CLI command
            write_metadata: Whether to write run metadata to a JSON file
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
            logger.info("Running DA3: %s", shlex.join(cmd))

        result = _run_da3(cmd, timeout_s=timeout_s)
        runtime_s = getattr(result, "_tp_elapsed_s", None)
        returncode = result.returncode

        metadata_path = None
        if write_metadata:
            try:
                in_path = Path(input_dir)
                metadata_path = output_dir / "da3_run_metadata.json"
                payload: Dict[str, object] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "device": self.device,
                    "command": cmd,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "input_dir": str(in_path),
                    "returncode": returncode,
                    "runtime_s": runtime_s,
                }
                _write_run_metadata(metadata_path, payload)
            except Exception:
                logger.exception("Failed to write DA3 run metadata")

        return DA3Result(
            success=result.returncode == 0,
            output_dir=output_dir,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=returncode,
            command=cmd,
            runtime_s=runtime_s,
            metadata_path=metadata_path,
        )
    
    def process_video(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float = 1.0,
        export_format: str = "glb-depth_vis",
        timeout_s: Optional[int] = None,
        write_metadata: bool = True,
        **kwargs
    ) -> DA3Result:
        """
        Process video file, extracting frames at specified FPS.

        Args:
            input_path: Path to video file
            output_dir: Directory for output files
            fps: Frame extraction rate
            export_format: Export format(s)
            timeout_s: Optional timeout in seconds for the DA3 CLI command
            write_metadata: Whether to write run metadata to a JSON file
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
            logger.info("Running DA3: %s", shlex.join(cmd))

        result = _run_da3(cmd, timeout_s=timeout_s)
        runtime_s = getattr(result, "_tp_elapsed_s", None)
        returncode = result.returncode

        metadata_path = None
        if write_metadata:
            try:
                in_path = Path(input_path)
                metadata_path = output_dir / "da3_run_metadata.json"
                payload: Dict[str, object] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "device": self.device,
                    "command": cmd,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "input_path": str(in_path),
                    "input_sha256": _sha256_file(in_path) if in_path.exists() else None,
                    "returncode": returncode,
                    "runtime_s": runtime_s,
                }
                _write_run_metadata(metadata_path, payload)
            except Exception:
                logger.exception("Failed to write DA3 run metadata")

        return DA3Result(
            success=result.returncode == 0,
            output_dir=output_dir,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=returncode,
            command=cmd,
            runtime_s=runtime_s,
            metadata_path=metadata_path,
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
