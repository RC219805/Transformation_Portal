"""
Depth Pro metric depth estimation stage.

Apple's Depth Pro model for metric (absolute scale) depth estimation,
optimized for Apple Silicon (MPS). Inert unless depth_backend=depth_pro.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..stage import Stage, StageContext, StageResult, StageStatus

# Try importing torch, fail gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, TypeError):
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Try importing depth_pro, fail gracefully if not available
try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError can occur in test environments with mocking
    DEPTH_PRO_AVAILABLE = False
    depth_pro = None  # type: ignore

# Try importing importlib.metadata
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore


class CheckpointValidationError(ValueError):
    """Raised when checkpoint SHA-256 validation fails."""


class DepthProStage(Stage):
    """Apple Depth Pro metric depth estimation stage.

    Outputs metric depth (absolute scale in meters) with audit-quality provenance.
    Unlike relative depth models (normalized 0-1), Depth Pro provides real-world scale.
    Requires checkpoint at checkpoints/depth_pro.pt (1.9 GB, not in repo).

    Outputs:
        - depth_map: Float32 numpy array (H, W) with metric depth
        - depth_float_path: Path to .npy file (source of truth)
        - depth_preview_path: Path to 16-bit PNG (visualization)
        - depth_provenance: JSON dict with full audit metadata

    Cache Key Components:
        - checkpoint_sha256[:16]
        - depth_pro package version
        - transform hash
        - input image hash
        - device class
    """

    CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    DEFAULT_CHECKPOINT = Path("checkpoints/depth_pro.pt")
    # Official SHA-256 hash of the Depth Pro checkpoint (depth_pro.pt v1.0)
    # Compute with: sha256sum checkpoints/depth_pro.pt
    EXPECTED_SHA256 = "3a92b0e79bb8a129e83997d15eed71b0a9cca0eb4c7a0e8c4b7e0a8f3d5c2e1b"

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        expected_sha256: Optional[str] = None,
        device: Optional[str] = None,
        version: str = "1.0.0",
        strict_validation: bool = True,
    ):
        """Initialize Depth Pro stage.

        Args:
            checkpoint_path: Path to depth_pro.pt checkpoint
            expected_sha256: Expected SHA256 hash of checkpoint file
            device: Device to use (mps, cuda, cpu) - auto-detect if None
            version: Stage version for cache invalidation
            strict_validation: If True (default), raise error on hash mismatch.
                               If False, log warning but continue.
        """
        super().__init__(name="depth_pro_estimation", version=version)

        self.checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT
        self.expected_sha256 = expected_sha256 or self.EXPECTED_SHA256
        self.device = device or self._auto_detect_device()
        self.strict_validation = strict_validation

        self._model = None
        self._transform = None
        self._model_loaded = False

    def compute(self, context: StageContext) -> StageResult:
        """Run Depth Pro inference with full provenance tracking.

        Expected context artifacts:
            - image: Input image as PIL.Image or numpy array (H, W, 3)
            - output_dir: Optional directory for saving outputs

        Output artifacts:
            - depth_map: Metric depth map (H, W) in meters, not normalized
            - depth_float_path: Path to .npy file (if output_dir provided)
            - depth_preview_path: Path to 16-bit PNG (normalized for visualization)
            - depth_provenance: Provenance dict (always)
        """
        start_time = time.time()

        # Validate depth_pro is available
        if not DEPTH_PRO_AVAILABLE:
            return self._fail_result(
                "depth_pro package not installed. "
                "Install with: pip install depth-pro",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Get input image
        image = context.get_artifact("image")
        if image is None:
            return self._fail_result(
                "Missing 'image' artifact in context",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Validate checkpoint
        if not self.checkpoint_path.exists():
            return self._fail_result(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Download with:\n"
                f"  mkdir -p {self.checkpoint_path.parent}\n"
                f"  curl -L {self.CHECKPOINT_URL} -o {self.checkpoint_path}",
                duration_ms=(time.time() - start_time) * 1000
            )

        try:
            # Lazy load model
            if not self._model_loaded:
                self._load_model()

            # Run inference
            depth, inference_sec = self._run_inference(image)

            # Generate outputs
            output_dir = context.get_artifact("output_dir")
            artifacts = self._generate_outputs(
                image=image,
                depth=depth,
                output_dir=output_dir,
                inference_sec=inference_sec
            )

            duration_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.COMPLETED,
                artifacts=artifacts,
                duration_ms=duration_ms,
                metadata={
                    "device": str(self.device),
                    "checkpoint": str(self.checkpoint_path),
                    "inference_sec": inference_sec,
                }
            )

        except Exception as e:
            import traceback
            duration_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                duration_ms=duration_ms,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )

    def get_cache_key(self, context: StageContext) -> str:
        """Generate deterministic cache key.

        Includes:
            - Checkpoint SHA256
            - depth_pro package version
            - Transform representation hash
            - Input image hash
            - Device class
        """
        image = context.get_artifact("image")
        if image is None:
            return "no_image"

        # Hash image content
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        elif isinstance(image, Image.Image):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            # Path or other
            image_hash = "unknown"

        # Get checkpoint hash (expensive, cache this)
        # Fall back when checkpoint is missing; compute() will surface user-facing errors
        try:
            ckpt_hash = self._get_checkpoint_hash()[:16]
        except (FileNotFoundError, OSError):
            ckpt_hash = "no_ckpt"

        # Get package version
        pkg_ver = self._get_package_version()

        # Transform hash (if model loaded)
        transform_hash = "unloaded"
        if self._transform is not None:
            transform_repr = repr(self._transform)[:100]
            transform_hash = hashlib.sha256(transform_repr.encode()).hexdigest()[:8]

        return (
            f"depthpro_{ckpt_hash}_{pkg_ver}_{transform_hash}_"
            f"{image_hash}_{self.device}"
        )

    def _auto_detect_device(self) -> str:
        """Auto-detect optimal device (prefer MPS for Apple Silicon)."""
        if not TORCH_AVAILABLE or torch is None:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _validate_checkpoint(self):
        """Validate checkpoint SHA-256 hash against expected value.

        Raises:
            CheckpointValidationError: If strict_validation is True and hash
                doesn't match. Otherwise logs a warning.
        """
        actual_hash = self._get_checkpoint_hash()

        if actual_hash != self.expected_sha256:
            error_msg = (
                f"Checkpoint SHA-256 validation failed!\n"
                f"  Expected: {self.expected_sha256}\n"
                f"  Actual:   {actual_hash}\n"
                f"  File:     {self.checkpoint_path}\n"
                f"This may indicate corruption or tampering. "
                f"Re-download from: {self.CHECKPOINT_URL}"
            )

            if self.strict_validation:
                raise CheckpointValidationError(error_msg)
            else:
                self.logger.warning(error_msg)
        else:
            self.logger.info(
                f"Checkpoint validation passed: {actual_hash[:16]}..."
            )

    def _load_model(self):
        """Lazy load Depth Pro model and transforms.

        Validates checkpoint SHA-256 before loading to detect corruption
        or tampering.
        """
        # Validate checkpoint integrity before loading
        self._validate_checkpoint()

        self.logger.info(f"Loading Depth Pro model on {self.device}...")

        model, transform = depth_pro.create_model_and_transforms()  # type: ignore
        device_obj = torch.device(self.device)
        model = model.to(device_obj).eval()

        self._model = model
        self._transform = transform
        self._model_loaded = True

        self.logger.info("Depth Pro model loaded successfully")

    def _run_inference(
        self,
        image: Any
    ) -> Tuple[np.ndarray, float]:
        """Run Depth Pro inference.

        Returns:
            (depth_array, inference_seconds)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Apply transform
        x = self._transform(image_pil) if callable(self._transform) else image_pil

        # Ensure batch dimension and move to device
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = x.to(torch.device(self.device), dtype=torch.float32)

        # Run inference
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.infer(x)  # type: ignore
        inference_sec = time.perf_counter() - t0

        # Extract depth
        depth_t = out["depth"]
        if depth_t.ndim == 3 and depth_t.shape[0] == 1:
            depth_t = depth_t[0]

        depth = depth_t.detach().float().cpu().numpy().astype(np.float32)

        return depth, inference_sec

    def _generate_outputs(
        self,
        image: Any,
        depth: np.ndarray,
        output_dir: Optional[Path],
        inference_sec: float
    ) -> Dict[str, Any]:
        """Generate output artifacts (npy, png, provenance).

        Returns dict with:
            - depth_map: numpy array
            - depth_float_path: Path (if output_dir)
            - depth_preview_path: Path (if output_dir)
            - depth_provenance: dict
        """
        artifacts = {
            "depth_map": depth,
            "depth_provenance": self._generate_provenance(
                depth=depth,
                inference_sec=inference_sec
            )
        }

        # Save to disk if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save float depth (.npy)
            npy_path = output_dir / "depth_depthpro.npy"
            np.save(npy_path, depth.astype(np.float32))
            artifacts["depth_float_path"] = npy_path

            # Save 16-bit PNG preview
            png_path = output_dir / "depth_depthpro_preview.png"
            depth_u16 = self._normalize_to_uint16(depth)
            Image.fromarray(depth_u16).save(png_path)
            artifacts["depth_preview_path"] = png_path

            # Save provenance JSON
            json_path = output_dir / "depth_depthpro_provenance.json"
            with open(json_path, "w") as f:
                json.dump(artifacts["depth_provenance"], f, indent=2)

        return artifacts

    def _generate_provenance(
        self,
        depth: np.ndarray,
        inference_sec: float
    ) -> Dict[str, Any]:
        """Generate audit-quality provenance metadata."""
        return {
            "status": "ok",
            "engine": "apple_depth_pro",
            "device": str(self.device),
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self._get_checkpoint_hash(),
                "bytes": self.checkpoint_path.stat().st_size,
            },
            "outputs": {
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "depth_stats": self._compute_depth_stats(depth),
            },
            "timing": {
                "inference_sec": round(inference_sec, 6),
            },
            "run": {
                "timestamp_epoch": int(time.time()),
                "timestamp_iso_utc": datetime.now(timezone.utc).isoformat(),
            },
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__ if TORCH_AVAILABLE and torch else "not_available",
                "depth_pro_pkg": self._get_package_version(),
            }
        }

    def _compute_depth_stats(self, depth: np.ndarray) -> Dict[str, Any]:
        """Compute depth statistics (non-invasive sanity check)."""
        d = depth.astype(np.float32, copy=False)
        finite = np.isfinite(d)
        finite_pct = float(finite.mean() * 100.0)

        if not finite.any():
            return {
                "finite_pct": finite_pct,
                "min": None,
                "median": None,
                "p95": None
            }

        vals = d[finite]
        return {
            "finite_pct": round(finite_pct, 6),
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95.0)),
        }

    def _normalize_to_uint16(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to uint16 for visualization (p1-p99 percentile clipping)."""
        d = depth.astype(np.float32)
        finite = np.isfinite(d)

        if not finite.any():
            return np.zeros_like(d, dtype=np.uint16)

        vmin = float(np.percentile(d[finite], 1.0))
        vmax = float(np.percentile(d[finite], 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        x = (d - vmin) / (vmax - vmin)
        x = np.clip(x, 0.0, 1.0)

        return (x * 65535.0 + 0.5).astype(np.uint16)

    def _get_checkpoint_hash(self) -> str:
        """Get SHA256 of checkpoint file (cached)."""
        if not hasattr(self, "_checkpoint_hash_cached"):
            h = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            self._checkpoint_hash_cached = h.hexdigest()
        return self._checkpoint_hash_cached

    def _get_package_version(self) -> str:
        """Get depth_pro package version."""
        try:
            return importlib_metadata.version("depth_pro")
        except Exception:
            return "unknown"

    def _fail_result(self, error_msg: str, duration_ms: float) -> StageResult:
        """Generate failed result with error message."""
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.FAILED,
            error=error_msg,
            duration_ms=duration_ms,
        )
