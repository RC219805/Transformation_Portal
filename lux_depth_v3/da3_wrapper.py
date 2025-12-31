"""Depth Anything 3 API wrapper.

This module provides three integration modes:
1. Official DA3 Python API (depth_anything_3.api.DepthAnything3)
2. Official DA3 CLI wrapper (da3 command)
3. Fallback placeholder (for testing when DA3 not available)

The Python API mode is recommended for most use cases as it provides full access
to all DA3 features including Gaussian Splatting, pose estimation, and feature extraction.

Notes on dependencies
---------------------
Some third‑party packages are *optional* and should not be imported at module import time.
This file therefore avoids unconditional imports of heavy/optional deps such as OpenCV
and requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import atexit
import importlib.util
import logging
import os
import shutil
import signal
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_da3_cli_available() -> bool:
    """Check if the DA3 CLI is available and usable in this environment.

    Note: A stale `da3` entrypoint can exist even when `depth_anything_3` is not
    importable, which would cause `da3` to crash at runtime. We treat that as
    unavailable to avoid false positives.
    """

    if shutil.which("da3") is None:
        return False
    return importlib.util.find_spec("depth_anything_3") is not None


@dataclass
class DA3Prediction:
    """Wrapper for DA3 prediction results.

    Provides unified interface for depth predictions from both Python API
    and CLI modes.
    """

    # Core outputs
    depth: np.ndarray  # (N, H, W) or (H, W)
    conf: Optional[np.ndarray] = None  # (N, H, W) confidence maps

    # Camera parameters
    extrinsics: Optional[np.ndarray] = None  # (N, 4, 4) camera extrinsics
    intrinsics: Optional[np.ndarray] = None  # (N, 3, 3) camera intrinsics

    # Additional outputs
    processed_images: Optional[np.ndarray] = None  # (N, H, W, 3)
    aux: Optional[Dict[str, Any]] = None  # Auxiliary outputs (GS, features, etc.)

    def __post_init__(self):
        """Validate shapes."""
        if self.depth.ndim not in (2, 3):
            raise ValueError(f"Depth must be 2D or 3D, got shape {self.depth.shape}")

        if self.conf is not None and self.conf.shape != self.depth.shape:
            raise ValueError(f"Confidence shape {self.conf.shape} != depth shape {self.depth.shape}")


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
    ):
        """Initialize backend manager.

        Args:
            model_dir: Path to DA3 model directory or HF id (as supported by the CLI)
            device: Device to use (cuda, mps, cpu)
            port: Port for backend service
            host: Host address for backend
        """
        self.model_dir = model_dir
        self.device = device
        self.port = port
        self.host = host
        self._process: Optional[subprocess.Popen] = None

        # Register cleanup on exit
        atexit.register(self.stop)

    def start(self, timeout: int = 30) -> None:
        """Start backend service if not running.

        Args:
            timeout: Seconds to wait for service to start

        Raises:
            RuntimeError: If backend fails to start
        """
        if self.is_running():
            logger.info("Backend already running at %s", self.get_url())
            return

        if not check_da3_cli_available():
            raise RuntimeError(
                "DA3 CLI not found or not usable. Ensure the `da3` entrypoint is on PATH and "
                "the `depth_anything_3` package is importable."
            )

        cmd = [
            "da3",
            "backend",
            "--model-dir",
            self.model_dir,
            "--device",
            self.device,
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]

        logger.info("Starting DA3 backend: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        # Keep output quiet by default. For debugging, change DEVNULL to PIPE.
        self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=env)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info("Backend started at %s", self.get_url())
                return
            time.sleep(0.5)

        self.stop()
        raise RuntimeError(f"Backend failed to start within {timeout}s")

    def stop(self) -> None:
        """Stop backend service."""
        if self._process is None:
            return

        logger.info("Stopping DA3 backend...")
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._process = None
        logger.info("Backend stopped")

    def is_running(self) -> bool:
        """Check if backend is running and healthy."""
        url = f"{self.get_url()}/status"
        try:
            with urlopen(url, timeout=1) as resp:
                return int(getattr(resp, "status", 0)) == 200
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
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
        """
        if not check_da3_cli_available():
            raise RuntimeError(
                "DA3 CLI not found or not usable. Ensure the `da3` entrypoint is on PATH and "
                "the `depth_anything_3` package is importable."
            )

        self.backend = backend

    def _build_base_cmd(self, subcommand: str, input_path: Optional[Path] = None, **kwargs) -> List[str]:
        """Build base command with common options."""
        cmd: List[str] = ["da3", subcommand]
        if input_path is not None:
            cmd.append(str(input_path))

        if self.backend is not None:
            cmd.append("--use-backend")
            cmd.extend(["--backend-url", self.backend.get_url()])

        for key, value in kwargs.items():
            if value is None:
                continue

            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        return cmd

    def _run_command(self, cmd: List[str], capture_output: bool = True) -> Dict[str, Any]:
        """Run CLI command and return stdout/stderr."""
        logger.info("Running: %s", " ".join(cmd))

        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        result = subprocess.run(cmd, capture_output=capture_output, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"DA3 CLI command failed:\nCommand: {' '.join(cmd)}\nError: {result.stderr}")

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def process_auto(self, input_path: Path, export_dir: Path, export_format: str = "mini_npz", **kwargs) -> Dict[str, Any]:
        cmd = self._build_base_cmd(
            "auto",
            input_path=input_path,
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs,
        )
        return self._run_command(cmd)

    def process_image(self, image_path: Path, export_dir: Path, export_format: str = "mini_npz", **kwargs) -> Dict[str, Any]:
        cmd = self._build_base_cmd(
            "image",
            input_path=image_path,
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs,
        )
        return self._run_command(cmd)

    def process_images(
        self,
        images_dir: Path,
        export_dir: Path,
        export_format: str = "mini_npz",
        image_extensions: str = "png,jpg,jpeg",
        **kwargs,
    ) -> Dict[str, Any]:
        cmd = self._build_base_cmd(
            "images",
            input_path=images_dir,
            export_dir=str(export_dir),
            export_format=export_format,
            image_extensions=image_extensions,
            **kwargs,
        )
        return self._run_command(cmd)

    def process_video(
        self, video_path: Path, export_dir: Path, fps: float = 1.0, export_format: str = "mini_npz", **kwargs
    ) -> Dict[str, Any]:
        cmd = self._build_base_cmd(
            "video",
            input_path=video_path,
            export_dir=str(export_dir),
            fps=str(fps),
            export_format=export_format,
            **kwargs,
        )
        return self._run_command(cmd)

    def process_colmap(
        self, colmap_dir: Path, export_dir: Path, export_format: str = "mini_npz-glb", **kwargs
    ) -> Dict[str, Any]:
        cmd = self._build_base_cmd(
            "colmap",
            input_path=colmap_dir,
            export_dir=str(export_dir),
            export_format=export_format,
            **kwargs,
        )
        return self._run_command(cmd)


class DepthAnything3Wrapper:
    """Wrapper for the official DepthAnything3 Python API."""

    AVAILABLE_MODELS = {
        "da3-giant": {
            "hf_id": "depth-anything/DA3-GIANT-1.1",
            "gs_capable": True,
            "description": "1.15B params, any-view with GS support",
        },
        "da3-large": {
            "hf_id": "depth-anything/DA3-LARGE-1.1",
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
            "hf_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "gs_capable": True,
            "description": "1.40B params, all features (any-view + metric + GS)",
        },
    }

    VARIANT_TO_API_NAME = {
        "DA3NESTED-GIANT-LARGE-1.1": "da3nested-giant-large",
        "DA3-GIANT-1.1": "da3-giant",
        "DA3-LARGE-1.1": "da3-large",
        "DA3NESTED-GIANT-LARGE": "da3nested-giant-large",
        "DA3-GIANT": "da3-giant",
        "DA3-LARGE": "da3-large",
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
        validate_license_strict: bool = False,
    ):
        self.model_name = model_name
        self.device = device

        # Avoid OpenMP duplicate library issue (common on macOS)
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore

            self.DepthAnything3 = DepthAnything3
            self.available = True
        except ImportError:
            self.DepthAnything3 = None
            self.available = False
            logger.warning("Official DA3 API not available. Install with: pip install depth-anything-3")

        # Lazily loaded to avoid triggering large downloads at import/init time.
        self.model = None

    def load_model(
        self,
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: Optional[bool] = None,
        force_reload: bool = False,
    ):
        """Load and cache the official DA3 model weights.

        By default, this method allows downloading weights from HuggingFace.
        Set `HF_HUB_OFFLINE=1` (or pass `local_files_only=True`) to enforce
        offline behavior.
        """
        if not self.available or self.DepthAnything3 is None:
            raise RuntimeError("DA3 API not available. Install with: pip install depth-anything-3")

        if self.model is not None and not force_reload:
            return self.model

        if local_files_only is None:
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"} or os.environ.get(
                "TRANSFORMERS_OFFLINE", ""
            ).lower() in {"1", "true", "yes"}

        hf_id = self._resolve_hf_id()
        api_model_name = self._resolve_api_model_name()

        logger.info(
            "Loading DA3 model: %s (HF: %s, API name: %s, local_only=%s)",
            self.model_name,
            hf_id,
            api_model_name,
            local_files_only,
        )

        cache_dir_str = str(cache_dir) if cache_dir is not None else None
        model = self.DepthAnything3.from_pretrained(hf_id, cache_dir=cache_dir_str, local_files_only=local_files_only)
        model = model.to(self.device)
        self.model = model
        logger.info("Model loaded on %s", self.device)
        return model

    def _load_model(self):
        """Backwards-compatible private loader (prefer `load_model`)."""
        return self.load_model()

    def _resolve_hf_id(self) -> str:
        if "/" in self.model_name:
            return self.model_name

        api_name = self._resolve_api_model_name()
        if api_name in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[api_name]["hf_id"]

        cli_aliases = {
            "nested-giant-large-v1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
            "giant-v1.1": "depth-anything/DA3-GIANT-1.1",
            "giant": "depth-anything/DA3-GIANT",
            "large-v1.1": "depth-anything/DA3-LARGE-1.1",
            "large": "depth-anything/DA3-LARGE",
            "base": "depth-anything/DA3-BASE",
            "small": "depth-anything/DA3-SMALL",
            "metric-large": "depth-anything/DA3METRIC-LARGE",
            "mono-large": "depth-anything/DA3MONO-LARGE",
        }
        alias = cli_aliases.get(self.model_name.lower())
        if alias is not None:
            return alias

        return f"depth-anything/{self.model_name.upper()}"

    def _resolve_api_model_name(self) -> str:
        key = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        return self.VARIANT_TO_API_NAME.get(key, key)

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cuda") -> "DepthAnything3Wrapper":
        # This wrapper uses short model aliases (e.g. "da3-giant") internally.
        # When given a HuggingFace ID, map it to our canonical alias.
        model_name = model_id
        if "/" in model_id:
            key = model_id.split("/")[-1]
            model_name = cls.VARIANT_TO_API_NAME.get(key, key)
        return cls(model_name=model_name, device=device)

    def inference(
        self,
        image: Optional[List[Union[np.ndarray, Image.Image, str, Path]]] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: Optional[np.ndarray] = None,
        render_ixts: Optional[np.ndarray] = None,
        render_hw: Optional[Tuple[int, int]] = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: Optional[Union[str, Path]] = None,
        export_format: str = "mini_npz",
        export_feat_layers: Optional[List[int]] = None,
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
        export_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DA3Prediction:
        if not self.available:
            raise RuntimeError("DA3 API not available. Install with: pip install depth-anything-3")

        if self.model is None:
            self.load_model()

        if image is None or len(image) == 0:
            raise ValueError("No images provided for inference")

        image_prepared, original_sizes = self._prepare_images_with_sizes(image)

        api_model_name = self._resolve_api_model_name()
        if infer_gs and api_model_name not in self.GS_CAPABLE_MODELS:
            raise ValueError(f"Gaussian Splatting requires {', '.join(self.GS_CAPABLE_MODELS)}, but got {api_model_name}")

        valid_strategies = ["first", "middle", "saddle_balanced", "saddle_sim_range"]
        if ref_view_strategy not in valid_strategies:
            raise ValueError(f"Invalid ref_view_strategy: {ref_view_strategy}. Must be one of {valid_strategies}")

        logger.info("Running DA3 inference on %d images", len(image_prepared))
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
            export_kwargs=export_kwargs or {},
        )

        depth_upsampled = self._upsample_depth_to_native(prediction.depth, original_sizes)

        conf_upsampled = None
        if hasattr(prediction, "conf") and prediction.conf is not None:
            conf_upsampled = self._upsample_depth_to_native(prediction.conf, original_sizes)

        return DA3Prediction(
            depth=depth_upsampled,
            conf=conf_upsampled,
            extrinsics=getattr(prediction, "extrinsics", None),
            intrinsics=getattr(prediction, "intrinsics", None),
            processed_images=getattr(prediction, "processed_images", None),
            aux=getattr(prediction, "aux", None),
        )

    def _prepare_images(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
    ) -> List[Union[np.ndarray, Image.Image, str]]:
        """Convert inputs to values accepted by the official DA3 API.

        This is a *pure* conversion step (no filesystem IO). It is used by tests
        and by `_prepare_images_with_sizes`.
        """
        from lux_depth_v3.input_manager import ImageInput

        prepared: List[Union[np.ndarray, Image.Image, str]] = []

        for img in images:
            if isinstance(img, ImageInput):
                if img.path is not None:
                    prepared.append(str(img.path))
                elif img.array is not None:
                    prepared.append(img.array)
                else:
                    raise ValueError("ImageInput has neither path nor array")

            elif isinstance(img, Path):
                prepared.append(str(img))

            elif isinstance(img, str):
                prepared.append(img)

            elif isinstance(img, Image.Image):
                prepared.append(img)

            elif isinstance(img, np.ndarray):
                prepared.append(img)

            # Test utilities sometimes pass mocks with an ndarray spec. Treat any
            # "array-like" input (has `.shape`) as a valid ndarray-like object.
            elif hasattr(img, "shape"):
                prepared.append(img)  # type: ignore[arg-type]

            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        return prepared

    def _prepare_images_with_sizes(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
    ) -> Tuple[List[Union[np.ndarray, Image.Image, str]], List[Optional[Tuple[int, int]]]]:
        """Convert inputs to API-compatible values and track original sizes.

        Size detection is best-effort:
        - If an image is provided as an array/PIL image, the size is always known.
        - If an image is provided as a path/string, we only read its size if the
          file exists and is readable.
        """
        from lux_depth_v3.input_manager import ImageInput

        prepared = self._prepare_images(images)
        sizes: List[Optional[Tuple[int, int]]] = []

        for img in images:
            if isinstance(img, ImageInput):
                if img.path is not None:
                    try:
                        if img.path.exists():
                            with Image.open(img.path) as pil_img:
                                sizes.append((pil_img.height, pil_img.width))
                        else:
                            sizes.append(None)
                    except Exception:
                        sizes.append(None)
                elif img.array is not None:
                    h, w = img.array.shape[:2]
                    sizes.append((h, w))
                else:
                    raise ValueError("ImageInput has neither path nor array")

            elif isinstance(img, Path):
                try:
                    if img.exists():
                        with Image.open(img) as pil_img:
                            sizes.append((pil_img.height, pil_img.width))
                    else:
                        sizes.append(None)
                except Exception:
                    sizes.append(None)

            elif isinstance(img, str):
                try:
                    p = Path(img)
                    if p.exists():
                        with Image.open(p) as pil_img:
                            sizes.append((pil_img.height, pil_img.width))
                    else:
                        sizes.append(None)
                except Exception:
                    sizes.append(None)

            elif isinstance(img, Image.Image):
                sizes.append((img.height, img.width))

            elif isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                sizes.append((h, w))

            elif hasattr(img, "shape"):
                try:
                    shape = img.shape  # type: ignore[attr-defined]
                    sizes.append((int(shape[0]), int(shape[1])))
                except Exception:
                    sizes.append(None)

            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        return prepared, sizes

    def _upsample_depth_to_native(self, depth: np.ndarray, original_sizes: List[Optional[Tuple[int, int]]]) -> np.ndarray:
        """Upsample depth/conf maps to native resolution.

        Prefers OpenCV if available (faster); otherwise falls back to torch bicubic.
        """
        is_batched = depth.ndim == 3
        if not is_batched:
            depth = depth[np.newaxis, ...]

        if len(original_sizes) != depth.shape[0]:
            raise ValueError(f"Mismatch: {len(original_sizes)} original sizes but {depth.shape[0]} depth maps")

        upsampled: List[np.ndarray] = []
        for i, size in enumerate(original_sizes):
            depth_map = depth[i].astype(np.float32, copy=False)
            if size is None:
                upsampled.append(depth_map)
                continue

            h_orig, w_orig = size
            h_low, w_low = depth_map.shape

            if (h_low, w_low) == (h_orig, w_orig):
                upsampled.append(depth_map)
                continue

            # Fast path: OpenCV if present
            try:
                import cv2  # type: ignore

                depth_up = cv2.resize(depth_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                upsampled.append(depth_up.astype(np.float32, copy=False))
                continue
            except Exception:
                pass

            # Fallback: torch bicubic (always available since torch is a hard dep)
            import torch.nn.functional as F

            t = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).float()
            t_up = F.interpolate(t, size=(h_orig, w_orig), mode="bicubic", align_corners=False)
            upsampled.append(t_up.squeeze(0).squeeze(0).cpu().numpy())

            logger.debug("Upsampled depth %d: %s -> %s", i, (h_low, w_low), (h_orig, w_orig))

        result = np.stack(upsampled, axis=0)
        if not is_batched:
            result = result[0]
        return result


class DepthAnything3(nn.Module):
    """Placeholder for Depth Anything 3 model.

    This is only used as a last-resort fallback for testing when the official
    DA3 API is not installed.
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
    ) -> "DepthAnything3":
        logger.warning(
            "[DA3 Wrapper] Loading placeholder model: %s (this is NOT the official DA3 implementation)",
            model_name,
        )
        return cls(model_name, device, dtype)

    def inference(
        self,
        images: torch.Tensor,
        mode: str = "monocular",
        poses: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self.eval()

        x = torch.relu(self.conv1(images))
        depth = self.sigmoid(self.conv2(x))  # (B, 1, H, W)

        result: Dict[str, Any] = {"depth": depth}

        if mode == "multi_view" and poses is not None:
            batch_size = images.shape[0]
            # Provide per-view depth maps shaped (H, W) for downstream compatibility.
            result["depths"] = [depth[i, 0] for i in range(batch_size)]
            result["point_cloud"] = None

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return self.sigmoid(self.conv2(x))


# Backwards-compatible alias (some scripts expect this name).
DA3Wrapper = DepthAnything3Wrapper
