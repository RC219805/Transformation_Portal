"""DA3 inference engine.

Wrapper for Depth Anything 3 models with unified API for monocular and
multi-view depth estimation. Supports both native Python API and official
CLI integration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lux_depth_v3.config import (
    DA3Config,
    DA3APIConfig,
    ModelVariant,
    InferenceMode,
    DeviceConfig,
)
from lux_depth_v3.preprocessing import Preprocessor
from lux_depth_v3.input_manager import ImageInput
from lux_depth_v3.da3_wrapper import (
    DA3Backend,
    DA3CLI,
    DepthAnything3Wrapper,
    DA3Prediction,
    check_da3_cli_available,
)
from lux_depth_v3.da3_model_backend import DA3ModelBackend

logger = logging.getLogger(__name__)

ENV_DISABLE_MODEL_BACKEND = "LUX_DA3_DISABLE_MODEL_BACKEND"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.strip().lower() in ("1", "true", "yes", "on")


class DA3InferenceEngine:
    """Inference engine for Depth Anything 3 models.

    Supports two modes:
    - Native Python API (default)
    - Official CLI wrapper (use_cli=True)
    """

    def __init__(
        self,
        config: DA3Config,
        commercial_use: bool = False,
        validate_license_strict: bool = False,
    ):
        """Initialize inference engine.

        Args:
            config: DA3 configuration
            commercial_use: Whether this is commercial use
            validate_license_strict: If True, raise error on license violation
        """
        self.config = config
        self.commercial_use = commercial_use
        self.validate_license_strict = validate_license_strict
        self.device = config.device.resolve_device()
        self.dtype = config.device.get_dtype()

        # Env kill-switch: disable model-level backend routing/initialization.
        # (Read once at init; CLI invocations are generally process-scoped anyway.)
        self.disable_model_backend = _env_truthy(ENV_DISABLE_MODEL_BACKEND)

        # Validate license before initializing
        # (Import lazily so environments that vendor/modify this package get a clearer error.)
        try:
            from lux_depth_v3.license import validate_license
        except ImportError as e:
            raise ImportError(
                "Missing lux_depth_v3.license module. If you intentionally removed license gating, "
                "either restore lux_depth_v3/license.py or remove the license validation call."
            ) from e

        validate_license(
            config.model_variant,
            commercial_use=commercial_use,
            strict=validate_license_strict,
        )

        # Initialize based on mode
        self.use_cli = config.cli.use_cli

        if self.use_cli:
            # CLI mode
            self._init_cli_mode()
        else:
            # Native Python API mode
            self._init_native_mode()

    def _init_cli_mode(self):
        """Initialize CLI mode."""
        if not check_da3_cli_available():
            print("WARNING: DA3 CLI not found, falling back to native mode")
            self.use_cli = False
            self._init_native_mode()
            return

        self.model = None
        self.preprocessor = None

        # Initialize backend if requested
        if self.config.cli.use_backend:
            model_dir = self.config.model_variant.info.huggingface_id
            self.backend = DA3Backend(
                model_dir=model_dir,
                device=str(self.device),
                port=self.config.cli.backend_port,
                host=self.config.cli.backend_host,
            )
        else:
            self.backend = None

        # Initialize CLI wrapper
        self.cli = DA3CLI(backend=self.backend)

        print(f"DA3 CLI mode initialized (backend: {self.config.cli.use_backend})")

    def _init_native_mode(self):
        """Initialize native Python API mode."""
        # Note: use the module-level `DepthAnything3Wrapper` so tests can patch it.
        model_name = self._get_model_name_from_variant()

        self.wrapper = DepthAnything3Wrapper(
            model_name=model_name,
            device=str(self.device),
            commercial_use=self.commercial_use,
            validate_license_strict=self.validate_license_strict,
        )

        if getattr(self.wrapper, "available", False):
            logger.info("DA3 Python API mode initialized with %s", model_name)
            self.model = self.wrapper.model
            self.model_backend = None
        else:
            # API wrapper missing (often due to optional deps like pycolmap/open3d).
            # Try model-level backend (HF config + safetensors) before conceding fallback.
            self.wrapper = None
            self.model = None
            self.model_backend = None

            if self.disable_model_backend:
                logger.warning(
                    "DA3 model-level backend disabled via %s; using fallback mode",
                    ENV_DISABLE_MODEL_BACKEND,
                )
            else:
                try:
                    # Configure backend from engine config
                    from lux_depth_v3.da3_model_backend import DA3ModelBackendConfig

                    backend_cfg = DA3ModelBackendConfig(
                        model_id=self.config.model_variant.value.huggingface_id,
                        device=self.device,
                        dtype="float32",  # DA3 requires float32 for quantile ops
                        max_side=896,
                        cache_dir=self.config.cache_dir / "models" if self.config.cache_dir else None,
                        offline=_env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE"),
                    )
                    mb = DA3ModelBackend(backend_cfg)
                    if mb.is_available():
                        self.model_backend = mb
                        logger.info("DA3 model-level backend available (no depth_anything_3.api).")
                    else:
                        logger.warning(
                            "DA3 API not available and model-level backend unavailable; using fallback mode"
                        )
                except Exception as e:
                    logger.warning(f"DA3 model-level backend init failed; using fallback mode ({e})")
                    self.model_backend = None

        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.backend = None
        self.cli = None

        # Model cache
        # NOTE: Keep existing behavior; config.cache_dir is expected to be set in production configs.
        self._model_cache_path = self.config.cache_dir / "models"
        self._model_cache_path.mkdir(parents=True, exist_ok=True)

    def _get_model_name_from_variant(self) -> str:
        """Resolve the configured ModelVariant to a HuggingFace model id."""
        return self.config.model_variant.info.huggingface_id

    def start_backend(self, timeout: int = 30):
        """Start backend service (CLI mode only).

        Args:
            timeout: Seconds to wait for service to start
        """
        if not self.use_cli or self.backend is None:
            print("Backend not available (not using CLI mode with backend)")
            return

        self.backend.start(timeout=timeout)

    def stop_backend(self):
        """Stop backend service (CLI mode only)."""
        if self.backend is not None:
            self.backend.stop()

    def infer(
        self,
        images: List[Path],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        export_dir: Optional[Path] = None,
        convert_to_metric: bool = False,
        focal_length_px: Optional[float] = None,
        fov_degrees: Optional[float] = None,
        **kwargs,
    ) -> DA3Prediction:
        """
        Run inference with full DA3 API support.

        This method provides access to all DA3 features including:
        - Multi-view depth estimation with pose estimation
        - Gaussian Splatting (3DGS)
        - Feature extraction
        - Multiple export formats
        - Optional metric depth conversion

        Args:
            images: List of image paths
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            export_dir: Export directory
            convert_to_metric: If True, convert depth to meters
            focal_length_px: Focal length in pixels (for metric conversion)
            fov_degrees: Horizontal FOV in degrees (for metric estimation)
            **kwargs: Additional API parameters (see DA3APIConfig for options)

        Returns:
            DA3Prediction with depth, poses, and auxiliary outputs
            If convert_to_metric=True, adds metric_depth and metric_depth_info attributes

        Raises:
            RuntimeError: If using CLI mode or API not available
        """
        if self.use_cli:
            return self._infer_cli(
                images,
                export_dir,
                convert_to_metric,
                focal_length_px,
                fov_degrees,
                **kwargs,
            )
        else:
            return self._infer_api(
                images,
                extrinsics,
                intrinsics,
                export_dir,
                convert_to_metric,
                focal_length_px,
                fov_degrees,
                **kwargs,
            )

    def _infer_api(
        self,
        images: List[Path],
        extrinsics: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        export_dir: Optional[Path],
        convert_to_metric: bool,
        focal_length_px: Optional[float],
        fov_degrees: Optional[float],
        **kwargs,
    ) -> DA3Prediction:
        """Inference using Python API."""

        if not hasattr(self, "wrapper") or self.wrapper is None:
            raise RuntimeError("DA3 Python API not available. Install with: pip install depth-anything-3")

        # Merge config with kwargs
        api_kwargs = self.config.api.to_api_kwargs()
        api_kwargs.update(kwargs)

        # Run inference
        prediction = self.wrapper.inference(
            image=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_dir=export_dir,
            **api_kwargs,
        )

        # Convert to metric depth if requested
        if convert_to_metric:
            from lux_depth_v3.metric_depth import convert_to_metric_depth

            # Get model name
            model_name = self.config.model_variant.info.display_name

            # Determine image width if needed for FOV estimation
            image_width = None
            if fov_degrees is not None and len(images) > 0:
                from PIL import Image

                with Image.open(images[0]) as img:
                    image_width = img.width

            metric_result = convert_to_metric_depth(
                depth=prediction.depth,
                model_name=model_name,
                intrinsics=(intrinsics if intrinsics is not None else getattr(prediction, "intrinsics", None)),
                focal_length_px=focal_length_px,
                image_width=image_width,
                fov_degrees=fov_degrees,
            )

            # Add metric depth to prediction
            prediction.metric_depth = metric_result.depth_meters
            prediction.metric_depth_info = metric_result

            logger = logging.getLogger(__name__)
            logger.info(f"Converted to metric depth: {metric_result.scale_factor:.4f}x scale factor")

        return prediction

    def _infer_cli(
        self,
        images: List[Path],
        export_dir: Optional[Path],
        convert_to_metric: bool,
        focal_length_px: Optional[float],
        fov_degrees: Optional[float],
        **kwargs,
    ) -> DA3Prediction:
        """Inference using CLI mode."""

        if self.cli is None:
            raise RuntimeError("CLI mode not initialized")

        export_dir_resolved = export_dir or Path("output")
        export_dir_resolved.mkdir(parents=True, exist_ok=True)

        # For multiple images, create a temp directory so we can preserve ordering
        # and avoid relying on the source directory layout.
        import tempfile

        input_path: Path
        tmp_ctx = tempfile.TemporaryDirectory() if len(images) > 1 else None
        try:
            if tmp_ctx is None:
                input_path = images[0]
            else:
                tmppath = Path(tmp_ctx.name)
                for i, img_path in enumerate(images):
                    dst = tmppath / f"{i:04d}{img_path.suffix}"
                    # Prefer symlinks (fast), but fall back to a physical copy when the platform
                    # or filesystem disallows symlinks.
                    try:
                        dst.symlink_to(img_path)
                    except OSError:
                        import shutil

                        shutil.copy2(img_path, dst)
                input_path = tmppath

            # Ensure we export an NPZ so we can load depth back into memory
            export_format = str(kwargs.pop("export_format", self.config.cli.export_format))
            if "mini_npz" not in export_format and "npz" not in export_format:
                export_format = f"mini_npz-{export_format}"

            result = self.cli.process_auto(
                input_path=input_path,
                export_dir=export_dir_resolved,
                export_format=export_format,
                model_dir=self.config.model_variant.info.huggingface_id,
                device=str(self.device),
                **kwargs,
            )

            # Load results (prefer full NPZ if requested)
            npz_path_candidates: List[Path] = []
            if "npz" in export_format:
                npz_path_candidates.append(export_dir_resolved / "exports" / "npz" / "results.npz")
            if "mini_npz" in export_format:
                npz_path_candidates.append(export_dir_resolved / "exports" / "mini_npz" / "results.npz")

            npz_path = next((p for p in npz_path_candidates if p.exists()), None)
            if npz_path is None:
                raise RuntimeError(f"DA3 CLI did not produce an NPZ in {export_dir_resolved}")

            data = np.load(npz_path)
            depth = data["depth"]
            conf = data["conf"] if "conf" in data else None
            extrinsics = data["extrinsics"] if "extrinsics" in data else None
            intrinsics = data["intrinsics"] if "intrinsics" in data else None

            prediction = DA3Prediction(
                depth=depth,
                conf=conf,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                processed_images=None,
                aux={**result, "npz_path": str(npz_path)},
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        # Convert to metric depth if requested
        if convert_to_metric:
            from lux_depth_v3.metric_depth import convert_to_metric_depth

            model_name = self.config.model_variant.info.display_name

            # Determine image width for FOV estimation
            image_width = None
            if fov_degrees is not None and len(images) > 0:
                from PIL import Image

                with Image.open(images[0]) as img:
                    image_width = img.width

            metric_result = convert_to_metric_depth(
                depth=prediction.depth,
                model_name=model_name,
                focal_length_px=focal_length_px,
                image_width=image_width,
                fov_degrees=fov_degrees,
            )

            prediction.metric_depth = metric_result.depth_meters
            prediction.metric_depth_info = metric_result

        return prediction

    def load_model(self, force_reload: bool = False):
        """Load DA3 model (native mode only).

        Args:
            force_reload: Force reload model even if already loaded

        Raises:
            ImportError: If depth_anything_v3 package not available
            RuntimeError: If model loading fails
        """
        if self.use_cli:
            print("CLI mode: model loading handled by CLI")
            return

        if self.model is not None and not force_reload:
            return

        try:
            # Import DA3 - this would be from the official package
            # For now, we'll create a placeholder that can be replaced
            # with the actual DA3 API when available
            from lux_depth_v3.da3_wrapper import DepthAnything3
        except ImportError:
            raise ImportError("Depth Anything 3 package not found. Install with: pip install depth-anything-v3")

        print(f"Loading model: {self.config.model_variant.value}")
        print(f"Device: {self.device}, Precision: {self.dtype}")

        try:
            # Load model using official API
            self.model = DepthAnything3.from_pretrained(
                self.config.model_variant.value,
                device=str(self.device),
                dtype=self.dtype,
                cache_dir=(str(self._model_cache_path) if self.config.enable_model_cache else None),
            )

            # Apply torch.compile if enabled (PyTorch 2.0+)
            if self.config.device.use_compile and hasattr(torch, "compile"):
                print("Applying torch.compile optimization...")
                self.model = torch.compile(self.model)

            print(f"Model loaded successfully: {self.config.model_variant.value}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def inference(
        self,
        inputs: Union[ImageInput, List[ImageInput]],
    ) -> Union["DepthResult", List["DepthResult"]]:
        """Run depth inference on input images.

        Args:
            inputs: Single image or list of images

        Returns:
            Depth estimation result(s)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If inference fails
        """
        # Handle single vs batch inputs
        single_input = not isinstance(inputs, list)
        if single_input:
            inputs = [inputs]

        # Validate inputs
        if not inputs:
            raise ValueError("No inputs provided")

        # Route to CLI or native inference
        if self.use_cli:
            results = self._inference_cli(inputs)
        else:
            # Prefer the official DA3 Python API wrapper only when its model is loaded.
            # This prevents unit tests (and offline environments) from inadvertently
            # triggering HuggingFace downloads during `engine.inference(...)`.
            if (
                getattr(self, "wrapper", None) is not None
                and getattr(self.wrapper, "available", False)
                and getattr(self.wrapper, "model", None) is not None
            ):
                results = self._inference_api(inputs)
            elif getattr(self, "model_backend", None) is not None and not self.disable_model_backend:
                # Use model-level backend (HF config + safetensors, no API wrapper)
                results = self._inference_model_backend(inputs)
            else:
                # Fallback to the legacy placeholder implementation.
                # Ensure model is loaded for native mode
                if self.model is None:
                    self.load_model()

                # Run inference based on mode
                if self.config.inference_mode == InferenceMode.MULTI_VIEW:
                    results = self._inference_multiview(inputs)
                else:
                    results = self._inference_monocular(inputs)

        return results[0] if single_input else results

    def predict(
        self,
        inputs: Union[ImageInput, List[ImageInput]],
    ) -> Union["DepthResult", List["DepthResult"]]:
        """Alias for inference() method for backward compatibility."""
        return self.inference(inputs)

    def _inference_cli(
        self,
        inputs: List[ImageInput],
    ) -> List["DepthResult"]:
        """Run inference via DA3 CLI."""
        # Start backend if configured and not running
        if self.backend is not None and not self.backend.is_running():
            print("Starting backend service...")
            self.backend.start()

        # Create temporary directory for CLI input/output
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_dir = tmpdir_path / "input"
            export_dir = tmpdir_path / "output"
            input_dir.mkdir()
            export_dir.mkdir()

            # Save inputs to temp directory
            for i, img_input in enumerate(inputs):
                image = img_input.load()
                from PIL import Image

                if img_input.path:
                    filename = img_input.path.name
                else:
                    filename = f"image_{i:04d}.jpg"

                Image.fromarray(image).save(input_dir / filename)

            # Run CLI command
            if len(inputs) == 1:
                # Single image
                cli_result = self.cli.process_image(
                    image_path=input_dir / filename,
                    export_dir=export_dir,
                    export_format=self.config.cli.export_format,
                )
            else:
                # Batch processing
                cli_result = self.cli.process_images(
                    images_dir=input_dir,
                    export_dir=export_dir,
                    export_format=self.config.cli.export_format,
                )

            # Parse results
            results = self._parse_cli_output(export_dir, inputs)

        return results

    def _parse_cli_output(
        self,
        export_dir: Path,
        inputs: List[ImageInput],
    ) -> List["DepthResult"]:
        """Parse CLI output into DepthResult objects."""
        results = []

        export_tokens = set(self.config.cli.export_format.split("-"))
        npz_candidates: List[Path] = []
        if "npz" in export_tokens:
            npz_candidates.append(export_dir / "exports" / "npz" / "results.npz")
        if "mini_npz" in export_tokens:
            npz_candidates.append(export_dir / "exports" / "mini_npz" / "results.npz")

        npz_path = next((p for p in npz_candidates if p.exists()), None)
        if npz_path is None:
            raise RuntimeError(f"DA3 CLI did not produce an NPZ in {export_dir}")

        data = np.load(npz_path)
        if "depth" not in data:
            raise RuntimeError(f"No 'depth' key in {npz_path}")

        depth = data["depth"]
        if depth.ndim == 2:
            depth = depth[np.newaxis, ...]

        if depth.shape[0] != len(inputs):
            raise RuntimeError(f"DA3 CLI returned {depth.shape[0]} depth maps for {len(inputs)} inputs")

        conf = data["conf"] if "conf" in data else None
        extrinsics = data["extrinsics"] if "extrinsics" in data else None
        intrinsics = data["intrinsics"] if "intrinsics" in data else None

        for i, img_input in enumerate(inputs):
            image = img_input.load()
            depth_map = depth[i]

            metadata = {
                **(img_input.metadata or {}),
                "model_variant": self.config.model_variant.info.display_name,
                "model_hf_id": self.config.model_variant.info.huggingface_id,
                "inference_mode": "cli",
                "input_path": str(img_input.path) if img_input.path else None,
                "cli_output": str(npz_path),
            }
            if conf is not None:
                conf_map = conf[i] if conf.ndim == 3 else conf
                metadata["conf_range"] = (float(conf_map.min()), float(conf_map.max()))
            if extrinsics is not None:
                metadata["extrinsics"] = extrinsics[i].tolist() if extrinsics.ndim == 3 else extrinsics.tolist()
            if intrinsics is not None:
                metadata["intrinsics"] = intrinsics[i].tolist() if intrinsics.ndim == 3 else intrinsics.tolist()

            results.append(DepthResult(depth_map=depth_map, original_image=image, metadata=metadata))

        return results

    def _inference_model_backend(self, inputs: List[ImageInput]) -> List["DepthResult"]:
        """Run inference via DA3ModelBackend (direct model access)."""
        if getattr(self, "model_backend", None) is None:
            raise RuntimeError("DA3 model backend not available")

        results: List[DepthResult] = []
        for img_input in inputs:
            # Load image via ImageInput for consistent TIFF/dtype handling
            image = img_input.load()

            # Convert to float32 [0,1] - handle different input dtypes
            orig_dtype = image.dtype
            rgb01 = image.astype(np.float32)

            if np.issubdtype(orig_dtype, np.uint8):
                rgb01 /= 255.0
            elif np.issubdtype(orig_dtype, np.uint16):
                rgb01 /= 65535.0
            elif np.issubdtype(orig_dtype, np.floating):
                if rgb01.max() > 1.5:
                    rgb01 /= 255.0

            rgb01 = np.clip(rgb01, 0.0, 1.0)

            # Predict depth
            depth_map = self.model_backend.predict_depth01_from_rgb01(rgb01)

            mv = self.config.model_variant.value  # ModelInfo
            metadata = {
                **(img_input.metadata or {}),
                "model_variant": mv.display_name,
                "model_hf_id": mv.huggingface_id,
                "inference_mode": "model_backend",
                "input_path": str(img_input.path) if img_input.path else None,
            }

            results.append(DepthResult(depth_map=depth_map, original_image=image, metadata=metadata))

        return results

    def _inference_api(self, inputs: List[ImageInput]) -> List["DepthResult"]:
        """Run inference via the official DA3 Python API wrapper."""
        if getattr(self, "wrapper", None) is None or not getattr(self.wrapper, "available", False):
            raise RuntimeError("DA3 Python API not available")

        api_kwargs = self.config.api.to_api_kwargs()

        prediction = self.wrapper.inference(image=inputs, export_dir=None, **api_kwargs)

        depth = prediction.depth
        if depth.ndim == 2:
            depth = depth[np.newaxis, ...]

        if depth.shape[0] != len(inputs):
            raise RuntimeError(f"DA3 returned {depth.shape[0]} depth maps for {len(inputs)} inputs")

        conf = prediction.conf
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics

        results: List[DepthResult] = []
        for i, img_input in enumerate(inputs):
            image = img_input.load()
            depth_map = depth[i]

            metadata = {
                **(img_input.metadata or {}),
                "model_variant": self.config.model_variant.info.display_name,
                "model_hf_id": self.config.model_variant.info.huggingface_id,
                "inference_mode": "api",
                "input_path": str(img_input.path) if img_input.path else None,
            }
            if conf is not None:
                conf_map = conf[i] if conf.ndim == 3 else conf
                metadata["conf_range"] = (float(conf_map.min()), float(conf_map.max()))
            if extrinsics is not None:
                metadata["extrinsics"] = extrinsics[i].tolist() if extrinsics.ndim == 3 else extrinsics.tolist()
            if intrinsics is not None:
                metadata["intrinsics"] = intrinsics[i].tolist() if intrinsics.ndim == 3 else intrinsics.tolist()

            results.append(DepthResult(depth_map=depth_map, original_image=image, metadata=metadata))

        return results

    def _inference_monocular(
        self,
        inputs: List[ImageInput],
    ) -> List["DepthResult"]:
        """Run monocular depth inference."""
        results = []

        for img_input in inputs:
            # Load and preprocess image
            image = img_input.load()
            preprocessed, metadata = self.preprocessor.preprocess(
                image,
                return_tensors=True,
            )

            # Add batch dimension and move to device
            input_tensor = preprocessed.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            )

            # Run inference
            with torch.no_grad():
                depth_output = self.model.inference(
                    input_tensor,
                    mode="monocular",
                )

            # Process output - squeeze batch and channel dimensions
            depth_map = depth_output["depth"].squeeze().cpu().numpy()  # (H, W)

            # Unpad if necessary
            if metadata["padding"] != (0, 0, 0, 0):
                depth_map = self.preprocessor.unpad(depth_map, metadata["padding"])

            # Resize to original size
            if metadata["target_size"] != metadata["original_size"]:
                depth_map = self._resize_depth(
                    depth_map,
                    metadata["original_size"],
                )

            # Create result
            result = DepthResult(
                depth_map=depth_map,
                original_image=image,
                metadata={
                    **metadata,
                    "model_variant": self.config.model_variant.value,
                    "inference_mode": self.config.inference_mode.value,
                    "input_path": str(img_input.path) if img_input.path else None,
                },
            )

            results.append(result)

        return results

    def _inference_multiview(
        self,
        inputs: List[ImageInput],
    ) -> List["DepthResult"]:
        """Run multi-view depth inference."""
        # Validate poses
        for img_input in inputs:
            if img_input.pose is None:
                raise ValueError("Multi-view inference requires camera poses")

        # Preprocess all images
        images_batch = []
        poses_batch = []
        metadata_list = []

        for img_input in inputs:
            image = img_input.load()
            preprocessed, metadata = self.preprocessor.preprocess(
                image,
                return_tensors=True,
            )

            images_batch.append(preprocessed)
            poses_batch.append(img_input.pose.to_matrix())
            metadata_list.append(metadata)

        # Stack to batch
        input_tensor = torch.stack(images_batch).to(
            device=self.device,
            dtype=self.dtype,
        )
        poses_tensor = torch.from_numpy(np.stack(poses_batch)).to(
            device=self.device,
            dtype=torch.float32,
        )

        # Run multi-view inference
        with torch.no_grad():
            depth_output = self.model.inference(
                input_tensor,
                poses=poses_tensor,
                mode="multi_view",
            )

        # Process outputs
        results = []
        for i, img_input in enumerate(inputs):
            # Some backends return per-view depths shaped (1, H, W). Squeeze to (H, W)
            # so downstream unpadding/resizing (which expects 2D) behaves correctly.
            depth_map = depth_output["depths"][i].squeeze().cpu().numpy()

            # Unpad and resize
            metadata = metadata_list[i]
            if metadata["padding"] != (0, 0, 0, 0):
                depth_map = self.preprocessor.unpad(depth_map, metadata["padding"])

            if metadata["target_size"] != metadata["original_size"]:
                depth_map = self._resize_depth(
                    depth_map,
                    metadata["original_size"],
                )

            result = DepthResult(
                depth_map=depth_map,
                original_image=img_input.load(),
                point_cloud=depth_output.get("point_cloud"),
                metadata={
                    **metadata,
                    "model_variant": self.config.model_variant.value,
                    "inference_mode": self.config.inference_mode.value,
                    "input_path": str(img_input.path) if img_input.path else None,
                    "pose": img_input.pose,
                },
            )

            results.append(result)

        return results

    def _resize_depth(
        self,
        depth_map: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize depth map to target size."""
        from PIL import Image

        h_target, w_target = target_size

        # Convert to PIL, resize with nearest neighbor to preserve depth values
        depth_uint16 = (depth_map * 65535).astype(np.uint16)
        pil_depth = Image.fromarray(depth_uint16, mode="I;16")
        pil_resized = pil_depth.resize((w_target, h_target), resample=Image.NEAREST)

        resized = np.array(pil_resized).astype(np.float32) / 65535.0
        return resized


@dataclass
class DepthResult:
    """Depth estimation result."""

    depth_map: np.ndarray  # (H, W) depth map in range [0, 1] or metric
    original_image: np.ndarray  # Original RGB image
    point_cloud: Optional[np.ndarray] = None  # (N, 3) point cloud (for multi-view)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> np.ndarray:
        """Alias for depth_map for backward compatibility."""
        return self.depth_map

    def get_depth_range(self) -> Tuple[float, float]:
        """Get depth value range."""
        return (float(self.depth_map.min()), float(self.depth_map.max()))

    def to_uint16(self, scale: float = 1000.0) -> np.ndarray:
        """Convert depth to uint16 (e.g., for PNG export)."""
        return (self.depth_map * scale).clip(0, 65535).astype(np.uint16)

    def to_colormap(self, colormap: str = "turbo") -> np.ndarray:
        """Convert depth to colormap visualization."""
        import matplotlib.pyplot as plt

        # Normalize to [0, 1]
        depth_norm = (self.depth_map - self.depth_map.min()) / (
            self.depth_map.max() - self.depth_map.min() + 1e-8
        )

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_norm)[:, :, :3]  # Remove alpha

        return (colored * 255).astype(np.uint8)
