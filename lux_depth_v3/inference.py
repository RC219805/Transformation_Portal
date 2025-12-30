"""DA3 inference engine.

Wrapper for Depth Anything 3 models with unified API for monocular and
multi-view depth estimation. Supports both native Python API and official
CLI integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

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


class DA3InferenceEngine:
    """Inference engine for Depth Anything 3 models.

    Supports two modes:
    - Native Python API (default)
    - Official CLI wrapper (use_cli=True)
    """

    def __init__(self, config: DA3Config, commercial_use: bool = False, validate_license_strict: bool = False):
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

        # Validate license before initializing
        from lux_depth_v3.license import validate_license

        validate_license(config.model_variant, commercial_use=commercial_use, strict=validate_license_strict)

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
            model_dir = str(self.config.cache_dir / "models" / self.config.model_variant.value)
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
        # Try to use official DA3 API wrapper
        try:
            from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper

            # Get model name from variant
            model_name = self._get_model_name_from_variant()

            self.wrapper = DepthAnything3Wrapper(
                model_name=model_name,
                device=str(self.device),
                commercial_use=self.commercial_use,
                validate_license_strict=self.validate_license_strict,
            )

            if self.wrapper.available:
                print(f"DA3 Python API mode initialized with {model_name}")
                self.model = self.wrapper.model
            else:
                print("DA3 API not available, using fallback mode")
                self.wrapper = None
                self.model = None
        except ImportError:
            self.wrapper = None
            self.model = None
            print("DA3 wrapper not available")

        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.backend = None
        self.cli = None

        # Model cache
        self._model_cache_path = self.config.cache_dir / "models"
        self._model_cache_path.mkdir(parents=True, exist_ok=True)

    def _get_model_name_from_variant(self) -> str:
        """Convert ModelVariant enum to DA3 API model name."""
        variant_to_name = {
            # v1.1 models (DA3 API doesn't use version suffixes)
            ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1: "da3nested-giant-large",
            ModelVariant.DA3_GIANT_V1_1: "da3-giant",
            ModelVariant.DA3_LARGE_V1_1: "da3-large",
            # v1.0 models (deprecated)
            ModelVariant.DA3_NESTED_GIANT_LARGE: "da3nested-giant-large",
            ModelVariant.DA3_GIANT: "da3-giant",
            ModelVariant.DA3_LARGE: "da3-large",
            # Legacy enums (backward compatibility)
            ModelVariant.NESTED_GIANT_LARGE: "da3nested-giant-large",
            ModelVariant.GIANT: "da3-giant",
            ModelVariant.LARGE: "da3-large",
            ModelVariant.BASE: "da3-base",
            ModelVariant.SMALL: "da3-small",
            # Apache-licensed models
            ModelVariant.DA3_BASE: "da3-base",
            ModelVariant.DA3_SMALL: "da3-small",
            ModelVariant.DA3_METRIC_LARGE: "da3metric-large",
            ModelVariant.DA3_MONO_LARGE: "da3mono-large",
            ModelVariant.METRIC_LARGE: "da3metric-large",
            ModelVariant.MONO_LARGE: "da3mono-large",
        }
        return variant_to_name.get(
            self.config.model_variant,
            "da3-large",  # Default fallback
        )

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

        Example:
            >>> engine = DA3InferenceEngine(config)
            >>> prediction = engine.infer(
            ...     images=[Path("img1.jpg"), Path("img2.jpg")],
            ...     export_dir=Path("output"),
            ...     export_format="mini_npz-glb",
            ...     infer_gs=False,
            ...     convert_to_metric=True,
            ...     focal_length_px=500.0
            ... )
        """
        if self.use_cli:
            return self._infer_cli(images, export_dir, convert_to_metric, focal_length_px, fov_degrees, **kwargs)
        else:
            return self._infer_api(
                images, extrinsics, intrinsics, export_dir, convert_to_metric, focal_length_px, fov_degrees, **kwargs
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
            image=images, extrinsics=extrinsics, intrinsics=intrinsics, export_dir=export_dir, **api_kwargs
        )

        # Convert to metric depth if requested
        if convert_to_metric:
            from lux_depth_v3.metric_depth import convert_to_metric_depth

            # Get model name
            model_name = self.config.model_variant.value.info.name

            # Determine image width if needed for FOV estimation
            image_width = None
            if fov_degrees is not None and len(images) > 0:
                from PIL import Image

                img = Image.open(images[0])
                image_width = img.width

            metric_result = convert_to_metric_depth(
                depth=prediction.depth,
                model_name=model_name,
                intrinsics=intrinsics if intrinsics is not None else getattr(prediction, "intrinsics", None),
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

        # Determine input type
        if len(images) == 1:
            result = self.cli.process_image(image_path=images[0], export_dir=export_dir or Path("output"), **kwargs)
        else:
            # For multiple images, create temp directory
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # Symlink images to temp dir
                tmppath = Path(tmpdir)
                for i, img_path in enumerate(images):
                    (tmppath / f"{i:04d}{img_path.suffix}").symlink_to(img_path)

                result = self.cli.process_images(images_dir=tmppath, export_dir=export_dir or Path("output"), **kwargs)

        # Load results from export_dir
        # This is a simplified version - actual implementation would parse
        # the exported files based on export_format
        depth = np.zeros((1, 100, 100))  # Placeholder

        prediction = DA3Prediction(depth=depth, conf=None, extrinsics=None, intrinsics=None, processed_images=None, aux=result)

        # Convert to metric depth if requested
        if convert_to_metric:
            from lux_depth_v3.metric_depth import convert_to_metric_depth

            model_name = self.config.model_variant.value.info.name

            # Determine image width for FOV estimation
            image_width = None
            if fov_degrees is not None and len(images) > 0:
                from PIL import Image

                img = Image.open(images[0])
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
                cache_dir=str(self._model_cache_path) if self.config.enable_model_cache else None,
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
    ) -> Union[DepthResult, List[DepthResult]]:
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
            # Ensure model is loaded for native mode
            if self.model is None:
                self.load_model()

            # Run inference based on mode
            if self.config.inference_mode == InferenceMode.MULTI_VIEW:
                results = self._inference_multiview(inputs)
            else:
                results = self._inference_monocular(inputs)

        return results[0] if single_input else results

    def _inference_cli(
        self,
        inputs: List[ImageInput],
    ) -> List[DepthResult]:
        """Run inference via DA3 CLI.

        Args:
            inputs: List of input images

        Returns:
            List of depth results
        """
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
    ) -> List[DepthResult]:
        """Parse CLI output into DepthResult objects.

        Args:
            export_dir: Directory with CLI output
            inputs: Original inputs

        Returns:
            List of depth results
        """
        results = []

        # Find npz files in export directory
        npz_files = sorted(export_dir.glob("*.npz"))

        for i, (npz_file, img_input) in enumerate(zip(npz_files, inputs)):
            # Load depth from npz
            data = np.load(npz_file)

            # DA3 CLI outputs "depth" key in npz
            if "depth" in data:
                depth_map = data["depth"]
            else:
                raise RuntimeError(f"No 'depth' key in {npz_file}")

            # Load original image
            image = img_input.load()

            # Create result
            result = DepthResult(
                depth_map=depth_map,
                original_image=image,
                metadata={
                    "model_variant": self.config.model_variant.value,
                    "inference_mode": "cli",
                    "input_path": str(img_input.path) if img_input.path else None,
                    "cli_output": str(npz_file),
                },
            )

            results.append(result)

        return results

    def _inference_monocular(
        self,
        inputs: List[ImageInput],
    ) -> List[DepthResult]:
        """Run monocular depth inference.

        Args:
            inputs: List of input images

        Returns:
            List of depth results
        """
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
    ) -> List[DepthResult]:
        """Run multi-view depth inference.

        Args:
            inputs: List of input images with camera poses

        Returns:
            List of depth results with 3D reconstruction
        """
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
            depth_map = depth_output["depths"][i].cpu().numpy()

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
        """Resize depth map to target size.

        Args:
            depth_map: Input depth map (H, W)
            target_size: Target size (height, width)

        Returns:
            Resized depth map
        """
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

    def get_depth_range(self) -> Tuple[float, float]:
        """Get depth value range."""
        return (float(self.depth_map.min()), float(self.depth_map.max()))

    def to_uint16(self, scale: float = 1000.0) -> np.ndarray:
        """Convert depth to uint16 (e.g., for PNG export).

        Args:
            scale: Scale factor (e.g., 1000 for mm)

        Returns:
            Uint16 depth map
        """
        return (self.depth_map * scale).clip(0, 65535).astype(np.uint16)

    def to_colormap(self, colormap: str = "turbo") -> np.ndarray:
        """Convert depth to colormap visualization.

        Args:
            colormap: Matplotlib colormap name

        Returns:
            RGB image (H, W, 3) in range [0, 255] uint8
        """
        import matplotlib.pyplot as plt

        # Normalize to [0, 1]
        depth_norm = (self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min() + 1e-8)

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_norm)[:, :, :3]  # Remove alpha

        return (colored * 255).astype(np.uint8)
