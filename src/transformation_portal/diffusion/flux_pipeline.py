"""FLUX diffusion pipeline for architectural image enhancement.

FLUX.1-dev: Open-source 12B parameter model with flow matching.
- Guidance distilled version for fast 1-4 step generation
- Superior architectural detail preservation
- Professional photography quality output

Integration with existing pipeline:
- Drop-in replacement for SDXL
- Compatible with existing ControlNet workflow
- Unified configuration system
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import (  # noqa: F401
        FluxPipeline,
        FluxControlNetPipeline,
        FluxControlNetModel,
        FlowMatchEulerDiscreteScheduler
    )
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    logging.warning(
        "FLUX not available. Install with: "
        "pip install diffusers>=0.30.0 transformers>=4.38.0 accelerate"
    )


logger = logging.getLogger(__name__)


class FLUXPipeline:
    """FLUX.1 pipeline for architectural image enhancement.

    Provides fast, high-quality img2img enhancement with architectural
    precision and photorealistic output.

    Example:
        >>> pipeline = FLUXPipeline(variant="dev")
        >>> result = pipeline.enhance(
        ...     image="luxury_kitchen.jpg",
        ...     prompt="luxury kitchen, professional architectural photography, 8k",
        ...     strength=0.45,
        ...     num_steps=4
        ... )
        >>> result.save("enhanced.jpg")
    """

    # Model variants
    VARIANTS = {
        "dev": "black-forest-labs/FLUX.1-dev",  # Main model
        "schnell": "black-forest-labs/FLUX.1-schnell",  # Fastest (1-4 steps)
    }

    # Default architectural prompts
    DEFAULT_ARCHITECTURAL_PROMPT = (
        "professional architectural photography, high detail, sharp focus, "
        "natural lighting, 8k resolution, photorealistic"
    )

    DEFAULT_NEGATIVE_PROMPT = (
        "oversaturated, artificial, fake, CGI, unrealistic, distorted, "
        "low quality, blurry, noise, artifacts, overexposed, underexposed"
    )

    def __init__(
        self,
        variant: str = "dev",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """Initialize FLUX pipeline.

        Args:
            variant: Model variant ("dev" or "schnell")
            device: Computation device (auto-detected if None)
            torch_dtype: Tensor dtype (bfloat16 recommended for FLUX)
            enable_cpu_offload: Enable CPU offload for memory efficiency
            enable_attention_slicing: Reduce memory usage
            cache_dir: Model cache directory

        Raises:
            ImportError: If FLUX dependencies not available
            ValueError: If invalid variant specified
        """
        if not FLUX_AVAILABLE:
            raise ImportError(
                "FLUX requires diffusers>=0.30.0. "
                "Install with: pip install diffusers>=0.30.0 transformers>=4.38.0"
            )

        if variant not in self.VARIANTS:
            raise ValueError(f"Invalid variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.model_id = self.VARIANTS[variant]
        self.device = device or self._detect_device()
        self.torch_dtype = torch_dtype

        logger.info(f"Initializing FLUX.1-{variant} on {self.device}")

        # Load pipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )

        # Optimize for memory/speed
        if enable_cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            logger.info("Enabled CPU offload for memory efficiency")
        else:
            self.pipe.to(self.device)

        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing")

        # Configure scheduler for fast generation
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        logger.info("FLUX pipeline initialized successfully")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def enhance(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        strength: float = 0.45,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Enhance image using FLUX img2img.

        Args:
            image: Input image
            prompt: Enhancement prompt (uses default if None)
            negative_prompt: Negative prompt (uses default if None)
            strength: Enhancement strength (0-1, higher = more change)
            num_steps: Number of diffusion steps (1-4 for schnell, 4-50 for dev)
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            output_size: Output dimensions (resizes if specified)

        Returns:
            Enhanced PIL Image
        """
        # Load and prepare image
        pil_image = self._load_image(image)

        # Resize if requested
        if output_size is not None:
            pil_image = pil_image.resize(output_size, Image.Resampling.LANCZOS)

        # Use defaults if not provided
        if prompt is None:
            prompt = self.DEFAULT_ARCHITECTURAL_PROMPT

        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Enhancing with FLUX (strength={strength}, steps={num_steps})")

        # Generate enhanced image
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                image=pil_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                negative_prompt=negative_prompt
            )

        enhanced_image = result.images[0]

        logger.info("Enhancement complete")

        return enhanced_image

    def enhance_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Enhance batch of images.

        Args:
            images: List of input images
            prompts: List of prompts (one per image, uses default if None)
            **kwargs: Additional arguments passed to enhance()

        Returns:
            List of enhanced PIL Images
        """
        if prompts is None:
            prompts = [None] * len(images)

        assert len(images) == len(prompts), "Images and prompts must have same length"

        enhanced_images = []

        for idx, (image, prompt) in enumerate(zip(images, prompts)):
            logger.info(f"Processing image {idx + 1}/{len(images)}")
            enhanced = self.enhance(image, prompt=prompt, **kwargs)
            enhanced_images.append(enhanced)

        return enhanced_images

    def enhance_with_controlnet(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        control_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        controlnet_type: str = "depth",
        prompt: Optional[str] = None,
        controlnet_conditioning_scale: float = 0.7,
        strength: float = 0.45,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Enhance with ControlNet guidance for structure preservation.

        Uses FLUXControlNet for generating control signals (depth, canny, normal)
        and applies structure-preserving enhancement. When official FLUX ControlNet
        models are available, this method will use the full diffusion pipeline.
        Currently, it combines control image generation with standard FLUX enhancement.

        Args:
            image: Input image to enhance
            control_image: Pre-computed control image (depth, canny, etc.).
                If None, will be generated automatically from input image.
            controlnet_type: Type of control ("depth", "canny", "normal")
            prompt: Enhancement prompt (uses default if None)
            controlnet_conditioning_scale: ControlNet influence strength (0-1)
            strength: Enhancement strength (0-1, higher = more change)
            num_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to control image generation

        Returns:
            Enhanced PIL Image with structural preservation

        Note:
            Full FLUX ControlNet pipeline integration depends on official
            model availability. This implementation provides the framework
            and falls back to standard enhancement with control-informed prompting.
        """
        from transformation_portal.diffusion.flux_controlnet import (
            FLUXControlNet,
            FLUX_CONTROLNET_AVAILABLE
        )

        # Load and prepare input image
        pil_image = self._load_image(image)

        # Initialize FLUXControlNet for control image generation
        if not FLUX_CONTROLNET_AVAILABLE:
            logger.warning(
                "FLUXControlNet dependencies not available. "
                "Falling back to standard enhancement."
            )
            return self.enhance(
                image=pil_image,
                prompt=prompt,
                strength=strength,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )

        # Create FLUXControlNet instance for control image processing
        controlnet = FLUXControlNet(
            control_types=[controlnet_type],
            device=self.device,
            torch_dtype=self.torch_dtype
        )

        # Generate control image if not provided
        # Note: control_img is prepared for future FluxControlNetPipeline integration
        if control_image is None:
            logger.info(f"Generating {controlnet_type} control image")
            control_img = controlnet.generate_control_image(
                pil_image,
                controlnet_type,
                **kwargs
            )
        else:
            control_img = self._load_image(control_image)

        # Log control image generation for debugging/validation
        logger.debug(
            f"Control image prepared: {control_img.size}, mode={control_img.mode}"
        )

        # Use defaults if prompt not provided
        if prompt is None:
            prompt = self.DEFAULT_ARCHITECTURAL_PROMPT

        # Build structure-aware prompt enhancement
        structure_prompt = self._build_controlnet_prompt(
            prompt, controlnet_type, controlnet_conditioning_scale
        )

        logger.info(
            f"Enhancing with {controlnet_type} ControlNet "
            f"(scale={controlnet_conditioning_scale}, strength={strength})"
        )

        # NOTE: When official FLUX ControlNet models are released, this will use
        # FluxControlNetPipeline directly with control_img. For now, we use the
        # standard pipeline with structure-aware prompting as a fallback.
        #
        # Future implementation with official models:
        # result = self.controlnet_pipe(
        #     prompt=structure_prompt,
        #     image=pil_image,
        #     control_image=control_img,
        #     controlnet_conditioning_scale=controlnet_conditioning_scale,
        #     strength=strength,
        #     num_inference_steps=num_steps,
        #     guidance_scale=guidance_scale,
        #     generator=generator,
        # )
        _ = control_img  # Explicitly mark as intentionally unused for now

        enhanced_image = self.enhance(
            image=pil_image,
            prompt=structure_prompt,
            strength=strength,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        logger.info("ControlNet-guided enhancement complete")

        return enhanced_image

    def _build_controlnet_prompt(
        self,
        prompt: str,
        controlnet_type: str,
        conditioning_scale: float
    ) -> str:
        """Build structure-aware prompt for ControlNet enhancement.

        Args:
            prompt: Base enhancement prompt
            controlnet_type: Type of control being applied
            conditioning_scale: How strongly to emphasize structure

        Returns:
            Enhanced prompt with structural preservation emphasis
        """
        # Structure preservation keywords based on control type
        structure_keywords = {
            "depth": "preserve spatial depth, maintain perspective",
            "canny": "preserve edges, maintain architectural lines",
            "normal": "preserve surface geometry, maintain material details"
        }

        structure_emphasis = structure_keywords.get(
            controlnet_type,
            "preserve structure"
        )

        # Scale the structural emphasis based on conditioning scale
        if conditioning_scale >= 0.8:
            emphasis = f"strictly {structure_emphasis}"
        elif conditioning_scale >= 0.5:
            emphasis = structure_emphasis
        else:
            emphasis = f"subtly {structure_emphasis}"

        # Combine with original prompt
        enhanced_prompt = f"{prompt}, {emphasis}"

        return enhanced_prompt

    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Load image as PIL Image.

        Args:
            image: Input in various formats

        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def get_optimal_steps(self) -> int:
        """Get recommended number of steps for variant.

        Returns:
            Optimal step count
        """
        if self.variant == "schnell":
            return 4  # Optimized for 1-4 steps
        else:  # dev
            return 25  # Can go higher for quality

    def get_memory_requirements(self) -> Dict[str, str]:
        """Get memory requirements for current configuration.

        Returns:
            Dictionary with memory information
        """
        requirements = {
            "model": "FLUX.1-" + self.variant,
            "parameters": "12 billion",
            "vram_minimum": "16GB (with CPU offload)",
            "vram_recommended": "24GB",
            "dtype": str(self.torch_dtype),
            "device": self.device
        }

        return requirements

    def __repr__(self) -> str:
        return (
            f"FLUXPipeline(variant='{self.variant}', "
            f"device='{self.device}', dtype={self.torch_dtype})"
        )
