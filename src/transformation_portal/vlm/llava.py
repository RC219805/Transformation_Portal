"""LLaVA-1.5 integration for vision-language understanding.

LLaVA-1.5 achieves 85.1% relative score vs GPT-4 with:
- CLIP ViT-L/14 vision encoder
- 2-layer MLP projection
- Vicuna-13B language model
- 665K instruction-following examples

For luxury real estate applications:
- Scene understanding and classification
- Quality assessment
- Material and style recognition
- Structural validation
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from transformation_portal.core.security.model_lock import resolve_model_lock_revision

logger = logging.getLogger(__name__)

MIN_TRANSFORMERS_VERSION = "4.40"
LLAVA_INSTALL_GUIDANCE = f"pip install transformers>={MIN_TRANSFORMERS_VERSION} accelerate bitsandbytes"

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not available. Install LLaVA runtime with: %s", LLAVA_INSTALL_GUIDANCE)

try:
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoProcessor = None
    BitsAndBytesConfig = None
    LlavaForConditionalGeneration = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning("LLaVA dependencies not available. Install with: %s", LLAVA_INSTALL_GUIDANCE)

LLAVA_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


class LLaVAProcessor:
    """LLaVA-1.5 processor for vision-language understanding.

    Provides scene understanding, quality assessment, and architectural analysis
    for luxury real estate imagery.

    Attributes:
        model_id: HuggingFace model identifier
        device: Computation device (cuda/mps/cpu)
        quantization: Whether to use 4-bit quantization for memory efficiency
        processor: HuggingFace processor for image+text inputs
        model: LLaVA conditional generation model
    """

    # Default prompts for luxury real estate analysis
    SCENE_ANALYSIS_PROMPT = """Analyze this architectural image and identify:
1. Room type or space category (interior/exterior/aerial)
2. Architectural style (modern, traditional, mediterranean, etc.)
3. Key materials visible (marble, wood, glass, metal, stone)
4. Notable luxury features
5. Lighting conditions (natural, artificial, golden hour, etc.)

Provide a structured analysis."""

    QUALITY_ASSESSMENT_PROMPT = """Evaluate this enhanced architectural image for:
1. Photographic realism (does it look like a real photograph?)
2. Structural accuracy (are architectural elements geometrically correct?)
3. Material consistency (do materials look natural and physically plausible?)
4. Lighting plausibility (is the lighting realistic and well-balanced?)
5. Overall aesthetic quality

Rate each aspect and identify any artifacts or implausible elements."""

    MATERIAL_VALIDATION_PROMPT = """Examine the materials in this architectural image:
1. Identify all visible materials (marble, granite, wood, metal, glass, fabric, etc.)
2. Assess material realism (proper reflections, textures, color)
3. Check material consistency across the image
4. Identify any material-related artifacts or implausibilities

Focus on luxury architectural materials."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-13b-hf",
        device: Optional[str] = None,
        quantization: bool = True,
        cache_dir: Optional[Path] = None,
        *,
        model_revision: Optional[str] = None,
        strict_model_lock: Optional[bool] = None,
    ):
        """Initialize LLaVA processor.

        Args:
            model_id: HuggingFace model ID (default: llava-1.5-13b-hf)
            device: Device to use (auto-detected if None)
            quantization: Use 4-bit quantization (reduces 24GB to ~8GB VRAM)
            cache_dir: Model cache directory
            model_revision: Optional immutable revision for model and processor
            strict_model_lock: Enforce pinned revisions for remote model loads.
                If None, uses ``TP_STRICT_MODEL_LOCK`` environment variable.

        Raises:
            ImportError: If LLaVA dependencies not available
            RuntimeError: If model loading fails
        """
        if not LLAVA_AVAILABLE:
            raise ImportError(
                "LLaVA requires PyTorch and "
                f"transformers>={MIN_TRANSFORMERS_VERSION}. Install with: {LLAVA_INSTALL_GUIDANCE}"
            )

        self.model_id = model_id
        self.device = device or self._detect_device()
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.model_revision = model_revision
        self.strict_model_lock = strict_model_lock

        logger.info(f"Initializing LLaVA-1.5 on device: {self.device}")
        logger.info(f"Model: {model_id}, Quantization: {quantization}")

        self.model_revision = resolve_model_lock_revision(
            self.model_id,
            self.model_revision,
            strict=self.strict_model_lock,
            context="LLaVAProcessor",
        )

        self.processor = None
        self.model = None
        self._load_model()

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load LLaVA model and processor."""
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(  # nosec B615
                self.model_id,
                revision=self.model_revision,
                cache_dir=self.cache_dir,
            )

            # Configure quantization for memory efficiency
            model_kwargs = {"cache_dir": self.cache_dir}

            if self.quantization and self.device == "cuda":
                # 4-bit quantization: 24GB -> ~8GB VRAM
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32

            # Load model
            self.model = LlavaForConditionalGeneration.from_pretrained(  # nosec B615
                self.model_id,
                revision=self.model_revision,
                **model_kwargs,
            )

            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model.to(self.device)

            self.model.eval()

            logger.info("LLaVA model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLaVA model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        """Analyze image with custom prompt.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Analysis prompt (uses SCENE_ANALYSIS_PROMPT if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter

        Returns:
            Analysis text from LLaVA

        Example:
            >>> processor = LLaVAProcessor()
            >>> analysis = processor.analyze_image(
            ...     "luxury_interior.jpg",
            ...     prompt="Describe the luxury features in this space"
            ... )
        """
        # Load and preprocess image
        pil_image = self._load_image(image)

        # Use default prompt if none provided
        if prompt is None:
            prompt = self.SCENE_ANALYSIS_PROMPT

        # Format prompt for LLaVA conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=pil_image, text=prompt_text, return_tensors="pt").to(self.device)

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Decode output
        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[
            0
        ]

        # Extract assistant response (after prompt)
        # LLaVA output includes the full conversation, extract just the response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text.strip()

        return response

    def assess_quality(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        custom_criteria: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess image quality for luxury real estate.

        Evaluates:
        - Photographic realism
        - Structural accuracy
        - Material consistency
        - Lighting plausibility
        - Overall aesthetic quality

        Args:
            image: Input image
            custom_criteria: Optional custom assessment criteria

        Returns:
            Dictionary with quality assessment results
        """
        prompt = custom_criteria or self.QUALITY_ASSESSMENT_PROMPT

        assessment = self.analyze_image(
            image,
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent assessment
        )

        return {"assessment": assessment, "prompt": prompt, "model": self.model_id}

    def validate_materials(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Validate material realism and consistency.

        Args:
            image: Input image

        Returns:
            Dictionary with material validation results
        """
        validation = self.analyze_image(image, prompt=self.MATERIAL_VALIDATION_PROMPT, temperature=0.1)

        return {
            "validation": validation,
            "prompt": self.MATERIAL_VALIDATION_PROMPT,
            "model": self.model_id,
        }

    def compare_images(
        self,
        original: Union[str, Path, Image.Image, np.ndarray],
        enhanced: Union[str, Path, Image.Image, np.ndarray],
        comparison_prompt: Optional[str] = None,
    ) -> str:
        """Compare original and enhanced images.

        Note: This requires sequential analysis as LLaVA-1.5 processes single images.
        For true multi-image comparison, use a multi-image VLM.

        Args:
            original: Original image
            enhanced: Enhanced image
            comparison_prompt: Custom comparison prompt

        Returns:
            Comparison analysis
        """
        if comparison_prompt is None:
            comparison_prompt = """Describe this architectural image focusing on:
1. Overall quality and realism
2. Architectural features and details
3. Materials and finishes
4. Lighting and atmosphere
5. Any notable characteristics"""

        # Analyze both images
        original_analysis = self.analyze_image(original, prompt=comparison_prompt)
        enhanced_analysis = self.analyze_image(enhanced, prompt=comparison_prompt)

        # Create comparison
        comparison = f"""Original Image Analysis:
{original_analysis}

Enhanced Image Analysis:
{enhanced_analysis}

Based on these analyses, the enhancement appears to have modified the image's characteristics."""

        return comparison

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Load image from various input formats.

        Args:
            image: Image in various formats

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __repr__(self) -> str:
        return f"LLaVAProcessor(model='{self.model_id}', " f"device='{self.device}', quantization={self.quantization})"
