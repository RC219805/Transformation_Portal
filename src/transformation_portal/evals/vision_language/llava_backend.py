"""Real LLaVA backend using Transformers multimodal chat templates.

This backend expects a manifest payload for a pinned HF model entry and loads the
processor/model from a verified local snapshot directory.

Key design choices:
- strict local loading after HF lock verification
- `apply_chat_template(...)` for multimodal message formatting
- single-turn evaluation with structured JSON-only prompts
- conservative generation settings suitable for eval workloads
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from transformation_portal.evals.vision_language.llava_loader import (
    LlavaLoadedArtifacts,
    load_llava_from_manifest_entry,
)
from transformation_portal.evals.vision_language.llava_prompts import (
    LlavaPromptSpec,
    build_segmentation_quality_prompt,
)
from transformation_portal.evals.vision_language.llava_schema import VQAResult, parse_vqa_result

logger = logging.getLogger(__name__)


class LlavaBackendError(RuntimeError):
    """Raised for LLaVA backend failures."""


@dataclass(frozen=True)
class LlavaGenerationConfig:
    """Configuration for LLaVA text generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate (default: 256)
        do_sample: Whether to use sampling (default: False for determinism)
        temperature: Sampling temperature (default: 0.0 for greedy)
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0


class LlavaQualityBackend:
    """LLaVA-based quality assessment backend.

    This backend provides visual quality assessment using LLaVA vision-language
    models loaded from manifest-verified local snapshots.

    Example usage:
        backend = LlavaQualityBackend(
            model_key="llava_quality_validation_primary",
            manifest_payload={
                "repo_id": "llava-hf/llava-v1.6-mistral-7b-hf",
                "revision": "abc123...",
            },
        )
        backend.load()
        result = backend.evaluate_images([Path("image.png")])
    """

    def __init__(
        self,
        *,
        model_key: str,
        manifest_payload: dict[str, Any],
        device_map: Optional[str | dict[str, Any]] = "auto",
        torch_dtype: Any = "auto",
        generation_config: Optional[LlavaGenerationConfig] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the LLaVA quality backend.

        Args:
            model_key: Manifest key identifying the model
            manifest_payload: Manifest payload with repo_id, revision, etc.
            device_map: Device mapping for model loading
            torch_dtype: Torch dtype for model weights
            generation_config: Generation configuration
            cache_dir: Optional HuggingFace cache directory
        """
        self.model_key = model_key
        self.manifest_payload = manifest_payload
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.generation_config = generation_config or LlavaGenerationConfig()
        self.cache_dir = cache_dir
        self._loaded: Optional[LlavaLoadedArtifacts] = None

    def load(self) -> None:
        """Load the LLaVA model and processor.

        Raises:
            LlavaBackendError: If loading fails
        """
        logger.info("Loading LLaVA backend for model key: %s", self.model_key)
        self._loaded = load_llava_from_manifest_entry(
            model_key=self.model_key,
            manifest_payload=self.manifest_payload,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
        )

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded is not None

    @property
    def loaded(self) -> LlavaLoadedArtifacts:
        """Get loaded artifacts, raising if not loaded."""
        if self._loaded is None:
            raise LlavaBackendError("LLaVA backend has not been loaded yet")
        return self._loaded

    def evaluate_images(
        self,
        image_paths: list[Path],
        prompt_spec: Optional[LlavaPromptSpec] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> VQAResult:
        """Evaluate images for quality issues.

        Args:
            image_paths: List of image paths to evaluate
            prompt_spec: Optional custom prompt specification
            context: Optional additional context for the prompt

        Returns:
            VQAResult with quality assessment results

        Raises:
            LlavaBackendError: If evaluation fails
        """
        if not image_paths:
            raise LlavaBackendError("evaluate_images requires at least one image path")

        # Lazy load if not already loaded
        if not self.is_loaded():
            self.load()

        prompt_spec = prompt_spec or build_segmentation_quality_prompt(context=context)
        messages = self._build_messages(image_paths=image_paths, prompt_spec=prompt_spec)
        raw_text = self._run_inference(messages=messages)
        return parse_vqa_result(model_key=self.model_key, raw_text=raw_text)

    def _build_messages(
        self,
        *,
        image_paths: list[Path],
        prompt_spec: LlavaPromptSpec,
    ) -> list[dict[str, Any]]:
        """Build multimodal message structure for chat template.

        Args:
            image_paths: List of image paths
            prompt_spec: Prompt specification

        Returns:
            List of message dictionaries for apply_chat_template
        """
        user_content: list[dict[str, Any]] = []

        # Add images first
        for image_path in image_paths:
            user_content.append({"type": "image", "image": str(image_path)})

        # Add text prompt
        user_content.append({"type": "text", "text": prompt_spec.user_text})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_spec.system_text}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _run_inference(self, messages: list[dict[str, Any]]) -> str:
        """Run inference on the messages.

        Args:
            messages: Multimodal message structure

        Returns:
            Generated text response

        Raises:
            LlavaBackendError: If inference fails
        """
        loaded = self.loaded

        try:
            import torch
        except ImportError as exc:
            raise LlavaBackendError(
                "torch is required for LLaVA inference"
            ) from exc

        # Apply chat template to get processed inputs
        processed = loaded.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Determine model device
        try:
            model_device = loaded.model.device
        except Exception:
            model_device = None

        # Move inputs to model device if possible
        if hasattr(processed, "to") and model_device is not None:
            processed = processed.to(model_device)

        input_len = processed["input_ids"].shape[-1]

        # Generate with no_grad for efficiency
        with torch.no_grad():
            generated_ids = loaded.model.generate(
                **processed,
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=self.generation_config.do_sample,
                temperature=self.generation_config.temperature,
            )

        # Extract only the generated continuation (not the input)
        continuation_ids = generated_ids[:, input_len:]

        # Decode the generated text
        generated_texts = loaded.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
        )

        if not generated_texts:
            raise LlavaBackendError("Model generation returned no decoded text")

        return generated_texts[0].strip()
