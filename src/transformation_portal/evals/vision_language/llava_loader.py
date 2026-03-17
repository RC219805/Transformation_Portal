"""Manifest-aware LLaVA loader built on strict HF lock resolution.

This module provides functions to load LLaVA models from verified local
snapshots after HF lock resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from transformation_portal.models.hf_manifest_loader import (
    HFResolvedLocalModel,
    resolve_manifest_model,
)

logger = logging.getLogger(__name__)


class LlavaLoaderError(RuntimeError):
    """Raised for LLaVA loader failures."""


@dataclass(frozen=True)
class LlavaLoadedArtifacts:
    """Loaded LLaVA model and processor artifacts.

    Attributes:
        model_key: Manifest key identifying this model
        local_root: Path to the local snapshot directory
        model: Loaded transformers model
        processor: Loaded transformers processor
        repo_id: HuggingFace repository ID
        revision: Pinned commit SHA
    """

    model_key: str
    local_root: Path
    model: Any
    processor: Any
    repo_id: str
    revision: str


def load_llava_from_manifest_entry(
    *,
    model_key: str,
    manifest_payload: dict[str, Any],
    device_map: Optional[str | dict[str, Any]] = "auto",
    torch_dtype: Any = "auto",
    cache_dir: Optional[str] = None,
) -> LlavaLoadedArtifacts:
    """Load LLaVA model and processor from a manifest entry.

    This function resolves the manifest entry to a local snapshot directory,
    verifies required files, and loads the model/processor from the local path.

    Args:
        model_key: Manifest key identifying the model
        manifest_payload: Manifest payload with repo_id, revision, etc.
        device_map: Device mapping for model loading (default: "auto")
        torch_dtype: Torch dtype for model weights (default: "auto")
        cache_dir: Optional HuggingFace cache directory

    Returns:
        LlavaLoadedArtifacts with model and processor

    Raises:
        LlavaLoaderError: If loading fails
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise LlavaLoaderError(
            "transformers is required for loading LLaVA artifacts. " "Install with: pip install transformers>=4.35"
        ) from exc

    logger.info("Resolving manifest entry for model key: %s", model_key)

    resolved: HFResolvedLocalModel = resolve_manifest_model(
        model_key=model_key,
        payload=manifest_payload,
        cache_dir=cache_dir,
    )

    logger.info(
        "Loading LLaVA processor from local root: %s",
        resolved.local_root,
    )

    try:
        processor = AutoProcessor.from_pretrained(
            str(resolved.local_root),
            trust_remote_code=False,
        )
    except Exception as exc:
        raise LlavaLoaderError(f"Failed to load processor from '{resolved.local_root}': {exc}") from exc

    logger.info(
        "Loading LLaVA model from local root: %s (device_map=%s, torch_dtype=%s)",
        resolved.local_root,
        device_map,
        torch_dtype,
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(resolved.local_root),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise LlavaLoaderError(f"Failed to load model from '{resolved.local_root}': {exc}") from exc

    logger.info(
        "Successfully loaded LLaVA model '%s' (%s@%s)",
        model_key,
        resolved.repo_id,
        resolved.revision[:8],
    )

    return LlavaLoadedArtifacts(
        model_key=model_key,
        local_root=resolved.local_root,
        model=model,
        processor=processor,
        repo_id=resolved.repo_id,
        revision=resolved.revision,
    )
