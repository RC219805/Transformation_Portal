"""Input discovery with hygiene filters to prevent processing depth artifacts as RGB inputs.

This module provides intelligent input discovery that excludes:
- Depth maps (*_depth.png, *_depthpro_depth16.png)
- PBR maps (*_normal.png, *_roughness.png, *_ao.png)
- Output directories (depth/, pbr/, v2/, manifests/, logs/)
- Hidden files/directories (.DS_Store, .cache/)
- Non-source directories (_non_source/)

Usage:
    from .input_discovery import discover_images, DiscoveryConfig

    config = DiscoveryConfig(strict_mode=False)
    images = discover_images(input_dir, config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Input discovery configuration.

    Attributes:
        exclude_path_patterns: Path segments to exclude (matched anywhere in path).
        exclude_stem_suffixes: Filename suffixes to exclude (before extension).
        exclude_hidden: Skip hidden files/directories (starting with '.').
        strict_mode: Fail with error if excluded files found (validation mode).
    """

    exclude_path_patterns: List[str] = field(
        default_factory=lambda: [
            "/_non_source/",
            "/output/",
            "/depth/",
            "/pbr/",
            "/v2/",
            "/manifests/",
            "/logs/",
            "/.depth_cache/",
            "/checkpoints/",
        ]
    )
    exclude_stem_suffixes: List[str] = field(
        default_factory=lambda: [
            "_depth",
            "_depthpro_depth16",
            "_normal",
            "_roughness",
            "_ao",
            "_pbr",
            "_zone",
        ]
    )
    exclude_hidden: bool = True
    strict_mode: bool = False  # Fail on excluded files if True


def discover_images(input_dir: Path, config: DiscoveryConfig, image_extensions: List[str] | None = None) -> List[Path]:
    """Discover valid RGB input images while excluding depth artifacts and outputs.

    Args:
        input_dir: Directory to scan for images (recursive).
        config: Discovery configuration with exclusion patterns.
        image_extensions: File extensions to include (default: common image formats).

    Returns:
        List of valid image paths to process.

    Raises:
        ValueError: If strict_mode=True and excluded artifacts are found.

    Example:
        >>> config = DiscoveryConfig(strict_mode=False)
        >>> images = discover_images(Path("./input"), config)
        INFO: Discovered 17 images, excluded 3 artifacts
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"]

    logger.debug(f"Scanning {input_dir} for images with extensions: {image_extensions}")
    logger.debug(f"Exclude path patterns: {config.exclude_path_patterns}")
    logger.debug(f"Exclude stem suffixes: {config.exclude_stem_suffixes}")

    # Collect all candidate files
    candidates = []
    for ext in image_extensions:
        candidates.extend(input_dir.rglob(f"*{ext}"))
        candidates.extend(input_dir.rglob(f"*{ext.upper()}"))

    valid_images = []
    excluded_artifacts = []

    for candidate in sorted(candidates):
        # Check for hidden files/directories
        if config.exclude_hidden and any(part.startswith(".") for part in candidate.parts):
            reason = "hidden file/directory"
            excluded_artifacts.append((candidate, reason))
            logger.debug(f"Skipped artifact: {candidate.name} (matched: {reason})")
            continue

        # Check for excluded path patterns (case-insensitive)
        candidate_str = candidate.as_posix().lower()
        matched_pattern = None
        for pattern in config.exclude_path_patterns:
            if pattern.lower() in candidate_str:
                matched_pattern = pattern
                break

        if matched_pattern:
            reason = f"path pattern: {matched_pattern}"
            excluded_artifacts.append((candidate, reason))
            logger.debug(f"Skipped artifact: {candidate.name} (matched: {reason})")
            continue

        # Check for excluded stem suffixes (case-insensitive)
        stem_lower = candidate.stem.lower()
        matched_suffix = None
        for suffix in config.exclude_stem_suffixes:
            if stem_lower.endswith(suffix.lower()):
                matched_suffix = suffix
                break

        if matched_suffix:
            reason = f"stem suffix: {matched_suffix}"
            excluded_artifacts.append((candidate, reason))
            logger.debug(f"Skipped artifact: {candidate.name} (matched: {reason})")
            continue

        # Valid image
        valid_images.append(candidate)

    # Summary logging
    logger.info(f"Discovered {len(valid_images)} images, excluded {len(excluded_artifacts)} artifacts")

    # Strict mode: fail if artifacts found
    if config.strict_mode and excluded_artifacts:
        error_msg = f"Strict mode: {len(excluded_artifacts)} excluded artifacts found in {input_dir}"
        logger.error(error_msg)
        for artifact, reason in excluded_artifacts[:10]:  # Show first 10
            logger.error(f"  - {artifact.name} ({reason})")
        if len(excluded_artifacts) > 10:
            logger.error(f"  ... and {len(excluded_artifacts) - 10} more")
        raise ValueError(error_msg)

    return valid_images
