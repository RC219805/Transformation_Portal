"""Security utilities for path sanitization and validation.

This module provides security utilities specific to the lux_depth_v3 pipeline,
building on the core security module for path validation and sanitization.

Functions:
- sanitize_file_stem: Sanitize file stems for safe filesystem use
- sanitize_path_component_nonlossy: Sanitize path components without losing info
- validate_device_spec: Validate device specifications (cpu/cuda/mps)
- validate_quantization_method: Validate quantization methods
- validate_depth_fallback: Validate depth fallback strategies
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

# Import core security utilities
from transformation_portal.core.security.path import safe_resolve_path
from transformation_portal.core.security.validation import ValidationError

logger = logging.getLogger(__name__)


class HashMode(Enum):
    """Hash computation modes for artifact integrity tracking."""

    ALWAYS = "always"
    IF_MANIFEST_EXISTS = "if_manifest_exists"
    NEVER = "never"


def sanitize_file_stem(stem: str, max_length: int = 200) -> str:
    """Sanitize file stem for safe filesystem use.

    Removes or replaces problematic characters that could cause issues
    on various filesystems (Windows, macOS, Linux).

    Args:
        stem: File stem to sanitize
        max_length: Maximum length for the sanitized stem (default: 200)

    Returns:
        Sanitized file stem, or 'unnamed' if stem is empty/invalid

    Examples:
        >>> sanitize_file_stem("my_image")
        'my_image'
        >>> sanitize_file_stem("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_file_stem("file<>with|bad:chars")
        'file_with_bad_chars'
    """
    if not stem:
        logger.warning("Empty stem provided, returning 'unnamed'")
        return "unnamed"

    # Replace problematic characters with underscores
    # Characters forbidden on Windows: < > : " / \ | ? *
    # Characters forbidden on macOS: / :
    # Additional problematic: null bytes, control characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)

    # Remove path traversal sequences
    sanitized = sanitized.replace("..", "_")

    # Ensure it doesn't start with dot (hidden file) or dash (CLI arg confusion)
    sanitized = re.sub(r"^[.-]+", "_", sanitized)

    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading/trailing underscores and spaces
    sanitized = sanitized.strip("_ ")

    # Truncate to max length while preserving meaningful suffix if possible
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.debug(f"Truncated stem to {max_length} chars")

    # Final validation - must have at least one valid character
    if not sanitized or not re.search(r"[\w]", sanitized):
        logger.warning("Stem %r sanitized to empty, returning 'unnamed'", stem)
        return "unnamed"

    return sanitized


def sanitize_path_component_nonlossy(component: str, max_length: int = 255) -> str:
    """Sanitize path component without losing information.

    Converts path separators to a stable delimiter and strips traversal segments.
    Unlike sanitize_file_stem, this function preserves nested path structure
    by joining path parts with a stable delimiter.

    Args:
        component: Path component to sanitize
        max_length: Maximum length for the result (default: 255)

    Returns:
        Sanitized path component

    Examples:
        >>> sanitize_path_component_nonlossy("dir1/dir2/file")
        'dir1__dir2__file'
        >>> sanitize_path_component_nonlossy("../../../etc/passwd")
        'etc__passwd'
        >>> sanitize_path_component_nonlossy("dir\\with\\backslash")
        'dir__with__backslash'
    """
    if not component:
        return "unnamed"

    # Normalize separators first so we can safely flatten to one component
    normalized = component.replace("\\", "/")

    # Split on path separator and filter out traversal and empty segments
    raw_parts = [part for part in normalized.split("/") if part not in ("", ".", "..")]

    sanitized_parts = []
    for part in raw_parts:
        # Remove forbidden characters while preserving structure
        # Characters that are dangerous on any filesystem
        sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)

        # Strip leading/trailing dots and spaces (problematic on some systems)
        sanitized = sanitized.strip(". ")

        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        if sanitized:
            sanitized_parts.append(sanitized)

    if not sanitized_parts:
        return "unnamed"

    # Join flattened parts with a deterministic delimiter to preserve ordering
    result = "__".join(sanitized_parts)

    # Truncate if necessary
    if len(result) > max_length:
        result = result[:max_length]
        # Clean up any trailing delimiter fragment
        if result.endswith("_"):
            result = result.rstrip("_")

    return result


def validate_device_spec(device: str) -> str:
    """Validate device specification.

    Ensures the device string is a valid PyTorch device specifier.

    Args:
        device: Device specification (cpu/cuda/mps/cuda:0, etc.)

    Returns:
        Validated and normalized device specification

    Raises:
        ValueError: If device specification is invalid

    Examples:
        >>> validate_device_spec("CPU")
        'cpu'
        >>> validate_device_spec("cuda:0")
        'cuda:0'
        >>> validate_device_spec("mps")
        'mps'
    """
    if not device:
        raise ValueError("Device specification cannot be empty")

    device = device.lower().strip()

    # Allow common device specs with explicit patterns
    valid_patterns = [
        (r"^cpu$", "cpu"),
        (r"^cuda$", "cuda"),
        (r"^cuda:(\d+)$", None),  # Keep cuda:N as-is
        (r"^mps$", "mps"),
        (r"^auto$", "auto"),
    ]

    for pattern, normalized in valid_patterns:
        if re.match(pattern, device):
            if normalized is not None:
                return normalized
            return device

    logger.warning("Invalid device specification rejected: %r", device)
    raise ValueError(f"Invalid device specification: {device}. " f"Expected one of: cpu, cuda, cuda:N, mps, auto")


def validate_quantization_method(method: str) -> str:
    """Validate quantization method.

    Ensures the quantization method is a supported option.

    Args:
        method: Quantization method (none/int8/fp16/fp32/auto)

    Returns:
        Validated quantization method

    Raises:
        ValueError: If method is invalid

    Examples:
        >>> validate_quantization_method("FP16")
        'fp16'
        >>> validate_quantization_method("none")
        'none'
    """
    if not method:
        raise ValueError("Quantization method cannot be empty")

    method = method.lower().strip()

    valid_methods = {"none", "int8", "fp16", "fp32", "auto"}

    if method in valid_methods:
        return method

    logger.warning("Invalid quantization method rejected: %r", method)
    raise ValueError(f"Invalid quantization method: {method}. " f"Expected one of: {', '.join(sorted(valid_methods))}")


def validate_depth_fallback(fallback: Optional[str]) -> Optional[str]:
    """Validate depth fallback strategy.

    Ensures the fallback strategy is a supported option.

    Args:
        fallback: Fallback strategy (fail/skip/v2-auto, or None)

    Returns:
        Validated fallback strategy or None

    Raises:
        ValueError: If fallback is invalid

    Examples:
        >>> validate_depth_fallback("fail")
        'fail'
        >>> validate_depth_fallback(None)
        None
    """
    if fallback is None:
        return None

    fallback = fallback.lower().strip()

    # Valid fallback strategies matching the documented interface
    valid_fallbacks = {"fail", "skip", "v2-auto"}

    if fallback in valid_fallbacks:
        return fallback

    logger.warning("Invalid depth fallback rejected: %r", fallback)
    raise ValueError(f"Invalid depth fallback: {fallback}. " f"Expected one of: {', '.join(sorted(valid_fallbacks))}")


def validate_preset_name(preset: str) -> str:
    """Validate preset name for safe use.

    Ensures the preset name doesn't contain path traversal or injection.

    Args:
        preset: Preset name to validate

    Returns:
        Validated preset name

    Raises:
        ValueError: If preset name is invalid or dangerous

    Examples:
        >>> validate_preset_name("default")
        'default'
        >>> validate_preset_name("../../../etc/passwd")
        ValueError: Preset name contains invalid characters
    """
    if not preset:
        raise ValueError("Preset name cannot be empty")

    # Check for path traversal attempts
    if ".." in preset or "/" in preset or "\\" in preset:
        logger.warning("Preset name rejected due to path traversal characters: %r", preset)
        raise ValueError(f"Preset name contains invalid characters: {preset}")

    # Check for null bytes or control characters
    if re.search(r"[\x00-\x1f]", preset):
        logger.warning("Preset name rejected due to control characters: %r", preset)
        raise ValueError(f"Preset name contains control characters: {preset}")

    # Allow alphanumeric, underscore, hyphen, dot
    # Note: \w includes underscore, but keeping explicit for clarity
    if not re.match(r"^[\w._-]+$", preset):
        logger.warning("Preset name rejected due to invalid characters: %r", preset)
        raise ValueError(
            f"Preset name contains invalid characters: {preset}. "
            "Only alphanumeric, underscore, hyphen, and dot are allowed."
        )

    return preset


# Re-export commonly used security functions from core for convenience
__all__ = [
    "HashMode",
    "sanitize_file_stem",
    "sanitize_path_component_nonlossy",
    "validate_device_spec",
    "validate_quantization_method",
    "validate_depth_fallback",
    "validate_preset_name",
    "safe_resolve_path",
    "ValidationError",
]
