"""Security utilities for path sanitization and validation.

STUB IMPLEMENTATION - Critical functions to enable package imports.
Full implementation pending.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional


class HashMode(Enum):
    """Hash computation modes."""

    ALWAYS = "always"
    IF_MANIFEST_EXISTS = "if_manifest_exists"
    NEVER = "never"


def sanitize_file_stem(stem: str) -> str:
    """Sanitize file stem for safe filesystem use.

    STUB: Basic implementation - removes problematic characters.

    Args:
        stem: File stem to sanitize

    Returns:
        Sanitized file stem
    """
    # Basic sanitization - replace problematic characters
    sanitized = re.sub(r"[^\w\-_.]", "_", stem)
    # Ensure it doesn't start with dot or dash
    sanitized = re.sub(r"^[.-]", "_", sanitized)
    return sanitized or "unnamed"


def sanitize_path_component_nonlossy(component: str) -> str:
    """Sanitize path component without losing information.

    Converts path separators to a stable delimiter and strips traversal segments.

    Args:
        component: Path component to sanitize

    Returns:
        Sanitized path component
    """
    # Normalize separators first so we can safely flatten to one component.
    normalized = component.replace("\\", "/")
    raw_parts = [part for part in normalized.split("/") if part not in ("", ".", "..")]

    sanitized_parts = []
    for part in raw_parts:
        sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)
        sanitized = sanitized.strip(". ")
        if sanitized:
            sanitized_parts.append(sanitized)

    if not sanitized_parts:
        return "unnamed"

    # Join flattened parts with a deterministic delimiter to preserve ordering.
    return "__".join(sanitized_parts)


def validate_device_spec(device: str) -> str:
    """Validate device specification.

    STUB: Basic validation.

    Args:
        device: Device specification (cpu/cuda/mps/cuda:0, etc.)

    Returns:
        Validated device specification

    Raises:
        ValueError: If device specification is invalid
    """
    device = device.lower().strip()

    # Allow common device specs
    valid_patterns = [
        r"^cpu$",
        r"^cuda(:\d+)?$",
        r"^mps$",
        r"^auto$",
    ]

    for pattern in valid_patterns:
        if re.match(pattern, device):
            return device

    raise ValueError(f"Invalid device specification: {device}. " f"Expected one of: cpu, cuda, cuda:N, mps, auto")


def validate_quantization_method(method: str) -> str:
    """Validate quantization method.

    STUB: Basic validation.

    Args:
        method: Quantization method (none/int8/fp16, etc.)

    Returns:
        Validated quantization method

    Raises:
        ValueError: If method is invalid
    """
    method = method.lower().strip()

    valid_methods = {"none", "int8", "fp16", "fp32", "auto"}

    if method in valid_methods:
        return method

    raise ValueError(f"Invalid quantization method: {method}. " f"Expected one of: {', '.join(sorted(valid_methods))}")


def validate_depth_fallback(fallback: Optional[str]) -> Optional[str]:
    """Validate depth fallback strategy.

    STUB: Basic validation.

    Args:
        fallback: Fallback strategy (none/zeros/previous, etc.)

    Returns:
        Validated fallback strategy

    Raises:
        ValueError: If fallback is invalid
    """
    if fallback is None:
        return None

    fallback = fallback.lower().strip()

    # Updated to match documented interface in config.py
    valid_fallbacks = {"fail", "skip", "v2-auto"}

    if fallback in valid_fallbacks:
        return fallback

    raise ValueError(f"Invalid depth fallback: {fallback}. " f"Expected one of: {', '.join(sorted(valid_fallbacks))}")
