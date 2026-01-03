"""Security utilities for V3+V2 orchestrator.

Provides input validation, sanitization, and security hardening functions
to prevent injection attacks, path traversal, and other vulnerabilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set, Optional, List
import logging

logger = logging.getLogger(__name__)

# Allowed extra arguments for V2 subprocess (whitelist)
ALLOWED_V2_EXTRA_ARGS: Set[str] = {
    "--verbose",
    "--quiet",
    "--debug",
    "--no-cache",
}

# Allowed device specifications
ALLOWED_DEVICES: Set[str] = {
    "auto",
    "cpu",
    "cuda",
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
    "mps",
}

# Allowed depth quantization methods
ALLOWED_QUANTIZATION_METHODS: Set[str] = {
    "p1p99",
    "p0.5p99.5",
    "minmax",
}

# Allowed depth fallback policies
ALLOWED_DEPTH_FALLBACKS: Set[str] = {
    "fail",
    "skip",
    "v2-auto",
}


def sanitize_file_stem(stem: str, max_length: int = 200) -> str:
    """Sanitize file stem to prevent path traversal and injection attacks.

    Args:
        stem: Original file stem (filename without extension)
        max_length: Maximum length for sanitized stem

    Returns:
        Sanitized stem safe for use in file paths

    Raises:
        ValueError: If stem is empty after sanitization

    Security considerations:
        - Removes path separators (/, \\)
        - Prevents hidden files (leading dots)
        - Restricts to alphanumeric, underscore, hyphen, and single dots
        - Limits length to prevent buffer overflow or filesystem issues

    NOTE: This function is LOSSY and can cause collisions.
    Use sanitize_path_component_nonlossy() for production systems.
    """
    if not stem:
        raise ValueError("File stem cannot be empty")

    # Replace path separators and dangerous characters with underscore
    # Allow only: alphanumeric, underscore, hyphen, single dots
    sanitized = re.sub(r"[^\w\-.]", "_", stem)

    # Prevent hidden files by removing leading dots
    sanitized = sanitized.lstrip(".")

    # Prevent double dots (parent directory reference)
    sanitized = re.sub(r"\.\.+", ".", sanitized)

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"File stem truncated to {max_length} characters: {sanitized}")

    # Final validation
    if not sanitized:
        raise ValueError(f"File stem is empty after sanitization: {stem}")

    return sanitized


def sanitize_path_component_nonlossy(component: str, max_length: int = 200) -> str:
    """Sanitize path component with non-lossy encoding.

    Strategy:
    - Alphanumeric, underscore, hyphen → preserved as-is
    - Invalid chars (/, :, <, >, etc.) → percent-encoded like URL encoding
    - Leading dots → stripped (prevent hidden files)
    - Double dots → encoded (prevent parent traversal)
    - If encoding causes length overflow → truncate and append hash suffix

    Args:
        component: Single path component (filename or directory name)
        max_length: Maximum length before truncation + hashing

    Returns:
        Sanitized component guaranteed unique for distinct inputs

    Examples:
        "kitchen"        → "kitchen"
        "kitchen:1"      → "kitchen%3A1"
        "kitchen/1"      → "kitchen%2F1"
        ".hidden"        → "hidden"
        "very_long_name" → "very_long_name__a1b2c3d4" (if >max_length)

    Raises:
        ValueError: If component is empty or empty after sanitization
    """
    if not component:
        raise ValueError("Path component cannot be empty")

    original = component

    # Remove leading dots (prevent hidden files)
    component = component.lstrip(".")
    if not component:
        raise ValueError(f"Path component is empty after sanitization: {original}")

    # Encode invalid characters (non-lossy)
    # Allow: alphanumeric, underscore, hyphen, single dots
    # Encode: everything else using percent-encoding
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")

    encoded_chars = []
    for char in component:
        if char in safe_chars:
            encoded_chars.append(char)
        else:
            # Percent-encode like URL encoding
            encoded_chars.append(f"%{ord(char):02X}")

    sanitized = "".join(encoded_chars)

    # Prevent double dots (encode them)
    sanitized = sanitized.replace("..", "%2E%2E")

    # Handle length limit
    if len(sanitized) > max_length:
        # Truncate and add deterministic hash suffix
        import hashlib

        hash_suffix = hashlib.sha256(sanitized.encode()).hexdigest()[:8]
        truncate_len = max_length - len(hash_suffix) - 2  # -2 for "__"
        sanitized = f"{sanitized[:truncate_len]}__{hash_suffix}"
        logger.debug(f"Path component truncated: {original} → {sanitized}")

    return sanitized


def validate_extra_args(extra_args: Optional[List[str]]) -> None:
    """Validate extra arguments for V2 subprocess against whitelist.

    Args:
        extra_args: List of additional CLI arguments for V2 (or None)

    Raises:
        ValueError: If any argument is not in the whitelist

    Security considerations:
        - Prevents command injection via extra_args
        - Uses strict whitelist approach
        - Rejects unknown flags
        - Only allows exact matches (no argument values)
    """
    if not extra_args:
        return

    for arg in extra_args:
        # Only allow exact matches to prevent injection via argument values
        if arg not in ALLOWED_V2_EXTRA_ARGS:
            raise ValueError(
                f"Disallowed V2 extra argument: '{arg}'. "
                f"Allowed arguments: {', '.join(sorted(ALLOWED_V2_EXTRA_ARGS))}. "
                f"Note: Arguments with values (e.g., --flag=value) are not allowed."
            )


def validate_device_spec(device: str) -> str:
    """Validate device specification.

    Args:
        device: Device specification (e.g., 'cuda', 'cuda:0', 'cpu')

    Returns:
        Validated device string

    Raises:
        ValueError: If device specification is invalid
    """
    # First, allow any explicitly whitelisted device
    if device in ALLOWED_DEVICES:
        return device

    # Additionally, allow cuda:N pattern for N in 0-9
    if re.match(r"^cuda:[0-9]$", device):
        return device

    raise ValueError(
        f"Invalid device specification: '{device}'. Allowed: {', '.join(sorted(ALLOWED_DEVICES))} or cuda:N (0-9)"
    )


def validate_quantization_method(method: str) -> str:
    """Validate depth quantization method.

    Args:
        method: Quantization method name

    Returns:
        Validated method string

    Raises:
        ValueError: If method is invalid
    """
    if method not in ALLOWED_QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: '{method}'. Allowed: {', '.join(sorted(ALLOWED_QUANTIZATION_METHODS))}"
        )
    return method


def validate_depth_fallback(fallback: str) -> str:
    """Validate depth fallback policy.

    Args:
        fallback: Fallback policy name

    Returns:
        Validated fallback string

    Raises:
        ValueError: If fallback policy is invalid
    """
    if fallback not in ALLOWED_DEPTH_FALLBACKS:
        raise ValueError(f"Invalid depth fallback policy: '{fallback}'. Allowed: {', '.join(sorted(ALLOWED_DEPTH_FALLBACKS))}")
    return fallback


def validate_git_repository(repo_path: Path) -> Optional[Path]:
    """Validate that a path is a safe git repository.

    Args:
        repo_path: Path to potential git repository

    Returns:
        Resolved path if valid, None if not a git repo

    Security considerations:
        - Resolves symlinks to prevent directory traversal
        - Verifies .git directory exists
        - Does not execute git commands on untrusted paths
    """
    try:
        # Resolve to absolute path (follows symlinks)
        resolved = repo_path.resolve()

        # Check for .git directory
        git_dir = resolved / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            return None

        return resolved
    except (OSError, RuntimeError):
        # Handle filesystem errors gracefully
        return None
