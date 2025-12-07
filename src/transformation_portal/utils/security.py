"""
Security Utilities for Transformation Portal

Provides validation functions to prevent common security vulnerabilities:
- Path traversal (CWE-22)
- Command injection (CWE-78)
- Resource exhaustion (CWE-400)

Usage:
    >>> from transformation_portal.utils.security import validate_filepath
    >>> safe_path = validate_filepath(
    ...     Path("user_input.jpg"),
    ...     allowed_dirs=[Path("/data")],
    ...     max_file_size=100*1024*1024  # 100MB
    ... )

See Also:
    - docs/architecture/adr/ADR-002-security-input-validation.md
    - SECURITY.md
"""

from pathlib import Path
from typing import List, Optional
import os


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


# File size limits (bytes)
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_VIDEO_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_CONFIG_SIZE = 10 * 1024 * 1024  # 10MB

# Allowed file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
CONFIG_EXTENSIONS = {'.yaml', '.yml', '.json', '.toml'}


def validate_filepath(
    filepath: Path,
    allowed_dirs: List[Path],
    max_file_size: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None,
    must_exist: bool = True
) -> Path:
    """
    Validate file path against security constraints.
    
    Prevents path traversal attacks by ensuring the resolved path
    is within allowed directories.
    
    Args:
        filepath: Path to validate
        allowed_dirs: List of allowed parent directories
        max_file_size: Maximum file size in bytes (None = no limit)
        allowed_extensions: Whitelist of allowed extensions (None = any)
        must_exist: Whether file must exist (default True)
        
    Returns:
        Resolved, validated Path object
        
    Raises:
        SecurityError: If validation fails
        
    Example:
        >>> safe_path = validate_filepath(
        ...     Path("../../../etc/passwd"),
        ...     allowed_dirs=[Path("/data")]
        ... )
        Traceback (most recent call last):
        SecurityError: Path outside allowed directories
    """
    # Convert to Path object if string
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    # Resolve to absolute path (follows symlinks)
    try:
        if must_exist:
            resolved = filepath.resolve(strict=True)
        else:
            resolved = filepath.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Cannot resolve path {filepath}: {e}")
    
    # Check file exists (if required)
    if must_exist and not resolved.exists():
        raise SecurityError(f"File does not exist: {filepath}")
    
    # Validate within allowed directories
    if not any(resolved.is_relative_to(d.resolve()) for d in allowed_dirs):
        raise SecurityError(
            f"Path {filepath} outside allowed directories: {allowed_dirs}"
        )
    
    # Check file size (if file exists)
    if must_exist and max_file_size and resolved.is_file():
        file_size = resolved.stat().st_size
        if file_size > max_file_size:
            raise SecurityError(
                f"File {filepath} exceeds size limit: "
                f"{file_size} > {max_file_size} bytes"
            )
    
    # Check extension
    if allowed_extensions:
        if resolved.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise SecurityError(
                f"File extension {resolved.suffix} not in whitelist: {allowed_extensions}"
            )
    
    return resolved


def validate_image_path(
    filepath: Path,
    allowed_dirs: List[Path],
    max_size: int = MAX_IMAGE_SIZE
) -> Path:
    """
    Validate image file path.
    
    Convenience wrapper for validate_filepath with image-specific defaults.
    
    Args:
        filepath: Path to image file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 100MB)
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(IMAGE_EXTENSIONS)
    )


def validate_video_path(
    filepath: Path,
    allowed_dirs: List[Path],
    max_size: int = MAX_VIDEO_SIZE
) -> Path:
    """
    Validate video file path.
    
    Args:
        filepath: Path to video file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 10GB)
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(VIDEO_EXTENSIONS)
    )


def validate_config_path(
    filepath: Path,
    allowed_dirs: List[Path],
    max_size: int = MAX_CONFIG_SIZE
) -> Path:
    """
    Validate configuration file path.
    
    Args:
        filepath: Path to config file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 10MB)
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(CONFIG_EXTENSIONS)
    )


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem operations.
    
    Removes/replaces dangerous characters while preserving extension.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length (default 255)
        
    Returns:
        Sanitized filename
        
    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("file<script>.jpg")
        'file_script_.jpg'
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    dangerous_chars = '<>:"|?*/\\;'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = 'unnamed'
    
    # Truncate if too long (preserve extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def build_safe_command(
    executable: str,
    args: List[str],
    dangerous_chars: Optional[set] = None
) -> List[str]:
    """
    Build safe command argument list for subprocess.
    
    Validates that arguments don't contain shell metacharacters
    that could be exploited via command injection.
    
    Args:
        executable: Command executable (e.g., 'ffmpeg')
        args: List of command arguments
        dangerous_chars: Set of characters to reject (default: shell metacharacters)
        
    Returns:
        List of command arguments (safe for subprocess.run without shell=True)
        
    Raises:
        SecurityError: If arguments contain dangerous characters
        
    Example:
        >>> cmd = build_safe_command('ffmpeg', ['-i', 'input.mp4', 'output.mp4'])
        >>> subprocess.run(cmd, check=True)  # Safe
        
        >>> cmd = build_safe_command('ffmpeg', ['-i', 'input.mp4; rm -rf /'])
        Traceback (most recent call last):
        SecurityError: Argument contains dangerous characters
    """
    if dangerous_chars is None:
        # Common shell metacharacters
        dangerous_chars = set(";&|`$()<>")
    
    # Validate executable
    if any(char in executable for char in dangerous_chars):
        raise SecurityError(
            f"Executable '{executable}' contains dangerous characters"
        )
    
    # Validate all arguments
    for arg in args:
        if any(char in str(arg) for char in dangerous_chars):
            raise SecurityError(
                f"Argument '{arg}' contains dangerous characters: {dangerous_chars}"
            )
    
    # Build command list
    return [executable] + [str(arg) for arg in args]


# Deprecated alias for backwards compatibility
def validate_file_path(*args, **kwargs):
    """Deprecated: Use validate_filepath instead."""
    import warnings
    warnings.warn(
        "validate_file_path is deprecated, use validate_filepath",
        DeprecationWarning,
        stacklevel=2
    )
    return validate_filepath(*args, **kwargs)
