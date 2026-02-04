"""Security and validation utilities."""

from pathlib import Path
from typing import Union


def validate_path(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """Validate path to prevent traversal attacks.

    Args:
        path: Path to validate
        base_dir: Base directory that path must be under

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path attempts to escape base_dir
    """
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()

    # Check if path is under base_dir
    try:
        path.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"Path traversal detected: {path} is not under {base_dir}")

    return path


def validate_image_extension(path: Path, allowed_extensions: tuple) -> None:
    """Validate image file extension.

    Args:
        path: Path to validate
        allowed_extensions: Tuple of allowed extensions (e.g., ('.jpg', '.png'))

    Raises:
        ValueError: If extension is not allowed
    """
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Invalid file extension: {path.suffix}. " f"Allowed: {allowed_extensions}")
