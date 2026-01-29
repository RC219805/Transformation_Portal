"""Atomic write operations for lux_depth_v3 artifact writers.

Provides unified atomic write primitives for all lux_depth_v3 artifact types:
- Depth maps (16-bit PNG via cv2)
- PBR maps (8-bit PNG via PIL)
- Future artifact types

All operations use atomic rename (os.replace) and guarantee:
- No partial writes visible to readers
- Deterministic FD cleanup (no leaks)
- No orphaned temp files on failure
- Temp files created in destination directory (same filesystem)
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional, BinaryIO
from contextlib import contextmanager

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore


@contextmanager
def atomic_temp_file(
    output_path: Path,
    suffix: str = ".tmp",
    prefix: str = ".tmp_"
):
    """Context manager for atomic temp file creation.

    Creates a temporary file in the same directory as output_path,
    then atomically renames it on successful exit.

    Args:
        output_path: Final destination path
        suffix: Temp file suffix (default: ".tmp")
        prefix: Temp file prefix (default: ".tmp_")

    Yields:
        Path to temporary file

    Ensures:
        - Temp file is in same directory as output_path
        - Atomic rename via os.replace on success
        - Cleanup of temp file on failure
        - No FD leaks

    Example:
        >>> with atomic_temp_file(Path("output.png"), suffix=".png") as temp_path:
        ...     temp_path.write_bytes(data)
        # output.png now exists atomically
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (same filesystem for atomic rename)
    temp_fd, temp_path_str = tempfile.mkstemp(
        suffix=suffix,
        dir=output_path.parent,
        prefix=prefix
    )
    temp_path = Path(temp_path_str)

    try:
        # Close FD immediately - caller will use path directly
        os.close(temp_fd)

        yield temp_path

        # Atomic rename on success
        os.replace(temp_path, output_path)

    except Exception:
        # Cleanup temp file on any failure
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(output_path: Path, data: bytes) -> Path:
    """Atomically write bytes to file.

    Args:
        output_path: Destination file path
        data: Bytes to write

    Returns:
        Path to written file (same as output_path)

    Raises:
        IOError: If write fails

    Example:
        >>> path = atomic_write_bytes(Path("output.bin"), b"hello")
        >>> assert path.read_bytes() == b"hello"
    """
    try:
        with atomic_temp_file(output_path) as temp_path:
            temp_path.write_bytes(data)
        return Path(output_path)
    except Exception as e:
        raise IOError(f"Failed to write {output_path}") from e


def atomic_write_pil_png(
    output_path: Path,
    pil_image: "Image.Image",
    optimize: bool = True,
    **save_kwargs
) -> Path:
    """Atomically write PIL Image as PNG.

    Args:
        output_path: Destination file path
        pil_image: PIL Image to save
        optimize: Whether to optimize PNG (default: True)
        **save_kwargs: Additional arguments for PIL Image.save()

    Returns:
        Path to written file (same as output_path)

    Raises:
        ImportError: If PIL not available
        IOError: If write fails

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100))
        >>> path = atomic_write_pil_png(Path("output.png"), img)
        >>> assert path.exists()
    """
    if not HAS_PIL:
        raise ImportError(
            "Pillow required for atomic_write_pil_png. Install with: pip install Pillow"
        )

    try:
        with atomic_temp_file(output_path, suffix=".png") as temp_path:
            # Save directly to temp file path
            pil_image.save(
                temp_path,
                format='PNG',
                optimize=optimize,
                **save_kwargs
            )
        return Path(output_path)
    except Exception as e:
        raise IOError(f"Failed to write PNG {output_path}") from e


def atomic_write_with_fd(
    output_path: Path,
    writer_func,
    suffix: str = ".tmp"
) -> Path:
    """Atomically write using a file descriptor-based writer function.

    For writers that need an open file descriptor (e.g., cv2.imwrite with fdopen).

    Args:
        output_path: Destination file path
        writer_func: Callable taking (fd, temp_path) and writing to FD
        suffix: Temp file suffix (default: ".tmp")

    Returns:
        Path to written file (same as output_path)

    Raises:
        IOError: If write fails

    Example:
        >>> def write_cv2_image(fd, temp_path):
        ...     # Close FD first if writer uses path
        ...     os.close(fd)
        ...     cv2.imwrite(str(temp_path), image_data)
        >>> path = atomic_write_with_fd(Path("out.png"), write_cv2_image, ".png")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd = None
    temp_path = None

    try:
        # Create temp file with FD
        temp_fd, temp_path_str = tempfile.mkstemp(
            suffix=suffix,
            dir=output_path.parent,
            prefix=".tmp_"
        )
        temp_path = Path(temp_path_str)

        # Call writer function - it's responsible for closing FD if needed
        writer_func(temp_fd, temp_path)

        # Close FD if writer didn't consume it
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass  # Already closed by writer

        # Atomic rename
        os.replace(temp_path, output_path)

        return Path(output_path)

    except Exception as e:
        # Cleanup
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass

        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

        raise IOError(f"Failed to write {output_path}") from e
