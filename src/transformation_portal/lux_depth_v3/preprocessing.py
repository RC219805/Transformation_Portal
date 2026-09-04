"""Preprocessing utilities for image normalization.

Standardizes image validation and normalization for V3 depth inference.

Design:
- PIL + NumPy only (no torch/transformers)
- Validates image integrity and format
- Normalizes to float32 RGB [0, 1]
- Enforces multiple-of-14 dimensions for Depth Anything V3
- Preserves original dimensions for post-processing
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image

from .raw_loader import RAW_EXTENSIONS, is_raw_file

logger = logging.getLogger(__name__)

try:
    import cv2 as _opencv  # type: ignore
except ImportError:
    _opencv = None  # type: ignore[assignment]

# Supported image formats (standard + RAW)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"} | RAW_EXTENSIONS

# Depth Anything V3 requires dimensions to be multiples of 14
DIMENSION_MULTIPLE = 14
MIN_DIMENSION = 14


def validate_image_format(image_path: Union[str, Path]) -> Path:
    """Validate image format and integrity.

    Checks:
    - File exists
    - Extension is supported
    - File can be opened by PIL
    - Image data is not corrupt

    Args:
        image_path: Path to image file

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format unsupported or image corrupt
    """
    image_path = Path(image_path)

    # Check existence
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check extension
    ext = image_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {ext}. " f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    # Verify image integrity
    try:
        # RAW files require different validation (PIL can't open them)
        if is_raw_file(image_path):
            # For RAW, just check file is readable
            # Full validation happens during canonical ingest decode.
            with open(image_path, "rb") as f:
                # Read first few bytes to ensure file is readable
                header = f.read(16)
                if len(header) < 16:
                    raise ValueError("RAW file too small or empty")
        else:
            # Standard image validation for non-RAW
            # Open and verify (checks file structure)
            with Image.open(image_path) as img:
                img.verify()

            # verify() invalidates the image object, reopen to test pixel load
            with Image.open(image_path) as img:
                img.load()  # Force load pixel data

    except Exception as e:
        raise ValueError(f"Image file corrupt or invalid: {image_path}") from e

    return image_path


def preprocess_image(
    image: Union[np.ndarray, Path, str],
    target_size: Optional[int] = None,
    raw_config: Optional[Any] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image for depth inference.

    Converts to RGB float32 [0, 1] and ensures dimensions are multiples of 14.

    Processing steps:
    1. Load/validate image
    2. Convert to RGB float32 [0, 1]
    3. Optionally resize to target_size (long edge)
    4. Enforce multiple-of-14 dimensions
    5. Return processed image + original shape

    Args:
        image: Input as numpy array, Path, or str
        target_size: Optional target size for long edge (maintains aspect)
        raw_config: Optional config object with RAW ingest knobs (
            raw_ingest_mode,
            raw_wb_mode, raw_demosaic
        ). Uses deterministic defaults when omitted.

    Returns:
        Tuple of:
            - Processed image (float32, HxWx3, [0, 1])
            - Original shape (H, W) before any resizing

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image invalid or unsupported format
    """
    # Load image
    if isinstance(image, (str, Path)):
        image_path = validate_image_format(image)

        # Phase C1: RAW ingest is routed through canonical
        # spatial_ai.ingest contract.
        if is_raw_file(image_path):
            from .ingest_adapter import decode_for_lux_depth

            ingest_cfg = raw_config or SimpleNamespace(
                raw_ingest_mode="auto",
                raw_wb_mode="camera",
                raw_demosaic="AHD",
            )  # type: ignore[arg-type]
            decoded_rgb = decode_for_lux_depth(image_path, cast(Any, ingest_cfg))
            return preprocess_from_linear_ingest(decoded_rgb, target_size=target_size)

        pil_img = Image.open(image_path).convert("RGB")
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL for consistent processing
        if image.ndim == 2:
            # Grayscale
            pil_img = Image.fromarray(image, mode="L")
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode="RGB")
            elif image.dtype == np.float32 or image.dtype == np.float64:
                # Convert float [0, 1] to uint8 for PIL
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode="RGB")
            else:
                raise ValueError(f"Unsupported array dtype: {image.dtype}")
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA - drop alpha channel
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode="RGBA")
            else:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode="RGBA")
        else:
            raise ValueError(f"Unsupported array shape: {image.shape}. " f"Expected (H, W), (H, W, 3), or (H, W, 4)")
    else:
        raise TypeError(f"Image must be np.ndarray, Path, or str. Got: {type(image)}")

    # Save original dimensions
    original_h, original_w = (pil_img.size[1], pil_img.size[0])  # PIL uses (W, H)

    # Convert to RGB (handles grayscale, RGBA)
    pil_img = pil_img.convert("RGB")

    # Optional resize to target size (long edge)
    if target_size is not None:
        pil_img = _resize_keep_aspect(pil_img, target_size)

    # Convert to numpy float32 [0, 1]
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    # Enforce multiple-of-14 dimensions
    img_array = _enforce_dimension_multiple(img_array, DIMENSION_MULTIPLE)

    return img_array, (original_h, original_w)


def preprocess_image_snapshot(
    image_path: Union[str, Path],
    target_size: Optional[int] = None,
    raw_config: Optional[Any] = None,
    *,
    opened_file_stat_validator: Optional[Callable[[os.stat_result], None]] = None,
    verify_snapshot: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int], Optional[str]]:
    """Decode one immutable source-byte snapshot and return its SHA-256.

    Inputs are opened without following a final symlink when the platform
    supports it, then copied once from that regular-file descriptor into an
    immutable snapshot. Standard images decode from an in-memory spool. RAW
    decoders require a pathname, so they consume a private, uniquely named
    temporary copy with the original suffix. The digest and decoded pixels
    therefore describe the exact same source bytes even if the source path is
    replaced after it is opened. ``opened_file_stat_validator`` is invoked
    before any source bytes are read. ``verify_snapshot`` runs PIL's strict
    verifier for standard images against the same immutable snapshot.
    """

    path = Path(image_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {extension}. " f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"Image input must be a regular non-symlink file: {path}")

    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, open_flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(f"Image input must be a regular non-symlink file: {path}") from exc
        raise
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Image input is not a regular file: {path}")

        def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                int(value.st_dev),
                int(value.st_ino),
                int(stat.S_IFMT(value.st_mode)),
                int(value.st_size),
                int(value.st_mtime_ns),
                int(value.st_ctime_ns),
            )

        # On platforms without O_NOFOLLOW, this comparison also detects an
        # ordinary replacement between the lexical lstat and descriptor open.
        if _stable_identity(path_stat) != _stable_identity(source_stat):
            raise ValueError(f"Image input changed before its byte snapshot was opened: {path}")
        if opened_file_stat_validator is not None:
            opened_file_stat_validator(source_stat)
        source = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    digest = hashlib.sha256()
    copied = 0

    def copy_source(snapshot: Any) -> None:
        nonlocal copied
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            snapshot.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
        copied_stat = os.fstat(source.fileno())
        if copied != source_stat.st_size or _stable_identity(copied_stat) != _stable_identity(source_stat):
            raise ValueError(f"Image input changed while its byte snapshot was captured: {path}")

    with source:
        if is_raw_file(path):
            with tempfile.TemporaryDirectory(prefix="tp-prepared-raw-") as snapshot_dir_name:
                snapshot_dir = Path(snapshot_dir_name)
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix="input-",
                    suffix=path.suffix,
                    dir=snapshot_dir,
                    delete=False,
                ) as raw_snapshot:
                    snapshot_path = Path(raw_snapshot.name)
                    copy_source(raw_snapshot)
                    raw_snapshot.flush()
                    os.fsync(raw_snapshot.fileno())
                    os.fchmod(raw_snapshot.fileno(), 0o400)
                    snapshot_stat = os.fstat(raw_snapshot.fileno())

                # RAW decoders require a pathname. Keep that pathname in a
                # non-writable private directory during decode, then reopen it
                # without following symlinks and rehash the exact inode before
                # allowing its pixels to authorize cache/evidence identity.
                os.chmod(snapshot_dir, 0o500)
                try:
                    processed, original_shape = preprocess_image(
                        snapshot_path,
                        target_size=target_size,
                        raw_config=raw_config,
                    )

                    snapshot_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
                    snapshot_flags |= getattr(os, "O_NOFOLLOW", 0)
                    snapshot_descriptor = os.open(snapshot_path, snapshot_flags)
                    try:
                        reopened_snapshot_stat = os.fstat(snapshot_descriptor)
                        if _stable_identity(reopened_snapshot_stat) != _stable_identity(snapshot_stat):
                            raise ValueError(f"RAW byte snapshot changed while it was decoded: {path}")
                        snapshot_digest = hashlib.sha256()
                        with os.fdopen(snapshot_descriptor, "rb", closefd=True) as reopened_snapshot:
                            snapshot_descriptor = -1
                            while True:
                                chunk = reopened_snapshot.read(1024 * 1024)
                                if not chunk:
                                    break
                                snapshot_digest.update(chunk)
                    finally:
                        if snapshot_descriptor >= 0:
                            os.close(snapshot_descriptor)

                    if snapshot_digest.hexdigest() != digest.hexdigest():
                        raise ValueError(f"RAW byte snapshot changed while it was decoded: {path}")
                    final_stat = os.fstat(source.fileno())
                    if _stable_identity(final_stat) != _stable_identity(source_stat):
                        raise ValueError(f"Image input changed while its byte snapshot was decoded: {path}")
                    return processed, original_shape, digest.hexdigest()
                finally:
                    os.chmod(snapshot_dir, 0o700)

        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b") as snapshot:
            copy_source(snapshot)
            try:
                snapshot.seek(0)
                if verify_snapshot:
                    with Image.open(snapshot) as verified_image:
                        verified_image.verify()
                    snapshot.seek(0)
                with Image.open(snapshot) as image:
                    image.load()
                    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            except Exception as exc:
                raise ValueError(f"Image file corrupt or invalid: {path}") from exc
            final_stat = os.fstat(source.fileno())
            if _stable_identity(final_stat) != _stable_identity(source_stat):
                raise ValueError(f"Image input changed while its byte snapshot was decoded: {path}")

    processed, original_shape = preprocess_image(rgb, target_size=target_size, raw_config=raw_config)
    return processed, original_shape, digest.hexdigest()


def probe_image_dimensions(
    image_path: Union[str, Path],
    *,
    raw_config: Optional[Any] = None,
) -> tuple[int, int]:
    """Read exact image dimensions without allocating decoded RGB pixels."""

    path = Path(image_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {extension}. " f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if is_raw_file(path):
        if raw_config is None:
            raise ValueError("RAW dimension probing requires the carried ingest configuration")
        from .ingest_adapter import probe_raw_dimensions

        return probe_raw_dimensions(path, cast(Any, raw_config))
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"Image header corrupt or invalid: {path}") from exc
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError(f"Image exposes invalid dimensions: {width}x{height}")
    return width, height


def preprocess_from_linear_ingest(
    image: np.ndarray,
    target_size: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess canonical decoded ingest tensors for depth inference.

    Args:
        image: Decoded float32 HxWx3 tensor.
        target_size: Optional target size for long edge.

    Returns:
        Tuple of processed image array and original (H, W).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Decoded ingest tensor must be (H, W, 3), got: {image.shape}")

    img_array = np.asarray(image, dtype=np.float32)
    original_h, original_w = img_array.shape[:2]

    # Canonical depth path expects [0,1] RGB floats.
    img_array = np.clip(img_array, 0.0, 1.0)

    if target_size is not None:
        image_uint8 = (img_array * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(image_uint8, mode="RGB")
        pil_img = _resize_keep_aspect(pil_img, target_size)
        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

    img_array = _enforce_dimension_multiple(img_array, DIMENSION_MULTIPLE)
    return img_array.astype(np.float32), (original_h, original_w)


def _resize_keep_aspect(pil_img: Image.Image, target_size: int) -> Image.Image:
    """Resize image to target size (long edge), maintaining aspect ratio.

    Args:
        pil_img: PIL Image
        target_size: Target size for long edge

    Returns:
        Resized PIL Image
    """
    w, h = pil_img.size

    # Determine scaling factor (long edge to target_size)
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    # Use LANCZOS for quality
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _enforce_dimension_multiple(img_array: np.ndarray, multiple: int) -> np.ndarray:
    """Enforce that image dimensions are multiples of a given value.

    Uses center crop + pad to preserve pixel fidelity rather than resampling.
    This avoids subtle quality degradation from resizing the entire image.

    Args:
        img_array: Image array (H, W, C) float32
        multiple: Required multiple (e.g., 14 for Depth Anything V3)

    Returns:
        Image array with compliant dimensions (may be cropped/padded)
    """
    h, w = img_array.shape[:2]

    # Compute target dimensions (nearest multiple)
    # Round down to nearest multiple, then ensure minimum
    new_h = max(multiple, (h // multiple) * multiple)
    new_w = max(multiple, (w // multiple) * multiple)

    # Only adjust if dimensions changed
    if (new_h, new_w) != (h, w):
        # Strategy: handle crop and pad independently per dimension
        # This supports mixed scenarios (e.g., 15x10 → 14x14: crop width, pad height)

        # Handle height dimension
        if new_h < h:
            # Crop height (center crop)
            crop_top = (h - new_h) // 2
            img_array = img_array[crop_top : crop_top + new_h, :]
        elif new_h > h:
            # Pad height (symmetric padding)
            pad_h = new_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            img_array = np.pad(
                img_array,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="edge",
            )

        # Handle width dimension
        if new_w < w:
            # Crop width (center crop)
            crop_left = (w - new_w) // 2
            img_array = img_array[:, crop_left : crop_left + new_w]
        elif new_w > w:
            # Pad width (symmetric padding)
            pad_w = new_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img_array = np.pad(
                img_array,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

        logger.debug(f"Enforced dimension multiple: ({h}, {w}) → ({new_h}, {new_w})")

    return img_array


def preprocess_image_linear(
    image: Union[np.ndarray, Path, str],
    target_size: Optional[int] = None,
    verify_linearity: bool = True,
    apex_strict_formats: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image with linear light preservation for APEX pipeline.

    This is the APEX-compliant preprocessing path that enforces:
    - Deterministic format boundary (RAW + TIFF only by default)
    - Linear light preservation (no gamma encoding)
    - Floating point dtype (no uint8/uint16 leakage)
    - Value range validation [0, 1]
    - 16-bit precision preservation for RAW/TIFF

    Per Spatial AI Foundation ROADMAP (Section I: Data Fidelity is Sacred):
    "Training inputs MUST preserve linear-light relationships."

    Processing steps:
    1. Validate format boundary (RAW + TIFF only unless apex_strict_formats=False)
    2. Load/validate image (RAW → linear uint16, TIFF → preserve bit depth)
    3. Convert to linear float32 [0, 1] preserving precision
    4. Validate linearity (dtype, range, gamma detection)
    5. Optionally resize to target_size (long edge)
    6. Enforce multiple-of-14 dimensions
    7. Return validated linear image + original shape

    Args:
        image: Input as numpy array, Path, or str
               For RAW: must be RAW file (linear output enforced)
               For TIFF: must be linear (gamma-encoded rejected)
               For JPEG/PNG: rejected by default (set apex_strict_formats=False to allow)
        target_size: Optional target size for long edge (maintains aspect)
        verify_linearity: Whether to run linear verification checks (default True)
        apex_strict_formats: Whether to enforce RAW + TIFF only (default True).
                            Set to False only if you understand the data-fidelity tradeoffs.

    Returns:
        Tuple of:
            - Processed image (float32, HxWx3, [0, 1], linear light)
            - Original shape (H, W) before any resizing

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image invalid, gamma-encoded, unsupported format,
            or violates format boundary
        DtypeViolationError: If dtype is uint8/uint16 after conversion
        RangeViolationError: If values outside [0, 1]
        LinearityViolationError: If gamma encoding detected

    Example:
        >>> # Load linear RAW file (default behavior)
        >>> img, orig_shape = preprocess_image_linear("photo.CR2")
        >>> # img is float32 [0, 1] linear light, verified

        >>> # Load 16-bit linear TIFF (default behavior)
        >>> img, orig_shape = preprocess_image_linear("render.tif")
        >>> # img is float32 [0, 1] linear light, preserving TIFF precision

        >>> # JPEG rejected by default (format boundary enforcement)
        >>> img, orig_shape = preprocess_image_linear("photo.jpg")
        >>> # → ValueError: APEX linear ingest only supports RAW + TIFF inputs

        >>> # Explicit escape hatch (discouraged)
        >>> img, orig_shape = preprocess_image_linear("photo.jpg", apex_strict_formats=False)
        >>> # Allowed but may violate linear-light preservation
    """
    from .linear_verify import DtypeViolationError, LinearityViolationError, RangeViolationError, verify_linear_ingest
    from .raw_loader import load_raw_as_rgb

    # Load image preserving bit depth and linearity
    if isinstance(image, (str, Path)):
        image_path = validate_image_format(image)

        # APEX ingest boundary: deterministic by format
        # Training ingest must be scene-referred by construction.
        # RAW files are inherently linear (sensor data).
        # TIFF files preserve bit depth and can carry linear data.
        # JPEG/PNG are display-referred and typically gamma-encoded.
        ext = image_path.suffix.lower()
        is_tiff = ext in {".tif", ".tiff"}

        if apex_strict_formats and (not is_raw_file(image_path)) and (not is_tiff):
            raise ValueError(
                f"APEX linear ingest only supports RAW + TIFF inputs. "
                f"Got: {ext or '<no extension>'}.\n"
                "Reason: JPEG/PNG are typically gamma-encoded (display-referred), "
                "violating linear-light preservation requirements.\n"
                "Use preprocess_image() for JPEG/PNG, or explicitly set "
                "apex_strict_formats=False if you understand the "
                "data-fidelity tradeoffs."
            )

        # Handle RAW files with linear output
        if is_raw_file(image_path):
            logger.debug(f"Loading RAW file with linear output: {image_path.name}")
            # load_raw_as_rgb with output_linear=True (default) gives uint16 linear
            rgb_array = load_raw_as_rgb(
                image_path,
                use_camera_wb=True,
                half_size=False,
                output_bps=16,  # 16-bit for precision
                output_linear=True,  # Linear RGB (enforced)
            )
            original_h, original_w = rgb_array.shape[:2]

            # Convert uint16 [0, 65535] → float32 [0, 1] preserving linearity
            img_array = rgb_array.astype(np.float32) / 65535.0

        else:
            # Handle standard formats (TIFF, PNG, JPEG)
            # Use tifffile for TIFF to preserve 16-bit
            if image_path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    import tifffile
                except ImportError as e:
                    raise ImportError(
                        "tifffile is required for linear TIFF processing "
                        "in APEX pipeline. "
                        "Install with: pip install tifffile\n"
                        "Or: pip install -e '.[tiff]'"
                    ) from e

                # Load TIFF preserving bit depth
                tiff_array = tifffile.imread(str(image_path))
                original_h, original_w = (
                    tiff_array.shape[0],
                    tiff_array.shape[1],
                )  # NumPy: (H, W, C)

                # Normalize based on dtype
                if tiff_array.dtype == np.uint8:
                    img_array = tiff_array.astype(np.float32) / 255.0
                elif tiff_array.dtype == np.uint16:
                    img_array = tiff_array.astype(np.float32) / 65535.0
                elif tiff_array.dtype in [np.float32, np.float64]:
                    img_array = tiff_array.astype(np.float32)
                    # Assume already in [0, 1] if float
                else:
                    raise ValueError(f"Unsupported TIFF dtype: {tiff_array.dtype}")

                # Ensure 3 channels
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    # Drop alpha
                    img_array = img_array[:, :, :3]
            else:
                # Use PIL for other formats (JPEG, PNG, etc.)
                pil_img = Image.open(image_path).convert("RGB")
                original_w, original_h = pil_img.size  # PIL returns (W, H)
                img_array = np.array(pil_img, dtype=np.float32) / 255.0

    elif isinstance(image, np.ndarray):
        # Handle numpy array input
        original_h, original_w = image.shape[0], image.shape[1]  # NumPy: (H, W, C)

        if image.dtype == np.uint8:
            img_array = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_array = image.astype(np.float32) / 65535.0
        elif image.dtype in [np.float32, np.float64]:
            img_array = image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported array dtype: {image.dtype}")

        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Unsupported array shape: {image.shape}")
    else:
        raise TypeError(f"Image must be np.ndarray, Path, or str. Got: {type(image)}")

    # Verify linearity before any further processing
    if verify_linearity:
        try:
            verify_linear_ingest(img_array)
            logger.debug("Linear ingest verification passed")
        except (DtypeViolationError, RangeViolationError, LinearityViolationError):
            # Let typed exceptions propagate for downstream orchestration
            raise
        except Exception as e:
            raise ValueError(
                f"Linear ingest verification failed: {e}\n"
                f"Image: {image if isinstance(image, (str, Path)) else 'numpy array'}\n"
                f"This indicates the input is gamma-encoded, has incorrect dtype, or invalid range."
            ) from e

    # Optional resize to target size (long edge)
    if target_size is not None:
        h, w = img_array.shape[:2]
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        # Resize using high-quality interpolation
        # Use OpenCV to avoid lossy float32→uint8→float32 conversion.
        if _opencv is None:
            raise ImportError(
                "OpenCV (cv2) is required for resizing in linear space for APEX pipeline. "
                "Install with: pip install opencv-python\n"
                "Or: pip install -e '.[cv2]'"
            )

        # OpenCV resize preserves dtype (float32 stays float32).
        # Use INTER_LANCZOS4 for high-quality resampling
        img_array = _opencv.resize(img_array, (new_w, new_h), interpolation=_opencv.INTER_LANCZOS4)  # type: ignore[assignment]

        # Clip to [0, 1] after resize (interpolation can introduce small out-of-range values)
        img_array = np.clip(img_array, 0.0, 1.0)

    # Enforce multiple-of-14 dimensions
    img_array = _enforce_dimension_multiple(img_array, DIMENSION_MULTIPLE)

    # Final verification after all processing
    if verify_linearity:
        verify_linear_ingest(img_array)

    return img_array, (original_h, original_w)


# Keep legacy stubs for backward compatibility (not yet used by V3 orchestrator)


def normalize_exif_orientation(input_path: Path, output_path: Path) -> None:
    """Normalize EXIF orientation by rotating image to upright position.

    Applies EXIF orientation tag to physically rotate/flip the image so it
    displays correctly regardless of viewer EXIF support. The output image
    will have orientation=1/Normal.

    Note: This function preserves EXIF metadata (with normalized orientation)
    but may re-encode the image. For JPEG inputs where no rotation is needed,
    the file is copied byte-for-byte to avoid lossy re-encoding.

    Args:
        input_path: Input image path
        output_path: Output image path (normalized). Can be same as input.

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If input_path is not a valid image
        OSError: If output cannot be written
    """
    import shutil

    from PIL import Image
    from PIL.ImageOps import exif_transpose

    # Validate input exists
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open image and check if transpose is needed
    try:
        img = Image.open(input_path)
    except (OSError, ValueError) as exc:
        # Normalize PIL-specific errors to a stable ValueError contract
        raise ValueError(f"Invalid image file: {input_path}") from exc

    try:
        # exif_transpose returns None if no transpose needed, else new image
        corrected = exif_transpose(img)

        if corrected is None:
            # No rotation needed - for JPEG, copy byte-for-byte to avoid lossy re-encode
            img.close()
            if input_path.suffix.lower() in {".jpg", ".jpeg"} and input_path != output_path:
                shutil.copy2(input_path, output_path)
                logger.debug("No EXIF rotation needed, copied file: %s -> %s", input_path, output_path)
                return
            elif input_path == output_path:
                # In-place, no changes needed
                logger.debug("No EXIF rotation needed, file unchanged: %s", input_path)
                return
            else:
                # Non-JPEG or same path - use PIL to copy (may re-encode)
                corrected = img.copy()
                img.close()
                img = Image.open(input_path)  # Re-open to get fresh copy

        # Extract and preserve EXIF data with normalized orientation
        exif_data = None
        try:
            exif_obj = img.getexif()
            if exif_obj:
                # Set orientation to Normal (1) since image is now physically rotated
                exif_obj[0x0112] = 1  # Orientation tag
                exif_data = exif_obj.tobytes()
        except Exception:
            # Image doesn't support EXIF or EXIF is malformed - continue without it
            pass

        # Save with preserved EXIF (orientation normalized)
        if exif_data:
            corrected.save(output_path, exif=exif_data)
        else:
            corrected.save(output_path)
        logger.debug("Normalized EXIF orientation: %s -> %s", input_path, output_path)

    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to process image: {input_path}") from exc
    finally:
        img.close()


def validate_depth_image_alignment(image_path: Path, depth_path: Path) -> bool:
    """Validate that depth map and image have matching dimensions.

    Checks that the RGB image and depth map dimensions are compatible.
    Allows tolerance for model-induced padding (Depth Anything V3 uses
    multiples of 14).

    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map (PNG, NPY, or TIFF)

    Returns:
        True if dimensions match (within padding tolerance), False otherwise

    Raises:
        FileNotFoundError: If either path doesn't exist
        ValueError: If files cannot be read as images/arrays
    """
    image_path = Path(image_path)
    depth_path = Path(depth_path)

    # Validate both exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Load RGB image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        raise ValueError(f"Cannot read image {image_path}: {e}") from e

    # Load depth map dimensions (supports .npy and image formats)
    try:
        if depth_path.suffix.lower() == ".npy":
            # Use memory-mapped loading so we can read the shape without
            # materializing potentially large depth arrays into memory.
            depth_arr = np.load(depth_path, mmap_mode="r")
            depth_height, depth_width = depth_arr.shape[:2]
        else:
            with Image.open(depth_path) as depth_img:
                depth_width, depth_height = depth_img.size
    except Exception as e:
        raise ValueError(f"Cannot read depth map {depth_path}: {e}") from e

    # Check exact match first
    if img_width == depth_width and img_height == depth_height:
        return True

    # Allow tolerance for DA3 padding (multiples of 14)
    # The depth map may be padded to the nearest multiple of 14
    # Note: This is a simple rounding helper. The _enforce_dimension_multiple()
    # function exists but operates on arrays. For integer arithmetic, local
    # helper is clearer and avoids unnecessary coupling.
    def round_to_multiple(val: int, multiple: int = DIMENSION_MULTIPLE) -> int:
        return ((val + multiple - 1) // multiple) * multiple

    padded_img_width = round_to_multiple(img_width)
    padded_img_height = round_to_multiple(img_height)

    if depth_width == padded_img_width and depth_height == padded_img_height:
        return True

    # Also check if image was padded from depth dimensions
    if img_width == round_to_multiple(depth_width) and img_height == round_to_multiple(depth_height):
        return True

    logger.debug(
        "Dimension mismatch: image=(%d, %d), depth=(%d, %d)",
        img_width,
        img_height,
        depth_width,
        depth_height,
    )
    return False
