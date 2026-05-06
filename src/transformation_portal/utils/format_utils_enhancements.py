"""Enhanced Format Utilities for Transformation Portal

This module provides advanced format detection, conversion, and TIFF handling
capabilities to extend the base format_utils.py functionality.

New Features:
- Option 2: Enhanced format detection (MIME types, magic numbers, integrity)
- Option 3: Format conversion utilities (single file, batch, smart convert)
- Option 4: Improved 16-bit TIFF handling (metadata, compression, multi-page)

Usage:
    Add these functions to your existing format_utils.py or import them:

    from format_utils_enhancements import (
        detect_format_from_content,
        validate_image_integrity,
        convert_image_format,
        batch_convert_directory,
        save_tiff_16bit,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

try:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully
except ImportError as exc:
    raise ImportError("Pillow is required. Install with: pip install Pillow") from exc

# Optional but recommended for advanced features
try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    warnings.warn("tifffile not available. 16-bit TIFF support limited. Install with: pip install tifffile")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    if TYPE_CHECKING:
        import numpy as np
    HAS_NUMPY = False
# ==============================================================================
# OPTION 2: Enhanced Format Detection
# ==============================================================================


def detect_format_from_content(path: Union[str, Path]) -> Optional[str]:
    """Detect image format from file content (magic numbers), not just extension.

    This is more reliable than extension-based detection and can identify
    files with wrong extensions or no extensions.

    Args:
        path: Path to image file

    Returns:
        Detected format string (e.g., 'JPEG', 'PNG', 'TIFF') or None if unknown

    Examples:
        >>> detect_format_from_content('image_no_ext')  # File with no extension
        'JPEG'
        >>> detect_format_from_content('wrong.txt')  # Actually a PNG
        'PNG'
        >>> detect_format_from_content('photo.jpg')
        'JPEG'
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return None

    try:
        with Image.open(path_obj) as img:
            return img.format
    except Exception:
        return None


def get_mime_type(path: Union[str, Path]) -> Optional[str]:
    """Get MIME type of image file.

    Args:
        path: Path to image file

    Returns:
        MIME type string (e.g., 'image/jpeg') or None

    Examples:
        >>> get_mime_type('photo.jpg')
        'image/jpeg'
        >>> get_mime_type('render.png')
        'image/png'
    """
    format_type = detect_format_from_content(path)

    if not format_type:
        return None

    mime_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "TIFF": "image/tif",
        "GIF": "image/gi",
        "BMP": "image/bmp",
        "WEBP": "image/webp",
        "ICO": "image/x-icon",
    }

    return mime_map.get(format_type.upper())


def validate_image_integrity(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """Validate that an image file is not corrupted and can be opened.

    Args:
        path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if image can be opened and is valid
        - error_message: None if valid, error description if invalid

    Examples:
        >>> validate_image_integrity('good_photo.jpg')
        (True, None)
        >>> validate_image_integrity('corrupted.jpg')
        (False, 'Cannot identify image file')
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return False, f"File does not exist: {path}"

    if not path_obj.is_file():
        return False, f"Path is not a file: {path}"

    try:
        with Image.open(path_obj) as img:
            # Try to load the image data to catch truncated files
            img.load()

            # Verify image has reasonable dimensions
            if img.width <= 0 or img.height <= 0:
                return False, f"Invalid dimensions: {img.width}x{img.height}"

            # Check if image is too large (over 100MP)
            if img.width * img.height > 100_000_000:
                warnings.warn(f"Very large image: {img.width}x{img.height} pixels")

            return True, None

    except Image.UnidentifiedImageError:
        return False, "Cannot identify image file (may be corrupted or unsupported format)"
    except OSError as e:
        return False, f"Cannot open image: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def get_image_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Extract comprehensive metadata from image file.

    Args:
        path: Path to image file

    Returns:
        Dictionary containing:
        - format: Image format (JPEG, PNG, etc.)
        - size: (width, height) tuple
        - mode: Color mode (RGB, RGBA, L, etc.)
        - bit_depth: Bits per channel
        - has_alpha: Whether image has alpha channel
        - file_size: File size in bytes
        - exif: EXIF data if available

    Examples:
        >>> meta = get_image_metadata('photo.jpg')
        >>> print(meta['size'])
        (4000, 3000)
        >>> print(meta['bit_depth'])
        8
    """
    path_obj = Path(path)
    metadata = {
        "format": None,
        "size": None,
        "mode": None,
        "bit_depth": None,
        "has_alpha": False,
        "file_size": path_obj.stat().st_size if path_obj.exists() else 0,
        "exif": None,
    }

    try:
        with Image.open(path_obj) as img:
            metadata["format"] = img.format
            metadata["size"] = img.size
            metadata["mode"] = img.mode
            metadata["has_alpha"] = "A" in img.mode

            # Calculate bit depth
            mode_bit_depth = {
                "1": 1,
                "L": 8,
                "P": 8,
                "RGB": 8,
                "RGBA": 8,
                "CMYK": 8,
                "YCbCr": 8,
                "LAB": 8,
                "HSV": 8,
                "I": 32,
                "F": 32,
                "I;16": 16,
                "I;16B": 16,
                "I;16L": 16,
                "I;16S": 16,
                "I;16BS": 16,
                "I;16LS": 16,
            }
            metadata["bit_depth"] = mode_bit_depth.get(img.mode, 8)

            # Try to get EXIF data
            try:
                exif = img.getexif()
                if exif:
                    metadata["exif"] = {k: str(v) for k, v in exif.items()}
            except Exception as exif_exc:
                warnings.warn(f"Failed to extract EXIF data from {path_obj}: {exif_exc}")
                metadata["exif_error"] = str(exif_exc)

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


# ==============================================================================
# OPTION 3: Format Conversion Utilities
# ==============================================================================


def convert_image_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    quality: int = 95,
    preserve_metadata: bool = True,
    optimize: bool = True,
) -> bool:
    """Convert image from one format to another with quality preservation.

    Args:
        input_path: Source image file
        output_path: Destination image file (extension determines format)
        quality: JPEG/WebP quality (1-100), ignored for lossless formats
        preserve_metadata: Whether to copy EXIF/metadata
        optimize: Whether to optimize output file size

    Returns:
        True if conversion successful, False otherwise

    Examples:
        >>> convert_image_format('photo.tif', 'photo.jpg', quality=95)
        True
        >>> convert_image_format('render.jpg', 'render.png')
        True
        >>> convert_image_format('image.bmp', 'image.webp', quality=90)
        True
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    try:
        with Image.open(input_path) as img:
            # Convert to RGB if saving to format that doesn't support alpha
            output_ext = output_path.suffix.lower()
            if output_ext in {".jpg", ".jpeg"} and img.mode in ("RGBA", "LA", "P"):
                # Create white background
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                rgb_img.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
                img = rgb_img

            # Prepare save parameters
            save_kwargs = {"optimize": optimize}

            # Format-specific parameters
            if output_ext in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = quality
                save_kwargs["subsampling"] = 0  # Best quality
            elif output_ext == ".webp":
                save_kwargs["quality"] = quality
            elif output_ext in {".ti", ".tiff"}:
                save_kwargs["compression"] = "tiff_lzw"  # Good lossless compression
            elif output_ext == ".png":
                save_kwargs["compress_level"] = 6  # Balance speed/size

            # Preserve metadata if requested
            if preserve_metadata:
                exif = img.getexif()
                if exif:
                    save_kwargs["exif"] = exif

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the image
            img.save(output_path, **save_kwargs)

            return True

    except Exception as e:
        warnings.warn(f"Failed to convert {input_path} to {output_path}: {str(e)}")
        return False


def batch_convert_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_format: str,
    quality: int = 95,
    recursive: bool = False,
    preserve_metadata: bool = True,
) -> Dict[str, int]:
    """Convert all images in directory to target format.

    Args:
        input_dir: Source directory with images
        output_dir: Destination directory for converted images
        target_format: Target extension (e.g., '.jpg', '.png', '.tif')
        quality: Quality for lossy formats (1-100)
        recursive: Whether to process subdirectories
        preserve_metadata: Whether to preserve EXIF data

    Returns:
        Dictionary with conversion statistics:
        - 'total': Total files found
        - 'success': Successfully converted
        - 'failed': Failed conversions
        - 'skipped': Already in target format

    Examples:
        >>> stats = batch_convert_directory('./raw', './jpg', '.jpg', quality=95)
        >>> print(f"Converted {stats['success']} of {stats['total']} images")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    target_format = target_format.lower()

    if not target_format.startswith("."):
        target_format = f".{target_format}"

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    # Get all image files
    pattern = "**/*" if recursive else "*"
    image_extensions = {".jpg", ".jpeg", ".png", ".ti", ".tif", ".bmp", ".webp", ".gif"}

    for input_file in input_dir.glob(pattern):
        if not input_file.is_file():
            continue

        if input_file.suffix.lower() not in image_extensions:
            continue

        stats["total"] += 1

        # Skip if already in target format
        if input_file.suffix.lower() == target_format:
            stats["skipped"] += 1
            continue

        # Construct output path preserving directory structure
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(target_format)

        # Convert
        if convert_image_format(input_file, output_file, quality, preserve_metadata):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def smart_convert(
    input_path: Union[str, Path], output_path: Union[str, Path], auto_quality: bool = True, preserve_bit_depth: bool = True
) -> bool:
    """Intelligently convert image choosing best quality settings.

    Automatically determines optimal quality settings based on:
    - Input format and quality
    - Output format capabilities
    - Image content (photos vs graphics)

    Args:
        input_path: Source image
        output_path: Destination image
        auto_quality: Automatically determine best quality
        preserve_bit_depth: Try to preserve 16-bit depth if possible

    Returns:
        True if successful

    Examples:
        >>> smart_convert('photo.tif', 'photo.jpg')  # Uses quality=95
        True
        >>> smart_convert('logo.png', 'logo.webp')  # Uses lossless
        True
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Get input metadata
    metadata = get_image_metadata(input_path)
    output_ext = output_path.suffix.lower()

    # Determine quality
    quality = 95  # Default high quality

    if auto_quality:
        # Use higher quality for photos, lower for graphics
        if metadata.get("format") == "JPEG":
            quality = 93  # Slight transcode
        elif metadata.get("has_alpha"):
            quality = 95  # Preserve transparency quality
        else:
            quality = 90  # Good balance for most content

    # Handle 16-bit preservation
    if preserve_bit_depth and metadata.get("bit_depth") == 16:
        if output_ext in {".ti", ".tif", ".png"}:
            # Use specialized converter for formats that support 16-bit (TIFF/PNG)
            return convert_tiff_preserve_depth(input_path, output_path)

    return convert_image_format(input_path, output_path, quality=quality)


# ==============================================================================
# OPTION 4: Improved 16-bit TIFF Handling
# ==============================================================================


def check_tifffile_available() -> bool:
    """Check if tifffile library is available for 16-bit support.

    Returns:
        True if tifffile is available, False otherwise
    """
    return HAS_TIFFFILE


def save_tiff_16bit(
    image_array: "np.ndarray", output_path: Union[str, Path], compression: str = "lzw", metadata: Optional[Dict] = None
) -> bool:
    """Save image as 16-bit TIFF with metadata and compression.

    Requires tifffile package for 16-bit support.

    Args:
        image_array: NumPy array with image data (uint16 or uint8)
        output_path: Where to save TIFF
        compression: 'none', 'lzw', 'jpeg', 'zip' (lzw recommended)
        metadata: Optional metadata dict to embed

    Returns:
        True if successful

    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 65536, (1000, 1000, 3), dtype=np.uint16)
        >>> save_tiff_16bit(img, 'output.tif', compression='lzw')
        True
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile required for 16-bit TIFF. Install with: pip install tifffile")

    if not HAS_NUMPY:
        raise ImportError("numpy required. Install with: pip install numpy")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Prepare metadata
        tiff_metadata = {}
        if metadata:
            tiff_metadata.update(metadata)

        # Save with tifffile
        tifffile.imwrite(
            output_path,
            image_array,
            compression=compression,
            metadata=tiff_metadata,
            photometric="rgb" if image_array.ndim == 3 else "minisblack",
        )

        return True

    except Exception as e:
        warnings.warn(f"Failed to save 16-bit TIFF: {str(e)}")
        return False


def load_tiff_preserve_depth(path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load TIFF preserving original bit depth.

    Args:
        path: Path to TIFF file

    Returns:
        Tuple of (array, bit_depth)
        - array: NumPy array with original bit depth
        - bit_depth: Original bit depth (8, 16, 32)

    Examples:
        >>> arr, depth = load_tiff_preserve_depth('photo.tiff')
        >>> print(f"Loaded {arr.shape} image with {depth}-bit depth")
    """
    path = Path(path)

    if HAS_TIFFFILE:
        try:
            with tifffile.TiffFile(path) as tif:
                array = tif.asarray()
                # Determine bit depth from dtype
                if array.dtype == np.uint8:
                    bit_depth = 8
                elif array.dtype == np.uint16:
                    bit_depth = 16
                elif array.dtype in (np.uint32, np.float32):
                    bit_depth = 32
                else:
                    bit_depth = 8  # Fallback

                return array, bit_depth
        except Exception as e:
            warnings.warn(f"tifffile failed, falling back to PIL: {str(e)}")

    # Fallback to PIL
    try:
        with Image.open(path) as img:
            array = np.array(img)
            # Determine bit depth from img.mode
            if img.mode == "I;16":
                bit_depth = 16
            elif img.mode in ("I", "F"):
                bit_depth = 32
            else:
                bit_depth = 8
            return array, bit_depth
    except Exception:
        return None, None


def convert_tiff_preserve_depth(input_path: Union[str, Path], output_path: Union[str, Path], compression: str = "lzw") -> bool:
    """Convert TIFF while preserving bit depth and metadata.

    Args:
        input_path: Source TIFF
        output_path: Destination TIFF
        compression: Compression method

    Returns:
        True if successful
    """
    array, bit_depth = load_tiff_preserve_depth(input_path)

    if array is None:
        return False

    if bit_depth == 16 and HAS_TIFFFILE:
        return save_tiff_16bit(array, output_path, compression)
    else:
        # Fallback to PIL for 8-bit
        return convert_image_format(input_path, output_path)


def get_tiff_compression_info(path: Union[str, Path]) -> Optional[str]:
    """Get compression method used in TIFF file.

    Args:
        path: Path to TIFF file

    Returns:
        Compression method string or None
    """
    if not HAS_TIFFFILE:
        return None

    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            compression = page.compression
            return str(compression)
    except Exception:
        return None


def optimize_tiff_compression(
    input_path: Union[str, Path], output_path: Union[str, Path], target_compression: str = "lzw"
) -> Tuple[bool, Optional[float]]:
    """Re-compress TIFF with optimal compression, preserving quality.

    Args:
        input_path: Source TIFF
        output_path: Output TIFF
        target_compression: Desired compression ('lzw', 'zip', 'jpeg', 'none')

    Returns:
        Tuple of (success, compression_ratio)
        - success: True if successful
        - compression_ratio: Size reduction (e.g., 0.5 = 50% smaller)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    original_size = input_path.stat().st_size

    array, bit_depth = load_tiff_preserve_depth(input_path)
    if array is None:
        return False, None

    if bit_depth == 16 and HAS_TIFFFILE:
        success = save_tiff_16bit(array, output_path, compression=target_compression)
    else:
        success = convert_image_format(input_path, output_path)

    if success and output_path.exists():
        new_size = output_path.stat().st_size
        compression_ratio = new_size / original_size
        return True, compression_ratio

    return False, None


# ==============================================================================
# Convenience Functions
# ==============================================================================


def get_optimal_format_for_use_case(use_case: str, has_alpha: bool = False, requires_16bit: bool = False) -> str:
    """Suggest optimal image format for specific use case.

    Args:
        use_case: 'web', 'print', 'editing', 'archival', 'preview'
        has_alpha: Whether transparency is needed
        requires_16bit: Whether 16-bit depth is required

    Returns:
        Recommended file extension

    Examples:
        >>> get_optimal_format_for_use_case('web', has_alpha=True)
        '.png'
        >>> get_optimal_format_for_use_case('print', requires_16bit=True)
        '.tif'
    """
    if requires_16bit:
        return ".tif"  # Only format that reliably supports 16-bit

    if use_case == "web":
        return ".png" if has_alpha else ".webp"
    elif use_case == "print":
        return ".tif"
    elif use_case == "editing":
        return ".tif" if requires_16bit else ".png"
    elif use_case == "archival":
        return ".tif"
    elif use_case == "preview":
        return ".jpg"
    else:
        return ".png"  # Safe default


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Example: Enhanced format detection
    print("=== Format Detection Examples ===")
    test_file = "example.jpg"

    if Path(test_file).exists():
        print(f"Format from content: {detect_format_from_content(test_file)}")
        print(f"MIME type: {get_mime_type(test_file)}")
        is_valid, error = validate_image_integrity(test_file)
        print(f"Valid: {is_valid}, Error: {error}")
        print(f"Metadata: {get_image_metadata(test_file)}")

    # Example: Format conversion
    print("\n=== Conversion Examples ===")
    # convert_image_format('input.tif', 'output.jpg', quality=95)
    # stats = batch_convert_directory('./input', './output', '.png')
    # print(f"Conversion stats: {stats}")

    # Example: 16-bit TIFF
    print("\n=== TIFF Support ===")
    print(f"tifffile available: {check_tifffile_available()}")
    print(f"Optimal format for print: {get_optimal_format_for_use_case('print', requires_16bit=True)}")
