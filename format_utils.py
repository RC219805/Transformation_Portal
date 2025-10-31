"""Format validation and utilities for Transformation Portal.

This module provides utilities for validating and working with supported
image and video file formats across all pipelines.

Functions:
    is_supported_image_format: Check if a file has a supported image extension
    is_supported_video_format: Check if a file has a supported video extension
    is_supported_tiff_format: Check if a file is a TIFF format
    validate_format: Validate format and raise error if unsupported
    get_format_info: Get detailed information about a file format
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

# Supported image extensions (case-insensitive)
SUPPORTED_IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.tif', '.tiff',
    '.webp', '.bmp', '.gif', '.ico',
    '.ppm', '.pgm', '.pbm', '.tga'
}

# Primary formats for luxury processing
LUXURY_IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png'}

# Supported video extensions (case-insensitive)
SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv'
}

# TIFF-specific extensions
TIFF_EXTENSIONS = {'.tif', '.tiff'}


class UnsupportedFormatError(ValueError):
    """Raised when a file format is not supported by the pipeline."""
    pass


def normalize_extension(path: Union[str, Path]) -> str:
    """Normalize file extension to lowercase with leading dot.
    
    Args:
        path: File path or extension string
        
    Returns:
        Normalized extension (e.g., '.png', '.tiff')
        
    Examples:
        >>> normalize_extension('image.PNG')
        '.png'
        >>> normalize_extension('.TIFF')
        '.tiff'
        >>> normalize_extension('photo.JPG')
        '.jpg'
    """
    if isinstance(path, str):
        path = Path(path)
    
    ext = path.suffix.lower()
    if not ext:
        # If no suffix, treat entire string as extension
        ext = str(path).lower()
        if not ext.startswith('.'):
            ext = '.' + ext
    
    return ext


def is_supported_image_format(path: Union[str, Path]) -> bool:
    """Check if file has a supported image extension.
    
    Args:
        path: File path to check
        
    Returns:
        True if format is supported, False otherwise
        
    Examples:
        >>> is_supported_image_format('render.jpg')
        True
        >>> is_supported_image_format('document.pdf')
        False
        >>> is_supported_image_format('photo.TIFF')
        True
    """
    ext = normalize_extension(path)
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_video_format(path: Union[str, Path]) -> bool:
    """Check if file has a supported video extension.
    
    Args:
        path: File path to check
        
    Returns:
        True if format is supported, False otherwise
        
    Examples:
        >>> is_supported_video_format('tour.mp4')
        True
        >>> is_supported_video_format('render.jpg')
        False
        >>> is_supported_video_format('video.MOV')
        True
    """
    ext = normalize_extension(path)
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_tiff_format(path: Union[str, Path]) -> bool:
    """Check if file is a TIFF format.
    
    Args:
        path: File path to check
        
    Returns:
        True if file is TIFF, False otherwise
        
    Examples:
        >>> is_supported_tiff_format('photo.tif')
        True
        >>> is_supported_tiff_format('photo.TIFF')
        True
        >>> is_supported_tiff_format('photo.jpg')
        False
    """
    ext = normalize_extension(path)
    return ext in TIFF_EXTENSIONS


def is_luxury_format(path: Union[str, Path]) -> bool:
    """Check if file is in a luxury/high-quality format (TIFF, PNG).
    
    Args:
        path: File path to check
        
    Returns:
        True if format is luxury-grade, False otherwise
        
    Examples:
        >>> is_luxury_format('render.tiff')
        True
        >>> is_luxury_format('photo.png')
        True
        >>> is_luxury_format('web.jpg')
        False
    """
    ext = normalize_extension(path)
    return ext in LUXURY_IMAGE_EXTENSIONS


def validate_format(
    path: Union[str, Path],
    allowed_types: str = 'image',
    raise_error: bool = True
) -> bool:
    """Validate that a file format is supported.
    
    Args:
        path: File path to validate
        allowed_types: 'image', 'video', or 'both'
        raise_error: If True, raise UnsupportedFormatError on invalid format
        
    Returns:
        True if format is valid, False otherwise
        
    Raises:
        UnsupportedFormatError: If raise_error=True and format is unsupported
        ValueError: If allowed_types is invalid
        
    Examples:
        >>> validate_format('render.jpg', 'image')
        True
        >>> validate_format('tour.mp4', 'video')
        True
        >>> validate_format('document.pdf', 'image', raise_error=False)
        False
    """
    if allowed_types not in {'image', 'video', 'both'}:
        raise ValueError(f"allowed_types must be 'image', 'video', or 'both', got '{allowed_types}'")
    
    path_obj = Path(path) if isinstance(path, str) else path
    ext = normalize_extension(path_obj)
    
    is_valid = False
    if allowed_types in {'image', 'both'}:
        is_valid = is_valid or is_supported_image_format(path_obj)
    if allowed_types in {'video', 'both'}:
        is_valid = is_valid or is_supported_video_format(path_obj)
    
    if not is_valid and raise_error:
        if allowed_types == 'image':
            supported = SUPPORTED_IMAGE_EXTENSIONS
        elif allowed_types == 'video':
            supported = SUPPORTED_VIDEO_EXTENSIONS
        else:
            supported = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
        
        supported_list = ', '.join(sorted(supported))
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}' for file: {path_obj.name}\n"
            f"Supported {allowed_types} formats: {supported_list}\n"
            f"See SUPPORTED_FILE_FORMATS.md for details."
        )
    
    return is_valid


def get_format_info(path: Union[str, Path]) -> Dict[str, Union[str, bool, List[str]]]:
    """Get detailed information about a file's format.
    
    Args:
        path: File path to analyze
        
    Returns:
        Dictionary with format information:
        - extension: Normalized extension
        - is_image: Whether it's a supported image format
        - is_video: Whether it's a supported video format
        - is_tiff: Whether it's a TIFF format
        - is_luxury: Whether it's a luxury/high-quality format
        - recommendations: List of processing recommendations
        
    Examples:
        >>> info = get_format_info('render.tiff')
        >>> info['is_luxury']
        True
        >>> info = get_format_info('video.mp4')
        >>> info['is_video']
        True
    """
    path_obj = Path(path) if isinstance(path, str) else path
    ext = normalize_extension(path_obj)
    
    is_image = is_supported_image_format(path_obj)
    is_video = is_supported_video_format(path_obj)
    is_tiff = is_supported_tiff_format(path_obj)
    is_luxury = is_luxury_format(path_obj)
    
    recommendations = []
    
    if is_tiff:
        recommendations.extend([
            "Best for 16-bit precision with Luxury TIFF Batch Processor",
            "Install 'tifffile' for full 16-bit support: pip install -e '.[tiff]'",
            "Preserves metadata (EXIF, IPTC, XMP)"
        ])
    elif ext == '.png':
        recommendations.extend([
            "Lossless format, good for web delivery",
            "Supports transparency (alpha channel)",
            "Recommended for depth maps and architectural renders"
        ])
    elif ext in {'.jpg', '.jpeg'}:
        recommendations.extend([
            "Lossy compression, good for web and fast preview",
            "Not recommended for 16-bit workflows",
            "Use quality=95 or higher for best results"
        ])
    elif ext == '.webp':
        recommendations.extend([
            "Modern format with good compression",
            "Supports both lossy and lossless modes",
            "Good alternative to JPEG for web"
        ])
    elif is_video:
        recommendations.extend([
            "Process with Luxury Video Master Grader",
            "FFmpeg required: sudo apt install ffmpeg (Linux) or brew install ffmpeg (macOS)",
            "HDR support available for PQ and HLG"
        ])
    
    return {
        'extension': ext,
        'is_image': is_image,
        'is_video': is_video,
        'is_tiff': is_tiff,
        'is_luxury': is_luxury,
        'is_supported': is_image or is_video,
        'recommendations': recommendations
    }


def suggest_output_format(
    input_path: Union[str, Path],
    preserve_quality: bool = True
) -> str:
    """Suggest an appropriate output format based on input format.
    
    Args:
        input_path: Input file path
        preserve_quality: If True, suggest lossless or high-quality formats
        
    Returns:
        Recommended output extension (e.g., '.tiff', '.png', '.jpg')
        
    Examples:
        >>> suggest_output_format('photo.jpg', preserve_quality=True)
        '.png'
        >>> suggest_output_format('render.tiff', preserve_quality=True)
        '.tiff'
        >>> suggest_output_format('web_image.png', preserve_quality=False)
        '.jpg'
    """
    path_obj = Path(input_path) if isinstance(input_path, str) else input_path
    ext = normalize_extension(path_obj)
    
    # TIFF stays TIFF for quality preservation
    if ext in TIFF_EXTENSIONS and preserve_quality:
        return '.tiff'
    
    # PNG is good lossless format
    if preserve_quality:
        return '.png'
    
    # For web/fast delivery, use JPEG
    return '.jpg'


def get_supported_formats_summary() -> Dict[str, List[str]]:
    """Get a summary of all supported formats.
    
    Returns:
        Dictionary with 'image' and 'video' keys containing lists of extensions
        
    Examples:
        >>> summary = get_supported_formats_summary()
        >>> '.png' in summary['image']
        True
        >>> '.mp4' in summary['video']
        True
    """
    return {
        'image': sorted(SUPPORTED_IMAGE_EXTENSIONS),
        'video': sorted(SUPPORTED_VIDEO_EXTENSIONS),
        'tiff': sorted(TIFF_EXTENSIONS),
        'luxury': sorted(LUXURY_IMAGE_EXTENSIONS)
    }


# Convenience function for CLI help text
def format_help_text(format_type: str = 'image') -> str:
    """Generate help text for supported formats.
    
    Args:
        format_type: 'image', 'video', or 'both'
        
    Returns:
        Formatted help text string
        
    Examples:
        >>> print(format_help_text('image'))
        Supported image formats: .bmp, .gif, .ico, .jpg, ...
    """
    summary = get_supported_formats_summary()
    
    if format_type == 'image':
        return f"Supported image formats: {', '.join(summary['image'])}"
    elif format_type == 'video':
        return f"Supported video formats: {', '.join(summary['video'])}"
    elif format_type == 'both':
        return (
            f"Supported image formats: {', '.join(summary['image'])}\n"
            f"Supported video formats: {', '.join(summary['video'])}"
        )
    else:
        raise ValueError(f"Invalid format_type: {format_type}")


if __name__ == '__main__':
    # Example usage and testing
    test_files = [
        'render.jpg',
        'photo.TIFF',
        'depth_map.png',
        'video.mp4',
        'document.pdf',
        'archive.WebP'
    ]
    
    print("Format Validation Examples:\n")
    for file in test_files:
        try:
            is_valid = validate_format(file, 'both', raise_error=False)
            info = get_format_info(file)
            
            print(f"File: {file}")
            print(f"  Valid: {is_valid}")
            print(f"  Extension: {info['extension']}")
            print(f"  Type: ", end='')
            if info['is_image']:
                print("Image", end='')
            if info['is_video']:
                print("Video", end='')
            if not info['is_image'] and not info['is_video']:
                print("Unsupported", end='')
            print()
            
            if info['recommendations']:
                print(f"  Recommendations:")
                for rec in info['recommendations']:
                    print(f"    - {rec}")
            print()
        except UnsupportedFormatError as e:
            print(f"File: {file}")
            print(f"  Error: {e}\n")
    
    print("\nSupported Formats Summary:")
    summary = get_supported_formats_summary()
    for category, formats in summary.items():
        print(f"  {category.capitalize()}: {len(formats)} formats")
