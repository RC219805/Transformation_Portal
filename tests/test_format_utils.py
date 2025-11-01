"""Tests for format validation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Format validation functions
from format_utils import (  # noqa: E402  # pylint: disable=wrong-import-position
    normalize_extension,
    is_supported_image_format,
    is_supported_video_format,
    is_supported_tiff_format,
    is_luxury_format,
    validate_format,
    get_format_info,
    suggest_output_format,
    get_supported_formats_summary,
    format_help_text,
    UnsupportedFormatError,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class TestNormalizeExtension:
    """Tests for extension normalization."""

    def test_lowercase_with_dot(self):
        assert normalize_extension('image.PNG') == '.png'
        assert normalize_extension('photo.JPEG') == '.jpeg'
        assert normalize_extension('render.TIF') == '.tif'

    def test_already_normalized(self):
        assert normalize_extension('photo.jpg') == '.jpg'
        assert normalize_extension('render.tiff') == '.tiff'

    def test_path_object(self):
        assert normalize_extension(Path('image.PNG')) == '.png'
        assert normalize_extension(Path('/path/to/photo.TIFF')) == '.tiff'

    def test_extension_only(self):
        assert normalize_extension('.PNG') == '.png'
        assert normalize_extension('.tiff') == '.tiff'

    def test_mixed_case(self):
        assert normalize_extension('photo.JpG') == '.jpg'
        assert normalize_extension('render.TiFf') == '.tiff'


class TestIsSupportedImageFormat:
    """Tests for image format detection."""

    def test_supported_formats(self):
        assert is_supported_image_format('render.jpg') is True
        assert is_supported_image_format('photo.png') is True
        assert is_supported_image_format('scan.tiff') is True
        assert is_supported_image_format('icon.bmp') is True
        assert is_supported_image_format('modern.webp') is True

    def test_case_insensitive(self):
        assert is_supported_image_format('IMAGE.PNG') is True
        assert is_supported_image_format('Photo.JPG') is True
        assert is_supported_image_format('Render.TIFF') is True

    def test_unsupported_formats(self):
        assert is_supported_image_format('document.pdf') is False
        assert is_supported_image_format('video.mp4') is False
        assert is_supported_image_format('archive.zip') is False

    def test_path_object(self):
        assert is_supported_image_format(Path('render.jpg')) is True
        assert is_supported_image_format(Path('/path/to/photo.png')) is True

    def test_all_supported_extensions(self):
        # Test all extensions in SUPPORTED_IMAGE_EXTENSIONS
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            filename = f'test{ext}'
            assert is_supported_image_format(filename) is True, f"Failed for {ext}"


class TestIsSupportedVideoFormat:
    """Tests for video format detection."""

    def test_supported_formats(self):
        assert is_supported_video_format('tour.mp4') is True
        assert is_supported_video_format('walkthrough.mov') is True
        assert is_supported_video_format('clip.avi') is True
        assert is_supported_video_format('video.mkv') is True

    def test_case_insensitive(self):
        assert is_supported_video_format('VIDEO.MP4') is True
        assert is_supported_video_format('Tour.MOV') is True

    def test_unsupported_formats(self):
        assert is_supported_video_format('photo.jpg') is False
        assert is_supported_video_format('document.pdf') is False

    def test_all_supported_extensions(self):
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            filename = f'test{ext}'
            assert is_supported_video_format(filename) is True, f"Failed for {ext}"


class TestIsSupportedTiffFormat:
    """Tests for TIFF format detection."""

    def test_tiff_formats(self):
        assert is_supported_tiff_format('photo.tif') is True
        assert is_supported_tiff_format('render.tiff') is True
        assert is_supported_tiff_format('scan.TIF') is True
        assert is_supported_tiff_format('image.TIFF') is True

    def test_non_tiff_formats(self):
        assert is_supported_tiff_format('photo.jpg') is False
        assert is_supported_tiff_format('render.png') is False
        assert is_supported_tiff_format('video.mp4') is False


class TestIsLuxuryFormat:
    """Tests for luxury format detection."""

    def test_luxury_formats(self):
        assert is_luxury_format('photo.tiff') is True
        assert is_luxury_format('render.tif') is True
        assert is_luxury_format('image.png') is True

    def test_non_luxury_formats(self):
        assert is_luxury_format('web.jpg') is False
        assert is_luxury_format('preview.webp') is False
        assert is_luxury_format('icon.bmp') is False


class TestValidateFormat:
    """Tests for format validation."""

    def test_valid_image_format(self):
        assert validate_format('render.jpg', 'image') is True
        assert validate_format('photo.png', 'image') is True
        assert validate_format('scan.tiff', 'image') is True

    def test_valid_video_format(self):
        assert validate_format('tour.mp4', 'video') is True
        assert validate_format('clip.mov', 'video') is True

    def test_both_types(self):
        assert validate_format('render.jpg', 'both') is True
        assert validate_format('video.mp4', 'both') is True

    def test_invalid_image_raises_error(self):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            validate_format('document.pdf', 'image', raise_error=True)

        error_msg = str(exc_info.value)
        assert 'document.pdf' in error_msg or '.pdf' in error_msg
        assert 'Unsupported' in error_msg

    def test_invalid_image_no_raise(self):
        assert validate_format('document.pdf', 'image', raise_error=False) is False

    def test_invalid_video_raises_error(self):
        with pytest.raises(UnsupportedFormatError):
            validate_format('photo.jpg', 'video', raise_error=True)

    def test_invalid_allowed_types(self):
        with pytest.raises(ValueError) as exc_info:
            validate_format('test.jpg', 'invalid')

        assert 'allowed_types' in str(exc_info.value)

    def test_error_message_includes_supported_formats(self):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            validate_format('test.xyz', 'image')

        error_msg = str(exc_info.value)
        # Check that some supported formats are mentioned
        assert '.png' in error_msg or '.jpg' in error_msg
        assert 'SUPPORTED_FILE_FORMATS.md' in error_msg


class TestGetFormatInfo:
    """Tests for format information retrieval."""

    def test_tiff_info(self):
        info = get_format_info('render.tiff')
        assert info['extension'] == '.tiff'
        assert info['is_image'] is True
        assert info['is_video'] is False
        assert info['is_tiff'] is True
        assert info['is_luxury'] is True
        assert info['is_supported'] is True
        assert len(info['recommendations']) > 0
        # Check for TIFF-specific recommendations
        recs = ' '.join(info['recommendations']).lower()
        assert '16-bit' in recs or 'tifffile' in recs

    def test_png_info(self):
        info = get_format_info('photo.png')
        assert info['extension'] == '.png'
        assert info['is_image'] is True
        assert info['is_tiff'] is False
        assert info['is_luxury'] is True
        assert 'lossless' in ' '.join(info['recommendations']).lower()

    def test_jpeg_info(self):
        info = get_format_info('web.jpg')
        assert info['extension'] == '.jpg'
        assert info['is_image'] is True
        assert info['is_luxury'] is False
        assert 'lossy' in ' '.join(info['recommendations']).lower()

    def test_video_info(self):
        info = get_format_info('tour.mp4')
        assert info['extension'] == '.mp4'
        assert info['is_video'] is True
        assert info['is_image'] is False
        assert info['is_supported'] is True
        recs = ' '.join(info['recommendations']).lower()
        assert 'video' in recs or 'ffmpeg' in recs

    def test_unsupported_info(self):
        info = get_format_info('document.pdf')
        assert info['extension'] == '.pdf'
        assert info['is_supported'] is False
        assert info['is_image'] is False
        assert info['is_video'] is False


class TestSuggestOutputFormat:
    """Tests for output format suggestions."""

    def test_tiff_preserve_quality(self):
        assert suggest_output_format('render.tiff', preserve_quality=True) == '.tiff'
        assert suggest_output_format('photo.tif', preserve_quality=True) == '.tiff'

    def test_tiff_no_preserve(self):
        result = suggest_output_format('render.tiff', preserve_quality=False)
        # Could be .jpg for smaller size
        assert result in {'.jpg', '.png'}

    def test_jpeg_preserve_quality(self):
        result = suggest_output_format('web.jpg', preserve_quality=True)
        # Should suggest lossless format
        assert result == '.png'

    def test_jpeg_no_preserve(self):
        assert suggest_output_format('web.jpg', preserve_quality=False) == '.jpg'

    def test_png_preserve_quality(self):
        assert suggest_output_format('render.png', preserve_quality=True) == '.png'

    def test_path_object(self):
        assert suggest_output_format(Path('render.tiff'), preserve_quality=True) == '.tiff'


class TestGetSupportedFormatsSummary:
    """Tests for supported formats summary."""

    def test_summary_structure(self):
        summary = get_supported_formats_summary()
        assert 'image' in summary
        assert 'video' in summary
        assert 'tiff' in summary
        assert 'luxury' in summary

    def test_summary_content(self):
        summary = get_supported_formats_summary()
        assert '.png' in summary['image']
        assert '.jpg' in summary['image']
        assert '.tiff' in summary['image']
        assert '.mp4' in summary['video']
        assert '.mov' in summary['video']

    def test_tiff_subset(self):
        summary = get_supported_formats_summary()
        assert '.tif' in summary['tiff']
        assert '.tiff' in summary['tiff']
        # TIFF should be subset of image
        for ext in summary['tiff']:
            assert ext in summary['image']

    def test_luxury_subset(self):
        summary = get_supported_formats_summary()
        # Luxury should be subset of image
        for ext in summary['luxury']:
            assert ext in summary['image']


class TestFormatHelpText:
    """Tests for help text generation."""

    def test_image_help(self):
        help_text = format_help_text('image')
        assert 'image formats' in help_text.lower()
        assert '.png' in help_text
        assert '.jpg' in help_text

    def test_video_help(self):
        help_text = format_help_text('video')
        assert 'video formats' in help_text.lower()
        assert '.mp4' in help_text
        assert '.mov' in help_text

    def test_both_help(self):
        help_text = format_help_text('both')
        assert 'image formats' in help_text.lower()
        assert 'video formats' in help_text.lower()
        assert '.png' in help_text
        assert '.mp4' in help_text

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            format_help_text('invalid')


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_extension(self):
        # File with no extension
        assert is_supported_image_format('filename') is False

    def test_multiple_dots(self):
        # File with multiple dots
        assert normalize_extension('my.photo.jpg') == '.jpg'
        assert is_supported_image_format('my.photo.jpg') is True

    def test_hidden_file(self):
        # Hidden file (starts with dot)
        assert normalize_extension('.hidden.jpg') == '.jpg'
        assert is_supported_image_format('.hidden.jpg') is True

    def test_path_with_spaces(self):
        assert is_supported_image_format('my photo.jpg') is True
        assert normalize_extension('my photo.PNG') == '.png'

    def test_unicode_filename(self):
        assert is_supported_image_format('фото.jpg') is True
        assert is_supported_image_format('画像.png') is True


class TestIntegration:
    """Integration tests using multiple functions together."""

    def test_validate_then_suggest(self):
        # Validate input, then suggest output
        input_file = 'render.jpg'
        assert validate_format(input_file, 'image') is True
        output_ext = suggest_output_format(input_file, preserve_quality=True)
        assert output_ext == '.png'  # Lossless upgrade

    def test_info_for_all_formats(self):
        # Get info for various formats
        test_files = [
            'photo.jpg',
            'render.tiff',
            'depth.png',
            'tour.mp4',
            'web.webp'
        ]

        for file in test_files:
            info = get_format_info(file)
            assert 'extension' in info
            assert 'is_supported' in info
            # Should have recommendations if supported
            if info['is_supported']:
                assert len(info['recommendations']) > 0

    def test_workflow_validation(self):
        # Simulate a typical workflow
        files = [
            ('render.jpg', 'image', True),
            ('tour.mp4', 'video', True),
            ('document.pdf', 'image', False),
        ]

        for file, type_check, expected_valid in files:
            is_valid = validate_format(file, type_check, raise_error=False)
            assert is_valid == expected_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
