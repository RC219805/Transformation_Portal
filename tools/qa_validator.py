#!/usr/bin/env python3
"""
QA Pre-Flight Validator
========================
Automated validation of input files before batch processing.
Checks format, bit depth, color space, resolution, HDR range, and metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


@dataclass
class ValidationIssue:
    """Represents a validation issue found in an image."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'format', 'bit_depth', 'color_space', 'resolution', 'hdr', 'metadata', 'corruption'
    message: str
    recommendation: Optional[str] = None


@dataclass
class ImageValidation:
    """Complete validation result for a single image."""
    filename: str
    path: str
    is_valid: bool
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]
    estimated_processing_time_min: float = 0.0


class QAValidator:
    """Pre-flight validation for image processing pipelines."""
    
    # Requirements
    MIN_RESOLUTION = (1000, 1000)  # Minimum width x height
    MAX_RESOLUTION = (15000, 15000)  # Maximum for reasonable processing
    SUPPORTED_FORMATS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
    RECOMMENDED_BIT_DEPTH = 16
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validations: List[ImageValidation] = []
    
    def validate_image(self, path: Path) -> ImageValidation:
        """Validate a single image file."""
        issues = []
        metadata = {
            'filename': path.name,
            'path': str(path),
            'file_size_mb': path.stat().st_size / (1024 * 1024)
        }
        
        # Check 1: File exists and readable
        try:
            if not path.exists():
                issues.append(ValidationIssue(
                    'error', 'format',
                    f"File not found: {path}",
                    "Verify file path and permissions"
                ))
                return ImageValidation(path.name, str(path), False, issues, metadata)
        except Exception as e:
            issues.append(ValidationIssue(
                'error', 'format',
                f"Cannot access file: {e}",
                "Check file permissions"
            ))
            return ImageValidation(path.name, str(path), False, issues, metadata)
        
        # Check 2: Format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            issues.append(ValidationIssue(
                'error', 'format',
                f"Unsupported format: {path.suffix}",
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            ))
            return ImageValidation(path.name, str(path), False, issues, metadata)
        
        # Try to load and analyze image
        try:
            img_array, img_metadata = self._load_and_analyze(path)
            metadata.update(img_metadata)
        except Exception as e:
            issues.append(ValidationIssue(
                'error', 'corruption',
                f"Cannot load image: {e}",
                "File may be corrupted or in unsupported format"
            ))
            return ImageValidation(path.name, str(path), False, issues, metadata)
        
        # Check 3: Resolution
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            issues.append(ValidationIssue(
                'warning', 'resolution',
                f"Low resolution: {width}x{height}",
                f"Minimum recommended: {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}"
            ))
        
        if width > self.MAX_RESOLUTION[0] or height > self.MAX_RESOLUTION[1]:
            issues.append(ValidationIssue(
                'warning', 'resolution',
                f"Very high resolution: {width}x{height}",
                "May require significant processing time and memory"
            ))
        
        # Check 4: Bit depth
        bit_depth = metadata.get('bit_depth', 0)
        
        if bit_depth == 8:
            issues.append(ValidationIssue(
                'warning', 'bit_depth',
                "8-bit image detected",
                "16-bit or 32-bit recommended for professional processing"
            ))
        elif bit_depth == 0:
            issues.append(ValidationIssue(
                'error', 'bit_depth',
                "Cannot determine bit depth",
                "Verify file integrity"
            ))
        else:
            issues.append(ValidationIssue(
                'info', 'bit_depth',
                f"{bit_depth}-bit image (excellent)",
                None
            ))
        
        # Check 5: Color space
        color_mode = metadata.get('color_mode', 'unknown')
        
        if color_mode not in ['RGB', 'RGBA']:
            issues.append(ValidationIssue(
                'warning', 'color_space',
                f"Unusual color mode: {color_mode}",
                "RGB or RGBA expected for processing"
            ))
        
        # Check 6: HDR analysis
        if metadata.get('is_hdr'):
            hdr_stats = metadata.get('hdr_stats', {})
            
            issues.append(ValidationIssue(
                'info', 'hdr',
                f"HDR data detected (negative: {hdr_stats.get('negative_pct', 0):.1f}%, >1.0: {hdr_stats.get('above_one_pct', 0):.1f}%)",
                "HDR tone mapping will be applied"
            ))
            
            # Check for extreme values
            if hdr_stats.get('max_value', 0) > 10:
                issues.append(ValidationIssue(
                    'warning', 'hdr',
                    f"Extreme HDR values detected (max: {hdr_stats.get('max_value', 0):.2f})",
                    "May require custom tone mapping parameters"
                ))
        
        # Check 7: Channel count
        channels = metadata.get('channels', 0)
        if channels == 4:
            issues.append(ValidationIssue(
                'info', 'format',
                "Alpha channel detected",
                "Alpha will be preserved or flattened based on settings"
            ))
        elif channels not in [3, 4]:
            issues.append(ValidationIssue(
                'warning', 'format',
                f"Unusual channel count: {channels}",
                "3 (RGB) or 4 (RGBA) channels expected"
            ))
        
        # Check 8: Metadata presence
        if not metadata.get('has_exif', False):
            issues.append(ValidationIssue(
                'info', 'metadata',
                "No EXIF metadata found",
                "Metadata preservation not applicable"
            ))
        
        # Check 9: File size sanity
        expected_size = (width * height * channels * (bit_depth / 8)) / (1024 * 1024)
        actual_size = metadata['file_size_mb']
        
        # TIFF with compression can be much smaller
        if path.suffix.lower() in ['.tif', '.tiff']:
            if actual_size > expected_size * 1.5:
                issues.append(ValidationIssue(
                    'warning', 'format',
                    f"Unusually large file size: {actual_size:.1f} MB (expected ~{expected_size:.1f} MB)",
                    "File may contain embedded thumbnails or uncompressed data"
                ))
        
        # Determine overall validity
        has_errors = any(issue.severity == 'error' for issue in issues)
        has_critical_warnings = any(
            issue.severity == 'warning' and issue.category in ['corruption', 'bit_depth']
            for issue in issues
        ) if self.strict_mode else False
        
        is_valid = not (has_errors or has_critical_warnings)
        
        # Estimate processing time (rough)
        megapixels = metadata.get('megapixels', 0)
        bit_multiplier = {8: 1.0, 16: 1.5, 32: 2.5}.get(bit_depth, 1.5)
        hdr_multiplier = 1.8 if metadata.get('is_hdr') else 1.0
        estimated_time = megapixels * 0.5 * bit_multiplier * hdr_multiplier / 60  # minutes
        
        validation = ImageValidation(
            filename=path.name,
            path=str(path),
            is_valid=is_valid,
            issues=issues,
            metadata=metadata,
            estimated_processing_time_min=estimated_time
        )
        
        self.validations.append(validation)
        return validation
    
    def _load_and_analyze(self, path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load image and extract detailed metadata."""
        metadata = {}
        
        if path.suffix.lower() in ['.tif', '.tiff'] and HAS_TIFFFILE:
            # Use tifffile for TIFF
            arr = tifffile.imread(path)
            
            metadata['width'] = arr.shape[1]
            metadata['height'] = arr.shape[0]
            metadata['channels'] = arr.shape[2] if len(arr.shape) > 2 else 1
            metadata['dtype'] = str(arr.dtype)
            metadata['color_mode'] = 'RGBA' if metadata['channels'] == 4 else 'RGB'
            
            # Bit depth
            if arr.dtype == np.uint8:
                metadata['bit_depth'] = 8
            elif arr.dtype == np.uint16:
                metadata['bit_depth'] = 16
            elif arr.dtype in [np.float32, np.float64]:
                metadata['bit_depth'] = 32
            else:
                metadata['bit_depth'] = 0
            
            # HDR analysis
            if arr.dtype in [np.float32, np.float64]:
                rgb = arr[:, :, :3] if arr.shape[2] >= 3 else arr
                negative_count = (rgb < 0).sum()
                above_one_count = (rgb > 1.0).sum()
                total_pixels = rgb.size
                
                metadata['is_hdr'] = bool(negative_count > 0 or above_one_count > 0)
                metadata['hdr_stats'] = {
                    'negative_pct': float(negative_count / total_pixels * 100),
                    'above_one_pct': float(above_one_count / total_pixels * 100),
                    'min_value': float(rgb.min()),
                    'max_value': float(rgb.max()),
                    'mean_value': float(rgb.mean())
                }
            else:
                metadata['is_hdr'] = False
            
            return arr, metadata
        else:
            # Use PIL
            img = Image.open(path)
            
            metadata['width'], metadata['height'] = img.size
            metadata['channels'] = len(img.getbands())
            metadata['color_mode'] = img.mode
            metadata['bit_depth'] = 8  # PIL typically 8-bit
            metadata['is_hdr'] = False
            
            # Check for EXIF
            metadata['has_exif'] = bool(img.getexif())
            
            # Convert to array for basic analysis
            arr = np.array(img)
            
            return arr, metadata
    
    def validate_batch(self, paths: List[Path]) -> Dict[str, Any]:
        """Validate a batch of images."""
        print(f"🔍 Validating batch of {len(paths)} images...")
        print()
        
        for i, path in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] Validating: {path.name}")
            validation = self.validate_image(path)
            
            # Print issues
            for issue in validation.issues:
                icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue.severity, '•')
                print(f"  {icon} [{issue.category}] {issue.message}")
                if issue.recommendation:
                    print(f"     → {issue.recommendation}")
            
            if validation.is_valid:
                print(f"  ✅ Valid (estimated: {validation.estimated_processing_time_min:.1f} min)")
            else:
                print(f"  ❌ INVALID - will be skipped")
            print()
        
        # Generate summary
        valid_count = sum(1 for v in self.validations if v.is_valid)
        invalid_count = len(self.validations) - valid_count
        
        total_time = sum(v.estimated_processing_time_min for v in self.validations if v.is_valid)
        total_size = sum(v.metadata.get('file_size_mb', 0) for v in self.validations if v.is_valid)
        
        # Issue statistics
        issue_counts = {}
        for validation in self.validations:
            for issue in validation.issues:
                key = f"{issue.severity}_{issue.category}"
                issue_counts[key] = issue_counts.get(key, 0) + 1
        
        summary = {
            'total_images': len(self.validations),
            'valid': valid_count,
            'invalid': invalid_count,
            'total_estimated_time_min': total_time,
            'total_estimated_time_hours': total_time / 60,
            'total_size_mb': total_size,
            'total_size_gb': total_size / 1024,
            'issue_summary': issue_counts,
            'validations': [asdict(v) for v in self.validations]
        }
        
        return summary
    
    def generate_report(self, output_path: Path, summary: Dict[str, Any]):
        """Generate pre-flight validation report."""
        with open(output_path, 'w') as f:
            f.write("# Pre-Flight Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Images:** {summary['total_images']}\n")
            f.write(f"- **Valid:** {summary['valid']} ✅\n")
            f.write(f"- **Invalid:** {summary['invalid']} ❌\n")
            f.write(f"- **Estimated Processing Time:** {summary['total_estimated_time_hours']:.2f} hours\n")
            f.write(f"- **Total Size:** {summary['total_size_gb']:.2f} GB\n\n")
            
            # Issue breakdown
            if summary['issue_summary']:
                f.write("## Issue Breakdown\n\n")
                f.write("| Severity | Category | Count |\n")
                f.write("|----------|----------|-------|\n")
                
                for key, count in sorted(summary['issue_summary'].items()):
                    severity, category = key.split('_', 1)
                    icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(severity, '•')
                    f.write(f"| {icon} {severity.title()} | {category} | {count} |\n")
                f.write("\n")
            
            # Go/No-Go decision
            f.write("## Go/No-Go Decision\n\n")
            
            if summary['invalid'] == 0:
                f.write("✅ **GO** - All images passed validation\n\n")
            elif summary['valid'] > summary['invalid']:
                f.write(f"⚠️ **CONDITIONAL GO** - {summary['invalid']} images will be skipped\n\n")
            else:
                f.write(f"❌ **NO-GO** - Too many invalid images ({summary['invalid']}/{summary['total_images']})\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Collect unique recommendations
            recommendations = set()
            for validation in self.validations:
                for issue in validation.issues:
                    if issue.recommendation and issue.severity in ['error', 'warning']:
                        recommendations.add(issue.recommendation)
            
            if recommendations:
                for rec in sorted(recommendations):
                    f.write(f"- {rec}\n")
            else:
                f.write("- No critical issues detected\n")
            f.write("\n")
            
            # Individual results
            f.write("## Individual Results\n\n")
            
            for validation in self.validations:
                status = "✅ Valid" if validation.is_valid else "❌ Invalid"
                f.write(f"### {validation.filename} - {status}\n\n")
                
                meta = validation.metadata
                f.write(f"- **Resolution:** {meta.get('width', 0)}x{meta.get('height', 0)} ({meta.get('megapixels', 0):.1f} MP)\n")
                f.write(f"- **Bit Depth:** {meta.get('bit_depth', 0)}-bit\n")
                f.write(f"- **Color Mode:** {meta.get('color_mode', 'unknown')}\n")
                f.write(f"- **File Size:** {meta.get('file_size_mb', 0):.1f} MB\n")
                
                if meta.get('is_hdr'):
                    f.write(f"- **HDR:** Yes\n")
                
                f.write(f"- **Estimated Processing Time:** {validation.estimated_processing_time_min:.1f} minutes\n\n")
                
                if validation.issues:
                    f.write("**Issues:**\n\n")
                    for issue in validation.issues:
                        icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue.severity, '•')
                        f.write(f"- {icon} **[{issue.category}]** {issue.message}\n")
                        if issue.recommendation:
                            f.write(f"  - *Recommendation: {issue.recommendation}*\n")
                
                f.write("\n---\n\n")
        
        print(f"📄 Validation report saved: {output_path}")


def main():
    """CLI for QA validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-flight validation for image processing")
    parser.add_argument("inputs", nargs='+', type=Path, help="Image files or directory")
    parser.add_argument("--output", type=Path, default=Path("qa_validation_report.md"), help="Output report path")
    parser.add_argument("--strict", action="store_true", help="Strict mode (treat warnings as errors)")
    parser.add_argument("--json", type=Path, help="Save results as JSON")
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    for input_path in args.inputs:
        if input_path.is_dir():
            for ext in QAValidator.SUPPORTED_FORMATS:
                image_paths.extend(input_path.glob(f"*{ext}"))
        else:
            image_paths.append(input_path)
    
    if not image_paths:
        print("❌ No images found")
        return
    
    print("="*80)
    print("🔍 QA PRE-FLIGHT VALIDATION")
    print("="*80)
    print()
    
    # Run validation
    validator = QAValidator(strict_mode=args.strict)
    summary = validator.validate_batch(image_paths)
    
    # Print summary
    print("="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    print(f"Total images: {summary['total_images']}")
    print(f"Valid: {summary['valid']} ✅")
    print(f"Invalid: {summary['invalid']} ❌")
    print(f"Estimated total time: {summary['total_estimated_time_hours']:.2f} hours")
    print(f"Total size: {summary['total_size_gb']:.2f} GB")
    print()
    
    # Decision
    if summary['invalid'] == 0:
        print("✅ GO - All images ready for processing")
    elif summary['valid'] > 0:
        print(f"⚠️ CONDITIONAL GO - {summary['invalid']} images will be skipped")
    else:
        print("❌ NO-GO - No valid images found")
    print("="*80)
    
    # Generate report
    validator.generate_report(args.output, summary)
    
    # Save JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 JSON results saved: {args.json}")


if __name__ == "__main__":
    main()
