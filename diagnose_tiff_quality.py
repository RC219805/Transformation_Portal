#!/usr/bin/env python3
"""
TIFF Quality Diagnostic Tool
=============================

Analyzes TIFF files to detect quality degradation issues:
- 8-bit vs 16-bit depth
- Improper scaling (8-bit data in 16-bit container)
- Compression artifacts
- Color space issues

Usage:
    python diagnose_tiff_quality.py <tiff_file_or_directory>
    python diagnose_tiff_quality.py output/  # Scan entire directory
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def analyze_tiff(tiff_path: Path) -> Dict:
    """
    Comprehensive TIFF quality analysis.

    Args:
        tiff_path: Path to TIFF file

    Returns:
        Dictionary with analysis results
    """
    try:
        import tifffile
        has_tifffile = True
    except ImportError:
        has_tifffile = False

    results = {
        'path': str(tiff_path),
        'file_size_mb': tiff_path.stat().st_size / (1024**2),
        'issues': [],
        'warnings': [],
        'status': 'OK'
    }

    # Open with PIL for basic info
    try:
        with Image.open(tiff_path) as img:
            results['size'] = img.size
            results['mode'] = img.mode
            results['format'] = img.format

            # Get PIL info
            pil_array = np.array(img)
            results['pil_dtype'] = str(pil_array.dtype)
            results['pil_shape'] = pil_array.shape
            results['pil_range'] = (int(pil_array.min()), int(pil_array.max()))
    except Exception as e:
        results['issues'].append(f"PIL load failed: {e}")
        results['status'] = 'ERROR'
        return results

    # Deep analysis with tifffile
    if has_tifffile:
        try:
            import tifffile

            with tifffile.TiffFile(str(tiff_path)) as tif:
                page = tif.pages[0]
                results['tiff_dtype'] = str(page.dtype)
                results['bits_per_sample'] = page.bitspersample
                results['compression'] = str(page.compression)
                results['photometric'] = str(page.photometric)

                # Load data
                data = page.asarray()
                results['data_range'] = (int(data.min()), int(data.max()))

                # Check for issues
                if page.dtype == np.uint8:
                    results['issues'].append("8-bit depth detected (should be 16-bit for masters)")
                    results['status'] = 'DEGRADED'

                if page.dtype == np.uint16:
                    # Check for improperly scaled data
                    max_val = data.max()
                    if max_val < 300:
                        results['issues'].append(
                            f"16-bit file but max value is {max_val} (likely 8-bit data scaled incorrectly)"
                        )
                        results['status'] = 'DEGRADED'
                    elif max_val < 10000:
                        results['warnings'].append(
                            f"Low dynamic range: max value is {max_val} / 65535"
                        )

                    # Check bit utilization
                    unique_count = len(np.unique(data))
                    utilization = (unique_count / 65536) * 100
                    results['unique_values'] = unique_count
                    results['bit_utilization_pct'] = utilization

                    if utilization < 0.5:
                        results['warnings'].append(
                            f"Low bit utilization: {utilization:.2f}% (may be upscaled 8-bit)"
                        )

                # Check for banding (common in 8-bit)
                if len(data.shape) == 3:
                    # Analyze first channel for gradient smoothness
                    channel = data[:, :, 0]
                    # Simple banding detection: check histogram for gaps
                    hist, _ = np.histogram(channel, bins=256)
                    zero_bins = np.sum(hist == 0)
                    if zero_bins > 50:
                        results['warnings'].append(
                            f"Possible banding detected ({zero_bins} empty histogram bins)"
                        )

        except Exception as e:
            results['warnings'].append(f"tifffile analysis failed: {e}")
    else:
        results['warnings'].append("tifffile not available - install for detailed analysis")

    return results


def print_analysis(results: Dict):
    """Print formatted analysis results."""
    print(f"\n{'='*80}")
    print(f"File: {Path(results['path']).name}")
    print(f"{'='*80}")

    print(f"Size: {results.get('size', 'unknown')}")
    print(f"File Size: {results['file_size_mb']:.2f} MB")
    print(f"PIL Mode: {results.get('mode', 'unknown')}")
    print(f"PIL dtype: {results.get('pil_dtype', 'unknown')}")
    print(f"PIL Range: {results.get('pil_range', 'unknown')}")

    if 'tiff_dtype' in results:
        print("\nTIFF Details:")
        print(f"  dtype: {results['tiff_dtype']}")
        print(f"  Bits per sample: {results.get('bits_per_sample', 'unknown')}")
        print(f"  Compression: {results.get('compression', 'unknown')}")
        print(f"  Data Range: {results.get('data_range', 'unknown')}")

        if 'unique_values' in results:
            print(f"  Unique values: {results['unique_values']:,}")
            print(f"  Bit utilization: {results['bit_utilization_pct']:.2f}%")

    # Print issues
    if results['issues']:
        print("\n❌ ISSUES FOUND:")
        for issue in results['issues']:
            print(f"  • {issue}")

    if results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  • {warning}")

    # Status
    status_symbols = {
        'OK': '✅',
        'DEGRADED': '❌',
        'ERROR': '⚠️'
    }
    symbol = status_symbols.get(results['status'], '?')
    print(f"\nStatus: {symbol} {results['status']}")


def scan_directory(directory: Path) -> List[Dict]:
    """Scan directory for TIFF files and analyze them."""
    tiff_files = list(directory.glob("**/*.tif")) + list(directory.glob("**/*.ti"))

    print(f"Found {len(tiff_files)} TIFF files in {directory}")

    results = []
    for tiff_path in sorted(tiff_files):
        result = analyze_tiff(tiff_path)
        results.append(result)
        print_analysis(result)

    return results


def print_summary(all_results: List[Dict]):
    """Print summary of all analyzed files."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total = len(all_results)
    ok = sum(1 for r in all_results if r['status'] == 'OK')
    degraded = sum(1 for r in all_results if r['status'] == 'DEGRADED')
    errors = sum(1 for r in all_results if r['status'] == 'ERROR')

    print(f"Total files: {total}")
    print(f"  ✅ OK: {ok}")
    print(f"  ❌ Degraded: {degraded}")
    print(f"  ⚠️  Errors: {errors}")

    # List degraded files
    if degraded > 0:
        print("\nDegraded files:")
        for r in all_results:
            if r['status'] == 'DEGRADED':
                print(f"  • {Path(r['path']).name}")
                for issue in r['issues']:
                    print(f"    - {issue}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_tiff_quality.py <tiff_file_or_directory>")
        print("\nExamples:")
        print("  python diagnose_tiff_quality.py output/kitchen_MASTER.tif")
        print("  python diagnose_tiff_quality.py output_premium_fixed/")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    if path.is_file():
        # Single file analysis
        result = analyze_tiff(path)
        print_analysis(result)
        sys.exit(0 if result['status'] == 'OK' else 1)

    elif path.is_dir():
        # Directory scan
        results = scan_directory(path)
        print_summary(results)
        sys.exit(0 if all(r['status'] == 'OK' for r in results) else 1)

    else:
        print(f"Error: Not a file or directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
