#!/usr/bin/env python3
"""
Source TIFF Audit Script
Batch-inspect all source TIFFs for dimensions, metadata, color profiles, and potential issues.
"""

import sys
from pathlib import Path
import numpy as np
from tifffile import TiffFile, imread
import json


def audit_tiff(filepath):
    """
    Comprehensive audit of a single TIFF file.
    
    Returns:
        dict: Audit results
    """
    print(f"\n{'='*80}")
    print(f"AUDITING: {filepath.name}")
    print('='*80)
    
    audit = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "exists": filepath.exists(),
    }
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        audit["status"] = "NOT_FOUND"
        return audit
    
    try:
        # Load with tifffile for detailed metadata
        with TiffFile(filepath) as tif:
            page = tif.pages[0]
            
            # Basic properties
            print("\n1. BASIC PROPERTIES")
            print("-" * 80)
            print(f"  Dimensions: {page.shape}")
            print(f"  Dtype: {page.dtype}")
            print(f"  Bits per sample: {page.bitspersample}")
            print(f"  Samples per pixel: {page.samplesperpixel}")
            print(f"  Photometric: {page.photometric.name}")
            
            audit["shape"] = page.shape
            audit["dtype"] = str(page.dtype)
            audit["bits_per_sample"] = page.bitspersample
            audit["samples_per_pixel"] = page.samplesperpixel
            audit["photometric"] = page.photometric.name
            
            # Check for expected dimensions
            expected_h, expected_w = 2250, 4000
            is_expected_size = (page.shape[0] == expected_h and page.shape[1] == expected_w)
            print(f"  Expected size (2250×4000): {'✅ YES' if is_expected_size else '❌ NO'}")
            audit["expected_dimensions"] = is_expected_size
            
            # Check for 16-bit depth
            is_16bit = page.bitspersample == 16
            print(f"  16-bit depth: {'✅ YES' if is_16bit else '❌ NO'}")
            audit["is_16bit"] = is_16bit
            
            # Orientation
            print("\n2. ORIENTATION")
            print("-" * 80)
            orientation = page.tags.get('Orientation')
            if orientation:
                print(f"  Orientation tag: {orientation.value} ({orientation.name if hasattr(orientation, 'name') else 'unknown'})")
                audit["orientation"] = orientation.value
            else:
                print("  Orientation tag: Not set (defaults to top-left)")
                audit["orientation"] = 1  # Default
            
            # Check for non-standard orientation
            is_standard_orientation = (orientation is None) or (orientation.value == 1)
            print(f"  Standard orientation: {'✅ YES' if is_standard_orientation else '⚠️  NO - may cause rotation/flip'}")
            audit["standard_orientation"] = is_standard_orientation
            
            # Color profile / ICC
            print("\n3. COLOR PROFILE")
            print("-" * 80)
            icc_profile = page.tags.get('InterColorProfile')
            if icc_profile:
                icc_data = icc_profile.value
                print(f"  ICC Profile: PRESENT ({len(icc_data)} bytes)")
                audit["icc_profile"] = "present"
                audit["icc_size"] = len(icc_data)
                # Try to parse profile description
                try:
                    desc_start = icc_data.find(b'desc')
                    if desc_start > 0:
                        desc_section = icc_data[desc_start:desc_start+200]
                        # Look for ASCII description
                        printable = ''.join(chr(b) if 32 <= b < 127 else '' for b in desc_section)
                        if printable:
                            print(f"  Profile hint: {printable[:80]}")
                except:
                    pass
            else:
                print("  ICC Profile: NOT PRESENT (unmanaged color)")
                audit["icc_profile"] = "none"
            
            # Photometric interpretation
            print("\n4. PHOTOMETRIC INTERPRETATION")
            print("-" * 80)
            print(f"  Photometric: {page.photometric.name}")
            is_rgb = page.photometric.name == 'RGB'
            print(f"  RGB color space: {'✅ YES' if is_rgb else '⚠️  NO'}")
            audit["is_rgb"] = is_rgb
            
            # Software / processing tags
            print("\n5. METADATA / PROCESSING TAGS")
            print("-" * 80)
            software = page.tags.get('Software')
            if software:
                print(f"  Software: {software.value}")
                audit["software"] = software.value
            else:
                print("  Software: Not set")
                audit["software"] = None
            
            datetime_tag = page.tags.get('DateTime')
            if datetime_tag:
                print(f"  DateTime: {datetime_tag.value}")
                audit["datetime"] = datetime_tag.value
            
            # Additional tags
            print("\n6. ADDITIONAL TAGS")
            print("-" * 80)
            interesting_tags = [
                'ImageDescription', 'Make', 'Model', 'XResolution', 'YResolution',
                'ResolutionUnit', 'Compression', 'PlanarConfiguration'
            ]
            for tag_name in interesting_tags:
                tag = page.tags.get(tag_name)
                if tag:
                    print(f"  {tag_name}: {tag.value}")
                    audit[tag_name.lower()] = str(tag.value)
            
        # Load pixel data for statistics
        print("\n7. PIXEL STATISTICS")
        print("-" * 80)
        img = imread(filepath)
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        
        if img.ndim == 3 and img.shape[2] >= 3:
            for ch, name in enumerate(['Red', 'Green', 'Blue']):
                ch_data = img[:, :, ch]
                print(f"  {name:6s}: min={ch_data.min():5d} max={ch_data.max():5d} "
                      f"mean={ch_data.mean():8.2f} std={ch_data.std():8.2f}")
                audit[f"{name.lower()}_min"] = int(ch_data.min())
                audit[f"{name.lower()}_max"] = int(ch_data.max())
                audit[f"{name.lower()}_mean"] = float(ch_data.mean())
                audit[f"{name.lower()}_std"] = float(ch_data.std())
        
        # File size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\n  File size: {file_size_mb:.2f} MB")
        audit["file_size_mb"] = file_size_mb
        
        # Verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        issues = []
        if not is_expected_size:
            issues.append("Unexpected dimensions")
        if not is_16bit:
            issues.append("Not 16-bit")
        if not is_standard_orientation:
            issues.append("Non-standard orientation")
        if not is_rgb:
            issues.append("Non-RGB photometric")
        
        if issues:
            print(f"⚠️  ISSUES DETECTED: {', '.join(issues)}")
            audit["status"] = "ISSUES"
            audit["issues"] = issues
        else:
            print("✅ CLEAN: No issues detected")
            audit["status"] = "CLEAN"
            audit["issues"] = []
        
        return audit
        
    except Exception as e:
        print(f"❌ ERROR auditing file: {e}")
        audit["status"] = "ERROR"
        audit["error"] = str(e)
        return audit


def main():
    source_dir = Path("projects/750_picacho_lane/Final_Production_UltraQuality")
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1
    
    # Find all TIFFs
    tiff_files = sorted(source_dir.glob("*.tif"))
    
    if not tiff_files:
        print(f"❌ No TIFF files found in {source_dir}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"SOURCE TIFF AUDIT")
    print(f"{'='*80}")
    print(f"\nFound {len(tiff_files)} TIFF files in {source_dir}")
    
    # Audit each file
    audits = []
    for tiff_file in tiff_files:
        audit = audit_tiff(tiff_file)
        audits.append(audit)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("AUDIT SUMMARY")
    print('='*80)
    print()
    
    clean_count = sum(1 for a in audits if a.get("status") == "CLEAN")
    issues_count = sum(1 for a in audits if a.get("status") == "ISSUES")
    error_count = sum(1 for a in audits if a.get("status") == "ERROR")
    
    print(f"Total files: {len(audits)}")
    print(f"  ✅ Clean: {clean_count}")
    print(f"  ⚠️  Issues: {issues_count}")
    print(f"  ❌ Errors: {error_count}")
    print()
    
    # Files with issues
    if issues_count > 0:
        print("Files with issues:")
        for audit in audits:
            if audit.get("status") == "ISSUES":
                print(f"  ⚠️  {audit['filename']}: {', '.join(audit['issues'])}")
        print()
    
    # Save full report
    report_path = Path("forensics/source_tiff_audit.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(audits, f, indent=2)
    print(f"Full audit report saved to: {report_path}")
    
    # Exit code
    if issues_count > 0 or error_count > 0:
        print("\n⚠️  AUDIT COMPLETED WITH ISSUES")
        return 1
    else:
        print("\n✅ AUDIT COMPLETED - ALL FILES CLEAN")
        return 0


if __name__ == "__main__":
    sys.exit(main())
