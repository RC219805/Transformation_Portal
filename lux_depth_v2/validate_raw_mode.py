#!/usr/bin/env python3
"""
Raw Mode Validation Script

Verifies that RAW mode implementation meets all requirements:
- Zero processing contamination
- CPU-only execution
- Deterministic output
- Pixel-perfect preservation
"""

import json
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def validate_raw_mode(test_image: str) -> bool:
    """Run comprehensive validation of RAW mode."""
    
    print("=" * 70)
    print("RAW MODE VALIDATION")
    print("=" * 70)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        
        # Run raw mode
        print("\n1. Running RAW mode...")
        cmd = [
            "python", "-m", "lux_depth_v2.cli",
            "--mode", "raw",
            "--input", test_image,
            "--output-dir", str(output_dir)
        ]
        
        exit_code, stdout, stderr = run_command(cmd)
        
        if exit_code != 0:
            print(f"❌ FAILED: Command exited with code {exit_code}")
            print(stderr)
            return False
        
        # Check for success message
        combined_output = stdout + stderr
        if "✓ RAW mode verified: pixel-perfect decode" in combined_output:
            print("✅ Pixel verification passed")
        else:
            print("❌ FAILED: No pixel verification message found")
            print("STDOUT:", stdout[:500])
            print("STDERR:", stderr[:500])
            return False
        
        # Load report
        print("\n2. Validating execution report...")
        stem = Path(test_image).stem
        report_path = output_dir / f"{stem}_report.json"
        
        if not report_path.exists():
            print(f"❌ FAILED: Report not found: {report_path}")
            return False
        
        with open(report_path) as f:
            report = json.load(f)
        
        # Validate critical fields
        checks = {
            "mode": ("raw", lambda v: v == "raw"),
            "device": ("cpu", lambda v: v == "cpu"),
            "precision": ("fp32", lambda v: v == "fp32"),
            "_preset_applied": (False, lambda v: v is False),
            "_config_locked": (False, lambda v: v is False),
            "enable_material": (False, lambda v: v is False),
            "upscale": (1, lambda v: v == 1),
            "preset": ("raw", lambda v: v == "raw"),
            "deterministic": (True, lambda v: v is True),
        }
        
        all_passed = True
        for field, (expected, validator) in checks.items():
            if field == "mode":
                value = report.get(field)
            elif field == "deterministic":
                value = report.get("pixel_verification", {}).get(field)
            else:
                # All other fields are in config
                value = report.get("config", {}).get(field)
            
            if value is None:
                print(f"  ❌ {field}: NOT FOUND")
                all_passed = False
            elif not validator(value):
                print(f"  ❌ {field}: {value} (expected {expected})")
                all_passed = False
            else:
                print(f"  ✅ {field}: {value}")
        
        # Validate stages executed
        print("\n3. Validating stages executed...")
        stages = report.get("stages_executed", [])
        expected_stages = {"io/read_input", "io/read_depth", "export_master", "verify_raw"}
        
        # Check that only expected stages ran (no grading, upscaling, material, etc.)
        forbidden_stages = {
            "grade/master", "material/segmentation", "upscale/base", 
            "upscale/torch", "upscale/realesrgan", "material/response"
        }
        
        forbidden_found = forbidden_stages & set(stages)
        if forbidden_found:
            print(f"  ❌ Forbidden stages found: {forbidden_found}")
            all_passed = False
        else:
            print(f"  ✅ No forbidden stages")
        
        # Check minimum required stages
        required_stages = {"io/read_input", "export_master"}
        missing = required_stages - set(stages)
        if missing:
            print(f"  ❌ Missing required stages: {missing}")
            all_passed = False
        else:
            print(f"  ✅ All required stages present")
        
        print(f"  Stages executed: {stages}")
        
        # Test determinism
        print("\n4. Testing determinism (2 runs)...")
        output_dir2 = Path(tmpdir) / "output2"
        
        cmd2 = [
            "python", "-m", "lux_depth_v2.cli",
            "--mode", "raw",
            "--input", test_image,
            "--output-dir", str(output_dir2)
        ]
        
        exit_code2, stdout2, stderr2 = run_command(cmd2)
        
        if exit_code2 != 0:
            print(f"  ❌ FAILED: Second run failed")
            all_passed = False
        else:
            # Compare outputs
            master1 = output_dir / f"{stem}_master16.tif"
            master2 = output_dir2 / f"{stem}_master16.tif"
            
            if not master1.exists() or not master2.exists():
                print(f"  ❌ FAILED: Output files not found")
                all_passed = False
            else:
                # Compare file sizes (should be identical)
                size1 = master1.stat().st_size
                size2 = master2.stat().st_size
                
                if size1 != size2:
                    print(f"  ❌ FAILED: File sizes differ ({size1} vs {size2})")
                    all_passed = False
                else:
                    # Binary compare
                    with open(master1, 'rb') as f1, open(master2, 'rb') as f2:
                        bytes1 = f1.read()
                        bytes2 = f2.read()
                        
                        if bytes1 != bytes2:
                            print(f"  ❌ FAILED: Files are not identical (binary diff)")
                            all_passed = False
                        else:
                            print(f"  ✅ Deterministic: outputs are identical")
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL VALIDATIONS PASSED")
            print("=" * 70)
            return True
        else:
            print("❌ VALIDATION FAILED")
            print("=" * 70)
            return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_raw_mode.py <test_image.tiff>")
        print("Example: python validate_raw_mode.py input_images/750Picacho_Kitchen_16bit.tiff")
        sys.exit(1)
    
    test_image = sys.argv[1]
    
    if not Path(test_image).exists():
        print(f"Error: Test image not found: {test_image}")
        sys.exit(1)
    
    success = validate_raw_mode(test_image)
    sys.exit(0 if success else 1)
