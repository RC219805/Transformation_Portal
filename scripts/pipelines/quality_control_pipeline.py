#!/usr/bin/env python3
"""
Quality Control Pipeline for 750 Picacho Lane
Ensures consistent, high-quality processing with verification
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image


class QualityControlPipeline:
    """Manages end-to-end quality control for luxury rendering pipeline"""

    def __init__(self, project_name: str = "750_Picacho"):
        self.project_name = project_name
        self.canonical_sources = []
        self.quality_report = {"project": project_name, "sources": {}, "outputs": {}, "verification": {}}

    def verify_source_integrity(self, source_dir: Path) -> Dict:
        """Verify all source files are present and valid"""
        expected_files = [
            "750Picacho_Aerial.jpg",
            "750Picacho_GreatRoom.jpg",
            "750Picacho_Kitchen.jpg",
            "750Picacho_Pool.jpg",
            "750Picacho_PrimaryBathroom.jpg",
            "750Picacho_PrimaryBedroom.jpg",
        ]

        results = {}
        for filename in expected_files:
            filepath = source_dir / filename
            if not filepath.exists():
                results[filename] = {"status": "MISSING", "valid": False}
                continue

            try:
                with Image.open(filepath) as img:
                    # Calculate file hash for integrity
                    file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()

                    results[filename] = {
                        "status": "OK",
                        "valid": True,
                        "size": img.size,
                        "mode": img.mode,
                        "format": img.format,
                        "file_size_mb": filepath.stat().st_size / (1024 * 1024),
                        "hash": file_hash,
                    }
            except Exception as e:
                results[filename] = {"status": f"ERROR: {e}", "valid": False}

        self.quality_report["sources"] = results
        return results

    def verify_output_quality(self, output_dir: Path, format_type: str) -> Dict:
        """Verify output files meet quality standards"""
        results = {}

        for source_file in self.canonical_sources:
            base_name = source_file.stem
            output_file = output_dir / f"{base_name}_luxury.{format_type}"

            if not output_file.exists():
                results[output_file.name] = {"status": "MISSING", "valid": False}
                continue

            try:
                with Image.open(output_file) as img:
                    # Quality checks
                    checks = {"exists": True, "readable": True, "size": img.size, "mode": img.mode, "format": img.format}

                    # Format-specific checks
                    if format_type == "ti":
                        checks["bit_depth"] = "16-bit" if img.mode in ["I;16", "I;16B", "RGB;16"] else img.mode
                        checks["tiff_quality_ok"] = img.mode not in ["L", "P"]  # Should be RGB or better

                    # Size validation (should match source approximately)
                    source_file_obj = next((s for s in self.canonical_sources if s.stem == base_name), None)
                    if source_file_obj and source_file_obj.exists():
                        with Image.open(source_file_obj) as src_img:
                            size_match = img.size == src_img.size
                            checks["size_matches_source"] = size_match

                    results[output_file.name] = {"status": "OK", "valid": True, **checks}

            except Exception as e:
                results[output_file.name] = {"status": f"ERROR: {e}", "valid": False}

        return results

    def run_pipeline(self, input_dir: Path, output_dir: Path, formats: List[str] = ["jpeg", "png", "tif"]):
        """Run the unified luxury pipeline with quality controls"""

        # Step 1: Verify sources
        print("Step 1: Verifying source integrity...")
        source_results = self.verify_source_integrity(input_dir)

        valid_sources = sum(1 for r in source_results.values() if r.get("valid"))
        print(f"  ✓ {valid_sources}/{len(source_results)} source files valid")

        if valid_sources != len(source_results):
            print("  ⚠ WARNING: Not all sources valid!")
            for filename, result in source_results.items():
                if not result.get("valid"):
                    print(f"    - {filename}: {result.get('status')}")

            if input("Continue anyway? (y/n): ").lower() != "y":
                return False

        # Store canonical sources for later verification
        self.canonical_sources = [input_dir / f for f in source_results.keys() if source_results[f].get("valid")]

        # Step 2: Run unified pipeline
        print("\nStep 2: Running unified luxury pipeline...")
        cmd = [
            sys.executable,
            "unified_luxury_pipeline.py",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--formats",
            *formats,
            "--preset",
            "luxury_estate",
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("  ✓ Pipeline completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Pipeline failed: {e}")
            print(e.stderr)
            return False

        # Step 3: Verify outputs
        print("\nStep 3: Verifying output quality...")
        for fmt in formats:
            ext = "ti" if fmt == "tif" else ("jpg" if fmt == "jpeg" else fmt)
            output_results = self.verify_output_quality(output_dir, ext)

            valid_outputs = sum(1 for r in output_results.values() if r.get("valid"))
            print(f"  {fmt.upper()}: {valid_outputs}/{len(output_results)} files valid")

            self.quality_report["outputs"][fmt] = output_results

            # Check for TIFF degradation issue
            if fmt == "tif":
                degraded = [
                    name
                    for name, result in output_results.items()
                    if result.get("valid") and not result.get("tiff_quality_ok", True)
                ]
                if degraded:
                    print(f"  ⚠ WARNING: {len(degraded)} TIFF files may be degraded:")
                    for name in degraded:
                        print(f"    - {name}")

        # Step 4: Save quality report
        report_path = output_dir / "quality_control_report.json"
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"\n✓ Quality report saved to: {report_path}")

        return True


def main():
    """Run quality-controlled pipeline for 750 Picacho Lane"""

    input_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "JPEGs"
    output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Final_Production"

    qc = QualityControlPipeline("750_Picacho_Lane")
    success = qc.run_pipeline(input_dir=input_dir, output_dir=output_dir, formats=["jpeg", "png", "tif"])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
