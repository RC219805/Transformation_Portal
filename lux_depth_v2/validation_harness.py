#!/usr/bin/env python3
"""
Edge Refinement Validation - Automated Test Harness

Executes 40-test validation matrix and computes quality metrics.
Designed for reproducibility and audit evidence.

Usage:
    python validation_harness.py --dataset-dir validation_images/ --output-dir validation_results/
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib


class ValidationHarness:
    """Automated validation for edge refinement feature."""

    PRESETS = ["baseline", "subtle", "balanced", "aggressive"]

    def __init__(self, dataset_dir: Path, output_dir: Path, checksum_file: Path = None):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.checksum_file = checksum_file
        self.results = []

    def verify_dataset_integrity(self) -> bool:
        """Verify dataset hasn't changed using checksums."""
        if not self.checksum_file or not self.checksum_file.exists():
            print("⚠️  No checksum file - skipping integrity check")
            return True

        print("🔒 Verifying dataset integrity...")
        with open(self.checksum_file) as f:
            expected = dict(line.strip().split(maxsplit=1)[::-1] for line in f if line.strip())

        for image_path in self.dataset_dir.glob("*.{tiff,tif,png,jpg}"):
            actual_hash = self._compute_sha256(image_path)
            expected_hash = expected.get(image_path.name)

            if expected_hash and actual_hash != expected_hash:
                print(f"❌ INTEGRITY FAILURE: {image_path.name}")
                print(f"   Expected: {expected_hash}")
                print(f"   Actual:   {actual_hash}")
                return False

        print("✅ Dataset integrity verified")
        return True

    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def run_validation_matrix(self) -> List[Dict]:
        """Execute all 40 validation runs (10 images × 4 presets)."""
        images = sorted(self.dataset_dir.glob("*.{tiff,tif,png}"))

        if len(images) < 10:
            print(f"⚠️  WARNING: Only {len(images)} images found (expected 10)")

        print(f"\n📋 Validation Matrix: {len(images)} images × 4 presets = {len(images) * 4} runs")

        run_count = 0
        for image_path in images:
            for preset in self.PRESETS:
                run_count += 1
                print(f"\n[{run_count}/{len(images) * 4}] {image_path.name} - {preset}")

                result = self._run_single_test(image_path, preset)
                self.results.append(result)

                # Save incremental results
                self._save_results()

        return self.results

    def _run_single_test(self, image_path: Path, preset: str) -> Dict:
        """Execute single validation test."""
        output_subdir = self.output_dir / preset / image_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Build command
        if preset == "baseline":
            cmd = [
                "lux-depth-v2",
                "--input",
                str(image_path),
                "--output-dir",
                str(output_subdir),
                "--preset",
                "interior_luxury",
            ]
        else:
            cmd = [
                "lux-depth-v2",
                "--input",
                str(image_path),
                "--output-dir",
                str(output_subdir),
                "--preset",
                "interior_luxury",
                "--edge-refinement",
                "--refinement-preset",
                preset,
            ]

        # Execute with timing
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per image
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0

        except subprocess.TimeoutExpired:
            elapsed = 300
            success = False
            result = type("obj", (object,), {"stdout": "", "stderr": "TIMEOUT"})()

        # Parse output files
        output_files = list(output_subdir.glob("*"))

        return {
            "image": image_path.name,
            "preset": preset,
            "success": success,
            "elapsed_seconds": elapsed,
            "output_dir": str(output_subdir),
            "output_files": [f.name for f in output_files],
            "stdout": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
            "stderr": result.stderr[-500:] if result.stderr else "",
        }

    def _save_results(self):
        """Save incremental results to JSON."""
        results_file = self.output_dir / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def compute_metrics(self):
        """Compute quality metrics (Edge F1, PSNR, SSIM)."""
        print("\n📊 Computing quality metrics...")

        # This is a placeholder - actual metric computation requires
        # image processing libraries (opencv, skimage, etc.)
        print("⚠️  Metric computation requires additional implementation")
        print("    See: validation_metrics.py (to be created)")

        return {"edge_f1": "PENDING", "psnr": "PENDING", "ssim": "PENDING"}

    def generate_report(self):
        """Generate validation summary report."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        avg_time = sum(r["elapsed_seconds"] for r in self.results) / total if total > 0 else 0

        report = {
            "summary": {
                "total_runs": total,
                "successful": successful,
                "failed": failed,
                "success_rate": (f"{(successful / total * 100):.1f}%" if total > 0 else "0%"),
                "avg_time_seconds": f"{avg_time:.2f}",
            },
            "by_preset": {},
        }

        for preset in self.PRESETS:
            preset_results = [r for r in self.results if r["preset"] == preset]
            preset_successful = sum(1 for r in preset_results if r["success"])

            report["by_preset"][preset] = {
                "runs": len(preset_results),
                "successful": preset_successful,
                "failed": len(preset_results) - preset_successful,
            }

        return report


def main():
    parser = argparse.ArgumentParser(description="Edge refinement validation harness")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing validation images")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--checksum-file", help="Optional: SHA256 checksum file for integrity check")
    parser.add_argument("--skip-integrity", action="store_true", help="Skip dataset integrity check")

    args = parser.parse_args()

    harness = ValidationHarness(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        checksum_file=Path(args.checksum_file) if args.checksum_file else None,
    )

    # Step 1: Verify dataset integrity
    if not args.skip_integrity:
        if not harness.verify_dataset_integrity():
            print("\n❌ Dataset integrity check failed - ABORTING")
            return 1

    # Step 2: Run validation matrix
    print("\n" + "=" * 60)
    print("VALIDATION MATRIX EXECUTION")
    print("=" * 60)

    results = harness.run_validation_matrix()

    # Step 3: Generate report
    report = harness.generate_report()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total runs:    {report['summary']['total_runs']}")
    print(f"Successful:    {report['summary']['successful']}")
    print(f"Failed:        {report['summary']['failed']}")
    print(f"Success rate:  {report['summary']['success_rate']}")
    print(f"Avg time:      {report['summary']['avg_time_seconds']}s")

    print("\nBy Preset:")
    for preset, stats in report["by_preset"].items():
        print(f"  {preset:12s}: {stats['successful']}/{stats['runs']} successful")

    # Save final report
    report_file = harness.output_dir / "validation_summary.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Validation complete")
    print(f"📄 Results: {harness.output_dir / 'validation_results.json'}")
    print(f"📊 Summary: {report_file}")

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
