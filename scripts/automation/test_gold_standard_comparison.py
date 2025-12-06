#!/usr/bin/env python3
"""
Gold Standard Pipeline Comparison Test
========================================

Comprehensive test comparing:
1. Gold Standard Lux Depth Pipeline (new, from Desktop)
2. Depth Integrated Luxury Pipeline Ultimate (existing best)
3. Unified Luxury Pipeline (existing production)

Test Image: 750 Picacho Pool (16-bit TIFF if available, else JPG)
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from PIL import Image
    import tifffile
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

@dataclass
class PipelineResult:
    """Results from a single pipeline run."""
    pipeline_name: str
    success: bool
    processing_time: float
    output_path: Optional[Path]
    output_size_mb: float
    output_dimensions: tuple
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PipelineComparison:
    """Test harness for comparing multiple pipelines."""
    
    def __init__(self, input_path: Path, test_name: str = "750_Picacho_Pool"):
        self.input_path = input_path
        self.test_name = test_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(f"output_gold_standard_test_{self.timestamp}")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Depth maps directory
        self.depth_dir = Path("output_750_Picacho_Depth_Maps")
        if not self.depth_dir.exists():
            print(f"⚠️  Warning: Depth maps not found at {self.depth_dir}")
        
        self.results: List[PipelineResult] = []
    
    def test_gold_standard_pipeline(self) -> PipelineResult:
        """Test the new gold standard pipeline."""
        print("\n" + "="*80)
        print("TEST 1: Gold Standard Lux Depth Pipeline")
        print("="*80)
        
        pipeline_name = "Gold Standard Lux Depth Pipeline"
        output_dir = self.base_output_dir / "gold_standard"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            
            # Build command
            cmd = [
                sys.executable,
                "gold_standard_lux_depth_pipeline.py",
                "--input", str(self.input_path),
                "--depth-dir", str(self.depth_dir),
                "--output-dir", str(output_dir),
                "--preset", "photo_realistic",
                "--upscale", "4",
                "--backend", "none",  # Start with no AI upscaling for baseline
                "--device", "auto",
            ]
            
            print(f"\n📋 Command: {' '.join(cmd)}\n")
            
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                # Find output files
                output_files = list(output_dir.glob("*_MASTER_16bit.tiff")) + \
                              list(output_dir.glob("*_UPSCALED_16bit.tiff"))
                
                if output_files:
                    output_path = output_files[0]
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    # Get dimensions
                    try:
                        if tifffile:
                            img = tifffile.imread(str(output_path))
                            dims = img.shape[:2][::-1]  # (width, height)
                        else:
                            img = Image.open(output_path)
                            dims = img.size
                    except Exception:
                        dims = (0, 0)
                    
                    # Parse report if available
                    report_path = output_dir / f"{self.input_path.stem}_report.json"
                    metrics = None
                    warnings = []
                    if report_path.exists():
                        with open(report_path) as f:
                            report = json.load(f)
                            metrics = report.get("metrics", {})
                            warnings = report.get("warnings", [])
                    
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=True,
                        processing_time=elapsed,
                        output_path=output_path,
                        output_size_mb=size_mb,
                        output_dimensions=dims,
                        metrics=metrics,
                        warnings=warnings
                    )
                else:
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=False,
                        processing_time=elapsed,
                        output_path=None,
                        output_size_mb=0,
                        output_dimensions=(0, 0),
                        error_message="No output files generated"
                    )
            else:
                return PipelineResult(
                    pipeline_name=pipeline_name,
                    success=False,
                    processing_time=elapsed,
                    output_path=None,
                    output_size_mb=0,
                    output_dimensions=(0, 0),
                    error_message=f"Exit code {result.returncode}: {result.stderr}"
                )
        
        except Exception as e:
            return PipelineResult(
                pipeline_name=pipeline_name,
                success=False,
                processing_time=0,
                output_path=None,
                output_size_mb=0,
                output_dimensions=(0, 0),
                error_message=str(e)
            )
    
    def test_depth_integrated_ultimate(self) -> PipelineResult:
        """Test the existing depth integrated luxury pipeline ultimate."""
        print("\n" + "="*80)
        print("TEST 2: Depth Integrated Luxury Pipeline Ultimate (Existing Best)")
        print("="*80)
        
        pipeline_name = "Depth Integrated Ultimate"
        output_dir = self.base_output_dir / "depth_integrated_ultimate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            
            cmd = [
                sys.executable,
                "depth_integrated_luxury_pipeline_ultimate.py",
                "--input", str(self.input_path),
                "--depth-maps", str(self.depth_dir),
                "--output", str(output_dir),
                "--preset", "signature_estate",
                "--upscale", "4",
            ]
            
            print(f"\n📋 Command: {' '.join(cmd)}\n")
            
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                output_files = list(output_dir.glob("*.tiff")) + list(output_dir.glob("*.tif"))
                if output_files:
                    output_path = output_files[0]
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    try:
                        if tifffile:
                            img = tifffile.imread(str(output_path))
                            dims = img.shape[:2][::-1]
                        else:
                            img = Image.open(output_path)
                            dims = img.size
                    except Exception:
                        dims = (0, 0)
                    
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=True,
                        processing_time=elapsed,
                        output_path=output_path,
                        output_size_mb=size_mb,
                        output_dimensions=dims
                    )
                else:
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=False,
                        processing_time=elapsed,
                        output_path=None,
                        output_size_mb=0,
                        output_dimensions=(0, 0),
                        error_message="No output files generated"
                    )
            else:
                return PipelineResult(
                    pipeline_name=pipeline_name,
                    success=False,
                    processing_time=elapsed,
                    output_path=None,
                    output_size_mb=0,
                    output_dimensions=(0, 0),
                    error_message=f"Exit code {result.returncode}: {result.stderr}"
                )
        
        except Exception as e:
            return PipelineResult(
                pipeline_name=pipeline_name,
                success=False,
                processing_time=0,
                output_path=None,
                output_size_mb=0,
                output_dimensions=(0, 0),
                error_message=str(e)
            )
    
    def test_unified_luxury(self) -> PipelineResult:
        """Test the unified luxury pipeline."""
        print("\n" + "="*80)
        print("TEST 3: Unified Luxury Pipeline (Production)")
        print("="*80)
        
        pipeline_name = "Unified Luxury Pipeline"
        output_dir = self.base_output_dir / "unified_luxury"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if pipeline exists and is callable
        if not Path("unified_luxury_pipeline.py").exists():
            return PipelineResult(
                pipeline_name=pipeline_name,
                success=False,
                processing_time=0,
                output_path=None,
                output_size_mb=0,
                output_dimensions=(0, 0),
                error_message="unified_luxury_pipeline.py not found"
            )
        
        try:
            import subprocess
            
            cmd = [
                sys.executable,
                "unified_luxury_pipeline.py",
                str(self.input_path),
                "--preset", "photo_realistic",
                "--output", str(output_dir),
            ]
            
            print(f"\n📋 Command: {' '.join(cmd)}\n")
            
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                output_files = list(output_dir.glob("*.tiff")) + list(output_dir.glob("*.tif"))
                if output_files:
                    output_path = output_files[0]
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    try:
                        if tifffile:
                            img = tifffile.imread(str(output_path))
                            dims = img.shape[:2][::-1]
                        else:
                            img = Image.open(output_path)
                            dims = img.size
                    except Exception:
                        dims = (0, 0)
                    
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=True,
                        processing_time=elapsed,
                        output_path=output_path,
                        output_size_mb=size_mb,
                        output_dimensions=dims
                    )
                else:
                    return PipelineResult(
                        pipeline_name=pipeline_name,
                        success=False,
                        processing_time=elapsed,
                        output_path=None,
                        output_size_mb=0,
                        output_dimensions=(0, 0),
                        error_message="No output files generated"
                    )
            else:
                return PipelineResult(
                    pipeline_name=pipeline_name,
                    success=False,
                    processing_time=elapsed,
                    output_path=None,
                    output_size_mb=0,
                    output_dimensions=(0, 0),
                    error_message=f"Exit code {result.returncode}: {result.stderr}"
                )
        
        except Exception as e:
            return PipelineResult(
                pipeline_name=pipeline_name,
                success=False,
                processing_time=0,
                output_path=None,
                output_size_mb=0,
                output_dimensions=(0, 0),
                error_message=str(e)
            )
    
    def run_all_tests(self):
        """Run all pipeline tests."""
        print("\n" + "🌟"*40)
        print("GOLD STANDARD PIPELINE COMPARISON TEST")
        print("🌟"*40)
        print(f"\nTest Image: {self.input_path}")
        print(f"Test Name: {self.test_name}")
        print(f"Output Directory: {self.base_output_dir}")
        print(f"Timestamp: {self.timestamp}")
        
        # Test 1: Gold Standard
        result1 = self.test_gold_standard_pipeline()
        self.results.append(result1)
        
        # Test 2: Depth Integrated Ultimate
        result2 = self.test_depth_integrated_ultimate()
        self.results.append(result2)
        
        # Test 3: Unified Luxury
        result3 = self.test_unified_luxury()
        self.results.append(result3)
        
        # Generate comparison report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80)
        
        # Summary table
        print("\n📊 PROCESSING SUMMARY\n")
        print(f"{'Pipeline':<35} {'Status':<10} {'Time (s)':<12} {'Output Size (MB)':<18} {'Dimensions':<15}")
        print("-" * 95)
        
        for result in self.results:
            status = "✅ Success" if result.success else "❌ Failed"
            time_str = f"{result.processing_time:.2f}" if result.success else "N/A"
            size_str = f"{result.output_size_mb:.2f}" if result.success else "N/A"
            dims_str = f"{result.output_dimensions[0]}x{result.output_dimensions[1]}" if result.success else "N/A"
            
            print(f"{result.pipeline_name:<35} {status:<10} {time_str:<12} {size_str:<18} {dims_str:<15}")
            
            if result.error_message:
                print(f"  ⚠️  Error: {result.error_message}")
            
            if result.warnings:
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"  ⚠️  Warning: {warning}")
        
        # Quality metrics comparison
        print("\n📈 QUALITY METRICS\n")
        for result in self.results:
            if result.metrics:
                print(f"\n{result.pipeline_name}:")
                for key, value in result.metrics.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
        
        # Output locations
        print("\n📁 OUTPUT LOCATIONS\n")
        for result in self.results:
            if result.output_path:
                print(f"{result.pipeline_name}:")
                print(f"  {result.output_path}")
        
        # Save JSON report
        report_path = self.base_output_dir / "comparison_report.json"
        report_data = {
            "test_name": self.test_name,
            "input_path": str(self.input_path),
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results]
        }
        
        # Convert Path objects to strings for JSON serialization
        for result in report_data["results"]:
            if result["output_path"]:
                result["output_path"] = str(result["output_path"])
        
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS\n")
        successful = [r for r in self.results if r.success]
        if successful:
            fastest = min(successful, key=lambda r: r.processing_time)
            print(f"⚡ Fastest: {fastest.pipeline_name} ({fastest.processing_time:.2f}s)")
            
            if any(r.metrics for r in successful):
                print(f"\n✨ For detailed quality comparison, examine output files:")
                for result in successful:
                    if result.output_path:
                        print(f"  - {result.output_path}")
        else:
            print("⚠️  All pipelines failed. Check error messages above.")
        
        print("\n" + "="*80)


def main():
    """Main entry point."""
    
    if not PIL_AVAILABLE:
        print("⚠️  Warning: PIL/tifffile not available. Install with: pip install Pillow tifffile")
    
    # Find test input
    test_candidates = [
        Path("/Users/rc/Desktop/750Picacho_Pool.jpg"),
        Path("/Users/rc/Desktop/750Picacho_Aerial750Picacho_Pool_Ultimate.tiff"),
        Path("input_images/750Picacho_Pool.tif"),
    ]
    
    test_input = None
    for candidate in test_candidates:
        if candidate.exists():
            test_input = candidate
            break
    
    if not test_input:
        print("❌ No test input found. Tried:")
        for c in test_candidates:
            print(f"  - {c}")
        return 1
    
    # Run comparison
    comparison = PipelineComparison(test_input, test_name="750_Picacho_Pool")
    comparison.run_all_tests()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
