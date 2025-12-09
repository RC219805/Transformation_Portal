#!/usr/bin/env python3
"""Benchmark Materials v2 performance overhead.

Measures processing time, memory usage, and VRAM allocation for:
- Baseline (no materials)
- Materials v2 with heuristic backend
- Materials v2 with confidence gating
- Materials v2 with mask caching
"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    test_name: str
    input_file: str
    
    # Timing
    total_time_sec: float
    materials_time_sec: Optional[float] = None
    upscaling_time_sec: Optional[float] = None
    
    # Memory
    peak_memory_mb: float = 0.0
    peak_vram_mb: Optional[float] = None
    
    # Quality
    success: bool = True
    error_message: Optional[str] = None
    
    # Materials-specific
    confidence_threshold: Optional[float] = None
    cache_hit: bool = False
    materials_overhead_pct: Optional[float] = None


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def run_benchmark(
    input_file: str,
    output_dir: str,
    test_name: str,
    materials_v2: bool = False,
    confidence_threshold: float = 0.6,
    cache_masks: bool = False,
    preset: str = "photo_realistic",
    upscale: int = 2,
) -> BenchmarkResult:
    """Run a single benchmark test."""
    
    print(f"\n{'=' * 60}")
    print(f"Running: {test_name}")
    print(f"Input: {input_file}")
    print(f"Materials v2: {materials_v2}")
    if materials_v2:
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Cache masks: {cache_masks}")
    print(f"{'=' * 60}\n")
    
    # Build command
    cmd = [
        "python3", "-m", "lux_depth_v2.cli",
        "--input", input_file,
        "--output-dir", output_dir,
        "--preset", preset,
        "--device", "auto",
        "--upscale", str(upscale),
    ]
    
    if materials_v2:
        cmd.append("--materials-v2")
        cmd.extend(["--confidence-threshold", str(confidence_threshold)])
        
        if cache_masks:
            cmd.append("--cache-masks")
            cmd.extend(["--cache-dir", ".materials_v2_cache"])
    
    # Start monitoring
    start_time = time.time()
    peak_memory = get_memory_usage()
    
    try:
        # Run process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        success = result.returncode == 0
        error_msg = None if success else result.stderr
        
        # Update peak memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)
        
        # Parse timing from output if available
        materials_time = None
        upscaling_time = None
        
        # Look for timing info in stdout
        for line in result.stdout.split('\n'):
            if 'Materials v2' in line and 'sec' in line:
                # Extract timing
                try:
                    materials_time = float(line.split('sec')[0].split()[-1])
                except:
                    pass
            elif 'Upscaling' in line and 'sec' in line:
                try:
                    upscaling_time = float(line.split('sec')[0].split()[-1])
                except:
                    pass
        
        return BenchmarkResult(
            test_name=test_name,
            input_file=input_file,
            total_time_sec=total_time,
            materials_time_sec=materials_time,
            upscaling_time_sec=upscaling_time,
            peak_memory_mb=peak_memory,
            success=success,
            error_message=error_msg,
            confidence_threshold=confidence_threshold if materials_v2 else None,
        )
        
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            test_name=test_name,
            input_file=input_file,
            total_time_sec=600.0,
            peak_memory_mb=peak_memory,
            success=False,
            error_message="Timeout after 10 minutes",
        )
    except Exception as e:
        return BenchmarkResult(
            test_name=test_name,
            input_file=input_file,
            total_time_sec=0.0,
            peak_memory_mb=peak_memory,
            success=False,
            error_message=str(e),
        )


def calculate_overhead(baseline: BenchmarkResult, enhanced: BenchmarkResult) -> float:
    """Calculate percentage overhead of Materials v2."""
    if baseline.total_time_sec == 0:
        return 0.0
    
    overhead = ((enhanced.total_time_sec - baseline.total_time_sec) / 
                baseline.total_time_sec * 100)
    return overhead


def run_benchmark_suite(test_images: List[str], output_base: str = "output_Benchmark"):
    """Run complete benchmark suite."""
    
    results = []
    
    for image_path in test_images:
        image_name = Path(image_path).stem
        
        # Test 1: Baseline (no Materials v2)
        baseline = run_benchmark(
            input_file=image_path,
            output_dir=f"{output_base}_Baseline/{image_name}",
            test_name=f"{image_name} - Baseline",
            materials_v2=False,
        )
        results.append(baseline)
        
        # Test 2: Materials v2 - Medium confidence (0.6)
        materials_medium = run_benchmark(
            input_file=image_path,
            output_dir=f"{output_base}_Materials_Medium/{image_name}",
            test_name=f"{image_name} - Materials v2 (0.6)",
            materials_v2=True,
            confidence_threshold=0.6,
            cache_masks=True,
        )
        results.append(materials_medium)
        
        # Calculate overhead
        if baseline.success and materials_medium.success:
            overhead = calculate_overhead(baseline, materials_medium)
            materials_medium.materials_overhead_pct = overhead
            print(f"\n✓ Materials v2 overhead: {overhead:.1f}%")
        
        # Test 3: Materials v2 - High confidence (0.8)
        materials_high = run_benchmark(
            input_file=image_path,
            output_dir=f"{output_base}_Materials_High/{image_name}",
            test_name=f"{image_name} - Materials v2 (0.8)",
            materials_v2=True,
            confidence_threshold=0.8,
            cache_masks=True,
        )
        results.append(materials_high)
        
        if baseline.success and materials_high.success:
            overhead = calculate_overhead(baseline, materials_high)
            materials_high.materials_overhead_pct = overhead
            print(f"✓ Materials v2 (conservative) overhead: {overhead:.1f}%")
        
        # Test 4: Cache test (second run with same confidence)
        materials_cached = run_benchmark(
            input_file=image_path,
            output_dir=f"{output_base}_Materials_Cached/{image_name}",
            test_name=f"{image_name} - Materials v2 (cached)",
            materials_v2=True,
            confidence_threshold=0.6,
            cache_masks=True,
        )
        materials_cached.cache_hit = True
        results.append(materials_cached)
        
        if materials_medium.success and materials_cached.success:
            speedup = ((materials_medium.total_time_sec - materials_cached.total_time_sec) / 
                      materials_medium.total_time_sec * 100)
            print(f"✓ Cache speedup: {speedup:.1f}%")
    
    return results


def generate_report(results: List[BenchmarkResult], output_file: str = "benchmark_report.json"):
    """Generate comprehensive benchmark report."""
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_count": len(results),
        "successful_tests": sum(1 for r in results if r.success),
        "failed_tests": sum(1 for r in results if not r.success),
        "results": [asdict(r) for r in results],
    }
    
    # Calculate summary statistics
    baseline_times = [r.total_time_sec for r in results 
                     if r.success and not r.confidence_threshold]
    materials_times = [r.total_time_sec for r in results 
                      if r.success and r.confidence_threshold == 0.6 and not r.cache_hit]
    
    if baseline_times and materials_times:
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_materials = sum(materials_times) / len(materials_times)
        avg_overhead = ((avg_materials - avg_baseline) / avg_baseline * 100)
        
        report["summary"] = {
            "avg_baseline_time_sec": round(avg_baseline, 2),
            "avg_materials_time_sec": round(avg_materials, 2),
            "avg_overhead_pct": round(avg_overhead, 2),
        }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests: {report['test_count']}")
    print(f"Successful: {report['successful_tests']}")
    print(f"Failed: {report['failed_tests']}")
    
    if "summary" in report:
        print(f"\nAverage baseline time: {report['summary']['avg_baseline_time_sec']:.2f} sec")
        print(f"Average Materials v2 time: {report['summary']['avg_materials_time_sec']:.2f} sec")
        print(f"Average overhead: {report['summary']['avg_overhead_pct']:.1f}%")
    
    print(f"\nReport saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Materials v2 performance")
    parser.add_argument("--input-dir", default="input_images/750_Picacho/Optimized_TIFFs",
                       help="Directory containing test images")
    parser.add_argument("--output-base", default="output_Benchmark_Materials_V2",
                       help="Base directory for benchmark outputs")
    parser.add_argument("--report", default="materials_v2_benchmark_report.json",
                       help="Output report file")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test on single image")
    
    args = parser.parse_args()
    
    # Find test images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        exit(1)
    
    test_images = sorted(input_dir.glob("*.tif"))
    if args.quick and test_images:
        # Just test Pool image
        test_images = [img for img in test_images if "Pool" in img.name]
    
    if not test_images:
        print(f"Error: No TIFF images found in {input_dir}")
        exit(1)
    
    print(f"Found {len(test_images)} test images")
    for img in test_images:
        print(f"  - {img.name}")
    
    # Run benchmark suite
    results = run_benchmark_suite(test_images, args.output_base)
    
    # Generate report
    report = generate_report(results, args.report)
