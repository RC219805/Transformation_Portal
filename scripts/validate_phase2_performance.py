#!/usr/bin/env python3
"""
Phase 2 Performance Validation Script

Automates benchmark testing of Phase 2 performance optimizations.
Measures actual performance gains against Phase 1 baseline.

Usage:
    python3 scripts/validate_phase2_performance.py --test-dir input_images/test_set
    python3 scripts/validate_phase2_performance.py --test baseline
    python3 scripts/validate_phase2_performance.py --test io-optimization
    python3 scripts/validate_phase2_performance.py --test parallel
    python3 scripts/validate_phase2_performance.py --test full-stack
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    test_name: str
    images_processed: int
    total_time_sec: float
    avg_time_per_image_sec: float
    throughput_img_per_hour: float
    success_count: int
    failure_count: int
    peak_memory_gb: float
    configuration: Dict[str, Any]
    individual_timings: List[Dict[str, float]]


class Phase2Validator:
    """Validates Phase 2 performance optimizations."""
    
    def __init__(self, test_dir: Path, output_base: Path):
        """
        Initialize validator.
        
        Args:
            test_dir: Directory containing test images
            output_base: Base directory for test outputs
        """
        self.test_dir = Path(test_dir)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        
    def run_benchmark(
        self,
        test_name: str,
        cli_args: List[str],
        config: Dict[str, Any]
    ) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Args:
            test_name: Name of the test
            cli_args: CLI arguments for lux-depth-v2
            config: Configuration metadata
        
        Returns:
            BenchmarkResult with timing and success metrics
        """
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        print(f"Configuration: {config}")
        print(f"CLI Args: {' '.join(cli_args)}\n")
        
        # Create output directory for this test
        output_dir = self.output_base / test_name.replace(' ', '_').lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            'python3', '-m', 'lux_depth_v2.cli',
            '--input-dir', str(self.test_dir),
            '--output-dir', str(output_dir),
        ] + cli_args
        
        # Run benchmark
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max
            )
            
            elapsed = time.time() - start_time
            
            # Parse output for metrics
            success_count = 0
            failure_count = 0
            individual_timings = []
            
            # Count successful outputs
            output_files = list(output_dir.glob('*_processed.tif'))
            success_count = len(output_files)
            
            # Estimate failures (simple heuristic)
            input_files = list(self.test_dir.glob('*.tif')) + list(self.test_dir.glob('*.tiff'))
            failure_count = max(0, len(input_files) - success_count)
            
            # Calculate metrics
            images_processed = success_count
            avg_time = elapsed / max(1, images_processed)
            throughput = (images_processed / elapsed) * 3600 if elapsed > 0 else 0
            
            benchmark_result = BenchmarkResult(
                test_name=test_name,
                images_processed=images_processed,
                total_time_sec=elapsed,
                avg_time_per_image_sec=avg_time,
                throughput_img_per_hour=throughput,
                success_count=success_count,
                failure_count=failure_count,
                peak_memory_gb=0.0,  # TODO: Extract from logs
                configuration=config,
                individual_timings=individual_timings
            )
            
            print(f"\n✅ Test complete: {test_name}")
            print(f"   Images processed: {images_processed}")
            print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"   Avg time/image: {avg_time:.1f}s")
            print(f"   Throughput: {throughput:.1f} images/hour")
            print(f"   Success rate: {success_count}/{success_count + failure_count}")
            
            self.results.append(benchmark_result)
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            print(f"❌ Test timeout: {test_name}")
            return None
        except Exception as e:
            print(f"❌ Test failed: {test_name} - {e}")
            return None
    
    def run_baseline_test(self):
        """Run Phase 1 baseline test (no optimizations)."""
        return self.run_benchmark(
            test_name="Phase 1 Baseline",
            cli_args=[
                '--preset', 'photo_realistic',
                # No Phase 2 optimizations
            ],
            config={
                'phase': 'Phase 1 (Baseline)',
                'async_io': False,
                'streaming_upscale': False,
                'parallel_workers': 1,
                'model_cache': False,
                'depth_cache': False,
            }
        )
    
    def run_io_optimization_test(self):
        """Run I/O optimization test."""
        return self.run_benchmark(
            test_name="Phase 2 I/O Optimization",
            cli_args=[
                '--preset', 'photo_realistic',
                '--async-io',
                '--streaming-upscale',
                '--tiff-compression', 'lzw',
            ],
            config={
                'phase': 'Phase 2 (I/O Optimization)',
                'async_io': True,
                'streaming_upscale': True,
                'tiff_compression': 'lzw',
                'parallel_workers': 1,
            }
        )
    
    def run_parallel_test(self):
        """Run parallel processing test."""
        return self.run_benchmark(
            test_name="Phase 2 Parallel Processing",
            cli_args=[
                '--preset', 'photo_realistic',
                '--parallel-workers', '2',
                '--model-cache',
                '--depth-cache',
            ],
            config={
                'phase': 'Phase 2 (Parallel Processing)',
                'parallel_workers': 2,
                'model_cache': True,
                'depth_cache': True,
            }
        )
    
    def run_full_stack_test(self):
        """Run full Phase 2 stack test (all optimizations)."""
        return self.run_benchmark(
            test_name="Phase 2 Full Stack",
            cli_args=[
                '--preset', 'photo_realistic',
                '--phase2-optimizations',
                '--parallel-workers', '2',
                '--async-io',
                '--streaming-upscale',
                '--tiff-compression', 'lzw',
                '--model-cache',
                '--depth-cache',
            ],
            config={
                'phase': 'Phase 2 (Full Stack)',
                'async_io': True,
                'streaming_upscale': True,
                'tiff_compression': 'lzw',
                'parallel_workers': 2,
                'model_cache': True,
                'depth_cache': True,
            }
        )
    
    def generate_report(self, output_file: Path):
        """Generate validation report."""
        if not self.results:
            print("No results to report")
            return
        
        # Calculate improvements
        baseline = next((r for r in self.results if 'Baseline' in r.test_name), None)
        
        report = {
            'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_directory': str(self.test_dir),
            'output_directory': str(self.output_base),
            'results': [asdict(r) for r in self.results],
            'summary': {}
        }
        
        if baseline:
            baseline_time = baseline.avg_time_per_image_sec
            baseline_throughput = baseline.throughput_img_per_hour
            
            improvements = []
            for result in self.results:
                if result.test_name == baseline.test_name:
                    continue
                
                speedup = baseline_time / result.avg_time_per_image_sec if result.avg_time_per_image_sec > 0 else 0
                throughput_gain = (result.throughput_img_per_hour / baseline_throughput - 1) * 100 if baseline_throughput > 0 else 0
                
                improvements.append({
                    'test': result.test_name,
                    'speedup': f"{speedup:.2f}x",
                    'throughput_gain': f"{throughput_gain:.1f}%",
                    'avg_time_sec': result.avg_time_per_image_sec,
                    'baseline_time_sec': baseline_time,
                })
            
            report['summary'] = {
                'baseline_avg_time_sec': baseline_time,
                'baseline_throughput_img_per_hour': baseline_throughput,
                'improvements': improvements
            }
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Validation Report: {output_file}")
        print(f"{'='*80}\n")
        
        # Print summary
        if baseline and 'improvements' in report['summary']:
            print(f"Baseline Performance:")
            print(f"  Avg time/image: {baseline_time:.1f}s ({baseline_time/60:.2f} min)")
            print(f"  Throughput: {baseline_throughput:.1f} images/hour\n")
            
            print(f"Phase 2 Improvements:")
            for imp in report['summary']['improvements']:
                print(f"  {imp['test']}:")
                print(f"    Speedup: {imp['speedup']}")
                print(f"    Throughput gain: {imp['throughput_gain']}")
                print(f"    Avg time: {imp['avg_time_sec']:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 2 performance optimizations"
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='input_images/test_set',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/phase2_validation',
        help='Output directory for test results'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['baseline', 'io-optimization', 'parallel', 'full-stack', 'all'],
        default='all',
        help='Which test to run'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='outputs/phase2_validation_report.json',
        help='Output file for validation report'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = Phase2Validator(
        test_dir=Path(args.test_dir),
        output_base=Path(args.output_dir)
    )
    
    # Run selected tests
    if args.test in ['baseline', 'all']:
        validator.run_baseline_test()
    
    if args.test in ['io-optimization', 'all']:
        validator.run_io_optimization_test()
    
    if args.test in ['parallel', 'all']:
        validator.run_parallel_test()
    
    if args.test in ['full-stack', 'all']:
        validator.run_full_stack_test()
    
    # Generate report
    validator.generate_report(Path(args.report))
    
    print(f"\n✅ Phase 2 validation complete!")
    print(f"   Report: {args.report}")
    print(f"   Outputs: {args.output_dir}")


if __name__ == '__main__':
    main()
