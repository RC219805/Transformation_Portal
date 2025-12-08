#!/usr/bin/env python3
"""Benchmark lux depth v2 CPU vs MPS performance."""
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent))

from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset

def benchmark(device: str, upscale: int, test_name: str):
    input_path = Path("input_images/750_Picacho/Ultimate_TIFFs_Base/750Picacho_Pool_Ultimate.tif")
    depth_dir = Path("output_750_Picacho_Depth_Maps_MaxQuality_20251206")
    output_dir = Path(f"output_benchmark_{device}_{upscale}x")
    
    config = PipelineConfig()
    config.preset = Preset.EXTERIOR_SHOWCASE
    config.device = device
    config.output_dir = output_dir
    config.depth_dir = depth_dir
    config.upscaler_backend = "torch"
    config.upscale = upscale
    config.save_upscaled = True
    config.save_marketing_png = False
    config.save_preview_jpg = False
    
    pipe = LuxPipelineV2(config)
    
    start = time.time()
    report = pipe.process_one(input_path)
    elapsed = time.time() - start
    
    throughput = 3600 / elapsed if elapsed > 0 else 0
    
    return {
        'test': test_name,
        'device': str(pipe.device),
        'upscale': upscale,
        'time': elapsed,
        'throughput': int(throughput),
        'ai_color_diff': report.get('ai_color_diff', 0),
        'ai_luma_diff': report.get('ai_luma_diff', 0),
    }

def main():
    print("=" * 80)
    print("PERFORMANCE BENCHMARK: Lux Depth V2 - CPU vs MPS (Apple Silicon)")
    print("=" * 80)
    print()
    
    results = []
    
    tests = [
        ('cpu', 2, 'CPU: 2x Upscale'),
        ('auto', 2, 'MPS: 2x Upscale'),
        ('cpu', 4, 'CPU: 4x Upscale'),
        ('auto', 4, 'MPS: 4x Upscale'),
    ]
    
    for device, upscale, name in tests:
        print(f"🔄 Running: {name}...")
        result = benchmark(device, upscale, name)
        results.append(result)
        print(f"   ✓ {result['time']:.2f}s ({result['throughput']} img/hr)")
        print()
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    print(f"{'Test':<25} {'Device':<8} {'Time':>8} {'Throughput':>12} {'Speedup':>10}")
    print("-" * 80)
    
    baseline = results[0]['time']
    for r in results:
        speedup = baseline / r['time']
        print(f"{r['test']:<25} {r['device']:<8} {r['time']:>7.2f}s {r['throughput']:>10} img/hr {speedup:>9.2f}x")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    cpu_2x = results[0]['time']
    mps_2x = results[1]['time']
    cpu_4x = results[2]['time']
    mps_4x = results[3]['time']
    
    print(f"\n2x Upscaling:")
    print(f"  MPS vs CPU: {cpu_2x/mps_2x:.2f}x {'faster' if mps_2x < cpu_2x else 'slower'}")
    
    print(f"\n4x Upscaling:")
    print(f"  MPS vs CPU: {cpu_4x/mps_4x:.2f}x {'faster' if mps_4x < cpu_4x else 'slower'}")
    
    print(f"\nUpscaling overhead (4x vs 2x):")
    print(f"  CPU: {cpu_4x - cpu_2x:.2f}s additional")
    print(f"  MPS: {mps_4x - mps_2x:.2f}s additional")
    
    print("\n✅ Benchmark complete!")
    
    # Cleanup
    import shutil
    for device, upscale, _ in tests:
        output_dir = Path(f"output_benchmark_{device}_{upscale}x")
        if output_dir.exists():
            shutil.rmtree(output_dir)
    print("   Cleaned up benchmark outputs")

if __name__ == "__main__":
    main()
