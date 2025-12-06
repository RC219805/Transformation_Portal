#!/usr/bin/env python3
"""
Performance Monitoring Example
===============================

Demonstrate telemetry collection and performance monitoring.

This example shows how to:
- Collect detailed timing metrics
- Track memory usage
- Export metrics to JSON
- (Optional) Export to Prometheus format
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


@dataclass
class PipelineMetrics:
    """Telemetry metrics for pipeline processing."""
    
    # Timing metrics
    total_time_s: float
    init_time_s: float
    avg_processing_time_s: float
    min_processing_time_s: float
    max_processing_time_s: float
    throughput_images_per_hour: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Processing statistics
    total_images: int
    successful: int
    failed: int
    skipped: int
    
    # Device info
    device: str
    autocast_enabled: bool
    
    # Configuration
    preset: str
    upscale_factor: int
    material_enabled: bool
    
    def to_json(self, path: Path) -> None:
        """Export metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP lux_pipeline_total_time_seconds Total processing time")
        lines.append("# TYPE lux_pipeline_total_time_seconds gauge")
        lines.append(f"lux_pipeline_total_time_seconds {self.total_time_s}")
        
        lines.append("# HELP lux_pipeline_throughput_images_per_hour Processing throughput")
        lines.append("# TYPE lux_pipeline_throughput_images_per_hour gauge")
        lines.append(f"lux_pipeline_throughput_images_per_hour {self.throughput_images_per_hour}")
        
        lines.append("# HELP lux_pipeline_peak_memory_mb Peak memory usage")
        lines.append("# TYPE lux_pipeline_peak_memory_mb gauge")
        lines.append(f"lux_pipeline_peak_memory_mb {self.peak_memory_mb}")
        
        lines.append("# HELP lux_pipeline_images_processed_total Total images processed")
        lines.append("# TYPE lux_pipeline_images_processed_total counter")
        lines.append(f'lux_pipeline_images_processed_total{{status="success"}} {self.successful}')
        lines.append(f'lux_pipeline_images_processed_total{{status="failed"}} {self.failed}')
        lines.append(f'lux_pipeline_images_processed_total{{status="skipped"}} {self.skipped}')
        
        return "\n".join(lines)


class MonitoredPipeline:
    """Wrapper for LuxPipelineV2 with telemetry."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.init_start = time.time()
        self.pipeline = LuxPipelineV2(config)
        self.init_time = time.time() - self.init_start
        
        self.memory_samples: List[float] = []
        self.processing_times: List[float] = []
        
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def process_directory(self) -> tuple[List[Dict[str, Any]], PipelineMetrics]:
        """Process directory with telemetry collection."""
        start_time = time.time()
        
        # Sample memory before processing
        initial_memory = self._get_memory_mb()
        self.memory_samples = [initial_memory]
        
        # Process images
        results = self.pipeline.process_directory()
        
        # Collect metrics
        for result in results:
            if result['status'] == 'ok':
                self.processing_times.append(result['timing_s'])
            
            # Sample memory periodically
            self.memory_samples.append(self._get_memory_mb())
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = sum(1 for r in results if r['status'] == 'ok')
        failed = sum(1 for r in results if r['status'] == 'error')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            throughput = 3600 / avg_time if avg_time > 0 else 0
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
        else:
            avg_time = throughput = min_time = max_time = 0.0
        
        # Memory statistics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0.0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0.0
        
        # Create metrics object
        metrics = PipelineMetrics(
            total_time_s=total_time,
            init_time_s=self.init_time,
            avg_processing_time_s=avg_time,
            min_processing_time_s=min_time,
            max_processing_time_s=max_time,
            throughput_images_per_hour=throughput,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            total_images=len(results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            device=str(self.pipeline.device),
            autocast_enabled=self.pipeline.autocast,
            preset=str(self.config.preset.value),
            upscale_factor=self.config.upscale,
            material_enabled=self.config.enable_material,
        )
        
        return results, metrics


def main():
    # Configure pipeline
    config = PipelineConfig(
        preset=Preset.PHOTO_REALISTIC,
        input_dir=Path("input"),
        output_dir=Path("output_monitored"),
        depth_dir=Path("depth_maps"),
        device="auto",
        upscale=4,
        upscaler_backend="none",
        enable_material=True,
    )
    
    print("=" * 60)
    print("Lux Depth V2 - Performance Monitoring")
    print("=" * 60)
    
    # Create monitored pipeline
    print("\nInitializing pipeline with telemetry...")
    monitored = MonitoredPipeline(config)
    print(f"Initialization time: {monitored.init_time:.2f}s")
    
    # Process with monitoring
    print(f"\nProcessing directory: {config.input_dir}")
    results, metrics = monitored.process_directory()
    
    # Display metrics
    print("\n" + "=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    
    print(f"\nTiming:")
    print(f"  Total time: {metrics.total_time_s:.2f}s ({metrics.total_time_s/60:.1f} min)")
    print(f"  Init time: {metrics.init_time_s:.2f}s")
    print(f"  Avg processing: {metrics.avg_processing_time_s:.2f}s/image")
    print(f"  Min processing: {metrics.min_processing_time_s:.2f}s")
    print(f"  Max processing: {metrics.max_processing_time_s:.2f}s")
    print(f"  Throughput: {metrics.throughput_images_per_hour:.1f} images/hour")
    
    print(f"\nMemory:")
    print(f"  Peak: {metrics.peak_memory_mb:.1f} MB")
    print(f"  Average: {metrics.avg_memory_mb:.1f} MB")
    
    print(f"\nStatistics:")
    print(f"  Total images: {metrics.total_images}")
    print(f"  Successful: {metrics.successful}")
    print(f"  Failed: {metrics.failed}")
    print(f"  Skipped: {metrics.skipped}")
    
    print(f"\nConfiguration:")
    print(f"  Device: {metrics.device}")
    print(f"  Autocast: {metrics.autocast_enabled}")
    print(f"  Preset: {metrics.preset}")
    print(f"  Upscale: {metrics.upscale_factor}x")
    print(f"  Material: {metrics.material_enabled}")
    
    # Export metrics
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    json_path = metrics_dir / "pipeline_metrics.json"
    metrics.to_json(json_path)
    print(f"\nMetrics exported to: {json_path}")
    
    prom_path = metrics_dir / "pipeline_metrics.prom"
    prom_path.write_text(metrics.to_prometheus())
    print(f"Prometheus metrics: {prom_path}")
    
    # Print sample Prometheus output
    print("\nPrometheus format sample:")
    print("-" * 60)
    print(metrics.to_prometheus()[:400] + "...")


if __name__ == "__main__":
    if not PSUTIL_AVAILABLE:
        print("Warning: psutil not installed. Memory metrics will be unavailable.")
        print("Install with: pip install psutil")
        print()
    
    main()
