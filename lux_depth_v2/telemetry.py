"""Performance telemetry and metrics collection for lux_depth_v2 pipeline.

This module provides instrumentation for tracking:
- Processing time per stage
- Memory usage
- GPU utilization (if available)
- Throughput metrics
- Error rates
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_delta_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ImageMetrics:
    """Metrics for processing a single image."""
    image_path: str
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_s: float = 0.0
    
    # Input characteristics
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    
    # Processing stages
    stages: List[StageMetrics] = field(default_factory=list)
    
    # Memory tracking
    peak_memory_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # Outputs
    zone_weights_source: str = ""
    material_mods_source: Optional[str] = None
    upscaler_used: str = ""
    
    # Quality metrics
    ai_color_drift: Optional[float] = None
    ai_luma_drift: Optional[float] = None
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    def add_stage(self, stage: StageMetrics) -> None:
        """Add a stage metrics to this image."""
        self.stages.append(stage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch of images."""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_s: float = 0.0
    
    # Batch statistics
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    # Timing statistics
    avg_processing_time_s: float = 0.0
    min_processing_time_s: float = 0.0
    max_processing_time_s: float = 0.0
    throughput_images_per_hour: float = 0.0
    
    # Memory statistics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Individual image metrics
    images: List[ImageMetrics] = field(default_factory=list)
    
    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def add_image(self, metrics: ImageMetrics) -> None:
        """Add image metrics to batch."""
        self.images.append(metrics)
        self.total_images += 1
        if metrics.success:
            self.successful += 1
        elif metrics.error:
            self.failed += 1
    
    def finalize(self) -> None:
        """Calculate aggregate statistics."""
        if not self.images:
            return
        
        # Filter successful images for timing stats
        successful = [m for m in self.images if m.success]
        
        if successful:
            times = [m.total_duration_s for m in successful]
            self.avg_processing_time_s = sum(times) / len(times)
            self.min_processing_time_s = min(times)
            self.max_processing_time_s = max(times)
            
            if self.avg_processing_time_s > 0:
                self.throughput_images_per_hour = 3600.0 / self.avg_processing_time_s
        
        # Memory statistics
        all_peaks = [m.peak_memory_mb for m in self.images if m.peak_memory_mb > 0]
        if all_peaks:
            self.peak_memory_mb = max(all_peaks)
            self.avg_memory_mb = sum(all_peaks) / len(all_peaks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self, path: Path) -> None:
        """Export metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        
        # Total duration
        lines.append("# HELP lux_batch_duration_seconds Total batch processing duration")
        lines.append("# TYPE lux_batch_duration_seconds gauge")
        lines.append(f"lux_batch_duration_seconds {self.total_duration_s:.3f}")
        
        # Throughput
        lines.append("# HELP lux_batch_throughput_images_per_hour Processing throughput")
        lines.append("# TYPE lux_batch_throughput_images_per_hour gauge")
        lines.append(f"lux_batch_throughput_images_per_hour {self.throughput_images_per_hour:.2f}")
        
        # Processing time statistics
        lines.append("# HELP lux_image_processing_seconds Image processing time statistics")
        lines.append("# TYPE lux_image_processing_seconds summary")
        lines.append(f"lux_image_processing_seconds{{quantile=\"0.0\"}} {self.min_processing_time_s:.3f}")
        lines.append(f"lux_image_processing_seconds{{quantile=\"0.5\"}} {self.avg_processing_time_s:.3f}")
        lines.append(f"lux_image_processing_seconds{{quantile=\"1.0\"}} {self.max_processing_time_s:.3f}")
        lines.append(f"lux_image_processing_seconds_sum {self.total_duration_s:.3f}")
        lines.append(f"lux_image_processing_seconds_count {self.total_images}")
        
        # Memory
        lines.append("# HELP lux_memory_usage_megabytes Memory usage statistics")
        lines.append("# TYPE lux_memory_usage_megabytes gauge")
        lines.append(f"lux_memory_usage_megabytes{{stat=\"peak\"}} {self.peak_memory_mb:.1f}")
        lines.append(f"lux_memory_usage_megabytes{{stat=\"avg\"}} {self.avg_memory_mb:.1f}")
        
        # Image counts
        lines.append("# HELP lux_images_processed_total Total images processed by status")
        lines.append("# TYPE lux_images_processed_total counter")
        lines.append(f'lux_images_processed_total{{status="success"}} {self.successful}')
        lines.append(f'lux_images_processed_total{{status="failed"}} {self.failed}')
        lines.append(f'lux_images_processed_total{{status="skipped"}} {self.skipped}')
        
        return "\n".join(lines) + "\n"


class MetricsCollector:
    """Collector for pipeline telemetry."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.current_image: Optional[ImageMetrics] = None
        self.current_stage: Optional[StageMetrics] = None
        self.batch: BatchMetrics = BatchMetrics()
    
    def start_batch(self, config_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Start collecting batch metrics."""
        if not self.enabled:
            return
        self.batch = BatchMetrics(
            start_time=time.time(),
            config_snapshot=config_snapshot or {}
        )
    
    def end_batch(self) -> BatchMetrics:
        """Finalize batch metrics and return."""
        if not self.enabled:
            return self.batch
        self.batch.end_time = time.time()
        self.batch.total_duration_s = self.batch.end_time - self.batch.start_time
        self.batch.finalize()
        return self.batch
    
    def start_image(self, image_path: str, width: int = 0, height: int = 0) -> None:
        """Start tracking a new image."""
        if not self.enabled:
            return
        self.current_image = ImageMetrics(
            image_path=image_path,
            start_time=time.time(),
            width=width,
            height=height,
            megapixels=(width * height) / 1e6 if width and height else 0.0,
            peak_memory_mb=self._get_memory_mb(),
            memory_samples=[self._get_memory_mb()],
        )
    
    def end_image(self, success: bool = True, error: Optional[str] = None) -> None:
        """Finalize current image metrics."""
        if not self.enabled or not self.current_image:
            return
        
        self.current_image.end_time = time.time()
        self.current_image.total_duration_s = self.current_image.end_time - self.current_image.start_time
        self.current_image.success = success
        self.current_image.error = error
        
        # Sample final memory
        final_mem = self._get_memory_mb()
        self.current_image.memory_samples.append(final_mem)
        self.current_image.peak_memory_mb = max(self.current_image.memory_samples)
        
        self.batch.add_image(self.current_image)
        self.current_image = None
    
    @contextmanager
    def stage(self, name: str):
        """Context manager for tracking a processing stage."""
        if not self.enabled or not self.current_image:
            yield
            return
        
        stage = StageMetrics(
            name=name,
            start_time=time.time(),
            memory_start_mb=self._get_memory_mb(),
        )
        
        try:
            yield stage
            stage.success = True
        except Exception as e:
            stage.success = False
            stage.error = str(e)
            raise
        finally:
            stage.end_time = time.time()
            stage.duration_s = stage.end_time - stage.start_time
            stage.memory_end_mb = self._get_memory_mb()
            stage.memory_delta_mb = stage.memory_end_mb - stage.memory_start_mb
            
            if self.current_image:
                self.current_image.add_stage(stage)
                # Update peak memory
                if stage.memory_end_mb > self.current_image.peak_memory_mb:
                    self.current_image.peak_memory_mb = stage.memory_end_mb
    
    def sample_memory(self) -> None:
        """Sample current memory usage."""
        if not self.enabled or not self.current_image:
            return
        mem = self._get_memory_mb()
        self.current_image.memory_samples.append(mem)
        if mem > self.current_image.peak_memory_mb:
            self.current_image.peak_memory_mb = mem
    
    def set_image_metadata(self, **kwargs) -> None:
        """Set metadata on current image."""
        if not self.enabled or not self.current_image:
            return
        for key, value in kwargs.items():
            if hasattr(self.current_image, key):
                setattr(self.current_image, key, value)
    
    def _get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def export_json(self, path: Path) -> None:
        """Export batch metrics to JSON."""
        self.batch.to_json(path)
    
    def export_prometheus(self, path: Path) -> None:
        """Export batch metrics in Prometheus format."""
        path.write_text(self.batch.to_prometheus())


# --- Observability additions (additive; safe to ignore if unused) -------------

from typing import Mapping, Optional

try:
    from lux_depth_v2.observability.context import get_request_id as current_request_id  # re-export
except Exception:  # pragma: no cover
    def current_request_id() -> Optional[str]:
        return None


def observe_pipeline_timings(timing_s: Optional[Mapping[str, float]]) -> None:
    """
    Optional bridge: pipe timing breakdowns into Prometheus histograms.
    Call this with result['timing_s'] (see migration docs) when available.
    """
    if not timing_s:
        return
    try:
        from lux_depth_v2.observability.metrics import get_metrics
        get_metrics().observe_pipeline_timings(dict(timing_s))
    except Exception:
        # Fully additive: telemetry must never fail the pipeline
        return
