"""Performance Profiler for Lux Depth V2 Pipeline.

Features:
- Automatic stage timing with context managers
- I/O vs compute time separation
- Memory snapshots per stage
- Bottleneck detection and reporting
- Performance report generation
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .logging_utils import setup_logging


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    elapsed_time: float = 0.0
    
    # Time breakdown
    io_time: float = 0.0
    compute_time: float = 0.0
    
    # Memory snapshots
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    
    # GPU memory (if available)
    gpu_memory_start_mb: Optional[float] = None
    gpu_memory_end_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def finalize(self):
        """Finalize metrics after stage completion."""
        if self.end_time and self.start_time:
            self.elapsed_time = self.end_time - self.start_time
        
        if self.memory_end_mb and self.memory_start_mb:
            self.memory_delta_mb = self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'elapsed_time': round(self.elapsed_time, 3),
            'io_time': round(self.io_time, 3),
            'compute_time': round(self.compute_time, 3),
            'memory_delta_mb': round(self.memory_delta_mb, 1) if self.memory_delta_mb else None,
            'memory_peak_mb': round(self.memory_peak_mb, 1) if self.memory_peak_mb else None,
            'gpu_memory_peak_mb': round(self.gpu_memory_peak_mb, 1) if self.gpu_memory_peak_mb else None,
            'metadata': self.metadata,
        }


@dataclass
class PerformanceReport:
    """Complete performance report for a task."""
    task_id: str
    total_time: float = 0.0
    stages: List[StageMetrics] = field(default_factory=list)
    
    # Breakdown
    total_io_time: float = 0.0
    total_compute_time: float = 0.0
    
    # Bottlenecks (stages taking >30% of total time)
    bottlenecks: List[Tuple[str, float]] = field(default_factory=list)
    
    # Memory
    peak_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    
    def finalize(self):
        """Finalize report after all stages complete."""
        self.total_time = sum(s.elapsed_time for s in self.stages)
        self.total_io_time = sum(s.io_time for s in self.stages)
        self.total_compute_time = sum(s.compute_time for s in self.stages)
        
        # Identify bottlenecks (>30% of total time)
        if self.total_time > 0:
            threshold = 0.30 * self.total_time
            self.bottlenecks = [
                (s.name, s.elapsed_time)
                for s in self.stages
                if s.elapsed_time > threshold
            ]
        
        # Track peak memory
        memory_peaks = [s.memory_peak_mb for s in self.stages if s.memory_peak_mb]
        if memory_peaks:
            self.peak_memory_mb = max(memory_peaks)
        
        gpu_memory_peaks = [s.gpu_memory_peak_mb for s in self.stages if s.gpu_memory_peak_mb]
        if gpu_memory_peaks:
            self.peak_gpu_memory_mb = max(gpu_memory_peaks)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'task_id': self.task_id,
            'total_time': round(self.total_time, 3),
            'total_io_time': round(self.total_io_time, 3),
            'total_compute_time': round(self.total_compute_time, 3),
            'io_compute_ratio': round(self.total_io_time / max(self.total_compute_time, 0.001), 2),
            'stages': [s.to_dict() for s in self.stages],
            'bottlenecks': [
                {'stage': name, 'time': round(time_val, 3), 'percent': round(100 * time_val / max(self.total_time, 0.001), 1)}
                for name, time_val in self.bottlenecks
            ],
            'peak_memory_mb': round(self.peak_memory_mb, 1) if self.peak_memory_mb else None,
            'peak_gpu_memory_mb': round(self.peak_gpu_memory_mb, 1) if self.peak_gpu_memory_mb else None,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Performance Report: {self.task_id}",
            f"  Total Time: {self.total_time:.2f}s",
            f"  I/O Time: {self.total_io_time:.2f}s ({100 * self.total_io_time / max(self.total_time, 0.001):.1f}%)",
            f"  Compute Time: {self.total_compute_time:.2f}s ({100 * self.total_compute_time / max(self.total_time, 0.001):.1f}%)",
        ]
        
        if self.bottlenecks:
            lines.append(f"  Bottlenecks ({len(self.bottlenecks)}):")
            for name, time_val in self.bottlenecks:
                pct = 100 * time_val / max(self.total_time, 0.001)
                lines.append(f"    - {name}: {time_val:.2f}s ({pct:.1f}%)")
        
        if self.peak_memory_mb:
            lines.append(f"  Peak Memory: {self.peak_memory_mb:.1f} MB")
        
        if self.peak_gpu_memory_mb:
            lines.append(f"  Peak GPU Memory: {self.peak_gpu_memory_mb:.1f} MB")
        
        return "\n".join(lines)


class PerformanceProfiler:
    """Performance profiler for pipeline stages.
    
    Features:
    - Automatic timing with context managers
    - Memory tracking (CPU and GPU)
    - I/O vs compute time separation
    - Bottleneck detection
    
    Usage:
        profiler = PerformanceProfiler(task_id="image_001")
        
        with profiler.stage("depth_load", is_io=True):
            # Load depth map
            pass
        
        with profiler.stage("material_segmentation"):
            # Segment materials
            pass
        
        report = profiler.get_report()
        print(report.summary())
    """
    
    def __init__(self, task_id: str, device: str = "auto", logger=None):
        self.task_id = task_id
        self.device = device
        self.logger = logger or setup_logging("INFO")
        
        self.report = PerformanceReport(task_id=task_id)
        self.current_stage: Optional[StageMetrics] = None
        
        # Track peak memory globally
        self._global_peak_memory_mb = 0.0
        self._global_peak_gpu_memory_mb = 0.0
        
        self.logger.debug(f"Profiler initialized for task: {task_id}")
    
    @contextmanager
    def stage(self, name: str, is_io: bool = False, **metadata):
        """Context manager for timing a stage.
        
        Args:
            name: Stage name
            is_io: Whether this is primarily I/O (vs compute)
            **metadata: Additional metadata to track
            
        Example:
            with profiler.stage("depth_load", is_io=True):
                depth = load_depth_map(path)
        """
        # Start stage
        metrics = StageMetrics(
            name=name,
            start_time=time.time(),
            memory_start_mb=self._get_memory_usage(),
            gpu_memory_start_mb=self._get_gpu_memory_usage(),
            metadata=metadata,
        )
        
        self.current_stage = metrics
        self.logger.debug(f"Stage started: {name}")
        
        try:
            yield metrics
        finally:
            # End stage
            metrics.end_time = time.time()
            metrics.memory_end_mb = self._get_memory_usage()
            metrics.gpu_memory_end_mb = self._get_gpu_memory_usage()
            
            # Track peak memory during stage
            metrics.memory_peak_mb = max(
                metrics.memory_start_mb or 0,
                metrics.memory_end_mb or 0,
                self._global_peak_memory_mb
            )
            metrics.gpu_memory_peak_mb = max(
                metrics.gpu_memory_start_mb or 0,
                metrics.gpu_memory_end_mb or 0,
                self._global_peak_gpu_memory_mb
            )
            
            # Finalize metrics
            metrics.finalize()
            
            # Categorize time (I/O vs compute)
            if is_io:
                metrics.io_time = metrics.elapsed_time
            else:
                metrics.compute_time = metrics.elapsed_time
            
            # Add to report
            self.report.stages.append(metrics)
            self.current_stage = None
            
            self.logger.debug(
                f"Stage completed: {name} | "
                f"elapsed={metrics.elapsed_time:.3f}s "
                f"memory_delta={metrics.memory_delta_mb:.1f}MB"
            )
    
    @contextmanager
    def io_operation(self, operation: str):
        """Context manager for tracking I/O operations within a stage.
        
        Args:
            operation: Description of I/O operation (e.g., "load_depth", "write_output")
            
        Example:
            with profiler.stage("export"):
                with profiler.io_operation("write_tiff"):
                    write_image(path, image)
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            if self.current_stage:
                self.current_stage.io_time += elapsed
                self.current_stage.compute_time = max(
                    0, self.current_stage.compute_time - elapsed
                )
                
                if 'io_operations' not in self.current_stage.metadata:
                    self.current_stage.metadata['io_operations'] = []
                
                self.current_stage.metadata['io_operations'].append({
                    'operation': operation,
                    'time': round(elapsed, 3)
                })
    
    def get_report(self) -> PerformanceReport:
        """Get complete performance report.
        
        Returns:
            PerformanceReport with all metrics
        """
        self.report.finalize()
        return self.report
    
    def save_report(self, output_path: Path):
        """Save performance report to JSON file.
        
        Args:
            output_path: Path to save report JSON
        """
        import json
        
        report = self.get_report()
        report_dict = report.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Performance report saved: {output_path}")
    
    def log_summary(self):
        """Log performance summary."""
        report = self.get_report()
        self.logger.info("\n" + report.summary())
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self._global_peak_memory_mb = max(self._global_peak_memory_mb, memory_mb)
            return memory_mb
        except Exception:
            return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        try:
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self._global_peak_gpu_memory_mb = max(self._global_peak_gpu_memory_mb, memory_mb)
                    return memory_mb
            elif self.device == "mps":
                import torch
                if torch.backends.mps.is_available():
                    memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024
                    self._global_peak_gpu_memory_mb = max(self._global_peak_gpu_memory_mb, memory_mb)
                    return memory_mb
        except Exception:
            pass
        
        return None


def analyze_batch_performance(reports: List[PerformanceReport]) -> Dict:
    """Analyze performance across multiple tasks.
    
    Args:
        reports: List of PerformanceReport objects
        
    Returns:
        Batch performance analysis dictionary
    """
    if not reports:
        return {}
    
    # Aggregate metrics
    total_time = sum(r.total_time for r in reports)
    avg_time = total_time / len(reports)
    min_time = min(r.total_time for r in reports)
    max_time = max(r.total_time for r in reports)
    
    # Stage breakdown
    stage_times = {}
    for report in reports:
        for stage in report.stages:
            if stage.name not in stage_times:
                stage_times[stage.name] = []
            stage_times[stage.name].append(stage.elapsed_time)
    
    stage_stats = {
        name: {
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total_pct': 100 * sum(times) / max(total_time, 0.001),
        }
        for name, times in stage_times.items()
    }
    
    # Identify common bottlenecks
    bottleneck_counts = {}
    for report in reports:
        for name, _ in report.bottlenecks:
            bottleneck_counts[name] = bottleneck_counts.get(name, 0) + 1
    
    common_bottlenecks = [
        (name, count, 100 * count / len(reports))
        for name, count in sorted(bottleneck_counts.items(), key=lambda x: -x[1])
    ]
    
    return {
        'task_count': len(reports),
        'total_time': round(total_time, 2),
        'avg_time_per_task': round(avg_time, 2),
        'min_time': round(min_time, 2),
        'max_time': round(max_time, 2),
        'throughput_images_per_hour': round(3600 / avg_time, 1) if avg_time > 0 else 0,
        'stage_stats': {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in stage_stats.items()},
        'common_bottlenecks': [
            {'stage': name, 'count': count, 'percent': round(pct, 1)}
            for name, count, pct in common_bottlenecks
        ],
    }
