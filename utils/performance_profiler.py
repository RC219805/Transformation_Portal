#!/usr/bin/env python3
"""
Enhanced Performance Profiler
==============================
Comprehensive performance monitoring for image processing pipelines.

Features:
- GPU utilization tracking (MPS/CUDA)
- Memory usage with peak detection
- Per-stage timing breakdowns
- Automatic bottleneck identification
- Performance reports with optimization suggestions
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import time
import json

import psutil


@dataclass
class StageMetrics:
    """Performance metrics for a processing stage."""
    name: str
    duration: float  # seconds
    memory_start: float  # MB
    memory_peak: float  # MB
    memory_end: float  # MB
    gpu_utilization: Optional[float] = None  # 0-100%
    throughput: Optional[float] = None  # items/second
    items_processed: int = 0


@dataclass
class SystemSnapshot:
    """System resource snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    available_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class PerformanceReport:
    """Complete performance profiling report."""
    session_id: str
    start_time: str
    end_time: str
    total_duration: float
    stages: List[StageMetrics]
    system_snapshots: List[SystemSnapshot]
    bottlenecks: List[str]
    optimization_suggestions: List[str]
    summary: Dict[str, Any]


class GPUMonitor:
    """Monitor GPU utilization and memory."""
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.available = self.backend is not None
    
    def _detect_backend(self) -> Optional[str]:
        """Detect available GPU backend."""
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                return 'cuda'
            elif (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps')
                  and torch.backends.mps.is_available()):
                return 'mps'
        except ImportError:
            # torch is not installed; GPU monitoring will be disabled.
            pass
        except Exception:
            # Catch any other exceptions during GPU detection.
            pass
        return None
    
    def get_memory_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if not self.available:
            return None
        
        try:
            import torch
            
            if self.backend == 'cuda':
                return torch.cuda.memory_allocated() / 1024**2
            elif self.backend == 'mps':
                return torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            # Intentionally ignore all exceptions: GPU memory monitoring is optional.
            pass
        
        return None
    
    def get_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.available:
            return None
        
        try:
            if self.backend == 'cuda':
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
        except Exception:
            # Intentionally ignore all exceptions: GPU utilization monitoring is optional.
            pass
        
        return None
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if not self.available:
            return
        
        try:
            import torch
            
            if self.backend == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            elif self.backend == 'mps':
                # MPS doesn't have peak stats reset
                pass
        except Exception:
            # Intentionally ignore all exceptions: GPU peak memory reset is optional.
            pass
    
    def get_peak_memory_mb(self) -> Optional[float]:
        """Get peak GPU memory usage in MB."""
        if not self.available:
            return None
        
        try:
            import torch
            
            if self.backend == 'cuda':
                return torch.cuda.max_memory_allocated() / 1024**2
            elif self.backend == 'mps':
                return torch.mps.driver_allocated_memory() / 1024**2
        except Exception:
            # Intentionally ignore all exceptions: GPU peak memory monitoring is optional.
            pass
        
        return None


class PerformanceProfiler:
    """
    Comprehensive performance profiler for image processing pipelines.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize profiler.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"profile_{int(time.time())}"
        self.start_time = time.time()
        self.stages: List[StageMetrics] = []
        self.snapshots: List[SystemSnapshot] = []
        self.current_stage: Optional[str] = None
        self.stage_start: Optional[float] = None
        self.stage_memory_start: Optional[float] = None
        self.stage_memory_peak: float = 0.0
        
        self.gpu_monitor = GPUMonitor()
        self.process = psutil.Process()
        
        # Baseline snapshot
        self._take_snapshot()
    
    def _take_snapshot(self) -> SystemSnapshot:
        """Take a system resource snapshot."""
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024**2
        
        vm = psutil.virtual_memory()
        
        snapshot = SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=self.process.cpu_percent(),
            memory_mb=mem_mb,
            memory_percent=vm.percent,
            available_memory_mb=vm.available / 1024**2,
            gpu_memory_mb=self.gpu_monitor.get_memory_mb(),
            gpu_utilization=self.gpu_monitor.get_utilization()
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    @contextmanager
    def stage(self, name: str, items: int = 0):
        """
        Context manager for profiling a processing stage.
        
        Args:
            name: Stage name
            items: Number of items processed (for throughput calculation)
        
        Example:
            with profiler.stage('depth_estimation', items=10):
                # Process 10 images
                pass
        """
        # Start stage
        self.current_stage = name
        self.stage_start = time.time()
        self.stage_memory_start = self.process.memory_info().rss / 1024**2
        self.stage_memory_peak = self.stage_memory_start
        self.gpu_monitor.reset_peak_memory()
        
        
        try:
            yield self
        finally:
            # End stage
            duration = time.time() - self.stage_start
            self._take_snapshot()
            
            memory_end = self.process.memory_info().rss / 1024**2
            memory_peak = max(self.stage_memory_peak, memory_end)
            
            gpu_util = self.gpu_monitor.get_utilization()
            throughput = items / duration if items > 0 and duration > 0 else None
            
            metrics = StageMetrics(
                name=name,
                duration=duration,
                memory_start=self.stage_memory_start,
                memory_peak=memory_peak,
                memory_end=memory_end,
                gpu_utilization=gpu_util,
                throughput=throughput,
                items_processed=items
            )
            
            self.stages.append(metrics)
            self.current_stage = None
    
    def update_peak_memory(self):
        """Update peak memory for current stage."""
        if self.current_stage:
            current_mem = self.process.memory_info().rss / 1024**2
            self.stage_memory_peak = max(self.stage_memory_peak, current_mem)
    
    def generate_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report.
        
        Returns:
            PerformanceReport with analysis and suggestions
        """
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        # Generate optimization suggestions
        suggestions = self._generate_suggestions()
        
        # Compute summary statistics
        summary = self._compute_summary(total_duration)
        
        return PerformanceReport(
            session_id=self.session_id,
            start_time=datetime.fromtimestamp(self.start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_duration=total_duration,
            stages=self.stages,
            system_snapshots=self.snapshots,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            summary=summary
        )
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if not self.stages:
            return bottlenecks
        
        # Find slowest stage
        slowest = max(self.stages, key=lambda s: s.duration)
        total_time = sum(s.duration for s in self.stages)
        
        if slowest.duration > 0.3 * total_time:
            bottlenecks.append(
                f"Stage '{slowest.name}' is the primary bottleneck "
                f"({slowest.duration:.2f}s, {100*slowest.duration/total_time:.1f}% of total time)"
            )
        
        # Find memory-intensive stages
        for stage in self.stages:
            mem_increase = stage.memory_peak - stage.memory_start
            if mem_increase > 1000:  # > 1GB
                bottlenecks.append(
                    f"Stage '{stage.name}' uses significant memory "
                    f"(+{mem_increase:.0f}MB peak increase)"
                )
        
        # Check for low GPU utilization
        if self.gpu_monitor.available:
            gpu_stages = [s for s in self.stages if s.gpu_utilization is not None]
            if gpu_stages:
                avg_gpu_util = sum(s.gpu_utilization for s in gpu_stages) / len(gpu_stages)
                if avg_gpu_util < 50:
                    bottlenecks.append(
                        f"Low GPU utilization detected (avg: {avg_gpu_util:.1f}%) - "
                        "consider batch processing or model optimization"
                    )
        
        # Check for slow throughput
        for stage in self.stages:
            if stage.throughput is not None and stage.throughput < 1.0:
                bottlenecks.append(
                    f"Stage '{stage.name}' has low throughput "
                    f"({stage.throughput:.2f} items/sec) - consider parallelization"
                )
        
        return bottlenecks
    
    def _generate_suggestions(self) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Memory optimization
        if self.snapshots:
            peak_mem = max(s.memory_mb for s in self.snapshots)
            if peak_mem > 8000:  # > 8GB
                suggestions.append(
                    "High memory usage detected. Consider:\n"
                    "  • Processing images in smaller batches\n"
                    "  • Using lower-resolution intermediate representations\n"
                    "  • Enabling incremental processing with caching"
                )
        
        # GPU optimization
        if self.gpu_monitor.available:
            gpu_stages = [s for s in self.stages if s.gpu_utilization is not None]
            if gpu_stages:
                avg_util = sum(s.gpu_utilization for s in gpu_stages) / len(gpu_stages)
                if avg_util < 70:
                    suggestions.append(
                        "GPU underutilized. Consider:\n"
                        "  • Increasing batch size for model inference\n"
                        "  • Using mixed precision (FP16) for faster processing\n"
                        "  • Pipelining CPU and GPU operations"
                    )
        
        # Parallelization opportunities
        sequential_stages = [s for s in self.stages if s.items_processed > 1]
        if sequential_stages:
            suggestions.append(
                "Parallelization opportunities detected:\n"
                "  • Use multiprocessing for independent image operations\n"
                "  • Implement async I/O for loading/saving\n"
                "  • Consider multi-GPU processing for large batches"
            )
        
        # Caching suggestions
        if len(self.stages) > 5:
            suggestions.append(
                "Complex pipeline detected. Consider:\n"
                "  • Caching intermediate results (depth maps, material masks)\n"
                "  • Implementing incremental processing\n"
                "  • Using LRU cache for repeated operations"
            )
        
        return suggestions
    
    def _compute_summary(self, total_duration: float) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {
            'total_stages': len(self.stages),
            'total_duration_seconds': total_duration,
            'total_items_processed': sum(s.items_processed for s in self.stages),
        }
        
        if self.stages:
            summary['slowest_stage'] = max(self.stages, key=lambda s: s.duration).name
            summary['fastest_stage'] = min(self.stages, key=lambda s: s.duration).name
            
            # Average throughput
            throughputs = [s.throughput for s in self.stages if s.throughput is not None]
            if throughputs:
                summary['avg_throughput_items_per_sec'] = sum(throughputs) / len(throughputs)
        
        if self.snapshots:
            summary['peak_memory_mb'] = max(s.memory_mb for s in self.snapshots)
            summary['avg_cpu_percent'] = sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)
            
            if self.gpu_monitor.available:
                gpu_mems = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
                if gpu_mems:
                    summary['peak_gpu_memory_mb'] = max(gpu_mems)
        
        return summary
    
    def print_report(self, report: PerformanceReport):
        """Print formatted performance report."""
        print(f"\n{'='*70}")
        print(f"Performance Profile Report - {report.session_id}")
        print(f"{'='*70}\n")
        
        print(f"Duration: {report.total_duration:.2f}s")
        print(f"Stages: {report.summary['total_stages']}")
        print(f"Items Processed: {report.summary.get('total_items_processed', 0)}")
        
        print(f"\n{'Stage Breakdown':-^70}")
        print(f"{'Stage':<30} {'Time':<10} {'Memory':<15} {'Throughput':<15}")
        print(f"{'-'*70}")
        
        for stage in report.stages:
            mem_str = f"{stage.memory_peak:.0f}MB"
            throughput_str = f"{stage.throughput:.2f}/s" if stage.throughput else "N/A"
            
            print(
                f"{stage.name:<30} "
                f"{stage.duration:>8.2f}s  "
                f"{mem_str:<15} "
                f"{throughput_str:<15}"
            )
        
        if report.bottlenecks:
            print(f"\n{'Bottlenecks Identified':-^70}")
            for i, bottleneck in enumerate(report.bottlenecks, 1):
                print(f"\n{i}. {bottleneck}")
        
        if report.optimization_suggestions:
            print(f"\n{'Optimization Suggestions':-^70}")
            for i, suggestion in enumerate(report.optimization_suggestions, 1):
                print(f"\n{i}. {suggestion}")
        
        print(f"\n{'System Resources':-^70}")
        print(f"Peak Memory: {report.summary['peak_memory_mb']:.0f}MB")
        print(f"Avg CPU: {report.summary['avg_cpu_percent']:.1f}%")
        
        if 'peak_gpu_memory_mb' in report.summary:
            print(f"Peak GPU Memory: {report.summary['peak_gpu_memory_mb']:.0f}MB")
        
        print(f"\n{'='*70}\n")
    
    def save_report(self, report: PerformanceReport, output_path: Path):
        """
        Save performance report to JSON file.
        
        Args:
            report: Performance report to save
            output_path: Path to output JSON file
        """
        # Convert to JSON-serializable format
        data = {
            'session_id': report.session_id,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'total_duration': report.total_duration,
            'stages': [
                {
                    'name': s.name,
                    'duration': s.duration,
                    'memory_start_mb': s.memory_start,
                    'memory_peak_mb': s.memory_peak,
                    'memory_end_mb': s.memory_end,
                    'gpu_utilization': s.gpu_utilization,
                    'throughput': s.throughput,
                    'items_processed': s.items_processed,
                }
                for s in report.stages
            ],
            'bottlenecks': report.bottlenecks,
            'optimization_suggestions': report.optimization_suggestions,
            'summary': report.summary,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Performance report saved: {output_path}")


def profile_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Profile a function call and return result + metrics.
    
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        (result, StageMetrics)
    """
    profiler = PerformanceProfiler()
    
    with profiler.stage(func.__name__):
        result = func(*args, **kwargs)
    
    report = profiler.generate_report()
    metrics = report.stages[0] if report.stages else None
    
    return result, metrics


# Example usage
if __name__ == '__main__':
    import numpy as np
    
    # Example: Profile image processing pipeline
    profiler = PerformanceProfiler(session_id="example_pipeline")
    
    # Stage 1: Image loading
    with profiler.stage('image_loading', items=5):
        images = []
        for i in range(5):
            img = np.random.rand(1000, 1000, 3)
            images.append(img)
            time.sleep(0.1)
            profiler.update_peak_memory()
    
    # Stage 2: Processing
    with profiler.stage('processing', items=5):
        processed = []
        for img in images:
            # Simulate processing
            result = img * 2.0
            processed.append(result)
            time.sleep(0.2)
            profiler.update_peak_memory()
    
    # Stage 3: Saving
    with profiler.stage('saving', items=5):
        for i, img in enumerate(processed):
            # Simulate saving
            time.sleep(0.05)
    
    # Generate and print report
    report = profiler.generate_report()
    profiler.print_report(report)
    
    # Save report
    profiler.save_report(report, Path('performance_report.json'))
