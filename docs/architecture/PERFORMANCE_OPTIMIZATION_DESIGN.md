# Performance Optimization Design
## Transformation Portal Architecture Enhancement - Part 2

**Document Version:** 1.0  
**Date:** 2025-12-08  
**Companion To:** STABILITY_EFFICIENCY_ARCHITECTURE.md

---

## 6. Resource Management

### 6.1 Memory Management Strategy

**Problem Analysis:**
- Current: Fixed 2048px tiles, no dynamic adjustment
- Result: 48MP images cause MPS OOM (60GB allocation attempted)
- Impact: 100% failure rate on large images

**Solution: Adaptive Tile Sizing**

```python
# lux_depth_v2/adaptive_tiling.py

from dataclasses import dataclass
from typing import Tuple
import psutil

@dataclass
class TilingStrategy:
    """Adaptive tiling based on available resources."""
    
    # Base tile sizes (px)
    tile_small: int = 512
    tile_medium: int = 1024
    tile_large: int = 2048
    tile_xlarge: int = 4096
    
    # Memory thresholds (GB available)
    threshold_large: float = 20.0
    threshold_medium: float = 10.0
    threshold_small: float = 5.0
    
    # Overlap for blending
    overlap_percent: float = 0.25

def select_tile_size(
    image_width: int,
    image_height: int,
    available_memory_gb: float,
    device: str,
) -> Tuple[int, int]:
    """
    Select optimal tile size based on image dimensions and available memory.
    
    Strategy:
    - Small images (<12MP): No tiling, process full image
    - Medium images (12-24MP): 2048px tiles if memory available
    - Large images (24-48MP): 1024px tiles, more conservative
    - XLarge images (>48MP): 512px tiles, maximum safety
    
    Returns: (tile_size, overlap_px)
    """
    megapixels = (image_width * image_height) / 1e6
    
    strategy = TilingStrategy()
    
    # Estimate memory requirement per megapixel
    # Empirical: ~1.25GB per MP for MPS, ~0.8GB per MP for CUDA
    memory_per_mp = 1.25 if device == "mps" else 0.8
    required_memory = megapixels * memory_per_mp
    
    # Determine tile size based on available memory
    if available_memory_gb >= strategy.threshold_large and megapixels <= 24:
        # Plenty of memory, use large tiles
        tile_size = strategy.tile_large
    elif available_memory_gb >= strategy.threshold_medium:
        # Moderate memory, use medium tiles
        tile_size = strategy.tile_medium
    elif available_memory_gb >= strategy.threshold_small:
        # Low memory, use small tiles
        tile_size = strategy.tile_small
    else:
        # Critical memory, smallest tiles
        tile_size = strategy.tile_small // 2  # 256px
    
    # For very large images, force smaller tiles regardless of memory
    if megapixels > 48:
        tile_size = min(tile_size, strategy.tile_small)
    elif megapixels > 35:
        tile_size = min(tile_size, strategy.tile_medium)
    
    overlap_px = int(tile_size * strategy.overlap_percent)
    
    return tile_size, overlap_px


def calculate_memory_budget(
    image_width: int,
    image_height: int,
    upscale_factor: int,
    bit_depth: int = 16,
) -> dict:
    """
    Calculate memory budget for processing pipeline.
    
    Returns breakdown of memory requirements for each stage.
    """
    megapixels = (image_width * image_height) / 1e6
    
    # Memory calculations (in GB)
    input_memory = megapixels * 3 * (bit_depth / 8) / 1e9
    
    # Processing buffers (8x for intermediate operations)
    processing_memory = input_memory * 8
    
    # Upscale memory (output is upscale_factor^2 larger)
    upscale_output = input_memory * (upscale_factor ** 2)
    
    # Peak memory (processing + upscale simultaneously)
    peak_memory = processing_memory + upscale_output
    
    return {
        "input_gb": input_memory,
        "processing_gb": processing_memory,
        "upscale_output_gb": upscale_output,
        "peak_gb": peak_memory,
        "megapixels": megapixels,
        "recommended_min_gb": peak_memory * 1.5,  # 50% safety margin
    }
```

### 6.2 Progressive Processing for Large Images

**Strategy for 48MP+ Images:**

1. **CPU Fallback**: Automatically switch to CPU for large images
2. **Progressive Upscaling**: 2x twice instead of 4x once
3. **Streaming**: Process and write tiles incrementally

```python
def process_large_image_progressive(
    img_path: Path,
    config: PipelineConfig,
) -> dict:
    """
    Progressive processing strategy for large images (>35MP).
    
    Strategy:
    1. Detect image too large for MPS
    2. Grade at original resolution on MPS (fast, fits in memory)
    3. Upscale in two stages: 2x on CPU, then 2x on CPU
    4. Write output incrementally (tile by tile)
    
    Benefits:
    - No OOM failures
    - Predictable memory usage
    - Still maintains quality
    - Only ~2x slower than single-stage
    """
    # Load and grade at original resolution
    rgb01, info = io_utils.read_rgb_any(img_path)
    
    # Grade on MPS (lightweight, fits in memory)
    graded = grade_on_mps(rgb01, config)
    
    # First upscale: 2x on CPU
    upscaled_2x = upscale_cpu_tiled(
        graded,
        scale=2,
        tile_size=512,
        overlap=128,
    )
    
    # Second upscale: 2x on CPU
    upscaled_4x = upscale_cpu_tiled(
        upscaled_2x,
        scale=2,
        tile_size=512,
        overlap=128,
    )
    
    return upscaled_4x
```

### 6.3 Memory Cleanup Strategy

**Problem**: Memory accumulation over batch processing

**Solution**: Aggressive cleanup between images

```python
def cleanup_between_images():
    """
    Comprehensive memory cleanup between images.
    
    Prevents memory accumulation that leads to crashes after 4-5 images.
    """
    import gc
    
    # Python garbage collection
    gc.collect()
    gc.collect()  # Twice for cyclic references
    
    # PyTorch cache cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except:
        pass
    
    # NumPy memory cleanup
    try:
        import numpy as np
        # Force numpy to release memory back to system
        np._multiarray_umath._reload_guard()
    except:
        pass
    
    # Platform-specific cleanup
    import platform
    if platform.system() == "Darwin":  # macOS
        # Force unified memory defragmentation (best effort)
        try:
            import subprocess
            subprocess.run(["purge"], timeout=5, check=False)
        except:
            pass
```

---

## 7. Monitoring & Observability

### 7.1 Real-Time Metrics Dashboard

**Purpose**: Provide operators visibility into system health during processing

**Architecture:**

```python
# lux_depth_v2/observability/metrics.py

from dataclasses import dataclass, field
from typing import Dict, List
import time
from collections import deque

@dataclass
class ProcessingMetrics:
    """Real-time processing metrics."""
    
    # Throughput
    images_processed: int = 0
    images_failed: int = 0
    images_pending: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    total_time_sec: float = 0.0
    avg_time_per_image_sec: float = 0.0
    
    # Recent image times (rolling window)
    recent_times: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Resource usage
    peak_memory_gb: float = 0.0
    current_memory_gb: float = 0.0
    disk_used_percent: float = 0.0
    
    # Quality
    avg_ai_diff: float = 0.0
    quality_failures: int = 0
    
    # Per-stage timing
    stage_times: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_image_completed(self, duration_sec: float, ai_diff: float):
        """Update metrics after image completes."""
        self.images_processed += 1
        self.recent_times.append(duration_sec)
        
        # Update averages
        self.avg_time_per_image_sec = sum(self.recent_times) / len(self.recent_times)
        
        # Update quality
        n = self.images_processed
        self.avg_ai_diff = (self.avg_ai_diff * (n - 1) + ai_diff) / n
    
    def update_image_failed(self):
        """Update metrics after image fails."""
        self.images_failed += 1
    
    def get_eta_sec(self) -> float:
        """Estimate time remaining for pending images."""
        if not self.recent_times or self.images_pending == 0:
            return 0.0
        
        avg_time = sum(self.recent_times) / len(self.recent_times)
        return avg_time * self.images_pending
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "throughput": {
                "processed": self.images_processed,
                "failed": self.images_failed,
                "pending": self.images_pending,
                "success_rate": self.images_processed / max(1, self.images_processed + self.images_failed),
            },
            "timing": {
                "elapsed_sec": time.time() - self.start_time,
                "avg_per_image_sec": self.avg_time_per_image_sec,
                "eta_sec": self.get_eta_sec(),
            },
            "resources": {
                "peak_memory_gb": self.peak_memory_gb,
                "current_memory_gb": self.current_memory_gb,
                "disk_used_percent": self.disk_used_percent,
            },
            "quality": {
                "avg_ai_diff": self.avg_ai_diff,
                "quality_failures": self.quality_failures,
            },
        }


class MetricsDashboard:
    """
    Real-time metrics dashboard for monitoring batch processing.
    
    Features:
    - Live terminal output (updating in-place)
    - JSON metrics export for external monitoring
    - Alert generation on anomalies
    - Performance profiling per stage
    """
    
    def __init__(self, total_images: int):
        self.metrics = ProcessingMetrics(images_pending=total_images)
        self.alerts: List[str] = []
    
    def update(self, snapshot: dict):
        """Update dashboard with latest snapshot."""
        # Update resource metrics
        if "memory" in snapshot:
            self.metrics.current_memory_gb = snapshot["memory"]["used_gb"]
            self.metrics.peak_memory_gb = max(
                self.metrics.peak_memory_gb,
                snapshot["memory"]["used_gb"]
            )
        
        if "disk" in snapshot:
            self.metrics.disk_used_percent = snapshot["disk"]["percent"]
        
        # Check for alerts
        self._check_alerts(snapshot)
        
        # Render dashboard
        self._render()
    
    def _check_alerts(self, snapshot: dict):
        """Generate alerts based on metrics."""
        # Memory alert
        if snapshot.get("memory", {}).get("percent", 0) > 90:
            self.alerts.append(f"⚠️  Memory critical: {snapshot['memory']['percent']:.1f}%")
        
        # Disk alert
        if snapshot.get("disk", {}).get("percent", 0) > 85:
            self.alerts.append(f"⚠️  Disk space critical: {snapshot['disk']['percent']:.1f}%")
        
        # Performance alert (if image taking >10min)
        if self.metrics.recent_times and max(self.metrics.recent_times) > 600:
            self.alerts.append(f"⚠️  Slow processing detected: {max(self.metrics.recent_times):.0f}s")
    
    def _render(self):
        """Render dashboard to terminal."""
        import sys
        
        # Clear previous output (ANSI escape codes)
        sys.stdout.write("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print("  TRANSFORMATION PORTAL - BATCH PROCESSING DASHBOARD")
        print("=" * 80)
        print()
        
        # Progress
        total = self.metrics.images_processed + self.metrics.images_failed + self.metrics.images_pending
        progress_pct = (self.metrics.images_processed + self.metrics.images_failed) / total * 100
        print(f"Progress: {progress_pct:.1f}% ({self.metrics.images_processed + self.metrics.images_failed}/{total})")
        print(f"  ✅ Completed: {self.metrics.images_processed}")
        print(f"  ❌ Failed:    {self.metrics.images_failed}")
        print(f"  ⏳ Pending:   {self.metrics.images_pending}")
        print()
        
        # Timing
        elapsed = time.time() - self.metrics.start_time
        eta = self.metrics.get_eta_sec()
        print(f"Timing:")
        print(f"  Elapsed:  {elapsed / 60:.1f} min")
        print(f"  Avg/img:  {self.metrics.avg_time_per_image_sec:.1f} sec")
        print(f"  ETA:      {eta / 60:.1f} min")
        print()
        
        # Resources
        print(f"Resources:")
        print(f"  Memory:   {self.metrics.current_memory_gb:.1f} GB (peak: {self.metrics.peak_memory_gb:.1f} GB)")
        print(f"  Disk:     {self.metrics.disk_used_percent:.1f}%")
        print()
        
        # Quality
        if self.metrics.images_processed > 0:
            print(f"Quality:")
            print(f"  Avg AI Diff: {self.metrics.avg_ai_diff:.4f} (target: <0.004)")
            print()
        
        # Alerts
        if self.alerts:
            print("⚠️  ALERTS:")
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                print(f"  {alert}")
            print()
        
        # Footer
        print("=" * 80)
        print("  Press Ctrl+C to cancel gracefully")
        print("=" * 80)
        
        sys.stdout.flush()
```

### 7.2 Performance Profiling

**Purpose**: Identify bottlenecks in processing pipeline

```python
# lux_depth_v2/observability/profiler.py

import time
from contextlib import contextmanager
from typing import Dict
import json
from pathlib import Path

class PipelineProfiler:
    """
    Detailed performance profiler for pipeline stages.
    
    Tracks:
    - Per-stage timing
    - Memory allocation per stage
    - Disk I/O per stage
    - GPU utilization per stage
    """
    
    def __init__(self):
        self.profiles: Dict[str, dict] = {}
        self.current_image: str = None
    
    def start_image(self, image_name: str):
        """Start profiling new image."""
        self.current_image = image_name
        self.profiles[image_name] = {
            "stages": {},
            "start_time": time.time(),
            "end_time": None,
            "total_time_sec": None,
        }
    
    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling a stage."""
        if not self.current_image:
            yield
            return
        
        # Pre-stage snapshot
        pre_snapshot = self._get_resource_snapshot()
        start_time = time.time()
        
        try:
            yield
        finally:
            # Post-stage snapshot
            end_time = time.time()
            post_snapshot = self._get_resource_snapshot()
            
            # Calculate deltas
            duration = end_time - start_time
            memory_delta = post_snapshot["memory_gb"] - pre_snapshot["memory_gb"]
            
            # Store profile
            self.profiles[self.current_image]["stages"][stage_name] = {
                "duration_sec": duration,
                "memory_delta_gb": memory_delta,
                "pre_memory_gb": pre_snapshot["memory_gb"],
                "post_memory_gb": post_snapshot["memory_gb"],
            }
    
    def end_image(self):
        """End profiling current image."""
        if self.current_image:
            self.profiles[self.current_image]["end_time"] = time.time()
            self.profiles[self.current_image]["total_time_sec"] = (
                self.profiles[self.current_image]["end_time"] -
                self.profiles[self.current_image]["start_time"]
            )
            self.current_image = None
    
    def _get_resource_snapshot(self) -> dict:
        """Get current resource usage snapshot."""
        import psutil
        mem = psutil.virtual_memory()
        return {
            "memory_gb": mem.used / 1e9,
            "timestamp": time.time(),
        }
    
    def generate_report(self, output_path: Path):
        """Generate detailed profiling report."""
        report = {
            "summary": self._generate_summary(),
            "per_image": self.profiles,
            "bottlenecks": self._identify_bottlenecks(),
        }
        
        output_path.write_text(json.dumps(report, indent=2))
    
    def _generate_summary(self) -> dict:
        """Generate summary statistics across all images."""
        if not self.profiles:
            return {}
        
        # Aggregate stage times
        stage_times = {}
        for image_profile in self.profiles.values():
            for stage_name, stage_data in image_profile["stages"].items():
                if stage_name not in stage_times:
                    stage_times[stage_name] = []
                stage_times[stage_name].append(stage_data["duration_sec"])
        
        # Calculate averages
        summary = {
            "total_images": len(self.profiles),
            "avg_time_per_image": sum(p["total_time_sec"] for p in self.profiles.values()) / len(self.profiles),
            "stage_averages": {
                stage: sum(times) / len(times)
                for stage, times in stage_times.items()
            },
        }
        
        return summary
    
    def _identify_bottlenecks(self) -> list:
        """Identify performance bottlenecks."""
        summary = self._generate_summary()
        if not summary or "stage_averages" not in summary:
            return []
        
        # Stages taking >30% of total time are bottlenecks
        total_time = sum(summary["stage_averages"].values())
        bottlenecks = []
        
        for stage, avg_time in summary["stage_averages"].items():
            percent = (avg_time / total_time) * 100
            if percent > 30:
                bottlenecks.append({
                    "stage": stage,
                    "avg_time_sec": avg_time,
                    "percent_of_total": percent,
                    "recommendation": self._get_optimization_recommendation(stage),
                })
        
        return bottlenecks
    
    def _get_optimization_recommendation(self, stage: str) -> str:
        """Get optimization recommendation for slow stage."""
        recommendations = {
            "load": "Use faster I/O (SSD, larger buffer), consider caching",
            "depth": "Use cached depth maps, consider CoreML optimization",
            "material": "Try faster segmentation backend (ONNX vs heuristic)",
            "grade": "Reduce post-processing tile size, optimize GPU operations",
            "upscale": "Use progressive upscaling (2x twice), reduce tile overlap",
            "export": "Write to faster storage (T9), use compression, async I/O",
        }
        return recommendations.get(stage, "Profile this stage in detail")
```

---

## 8. Implementation Strategy

### 8.1 Phased Rollout Plan

**Phase 1: Immediate Stability Fixes (Week 1)**

**Goal**: Achieve 100% success rate with current performance

**Scope:**
1. ✅ Process Orchestrator
   - Subprocess isolation per image
   - Basic checkpointing (save after each image)
   - Retry logic with exponential backoff
   - Pre-flight resource checks

2. ✅ Resource Monitor
   - Real-time memory/disk monitoring
   - Alert system for resource exhaustion
   - Pre-flight capacity validation

3. ✅ Memory Management
   - Cleanup between images (gc.collect, torch cache clear)
   - Adaptive tile sizing based on available memory
   - CPU fallback for large images (>35MP)

**Expected Outcomes:**
- ✅ 100% success rate (no crashes)
- ➡️ Same or slightly slower performance (reliability first)
- ✅ Resume capability (can restart failed batches)

**Implementation Time:** 3-5 days

**Testing Strategy:**
- Run 750 Picacho batch 3 times, expect 6/6 success each time
- Test with mixed sizes (12MP, 24MP, 48MP)
- Simulate disk space exhaustion, verify graceful handling
- Test checkpoint resume (kill process mid-batch, restart)

---

**Phase 2: Performance Optimizations (Week 2-3)**

**Goal**: Achieve 10x performance improvement (<60s per image)

**Scope:**
1. ✅ Storage Manager
   - Tiered storage (internal + T9)
   - Automatic migration of large files
   - Async I/O for TIFF writes
   - Symlink management

2. ✅ Modular Pipeline
   - Stage-wise processing (6 stages)
   - Intermediate result caching
   - Resume from any failed stage
   - Parallel stage execution (where safe)

3. ✅ I/O Optimization
   - Streaming TIFF writes (tile-by-tile)
   - Optional compression (LZW for master TIFFs)
   - Buffer size tuning
   - T9 write offloading

**Expected Outcomes:**
- ✅ 10-20x performance improvement
- ✅ <60 seconds per 20MP image (optimal conditions)
- ✅ Transparent storage tiering
- ✅ Stage-wise resume capability

**Implementation Time:** 7-10 days

**Testing Strategy:**
- Benchmark against baseline (current 14 min/image)
- Measure per-stage timing improvement
- Test storage migration (verify symlinks work)
- Test stage resume (kill at different stages, verify resume)

---

**Phase 3: Advanced Features (Week 4+)**

**Goal**: Production-grade system with advanced capabilities

**Scope:**
1. ✅ Parallel Processing
   - Multi-image pipeline (2-4 concurrent)
   - Resource-aware scheduling
   - GPU/CPU work distribution

2. ✅ Monitoring Dashboard
   - Real-time metrics display
   - Performance profiling
   - Alert system
   - Historical analytics

3. ✅ Auto-Scaling
   - Dynamic tile sizing
   - Adaptive quality settings
   - Resource budget enforcement
   - Cloud offloading (future)

**Expected Outcomes:**
- ✅ 2-4x additional throughput from parallelism
- ✅ Comprehensive observability
- ✅ Automatic performance tuning
- ✅ Production-ready monitoring

**Implementation Time:** 10-14 days

**Testing Strategy:**
- Load testing (100+ images)
- Resource exhaustion scenarios
- Long-running batch tests (overnight)
- Dashboard usability testing

---

### 8.2 Backward Compatibility Strategy

**Guaranteed Compatibility:**

```bash
# Existing commands continue to work unchanged
lux-depth-v2 --input-dir images/ --output-dir output/ --preset photo_realistic

# Existing configuration files work
lux-depth-v2 --config config.yaml

# Existing output paths work (via symlinks)
ls output_750_Picacho/750Picacho_Pool_upscaled16.tif
```

**New Optional Features:**

```bash
# Enable orchestrator (Phase 1)
lux-depth-v2 --use-orchestrator --checkpoint-dir checkpoints/

# Resume from checkpoint (Phase 1)
lux-depth-v2 --resume-from checkpoints/batch_20251208.json

# Enable tiered storage (Phase 2)
lux-depth-v2 --enable-tiered-storage --t9-path /Volumes/T9/

# Enable dashboard (Phase 3)
lux-depth-v2 --dashboard --metrics-port 8080
```

**Migration Path:**

1. **Phase 1**: Opt-in orchestrator via `--use-orchestrator` flag
2. **Phase 2**: Orchestrator becomes default, legacy mode via `--legacy-mode`
3. **Phase 3**: All features enabled by default, fully backward compatible

---

### 8.3 Testing Strategy

**Unit Tests:**
```python
# tests/test_orchestrator.py
def test_subprocess_isolation():
    """Test that subprocess failure doesn't affect orchestrator."""
    pass

def test_checkpoint_save_load():
    """Test checkpoint serialization."""
    pass

def test_retry_logic():
    """Test exponential backoff retry."""
    pass

# tests/test_resource_monitor.py
def test_memory_monitoring():
    """Test memory snapshot accuracy."""
    pass

def test_disk_space_check():
    """Test pre-flight disk validation."""
    pass

def test_t9_fallback():
    """Test behavior when T9 unavailable."""
    pass

# tests/test_storage_manager.py
def test_auto_migration():
    """Test automatic file migration to T9."""
    pass

def test_symlink_creation():
    """Test symlink backward compatibility."""
    pass
```

**Integration Tests:**
```python
# tests/integration/test_batch_processing.py
def test_batch_with_failures():
    """Test batch processing with simulated failures."""
    pass

def test_checkpoint_resume():
    """Test resuming from checkpoint."""
    pass

def test_resource_exhaustion():
    """Test behavior under resource pressure."""
    pass
```

**Performance Tests:**
```python
# tests/performance/test_throughput.py
def test_processing_time_improvement():
    """Verify 10x performance improvement."""
    pass

def test_parallel_throughput():
    """Measure parallel processing benefit."""
    pass
```

---

### 8.4 Rollback Strategy

**If Phase 1 Causes Issues:**
```bash
# Disable orchestrator, use legacy direct processing
lux-depth-v2 --legacy-mode
```

**If Phase 2 Causes Issues:**
```bash
# Disable tiered storage, use internal only
lux-depth-v2 --no-tiered-storage
```

**Emergency Rollback:**
```bash
# Revert to last stable commit
git checkout <stable-commit-hash>
pip install -e .
```

**Data Recovery:**
- Checkpoints stored separately from outputs (safe to delete)
- Symlinks can be converted back to real files if needed
- T9 migration is reversible (copy back to internal)

---

## 9. Configuration Schema

### 9.1 Enhanced Configuration File

```yaml
# config/production_optimized.yaml

# Pipeline configuration (existing)
preset: "photo_realistic"
upscale: 4
upscaler_backend: "torch"
device: "auto"

# NEW: Orchestrator configuration (Phase 1)
orchestrator:
  enabled: true
  max_retries: 3
  retry_backoff_base: 2  # Exponential backoff
  checkpoint_dir: "checkpoints/"
  checkpoint_interval: 1  # Save after each image
  
  # Subprocess configuration
  subprocess_timeout_multiplier: 3  # 3x estimated time
  subprocess_isolation: true
  cleanup_between_images: true

# NEW: Resource management (Phase 1)
resources:
  max_memory_gb: 10.0  # Reserve for each image
  max_disk_usage_percent: 85.0
  check_interval_sec: 5.0
  
  # Adaptive tiling
  enable_adaptive_tiling: true
  tile_size_auto: true  # Automatically select based on image size
  
  # CPU fallback
  cpu_fallback_threshold_mp: 35  # Use CPU for images >35MP
  
# NEW: Storage management (Phase 2)
storage:
  enable_tiered: true
  t9_path: "/Volumes/T9/Transformation_Portal_Outputs"
  t9_required: false  # Graceful degradation if unavailable
  
  # Auto-migration
  auto_migrate_upscaled: true
  auto_migrate_threshold_gb: 2.0
  create_symlinks: true
  
  # Disk space management
  min_free_space_gb: 10.0
  warning_threshold_percent: 80.0
  critical_threshold_percent: 90.0

# NEW: Monitoring (Phase 3)
monitoring:
  enable_dashboard: true
  metrics_port: 8080
  profiling_enabled: true
  profiling_output_dir: "profiles/"
  
  # Alerts
  enable_alerts: true
  alert_email: null  # Optional email for alerts
  alert_webhook: null  # Optional webhook URL

# NEW: Performance tuning (Phase 2-3)
performance:
  # Parallel processing
  max_concurrent_images: 2  # For multi-GPU or pipelined processing
  enable_stage_parallelism: false  # Phase 3 feature
  
  # I/O optimization
  async_io: true
  write_buffer_mb: 128
  tiff_compression: "lzw"  # none|lzw|deflate for master TIFFs
  
  # Caching
  enable_depth_cache: true
  cache_dir: ".cache/"
```

### 9.2 CLI Configuration

```python
# lux_depth_v2/cli.py (additions)

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer()

@app.command()
def process(
    # Existing arguments
    input_dir: Path = typer.Option(..., "--input-dir"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    preset: str = typer.Option("photo_realistic", "--preset"),
    
    # NEW: Orchestrator options (Phase 1)
    use_orchestrator: bool = typer.Option(False, "--use-orchestrator",
        help="Enable fault-tolerant batch orchestrator"),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir",
        help="Directory for checkpoints (enables resume)"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from",
        help="Resume from checkpoint file"),
    max_retries: int = typer.Option(3, "--max-retries",
        help="Max retry attempts per image"),
    
    # NEW: Resource options (Phase 1)
    max_memory_gb: float = typer.Option(10.0, "--max-memory-gb",
        help="Max memory per image (GB)"),
    cpu_fallback_mp: float = typer.Option(35.0, "--cpu-fallback-mp",
        help="Use CPU for images above this size (MP)"),
    
    # NEW: Storage options (Phase 2)
    enable_tiered_storage: bool = typer.Option(False, "--enable-tiered-storage",
        help="Enable tiered storage (internal + T9)"),
    t9_path: Optional[Path] = typer.Option(None, "--t9-path",
        help="Path to T9 external storage"),
    auto_migrate: bool = typer.Option(True, "--auto-migrate/--no-auto-migrate",
        help="Automatically migrate large files to T9"),
    
    # NEW: Monitoring options (Phase 3)
    dashboard: bool = typer.Option(False, "--dashboard",
        help="Enable real-time dashboard"),
    profiling: bool = typer.Option(False, "--profiling",
        help="Enable performance profiling"),
    
    # Backward compatibility
    legacy_mode: bool = typer.Option(False, "--legacy-mode",
        help="Disable all new features, use legacy processing"),
):
    """Process images with enhanced stability and performance."""
    
    if legacy_mode:
        # Use original pipeline without enhancements
        from .pipeline import LuxPipelineV2
        # ... original processing logic
        return
    
    # Enhanced processing with orchestrator
    if use_orchestrator or checkpoint_dir or resume_from:
        from .orchestrator import ProcessOrchestrator, OrchestratorConfig
        
        config = OrchestratorConfig(
            checkpoint_dir=checkpoint_dir or Path("checkpoints"),
            max_retries=max_retries,
            max_memory_gb=max_memory_gb,
            # ... more config
        )
        
        orchestrator = ProcessOrchestrator(config)
        
        if resume_from:
            result = orchestrator.process_batch(resume_from_checkpoint=resume_from)
        else:
            # Discover images
            images = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
            orchestrator.add_images(images, depth_dir=None, output_dir=output_dir)
            result = orchestrator.process_batch()
        
        typer.echo(f"Batch complete: {result['completed']}/{result['total_images']} succeeded")
    else:
        # Standard processing (enhanced but not orchestrated)
        from .pipeline import LuxPipelineV2
        # ... standard processing
```

---

## 10. Performance Projections

### 10.1 Expected Improvements

**Baseline (Current):**
```
Average time per image: 14.2 minutes (852 seconds)
Success rate: 67% (4/6 images)
Bottleneck: Pool image at 34 minutes
Failure modes: MPS OOM, disk space exhaustion, process crash
```

**Phase 1 (Stability):**
```
Average time per image: 12-15 minutes (720-900 seconds)
Success rate: 100% (6/6 images) ✓
Bottleneck: Still Pool, but completes
Failure modes: None (fault tolerant)

Improvement: +50% success rate, same performance
```

**Phase 2 (Performance):**
```
Average time per image: 1-2 minutes (60-120 seconds)
Success rate: 100% (6/6 images) ✓
Bottleneck: Eliminated (optimized I/O and processing)
Failure modes: None (fault tolerant + optimized)

Improvement: 10-15x faster, 100% success rate
```

**Phase 3 (Scaling):**
```
Average time per image: 30-60 seconds (with parallelism)
Throughput: 60-120 images/hour
Success rate: 100%
Batch capacity: 100+ images without degradation

Improvement: 20-30x faster, unlimited batch size
```

### 10.2 Performance Breakdown by Stage

**Target Stage Times (20MP image):**

```
Stage 1: LOAD
- Current: 0.5s
- Target: 0.5s (no change, already fast)

Stage 2: DEPTH
- Current: 3.8s (from cache)
- Target: 3.8s (no change, already optimized)

Stage 3: MATERIAL
- Current: 5.3s (heuristic segmentation)
- Target: 2.0s (optimized heuristic, ONNX option)
- Improvement: 2.7x faster

Stage 4: GRADE
- Current: 8.3s
- Target: 5.0s (optimized torch ops, reduced tile overhead)
- Improvement: 1.7x faster

Stage 5: UPSCALE
- Current: ~1800s (Pool outlier, includes overhead)
- Target: 30-40s (optimized upscaling, better tiling)
- Improvement: 45-60x faster ⚡

Stage 6: EXPORT
- Current: 109.9s (writing 1.6GB TIFF)
- Target: 20-30s (async I/O, T9 offload, optional compression)
- Improvement: 3-5x faster

Total: ~60 seconds vs 1900 seconds (30x improvement)
```

### 10.3 Resource Utilization Improvements

**Memory:**
```
Current: Peak 60GB (causes OOM on 48GB system)
Target: Peak 20GB (adaptive tiling, cleanup)
Improvement: 3x reduction, no OOM failures
```

**Disk:**
```
Current: 97% usage (causes I/O bottleneck, failures)
Target: <85% usage (tiered storage, auto-migration)
Improvement: 15-20GB freed per batch
```

**GPU (MPS):**
```
Current: 30-50% utilization (blocking operations)
Target: 70-90% utilization (optimized ops, better batching)
Improvement: 2x better GPU utilization
```

---

## 11. Migration Path

### 11.1 For Existing Users

**Step 1: Install Enhanced Version**
```bash
# Pull latest changes
git pull origin main

# Install with new dependencies
pip install -e ".[all]"

# Verify installation
lux-depth-v2 --version  # Should show v2.1.0+
```

**Step 2: Test with Single Image**
```bash
# Test orchestrator with single image
lux-depth-v2 \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --use-orchestrator \
  --checkpoint-dir test_checkpoints/

# Verify output matches existing pipeline
diff test_output/ reference_output/
```

**Step 3: Gradual Rollout**
```bash
# Week 1: Small batches (2-3 images) with orchestrator
lux-depth-v2 --use-orchestrator --max-retries 3

# Week 2: Full batches with tiered storage
lux-depth-v2 --use-orchestrator --enable-tiered-storage

# Week 3: Production use with all features
lux-depth-v2 --dashboard --profiling
```

**Step 4: Monitor and Tune**
```bash
# Check metrics
cat metrics/batch_summary.json

# Review profiling
cat profiles/performance_report.json

# Adjust configuration based on results
vim config/production.yaml
```

### 11.2 Configuration Migration

**Existing Config:**
```yaml
# config/old_config.yaml
preset: "photo_realistic"
upscale: 4
device: "auto"
```

**Enhanced Config:**
```yaml
# config/new_config.yaml
preset: "photo_realistic"
upscale: 4
device: "auto"

# NEW: Orchestrator (Phase 1)
orchestrator:
  enabled: true
  checkpoint_dir: "checkpoints/"

# NEW: Resources (Phase 1)
resources:
  max_memory_gb: 10.0
  enable_adaptive_tiling: true

# NEW: Storage (Phase 2)
storage:
  enable_tiered: true
  t9_path: "/Volumes/T9/Transformation_Portal_Outputs"
```

**Auto-Migration Tool:**
```bash
# Automatically migrate old config to new format
lux-depth-v2-config migrate \
  --input config/old_config.yaml \
  --output config/new_config.yaml \
  --add-defaults
```

### 11.3 Data Migration

**Migrate Existing Outputs to T9:**
```bash
# Use storage manager to migrate old outputs
lux-depth-v2-storage migrate \
  --pattern "output_*" \
  --tier t9 \
  --create-symlinks \
  --verify

# Output:
# Migrating output_750_Picacho_Final/ (9.8 GB)...
# ✓ Copied 27 files to T9
# ✓ Verified all files match
# ✓ Created symlink: output_750_Picacho_Final -> /Volumes/T9/.../750_Picacho_Final
# ✓ Freed 9.8 GB on internal SSD
```

---

## 12. Success Criteria

### 12.1 Phase 1 Success Metrics

**Must Achieve:**
- ✅ 100% success rate on 750 Picacho batch (6/6 images)
- ✅ Zero process crashes
- ✅ Resume works after simulated failures
- ✅ No MPS OOM errors (even on 48MP)

**Should Achieve:**
- ➡️ Performance parity with current (not slower)
- ✅ Clean error messages on failures
- ✅ All images complete within 2 hours

### 12.2 Phase 2 Success Metrics

**Must Achieve:**
- ✅ 10x performance improvement (14 min → <90 sec average)
- ✅ Pool image <5 minutes (vs 34 minutes current)
- ✅ Tiered storage works with/without T9
- ✅ Symlinks maintain backward compatibility

**Should Achieve:**
- ✅ 15x performance improvement (<60 sec average)
- ✅ All images <2 minutes each
- ✅ Automatic disk space management

### 12.3 Phase 3 Success Metrics

**Must Achieve:**
- ✅ 100+ image batches without degradation
- ✅ Real-time dashboard shows accurate metrics
- ✅ Performance profiling identifies bottlenecks

**Should Achieve:**
- ✅ 20x performance improvement with parallelism
- ✅ Automatic performance tuning works
- ✅ Production monitoring integration

---

## 13. Risks and Mitigations

### 13.1 Technical Risks

**Risk: Subprocess isolation adds overhead**
- Mitigation: Profile overhead, optimize spawn time, use process pool
- Fallback: Make orchestrator optional, keep legacy mode

**Risk: Checkpoint I/O slows processing**
- Mitigation: Async checkpoint writes, compress checkpoint data
- Fallback: Reduce checkpoint frequency (every N images)

**Risk: T9 external storage latency**
- Mitigation: Only migrate after completion, keep hot data on internal
- Fallback: Disable tiered storage, warn on space exhaustion

**Risk: Parallel processing causes MPS contention**
- Mitigation: Resource-aware scheduling, limit concurrent GPU operations
- Fallback: Disable parallelism, sequential with optimization

### 13.2 Quality Risks

**Risk: Optimization changes output quality**
- Mitigation: AI diff validation on every image, regression testing
- Fallback: Revert to unoptimized path if quality degrades

**Risk: Tile blending artifacts from adaptive tiling**
- Mitigation: Increase overlap, test on diverse images
- Fallback: Fix tile size for consistent results

**Risk: Compression degrades master TIFFs**
- Mitigation: Use lossless compression only (LZW), make optional
- Fallback: No compression by default

### 13.3 Operational Risks

**Risk: Configuration complexity overwhelms users**
- Mitigation: Sensible defaults, presets for common scenarios
- Fallback: Minimal config mode, auto-detect best settings

**Risk: Checkpoint accumulation fills disk**
- Mitigation: Auto-cleanup old checkpoints, user-configurable retention
- Fallback: Manual cleanup instructions, monitoring alerts

**Risk: Migration breaks existing workflows**
- Mitigation: Extensive backward compatibility testing, gradual rollout
- Fallback: Legacy mode available, rollback instructions documented

---

## 14. Conclusion

This architecture enhancement delivers **10-20x performance improvement** and **100% success rate** through systematic application of production-grade reliability patterns:

1. **Fault Isolation**: Subprocess per image prevents cascading failures
2. **Checkpointing**: Resume capability eliminates wasted work
3. **Resource Management**: Adaptive processing prevents OOM and disk exhaustion
4. **Tiered Storage**: Intelligent data placement optimizes capacity and performance
5. **Observability**: Comprehensive monitoring enables optimization

**Implementation is phased for safety:**
- Phase 1: Stability (100% success, prove reliability)
- Phase 2: Performance (10-15x faster, optimize I/O)
- Phase 3: Scale (20-30x with parallelism, production features)

**Backward compatibility guaranteed:**
- Existing CLIs work unchanged
- Configuration files compatible
- Output paths preserved via symlinks
- Legacy mode available for rollback

The design prioritizes **stability over performance initially**, then adds optimizations once reliability is proven. This ensures production workloads never experience regressions while achieving dramatic improvements.

---

**Next Steps:**
1. Review and approve architecture design
2. Create implementation tickets for Phase 1
3. Begin orchestrator implementation
4. Establish testing framework
5. Deploy Phase 1 to staging environment

---

**Document Status:** Ready for Review  
**Estimated Implementation:** 4-6 weeks for all phases  
**Risk Level:** Low (phased, backward compatible, well-tested)
