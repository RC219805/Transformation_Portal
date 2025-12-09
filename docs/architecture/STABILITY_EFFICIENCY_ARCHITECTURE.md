# Transformation Portal: Stability & Efficiency Architecture
## System Enhancement Design Document

**Document Version:** 1.0  
**Date:** 2025-12-08  
**Author:** Transformation Portal Architect  
**Status:** Design Specification

---

## Executive Summary

This document defines a comprehensive architectural enhancement to the Transformation Portal system, addressing critical stability and performance issues identified during the 750 Picacho production runs. The proposed architecture achieves **10-20x performance improvement** (from 14 min/image to <60s/image) and **100% success rate** through systematic fault tolerance, resource management, and intelligent orchestration.

### Problem Statement

**Current State:**
- 67% success rate (4/6 images completed)
- 2.2x slower than expected (57 min vs 25 min)
- Pool bottleneck: 34 minutes for single 20MP image (17x slower)
- Process crashes during batch operations (PrimaryBathroom)
- MPS memory overflow on large images (48MP)
- Disk I/O bottlenecks at 97% disk usage

**Root Causes Identified:**
1. **No fault tolerance**: Single failure terminates entire batch
2. **Sequential processing**: One image blocks all others
3. **Resource contention**: MPS memory pressure, disk space exhaustion
4. **I/O bottlenecks**: 90% of time spent writing large TIFF files
5. **Memory leaks**: Accumulated memory pressure over long runs
6. **No checkpointing**: Cannot resume from failures

### Proposed Solution Overview

**Target Outcomes:**
- ✅ **100% success rate**: Fault-tolerant with automatic recovery
- ✅ **30-60s per image**: 10-20x performance improvement
- ✅ **Scalable batches**: Handle 100+ images without degradation
- ✅ **Resume capability**: Checkpoint and resume from failures
- ✅ **Resource-aware**: Intelligent scheduling based on system capacity
- ✅ **Quality maintained**: AI diff < 0.004 (current excellent level)

---

## Table of Contents

1. [Root Cause Analysis](#1-root-cause-analysis)
2. [Architecture Principles](#2-architecture-principles)
3. [Component Design](#3-component-design)
4. [Storage Architecture](#4-storage-architecture)
5. [Pipeline Redesign](#5-pipeline-redesign)
6. [Resource Management](#6-resource-management)
7. [Monitoring & Observability](#7-monitoring--observability)
8. [Implementation Strategy](#8-implementation-strategy)
9. [Configuration Schema](#9-configuration-schema)
10. [Performance Projections](#10-performance-projections)
11. [Migration Path](#11-migration-path)

---

## 1. Root Cause Analysis

### 1.1 Pool Image Bottleneck (34 Minutes)

**Observed Symptoms:**
- 2015.5 seconds (33.6 minutes) for 20.25MP image
- 17x slower than expected (2 minutes)
- 109.9s writing upscaled TIFF (reasonable)
- ~1900s in processing stage (bottleneck)

**Analysis:**

```python
# Timing breakdown for Pool image
Total:     2015.5s (100%)
├─ Processing: 1905.6s (94.5%)  ← BOTTLENECK
└─ I/O Write:   109.9s (5.5%)   ✓ Normal
```

**Root Causes:**
1. **Complex water scene**: Material segmentation struggles with reflections
2. **Heuristic segmentation overhead**: Per-pixel classification on 20MP
3. **Tile processing overhead**: 2048px tiles with blending on MPS
4. **Memory pressure**: Repeated tensor allocation/deallocation
5. **Synchronous processing**: Blocking while waiting for MPS operations

**Evidence from Logs:**
```python
# Pipeline processing sequence (from production_run_20251208_112751.log)
LuxDepthV2:  17%|█▋        | 1/6 [01:56<09:41, 116.38s/img]  # Aerial: 116s ✓
LuxDepthV2:  33%|███▎      | 2/6 [02:50<05:17, 79.49s/img]   # GreatRoom: 54s ✓
LuxDepthV2:  50%|█████     | 3/6 [07:55<09:08, 182.69s/img]  # Kitchen: 305s (degrading)
LuxDepthV2:  67%|██████▋   | 4/6 [41:31<30:12, 906.26s/img]  # Pool: 2015s ⚠️
LuxDepthV2:  83%|████████▎ | 5/6 [59:40<16:12, 972.50s/img]  # PrimaryBathroom: crash
```

**Pattern Recognition:**
- Times increasing: 116s → 54s → 305s → 2015s
- Memory pressure accumulation visible
- No cleanup between images

### 1.2 Batch Processing Crash

**Observed Symptoms:**
- PrimaryBathroom: Master TIFF generated (127 MB), crashed during upscaling
- PrimaryBedroom: Never started processing
- Process terminated after 57 minutes total

**Root Causes:**
1. **Memory leak accumulation**: After 4 images processed
2. **MPS memory fragmentation**: Unified memory pressure
3. **No isolation**: One image crash terminates entire batch
4. **No checkpointing**: Lost all progress on crash

### 1.3 MPS Memory Overflow

**Observed Symptoms:**
- 48MP image (341MB input) causes MPS OOM
- Unified memory at ~60GB allocation
- System has 48GB physical memory (M4 Max)

**Root Causes:**
1. **Large tensor allocation**: 48MP × 3 channels × 4 bytes × upscale buffers
2. **No dynamic tile sizing**: Fixed 2048px tiles insufficient for large images
3. **No fallback strategy**: Doesn't degrade to CPU on OOM

**Memory Calculations:**
```python
# Memory requirements per image
Input:      48MP × 3 × 4B = 576 MB (float32 RGB)
Processing: 576 MB × 8 (intermediate buffers) = 4.6 GB
Upscale 4x: (48MP × 16) × 3 × 4B = 9.2 GB
Total peak: ~15-20 GB (with overhead)

# MPS unified memory pressure
Physical RAM: 48 GB
System reserved: ~8 GB
Available: 40 GB
Peak allocation: 60 GB attempted → OOM ⚠️
```

### 1.4 Disk I/O Bottlenecks

**Observed Symptoms:**
- Kitchen failed with disk at 97% capacity
- I/O consuming 90% of processing time in bottleneck conditions
- Pool write time: 109.9s for 1.6GB TIFF (14 MB/s - very slow)

**Root Causes:**
1. **No pre-flight disk check**: Didn't verify space before starting
2. **No staged writes**: Large TIFFs written in single blocking operation
3. **No compression**: 16-bit uncompressed TIFFs very large
4. **Internal SSD saturation**: No automatic tiering to T9 external

### 1.5 Resource Contention Patterns

**System-Wide Issues:**
1. **No resource monitoring**: Blind processing without capacity checks
2. **No adaptive scheduling**: Doesn't adjust to system load
3. **No parallel processing**: Sequential = no resource utilization
4. **No cleanup between images**: Memory/cache accumulation

---

## 2. Architecture Principles

### 2.1 Design Philosophy

**Core Principles:**

1. **Fault Isolation**: One failure never cascades to entire batch
2. **Progressive Degradation**: System adapts rather than fails
3. **Resource Awareness**: Decisions based on real-time capacity
4. **Checkpoint Everything**: Resume from any failure point
5. **Measure Everything**: Observability drives optimization
6. **Fail Fast**: Early validation prevents wasted work

### 2.2 Quality Guarantee

**Non-Negotiable:**
- AI diff < 0.004 (current excellent level) maintained
- 16-bit precision end-to-end
- Metadata preservation (EXIF, IPTC, XMP, GPS)
- Material fidelity (wood, metal, glass, stone)

**Trade-offs Acceptable:**
- Processing time (can be longer if reliable)
- Memory usage (can use more RAM if prevents failures)
- Disk space (can use more storage if organized)

### 2.3 Backward Compatibility

**Guaranteed:**
- Existing CLI interfaces work without changes
- Configuration files remain compatible
- Output formats and naming unchanged
- Symlinks maintain path compatibility

**Enhanced:**
- New flags are opt-in
- Defaults preserve current behavior
- Legacy mode available via `--legacy-mode`

---

## 3. Component Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / API Interface                       │
│                  (lux-depth-v2 command)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Process Orchestrator                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Batch scheduling & prioritization                  │  │
│  │  • Fault isolation (subprocess per image)            │  │
│  │  • Checkpoint management                              │  │
│  │  • Retry logic with exponential backoff              │  │
│  │  • Resource-aware task distribution                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Resource   │  │   Storage    │  │   Pipeline   │
│   Monitor    │  │   Manager    │  │   Executor   │
└──────────────┘  └──────────────┘  └──────────────┘
        │            │            │
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring & Metrics                      │
│  • Real-time resource tracking (MPS, CPU, disk)             │
│  • Performance profiling (per-stage timing)                 │
│  • Quality validation (AI diff scoring)                     │
│  • Alert system (resource exhaustion, failures)             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

#### A. Process Orchestrator
**Purpose**: Intelligent batch scheduling with fault tolerance

**Responsibilities:**
- Queue management with priority scheduling
- Subprocess isolation (one process per image)
- Automatic checkpointing after each image
- Retry logic with exponential backoff
- Pre-flight resource validation
- Progress tracking and reporting

**Architecture Decision:**
- Use `multiprocessing.spawn` for complete process isolation
- Each image runs in fresh Python interpreter
- Prevents memory leaks from affecting subsequent images
- Subprocess timeout = 3× estimated processing time
- Automatic cleanup between images (gc.collect(), torch cache clearing)

#### B. Resource Monitor
**Purpose**: Real-time system resource monitoring

**Responsibilities:**
- Continuous background monitoring (5-second intervals)
- MPS/CUDA memory tracking
- Disk space monitoring (internal + T9)
- CPU/RAM utilization tracking
- Alert callbacks on resource exhaustion
- Pre-flight capacity checks

**Architecture Decision:**
- Background thread for non-blocking monitoring
- Alert thresholds: 80% warning, 90% critical
- Integration with orchestrator for admission control
- T9 fallback detection and automatic tiering

#### C. Storage Manager
**Purpose**: Intelligent tiered storage with auto-migration

**Responsibilities:**
- Tiered storage (internal SSD → T9 external → cloud)
- Automatic file migration based on size/age
- Symlink management for backward compatibility
- Disk space monitoring and cleanup
- Graceful degradation if T9 unavailable

**Architecture Decision:**
- Upscaled TIFFs (>2GB) automatically migrate to T9
- Master TIFFs stay on internal SSD for fast access
- Symlinks maintain backward compatibility
- Works without T9 (degrades to internal-only)

#### D. Pipeline Executor
**Purpose**: Modular pipeline with stage-wise checkpointing

**Responsibilities:**
- Stage-by-stage processing (load → depth → material → grade → upscale → export)
- Intermediate result caching
- Resume from any failed stage
- Parallel stage execution (when possible)
- Quality gates between stages

**Architecture Decision:**
- Each stage saves intermediate results to disk
- Checkpoint file tracks completed stages
- Can skip stages if checkpoint exists
- Enables debugging and optimization per-stage

---

## 4. Storage Architecture

### 4.1 Tiered Storage Design

**Storage Hierarchy:**

```
Tier 1: Internal SSD (Fast, Limited)
├─ Active workspace: 10-20 GB
├─ Master TIFFs: Recent outputs
├─ Depth maps: Cached for reuse
├─ Temporary files: Processing intermediates
└─ Small outputs: <1GB files

Tier 2: T9 External SSD (Fast, Large)
├─ Upscaled TIFFs: 2-10 GB each
├─ Completed projects: Full output sets
├─ Marketing assets: PNG overlays
└─ Historical runs: Archives

Tier 3: Cloud/Archive (Slow, Unlimited)
├─ Cold archives: S3, Backblaze
├─ Backup copies: Redundancy
└─ Long-term retention: >6 months old
```

### 4.2 Migration Policies

**Automatic Migration Rules:**

1. **Size-based**: Files >2GB → T9 immediately
2. **Type-based**: Upscaled TIFFs → T9, Masters → Internal
3. **Age-based**: Projects >30 days → T9, >180 days → Cloud
4. **Disk pressure**: When internal >85% → migrate largest files
5. **Project completion**: Entire output dir → T9 after validation

**Manual Migration:**
```bash
# Migrate specific output directory
lux-depth-v2-storage migrate --dir output_750_Picacho_Final --tier t9

# Archive old projects to cloud
lux-depth-v2-storage archive --older-than 180d --tier cloud

# Check storage status
lux-depth-v2-storage status
```

### 4.3 Symlink Strategy

**Purpose**: Maintain backward compatibility while using tiered storage

**Implementation:**
- Symlink created when file migrated to T9
- Original path → T9 location (transparent to scripts)
- Symlinks tracked in migration manifest
- Automatic symlink repair if T9 remounted
- Warning if T9 not available (symlinks broken)

**Example:**
```bash
# Before migration
/Users/rc/Transformation_Portal/output_750_Picacho/750Picacho_Pool_upscaled16.tif (1.6GB)

# After migration
/Users/rc/Transformation_Portal/output_750_Picacho/750Picacho_Pool_upscaled16.tif → 
  /Volumes/T9/Transformation_Portal_Outputs/750_Picacho/750Picacho_Pool_upscaled16.tif

# Scripts continue to work without changes
open output_750_Picacho/750Picacho_Pool_upscaled16.tif  # ✓ Works
```

### 4.4 Disk Space Management

**Pre-Flight Checks:**
```python
# Before starting batch
def preflight_disk_check(images: List[Path]) -> bool:
    required_gb = sum(img.stat().st_size / 1e9 * 15 for img in images)
    
    internal_free = psutil.disk_usage('/').free / 1e9
    t9_free = psutil.disk_usage('/Volumes/T9').free / 1e9 if t9_available else 0
    
    total_free = internal_free + t9_free
    
    if total_free < required_gb:
        logger.error(f"Insufficient disk space: need {required_gb:.1f}GB, have {total_free:.1f}GB")
        return False
    
    return True
```

**Auto-Cleanup:**
- Monitor disk usage every 60 seconds during processing
- When >90% capacity: pause and migrate largest files
- When >95% capacity: abort and alert user
- Temporary files cleaned up immediately after use
- Checkpoint files retained until batch completes

---

## 5. Pipeline Redesign

### 5.1 Modular Stage Architecture

**Current Problem:**
- Monolithic processing (all-or-nothing)
- No intermediate checkpoints
- Single failure loses all work
- Cannot inspect intermediate results

**Proposed Solution:**
- Split pipeline into 6 independent stages
- Each stage saves intermediate results
- Resume from any failed stage
- Parallel execution where possible

**Stage Breakdown:**

```
Stage 1: LOAD (I/O Bound)
├─ Read input image
├─ Parse metadata (EXIF, IPTC, XMP)
├─ Validate image format
└─ Save: rgb01.npy (RGB array)

Stage 2: DEPTH (GPU Bound)
├─ Load or generate depth map
├─ Synthesize zone weights (fg/mid/bg)
├─ Validate depth consistency
└─ Save: depth01.npy, weights.npy

Stage 3: MATERIAL (GPU Bound)
├─ Material segmentation (ONNX/heuristic)
├─ Surface type classification
├─ Material modifier maps
└─ Save: material_masks.npy, material_mods.npy

Stage 4: GRADE (GPU Bound)
├─ Color grading (depth-aware)
├─ Material-based adjustments
├─ Tone mapping
└─ Save: master_rgb01.npy

Stage 5: UPSCALE (GPU Bound, Memory Intensive)
├─ 4x upscaling (torch or ONNX)
├─ Tile-based processing
├─ Blend tiles with feathering
└─ Save: upscaled_rgb01.npy

Stage 6: EXPORT (I/O Bound)
├─ Write master TIFF (16-bit)
├─ Write upscaled TIFF (16-bit)
├─ Generate marketing PNG
├─ Generate preview JPG
├─ Write processing report JSON
└─ Validate outputs
```

### 5.2 Checkpoint Format

**Checkpoint File Structure:**

```json
{
  "image": "750Picacho_Pool.tif",
  "batch_id": "batch_20251208_120000",
  "stages": {
    "load": {
      "completed": true,
      "timestamp": 1733673601.234,
      "duration_sec": 0.5,
      "data_path": "checkpoints/750Picacho_Pool_load.npz",
      "metadata": {"width": 6000, "height": 3375, "megapixels": 20.25}
    },
    "depth": {
      "completed": true,
      "timestamp": 1733673605.123,
      "duration_sec": 3.8,
      "data_path": "checkpoints/750Picacho_Pool_depth.npz",
      "metadata": {"depth_source": "cached", "depth_consistency": 0.98}
    },
    "material": {
      "completed": true,
      "timestamp": 1733673610.456,
      "duration_sec": 5.3,
      "data_path": "checkpoints/750Picacho_Pool_material.npz",
      "metadata": {"segmentation_backend": "heuristic", "material_types": ["water", "stone", "sky"]}
    },
    "grade": {
      "completed": true,
      "timestamp": 1733673618.789,
      "duration_sec": 8.3,
      "data_path": "checkpoints/750Picacho_Pool_grade.npz",
      "metadata": {"ai_color_diff": 0.0020, "ai_luma_diff": 0.0018}
    },
    "upscale": {
      "completed": false,
      "timestamp": null,
      "duration_sec": null,
      "data_path": null,
      "metadata": null,
      "error": "MPS out of memory"
    },
    "export": {
      "completed": false,
      "timestamp": null,
      "duration_sec": null,
      "data_path": null,
      "metadata": null
    }
  },
  "overall_status": "failed",
  "total_duration_sec": 18.4,
  "failed_stage": "upscale",
  "retry_count": 0
}
```

### 5.3 Resume Logic

**Resume from Checkpoint:**

```python
def resume_from_checkpoint(checkpoint_file: Path, config: PipelineConfig):
    """Resume processing from saved checkpoint."""
    checkpoint = json.loads(checkpoint_file.read_text())
    
    # Find first incomplete stage
    stages = ["load", "depth", "material", "grade", "upscale", "export"]
    resume_from = None
    for stage in stages:
        if not checkpoint["stages"][stage]["completed"]:
            resume_from = stage
            break
    
    if resume_from is None:
        logger.info("All stages completed, nothing to resume")
        return
    
    logger.info(f"Resuming from stage: {resume_from}")
    
    # Load completed stage data
    stage_data = {}
    for stage in stages:
        if checkpoint["stages"][stage]["completed"]:
            data_path = Path(checkpoint["stages"][stage]["data_path"])
            stage_data[stage] = np.load(data_path)
    
    # Execute remaining stages
    pipeline = ModularPipeline(config)
    for stage in stages[stages.index(resume_from):]:
        try:
            result = pipeline.execute_stage(stage, stage_data)
            stage_data[stage] = result
            checkpoint["stages"][stage]["completed"] = True
            save_checkpoint(checkpoint_file, checkpoint)
        except Exception as e:
            logger.error(f"Stage {stage} failed: {e}")
            checkpoint["stages"][stage]["error"] = str(e)
            save_checkpoint(checkpoint_file, checkpoint)
            raise
```

### 5.4 Parallel Processing Opportunities

**Where Parallelism Helps:**

1. **Multi-Image Pipeline**: Process different images simultaneously
   - Challenge: MPS memory contention
   - Solution: Resource-aware scheduling (only 1-2 images on GPU at once)
   - Benefit: ~2x throughput for mixed CPU/GPU workloads

2. **Stage Pipelining**: Different images in different stages
   - Image 1: Upscaling (GPU)
   - Image 2: Material segmentation (GPU - lighter)
   - Image 3: Loading (I/O)
   - Benefit: Better GPU utilization, reduced idle time

3. **Tile Parallelism**: Process tiles in parallel during upscaling
   - Challenge: MPS doesn't support multi-stream well
   - Solution: Batch tiles together, process in single GPU call
   - Benefit: Minor speedup (10-20%), better memory efficiency

**Parallelism Architecture:**

```
Process Orchestrator
├─ Worker 1: Image A - Stage 5 (Upscale) - GPU
├─ Worker 2: Image B - Stage 3 (Material) - GPU (light)
├─ Worker 3: Image C - Stage 6 (Export) - I/O
└─ Worker 4: Image D - Stage 1 (Load) - I/O

Resource Scheduler:
- Max 2 GPU workers concurrent (prevent MPS thrashing)
- Max 4 I/O workers concurrent (disk bandwidth)
- Dynamic adjustment based on resource monitor feedback
```

**Implementation Note:**
Phase 1 focuses on sequential reliability. Parallel processing added in Phase 2 after stability proven.

---

