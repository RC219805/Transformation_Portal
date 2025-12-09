# Stability Architecture - Quick Reference Card
## For Developers & Operators

**Version:** 1.0 | **Date:** 2025-12-08

---

## 🚀 Quick Start

### Using the Enhanced Pipeline (Phase 1+)

```bash
# Basic usage (with orchestrator)
lux-depth-v2 \
  --input-dir images/ \
  --output-dir output/ \
  --use-orchestrator \
  --checkpoint-dir checkpoints/

# Resume from checkpoint
lux-depth-v2 \
  --resume-from checkpoints/batch_20251208_120000.json

# With all Phase 2 features
lux-depth-v2 \
  --input-dir images/ \
  --output-dir output/ \
  --use-orchestrator \
  --enable-tiered-storage \
  --t9-path /Volumes/T9/

# Full production mode (Phase 3)
lux-depth-v2 \
  --input-dir images/ \
  --output-dir output/ \
  --dashboard \
  --profiling \
  --max-concurrent 2
```

---

## 🎯 Key Features by Phase

### Phase 1: Stability ✅

**Orchestrator:**
- `--use-orchestrator`: Enable fault-tolerant processing
- `--checkpoint-dir PATH`: Save checkpoints for resume
- `--resume-from PATH`: Resume from checkpoint file
- `--max-retries N`: Retry failed images N times (default: 3)

**Resource Management:**
- `--max-memory-gb N`: Reserve N GB per image (default: 10)
- `--cpu-fallback-mp N`: Use CPU for images >N MP (default: 35)
- `--max-disk-usage-percent N`: Alert at N% disk usage (default: 85)

**Example:**
```bash
lux-depth-v2 \
  --input-dir images/ \
  --use-orchestrator \
  --max-memory-gb 8 \
  --cpu-fallback-mp 30
```

---

### Phase 2: Performance ⚡

**Storage Management:**
- `--enable-tiered-storage`: Use internal + T9 tiering
- `--t9-path PATH`: Path to T9 external storage
- `--auto-migrate`: Automatically migrate large files (default: true)
- `--no-tiered-storage`: Disable tiering (use internal only)

**Pipeline:**
- `--modular-pipeline`: Use stage-wise processing
- `--resume-from-stage STAGE`: Resume from specific stage

**I/O:**
- `--async-io`: Enable async TIFF writes (default: true)
- `--tiff-compression TYPE`: none|lzw|deflate (default: lzw)

**Example:**
```bash
lux-depth-v2 \
  --input-dir images/ \
  --enable-tiered-storage \
  --t9-path /Volumes/T9/ \
  --tiff-compression lzw
```

---

### Phase 3: Scale 📊

**Parallel Processing:**
- `--max-concurrent N`: Process N images in parallel (default: 2)
- `--enable-parallelism`: Enable multi-image pipeline

**Monitoring:**
- `--dashboard`: Show real-time metrics dashboard
- `--profiling`: Enable performance profiling
- `--metrics-port N`: Dashboard port (default: 8080)

**Example:**
```bash
lux-depth-v2 \
  --input-dir large_batch/ \
  --dashboard \
  --profiling \
  --max-concurrent 4
```

---

## 📋 Checkpoint Management

### Checkpoint File Structure

```json
{
  "batch_id": "batch_20251208_120000",
  "tasks": [
    {
      "image": "Pool.tif",
      "status": "completed",
      "attempts": 1
    },
    {
      "image": "Kitchen.tif",
      "status": "failed",
      "attempts": 3,
      "error": "MPS out of memory"
    }
  ],
  "completed_count": 4,
  "failed_count": 2
}
```

### Resume from Checkpoint

```bash
# Automatic resume (finds latest checkpoint)
lux-depth-v2 --resume

# Resume from specific checkpoint
lux-depth-v2 --resume-from checkpoints/batch_20251208_120000.json

# Resume from specific stage (Phase 2+)
lux-depth-v2 --resume-from-stage upscale --checkpoint batch.json
```

### Clean Up Checkpoints

```bash
# Remove old checkpoints (>7 days)
find checkpoints/ -name "*.json" -mtime +7 -delete

# Archive completed batch checkpoints
mv checkpoints/batch_20251208*.json archives/
```

---

## 💾 Storage Management

### Tiered Storage Tiers

```
Tier 1: Internal SSD (Fast, Limited)
├─ Master TIFFs
├─ Active processing
└─ Small outputs (<1GB)

Tier 2: T9 External (Fast, Large)
├─ Upscaled TIFFs (>2GB)
├─ Completed projects
└─ Archives

Tier 3: Cloud (Slow, Unlimited)
├─ Long-term archives (>6 months)
└─ Backup copies
```

### Storage CLI

```bash
# Check storage status
lux-depth-v2-storage status

# Migrate directory to T9
lux-depth-v2-storage migrate \
  --dir output_750_Picacho/ \
  --tier t9 \
  --create-symlinks

# Archive to cloud
lux-depth-v2-storage archive \
  --dir old_projects/ \
  --tier cloud

# Check disk space
lux-depth-v2-storage check-space
```

### Symlink Management

```bash
# Verify symlinks
ls -l output_*/ | grep " -> "

# Repair broken symlinks (if T9 remounted)
lux-depth-v2-storage repair-symlinks

# Convert symlinks back to files (if needed)
lux-depth-v2-storage materialize --dir output_750_Picacho/
```

---

## 🔍 Monitoring & Debugging

### Real-Time Dashboard (Phase 3)

```bash
# Start with dashboard
lux-depth-v2 --dashboard

# View in browser
open http://localhost:8080/metrics
```

**Dashboard Shows:**
- Progress: Completed / Failed / Pending
- Timing: Elapsed, ETA, Avg per image
- Resources: Memory, Disk, GPU utilization
- Alerts: Resource warnings, slow processing

### Performance Profiling

```bash
# Enable profiling
lux-depth-v2 --profiling --profiling-output profiles/

# View profiling report
cat profiles/performance_report.json

# Identify bottlenecks
python -m lux_depth_v2.profiler analyze \
  --input profiles/performance_report.json
```

### Logs and Metrics

```bash
# View processing logs
tail -f lux_depth_v2.log

# View metrics (JSON format)
cat output_dir/batch_summary.json

# View per-image reports
cat output_dir/750Picacho_Pool_report.json
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "MPS out of memory"**
```bash
# Solution: Use CPU fallback for large images
lux-depth-v2 --cpu-fallback-mp 30

# Or: Reduce tile size
lux-depth-v2 --post-tile 1024
```

**Issue: "Insufficient disk space"**
```bash
# Solution: Enable tiered storage
lux-depth-v2 --enable-tiered-storage --t9-path /Volumes/T9/

# Or: Manually free space
lux-depth-v2-storage migrate --pattern "output_*" --tier t9
```

**Issue: "Checkpoint not found"**
```bash
# Solution: Check checkpoint directory
ls -lh checkpoints/

# Or: Start fresh (no resume)
lux-depth-v2 --no-resume
```

**Issue: "T9 not available"**
```bash
# Solution: Check T9 is mounted
ls /Volumes/T9/

# Or: Disable tiered storage
lux-depth-v2 --no-tiered-storage
```

**Issue: "Pool image extremely slow"**
```bash
# Solution: Use ONNX segmentation (faster than heuristic)
lux-depth-v2 --seg-backend onnx

# Or: Skip material segmentation
lux-depth-v2 --no-material-segmentation
```

### Emergency Commands

```bash
# Stop processing gracefully (saves checkpoint)
# Press Ctrl+C once

# Force stop (no checkpoint)
# Press Ctrl+C twice

# Rollback to legacy mode
lux-depth-v2 --legacy-mode

# Clear all caches
rm -rf .cache/ checkpoints/
```

---

## 📊 Performance Benchmarks

### Expected Times (20MP Image)

| Phase | Time/Image | Notes |
|-------|-----------|-------|
| Baseline | 14 minutes | Current (before enhancements) |
| Phase 1 | 12-15 minutes | Stability (no performance gain) |
| Phase 2 | 1-2 minutes | ⚡ 10-15x faster |
| Phase 3 | 30-60 seconds | ⚡ 20-30x faster (parallel) |

### Stage Breakdown (20MP, Phase 2+)

| Stage | Time | % of Total |
|-------|------|-----------|
| Load | 0.5s | 1% |
| Depth | 3.8s | 6% |
| Material | 2.0s | 3% |
| Grade | 5.0s | 8% |
| Upscale | 40s | 67% |
| Export | 9s | 15% |
| **Total** | **60s** | **100%** |

---

## 🔧 Configuration Examples

### Minimal (Phase 1)

```yaml
# config/minimal.yaml
preset: "photo_realistic"
upscale: 4
device: "auto"

orchestrator:
  enabled: true
  checkpoint_dir: "checkpoints/"
```

### Production (Phase 2)

```yaml
# config/production.yaml
preset: "photo_realistic"
upscale: 4
device: "auto"

orchestrator:
  enabled: true
  checkpoint_dir: "checkpoints/"
  max_retries: 3

resources:
  max_memory_gb: 10.0
  cpu_fallback_threshold_mp: 35

storage:
  enable_tiered: true
  t9_path: "/Volumes/T9/Transformation_Portal_Outputs"
  auto_migrate: true
```

### Advanced (Phase 3)

```yaml
# config/advanced.yaml
preset: "photo_realistic"
upscale: 4
device: "auto"

orchestrator:
  enabled: true
  checkpoint_dir: "checkpoints/"

resources:
  max_memory_gb: 10.0
  enable_adaptive_tiling: true

storage:
  enable_tiered: true
  t9_path: "/Volumes/T9/"

performance:
  max_concurrent_images: 2
  async_io: true

monitoring:
  enable_dashboard: true
  profiling_enabled: true
```

---

## 📚 Related Documents

- **STABILITY_EFFICIENCY_ARCHITECTURE.md**: Full architecture design
- **PERFORMANCE_OPTIMIZATION_DESIGN.md**: Detailed implementation
- **IMPLEMENTATION_ROADMAP.md**: Week-by-week plan
- **STABILITY_ARCHITECTURE_EXECUTIVE_SUMMARY.md**: Executive overview

---

## 🆘 Getting Help

### Documentation
```bash
# General help
lux-depth-v2 --help

# Orchestrator help
lux-depth-v2 orchestrator --help

# Storage help
lux-depth-v2-storage --help
```

### Support
- GitHub Issues: `https://github.com/RC219805/Transformation_Portal/issues`
- Architecture Questions: See `docs/architecture/`
- Performance Issues: Enable `--profiling` and share report

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2025-12-08  
**Status:** Ready for Use (Phase 1 implementation starting)
