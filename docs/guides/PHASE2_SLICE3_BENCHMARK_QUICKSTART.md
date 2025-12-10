# Phase 2 Slice 3: Benchmark Quick Start Guide

**Status**: Ready to Execute 🚀  
**Date**: 2025-12-10  
**Estimated Time**: 30-60 minutes (automated)

---

## Prerequisites

### 1. Environment Setup
```bash
cd ~/Transformation_Portal
git checkout main
git pull origin main

# Activate virtual environment
source .venv/bin/activate  # or your environment

# Verify Python version (3.11+ recommended)
python --version
```

### 2. Test Images Available
Ensure Picacho test images exist:
```bash
ls -lh input_images/750_Picacho/*.tif
```

Expected files:
- `Pool.tif` (~72 MP)
- `Aerial.tif` (~96 MP)
- `GreatRoom.tif` (~65 MP)
- `Kitchen.tif` (~58 MP)

If missing, use any high-resolution 16-bit TIFF images (50-100 MP recommended).

### 3. Clean System State (Optional but Recommended)
For best benchmark accuracy:
```bash
# Close heavy applications (browsers, IDEs)
# Disable Spotlight indexing temporarily:
sudo mdutil -a -i off

# Stop Time Machine backups:
tmutil disable

# Check disk space (need ~10GB free for outputs)
df -h
```

---

## Quick Start: Single Command

Run complete benchmark suite:

```bash
python scripts/run_benchmark_suite.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark \
    --images Pool Aerial GreatRoom Kitchen \
    --runs 3
```

**What This Does**:
- Runs 4 images × 4 modes × 3 runs = **48 benchmarks**
- Total time: ~30-60 minutes (depends on hardware)
- Auto-generates all results and documentation
- No manual intervention needed

---

## What Happens During Execution

### Phase 1: Baseline Runs (Ground Truth)
```
Running: Pool [baseline]
  Run 1/3... Export time: X.XXs, File size: XXX.X MB
  Run 2/3... Export time: X.XXs, File size: XXX.X MB
  Run 3/3... Export time: X.XXs, File size: XXX.X MB
✅ Benchmark complete!

Running: Aerial [baseline]
...
```

### Phase 2: Tiled Runs (Optimized)
```
Running: Pool [tiled]
  Config: tiff_tile_size=512, compression=lzw
  Run 1/3... Export time: X.XXs, File size: XXX.X MB
  ...
```

### Phase 3: Tiled+Atomic Runs
```
Running: Pool [tiled_atomic]
  Config: tiled + atomic_writes=True
  ...
```

### Phase 4: Full Optimized Runs
```
Running: Pool [full_optimized]
  Config: tiled + atomic + tiered_storage
  ...
```

### Final: Results Generation
```
✅ Results saved to: output_benchmark/all_results.json
✅ CSV saved to: output_benchmark/comparison.csv
✅ Results markdown updated: docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md

✅ BENCHMARK SUITE COMPLETE
```

---

## Output Files

After completion, you'll have:

### 1. Per-Image, Per-Mode Results
```
output_benchmark/
  pool_baseline/
    results.json          # Aggregated 3-run average
    run_1/
      Pool_master16.tif
      Pool_upscaled16.tif
      Pool__REPORT.json
    run_2/
    run_3/
  pool_tiled/
    results.json
    run_1/
    ...
  pool_tiled_atomic/
  pool_full_optimized/
  
  aerial_baseline/
  aerial_tiled/
  ...
```

### 2. Aggregated Results
```
output_benchmark/
  all_results.json       # Complete nested JSON dataset
  comparison.csv         # Flat CSV for Excel/R/Python
```

### 3. Auto-Generated Markdown Report
```
docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md
```

This file will be **automatically populated** with:
- Executive summary
- Per-image comparison tables
- Aggregate statistics
- Performance gain calculations
- Target met/miss indicators (✅/❌)

---

## Reading the Results

### Key Metrics to Look For

#### 1. Export Latency Reduction
**Target**: 30-50% faster on 50MP+ images

Example result:
```
| Scene  | Baseline (s) | Tiled (s) | Reduction (%) | Target Met? |
|--------|--------------|-----------|---------------|-------------|
| Pool   | 15.30        | 8.45      | 44.8%         | ✅           |
```

**Interpretation**: 
- ✅ 44.8% faster → Target met (30-50% range)
- This is a **production-ready optimization**

#### 2. File Size Reduction
**Target**: 20-40% smaller with compression

Example result:
```
| Scene | Baseline (MB) | Tiled+LZW (MB) | Reduction (%) | Target Met? |
|-------|---------------|----------------|---------------|-------------|
| Pool  | 825.3         | 512.7          | 37.9%         | ✅           |
```

**Interpretation**:
- ✅ 37.9% smaller → Target met (20-40% range)
- Compression ratio: 1.61x
- **Significant storage savings**

#### 3. Memory Usage
**Target**: Neutral or reduced (no increase)

Example result:
```
| Mode           | Peak RSS (MB) | vs Baseline |
|----------------|---------------|-------------|
| Baseline       | 3421          | -           |
| Tiled          | 3215          | -6.0%       |
```

**Interpretation**:
- ✅ 6% less memory → Tiling helps
- **No memory regressions**

#### 4. Throughput
**Target**: 50-100% increase (images/hour)

Example result:
```
| Mode     | Images/Hour | vs Baseline |
|----------|-------------|-------------|
| Baseline | 235         | -           |
| Tiled    | 425         | +80.9%      |
```

**Interpretation**:
- ✅ 81% faster throughput → Target met
- **Batch processing dramatically improved**

---

## Interpreting Target Met Indicators

### ✅ All Green: Ready for Default Rollout
If all 4 images show ✅ for latency and file size:
- **Recommendation**: Enable by default for all images >50 MP
- **Confidence**: High (validated across diverse scenes)
- **Rollout**: Can proceed to Phase 3 (default ON)

### ✅ Mostly Green (3/4): Ready for Gradual Rollout
If 3+ images meet targets:
- **Recommendation**: Enable for images >80 MP (conservative)
- **Confidence**: Medium-High
- **Rollout**: Phase 1 (>80MP), then Phase 2 (>50MP) after monitoring

### ❌ Mixed Results (2/4): Investigate & Optimize
If only 2 images meet targets:
- **Action Required**: 
  - Check for I/O bottlenecks (slow disk, network storage)
  - Try different compression (zstd vs lzw)
  - Adjust tile size (256, 512, 1024)
- **Rollout**: Defer until optimization complete

### ❌ Poor Results (<2/4): Revisit Design
If less than 2 images meet targets:
- **Action Required**: Architectural review
- Possible causes:
  - Disk I/O bound (not CPU bound)
  - Compression overhead exceeds savings
  - System configuration issues
- **Rollout**: Do not proceed

---

## Common Issues & Solutions

### Issue: "Input file not found"
```
Warning: Input not found: input_images/750_Picacho/Pool.tif
```

**Solution**: Create symlink or copy images to expected path:
```bash
mkdir -p input_images/750_Picacho
cp /path/to/your/images/*.tif input_images/750_Picacho/
```

### Issue: Benchmark taking too long
**If >2 hours**:
- Reduce `--runs` to 1 (less statistical validity but faster)
- Test fewer images (just Pool and Aerial)
- Check disk I/O (slow network storage?)

### Issue: Out of disk space
**Need ~10GB free**:
```bash
# Clean up old benchmark runs
rm -rf output_benchmark_old

# Use external SSD for output
python scripts/run_benchmark_suite.py \
    --output-dir /Volumes/ExternalSSD/benchmark \
    ...
```

### Issue: High variance between runs
**If run-to-run timing varies >10%**:
- System under load (close other apps)
- Background processes (indexing, backups)
- Thermal throttling (let system cool down)
- Re-run with `--runs 5` for better averaging

---

## Advanced Options

### Test with Tiered Storage
Add scratch directory for SSD acceleration:
```bash
python scripts/run_benchmark_suite.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark \
    --images Pool Aerial GreatRoom Kitchen \
    --runs 3 \
    --scratch /tmp/scratch
```

### Test Subset of Images (Quick Validation)
```bash
# Just Pool and Aerial (fastest + largest)
python scripts/run_benchmark_suite.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark_quick \
    --images Pool Aerial \
    --runs 2
```

### Custom Results Path
```bash
python scripts/run_benchmark_suite.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark \
    --images Pool Aerial GreatRoom Kitchen \
    --runs 3 \
    --results-md docs/guides/MY_CUSTOM_RESULTS.md
```

---

## After Benchmark Completion

### 1. Review Results
```bash
# Open auto-generated markdown
open docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md

# Or view in terminal
cat docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md
```

### 2. Analyze CSV (Optional)
```bash
# Open in Excel
open output_benchmark/comparison.csv

# Or quick terminal view
column -t -s, output_benchmark/comparison.csv | less -S
```

### 3. Commit Results to Repo
```bash
git add docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md
git add output_benchmark/all_results.json
git add output_benchmark/comparison.csv
git commit -m "Phase 2 Slice 3: benchmark results"
git push
```

### 4. Make Rollout Decision

Based on results, choose rollout strategy:

#### Option A: Aggressive (All targets met)
```python
# Enable by default for all images >50 MP
DEFAULT_EXPORT_CONFIG = ExportConfig(
    output_dir=output_dir,
    tiff_tile_size=512,
    tiff_compression="lzw",
    use_atomic_image_writes=True,
)
```

#### Option B: Conservative (Mixed results)
```python
# Enable only for very large images
if image_size_mp > 80:
    cfg.tiff_tile_size = 512
    cfg.tiff_compression = "lzw"
```

#### Option C: Gradual (Recommended)
```python
# Week 1-2: >80 MP
# Week 3-4: >50 MP
# Week 5+: All images
```

---

## Re-Enable System Features (If Disabled)

After benchmarks complete:

```bash
# Re-enable Spotlight indexing
sudo mdutil -a -i on

# Re-enable Time Machine
tmutil enable
```

---

## Timeline

| Step | Duration | Action |
|------|----------|--------|
| **Setup** | 5 min | Verify environment, test images |
| **Execute** | 30-60 min | Run automated suite (hands-off) |
| **Review** | 10 min | Read generated PERFORMANCE_RESULTS.md |
| **Decide** | 5 min | Choose rollout strategy |
| **Total** | **~1 hour** | **Complete validation** |

---

## Success Checklist

Before proceeding to rollout:

- ✅ All 4 test images benchmarked successfully
- ✅ Results populated in PERFORMANCE_RESULTS.md
- ✅ At least 3/4 images meet latency target (30-50% reduction)
- ✅ At least 3/4 images meet file size target (20-40% reduction)
- ✅ No memory regressions observed
- ✅ Results committed to repository
- ✅ Rollout strategy decided

---

## Ready to Execute?

Run the single command:

```bash
python scripts/run_benchmark_suite.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark \
    --images Pool Aerial GreatRoom Kitchen \
    --runs 3
```

**Then walk away and let it run!** ☕

Results will be waiting for you in:
- `output_benchmark/all_results.json`
- `output_benchmark/comparison.csv`
- `docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md`

---

**Questions or issues?** Check the [troubleshooting section](#common-issues--solutions) or open an issue.

🚀 **Ready to validate Slice 3 performance!**
