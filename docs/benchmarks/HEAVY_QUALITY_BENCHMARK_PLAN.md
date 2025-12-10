# Heavy Quality Benchmark Plan

**Purpose**: Establish "max quality" performance baseline after marketing export optimization.

**Context**: With marketing export optimized (84% faster), front-half stages (depth, materials, grading, upscaling) are now visible. This benchmark measures the true cost of turning on all heavy features.

---

## Why Now?

1. **Marketing export no longer dominates**: Was 95% of export time, now ~16% (12s of ~55s total)
2. **Autotune in place**: Export is tunable and measurable
3. **Clean baseline**: Before adding more heuristics/complexity
4. **Inform future decisions**: Understand quality vs speed trade-offs empirically

---

## Benchmark Matrix

### Test Images
- **Aerial**: Exterior/overhead, 21.6 MP (6000×3600)
- **Pool**: Complex water/reflections, 20.25 MP (6000×3375)
- **GreatRoom**: Interior, 12 MP (4000×3000)

### Configurations

#### Baseline (Current Production)
- Marketing PNG compression: Level 1 (new default)
- Materials v2: OFF
- Segmentation: Default resolution (768px long side)
- Depth: Uniform weights (no depth maps)
- Upscaling: 4× progressive with current tile settings

#### Heavy (Max Quality)
- Marketing PNG compression: Level 1 (keep optimized export)
- **Materials v2: ON**
- **Segmentation: 1536px long side** (higher resolution)
- **Mask caching: Enabled**
- Depth: Uniform weights (depth maps not yet available)
- Upscaling: Same 4× progressive (isolate materials v2 cost)

### What We're Measuring

**Stage-by-stage timings**:
- `io/read_input`
- `depth/estimate` or `io/read_depth`
- `material/segmentation` ← **Heavy mode impact**
- `material/response` ← **Heavy mode impact**
- `grade/master`
- `upscale/*` stages
- `export_upscaled`
- `export_marketing`

**Resource usage**:
- Peak RSS memory
- MPS/GPU memory (if available)
- CPU utilization during segmentation

**Quality proxies** (manual inspection):
- Material detection accuracy
- Segmentation mask quality
- Final render fidelity

---

## Success Criteria

### Performance Ceiling
- **Acceptable**: Heavy mode adds ≤50% total time vs baseline
- **Concerning**: Heavy mode adds >100% total time
- **Blocker**: Heavy mode adds >200% total time or causes OOM

### Stage Cost Distribution
- Identify which stages dominate in heavy mode
- Quantify materials v2 segmentation overhead
- Understand if 1536px segmentation is practical

### Decision Thresholds
- If heavy mode cost is ≤20s extra: **Make it default**
- If 20-50s extra: **Offer as optional "high quality" preset**
- If >50s extra: **Optimize segmentation or keep OFF by default**

---

## Execution Plan

### 1. Run Baseline (3 images)
```bash
for img in Aerial Pool GreatRoom; do
  lux-depth-v2 \
    --input input_images/750_Picacho/${img}.tif \
    --output-dir benchmarks/heavy/baseline_${img}/ \
    --preset exterior_showcase \
    --marketing-png-compression 1 \
    --no-autotune-export
done
```

**Expected time**: ~3-5 minutes per image (~10-15 min total)

### 2. Run Heavy (3 images)
```bash
for img in Aerial Pool GreatRoom; do
  lux-depth-v2 \
    --input input_images/750_Picacho/${img}.tif \
    --output-dir benchmarks/heavy/heavy_${img}/ \
    --preset exterior_showcase \
    --marketing-png-compression 1 \
    --no-autotune-export \
    --materials-v2 \
    --max-segmentation-side 1536 \
    --cache-masks
done
```

**Expected time**: Unknown (this is what we're measuring!)
- **Hypothesis**: 3-7 minutes per image if materials v2 is expensive
- **Best case**: <2× baseline (~6-10 min per image)
- **Worst case**: >3× baseline (OOM or >15 min per image)

### 3. Analyze Results
```bash
python scripts/analyze_heavy_benchmark.py benchmarks/heavy/
```

**Output**:
- Median total time: baseline vs heavy
- Stage-by-stage breakdown
- Top 5 cost increases
- Memory usage deltas

### 4. Automated Runner (All-in-One)
```bash
bash scripts/run_heavy_quality_benchmark.sh
```

Runs both configurations on all 3 images automatically.

---

## Key Questions to Answer

### Performance
1. **What is the total cost of materials v2 + high-res segmentation?**
   - Measured as: `heavy_total - baseline_total`
   - Acceptable: ≤50% increase (~30s)
   - Concerning: >100% increase (>60s)

2. **Which stage(s) dominate in heavy mode?**
   - Is it `material/segmentation`? (likely)
   - Is it `material/response`? (shouldn't be)
   - Any unexpected bottlenecks?

3. **Is 1536px segmentation practical?**
   - vs 768px default
   - Memory impact?
   - Quality improvement visible?

4. **Does mask caching help?**
   - Should be negligible cost
   - Confirm it doesn't break anything

### Resource Usage
5. **Memory ceiling with heavy features?**
   - Peak RSS in GB
   - Fits in 64GB M4 Max budget?
   - Margin for safety?

6. **CPU vs GPU bound?**
   - Segmentation should be GPU-bound (MPS)
   - If CPU-bound, investigate why

### Quality
7. **Visual improvement with materials v2?**
   - Pool: Better water material detection?
   - Aerial: Sky/terrain/vegetation separation?
   - GreatRoom: Wood/metal/fabric distinction?

8. **Is the quality delta worth the time cost?**
   - Subjective but important
   - Could inform "fast vs quality" preset split

---

## Risk Mitigation

### Potential Issues
1. **OOM on heavy mode**: 1536px segmentation may be too large
   - Mitigation: Monitor memory, fallback to 1024px if needed
2. **Extremely slow**: Materials v2 may have unexpected bottleneck
   - Mitigation: Profile individual stages, optimize or disable
3. **Quality regression**: Heavy mode introduces artifacts
   - Mitigation: Visual inspection, compare to baseline renders

### Safety Checks
- Monitor disk space (upscaled TIFFs are ~1.7GB each)
- Check for memory warnings in logs
- Validate output files are complete and uncorrupted

---

## Expected Outcomes

### Best Case
- Heavy mode adds 10-20s per image (~30-50% increase)
- Materials v2 provides visible quality improvement
- Memory usage stays well under 64GB
- **Decision**: Make heavy mode default or "recommended"

### Likely Case
- Heavy mode adds 30-40s per image (~50-100% increase)
- Materials v2 improves Pool/GreatRoom, marginal for Aerial
- Memory usage acceptable but tighter
- **Decision**: Offer heavy mode as optional preset ("archival_quality")

### Worst Case
- Heavy mode adds >60s per image (>100% increase)
- Materials v2 segmentation is too slow or memory-intensive
- Minimal visible quality benefit
- **Decision**: Keep materials v2 OFF by default, revisit optimization

---

## Post-Benchmark Actions

### If Heavy Mode is Fast Enough (≤50% overhead)
1. Update defaults: `materials_v2_enabled = True` in appropriate presets
2. Document quality improvements in README
3. Add visual examples to docs
4. Consider further segmentation optimization (if time allows)

### If Heavy Mode is Expensive (>50% overhead)
1. Create separate "archival_quality" preset with heavy features
2. Document trade-offs clearly (time vs quality)
3. Investigate segmentation optimization:
   - Try 1024px instead of 1536px
   - Profile segmentation backend (ONNX vs heuristic)
   - Consider caching strategies
4. Add `--fast` CLI flag to explicitly disable heavy features

### If Heavy Mode Causes Problems
1. Keep materials v2 OFF by default
2. Flag as "experimental" if exposed at all
3. Focus optimization efforts on segmentation before enabling

---

## Timeline

- **Benchmark execution**: 30-60 minutes (automated)
- **Analysis**: 15-30 minutes
- **Decision + documentation**: 30 minutes
- **Implementation** (if needed): 1-2 hours

**Total**: 2-4 hours end-to-end

---

## Success Definition

**This benchmark succeeds when we can confidently answer**:

> "What does 'max quality' cost in seconds, and is it worth it?"

With data-driven answers to guide:
- Default configuration choices
- Preset definitions (fast vs quality)
- Future optimization priorities
- User-facing quality vs speed recommendations

---

## References

- Marketing export optimization: 84% speedup (level 1 PNG)
- Current baseline: ~55-60s per image (Aerial)
- Materials v2 implementation: `lux_depth_v2/materials_v2.py`
- Segmentation backends: ONNX, SegFormer, heuristic
- Analysis tool: `scripts/analyze_heavy_benchmark.py`
