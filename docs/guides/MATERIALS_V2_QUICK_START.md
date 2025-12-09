# Materials v2 Quick Start Guide

**Ready to integrate and test Materials v2 in 3 steps!**

---

## Step 1: Complete Pipeline Integration (30-60 min) ⚠️

### Use Integration Helper

```bash
python3 scripts/integrate_materials_v2_pipeline.py --analyze
python3 scripts/integrate_materials_v2_pipeline.py --backup
```

### Manual Integration

**File:** `lux_depth_v2/pipeline.py`

1. **Add import** (line ~10):
   ```python
   from .materials_v2 import MaterialsV2Engine
   ```

2. **Initialize in `__init__`** (line ~150):
   ```python
   self.materials_engine = None
   if cfg.materials_v2 and cfg.materials_v2.enabled:
       self.materials_engine = MaterialsV2Engine(
           config=cfg.materials_v2,
           device=self.device,
       )
   ```

3. **Process in `_process_one`** (before upscaling):
   ```python
   if self.materials_engine:
       logger.info("Applying Materials v2...")
       with self.materials_engine.vram_manager.context_manager():
           img = self.materials_engine.process(img, depth_map, zones)
       logger.info("Materials v2 complete")
   ```

---

## Step 2: Quick Test (15 min)

```bash
# Test single image
python3 -m lux_depth_v2.cli \
  --input input_images/750_Picacho/Optimized_TIFFs/750Picacho_Pool_Ultimate.tif \
  --output-dir output_Materials_V2_Test \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks

# Verify success
ls -lh output_Materials_V2_Test/
ls -lh .materials_v2_cache/
```

**Expected:**
- ✓ "Applying Materials v2..." in logs
- ✓ "Materials v2 complete" in logs
- ✓ Output images created
- ✓ Cache files in `.materials_v2_cache/`

---

## Step 3: Full Test Suite (2-3 hours)

```bash
# Run comprehensive tests
./scripts/run_materials_v2_tests.sh

# Analyze results
python3 scripts/compare_materials_quality.py \
  --baseline-dir output_Materials_V2_Tests_*/Baseline \
  --enhanced-dir output_Materials_V2_Tests_*/Enhanced_0.6 \
  --output quality_report.json

# Run benchmarks
python3 scripts/benchmark_materials_v2.py \
  --input-dir input_images/750_Picacho/Optimized_TIFFs \
  --report benchmark_report.json
```

---

## CLI Quick Reference

### Basic Usage
```bash
--materials-v2                    # Enable Materials v2
--confidence-threshold 0.6        # Default threshold
--cache-masks                     # Enable caching
```

### Confidence Levels
```bash
--confidence-threshold 0.4        # Aggressive (more coverage)
--confidence-threshold 0.6        # Balanced (default)
--confidence-threshold 0.8        # Conservative (high certainty)
```

### Material-Specific Scenarios
```bash
# Water features (Pool)
--confidence-threshold 0.45

# Glass surfaces (Bathroom)
--confidence-threshold 0.55

# Mixed materials (Kitchen)
--confidence-threshold 0.65

# Wood-heavy (Bedroom)
--confidence-threshold 0.7
```

---

## Success Checklist

### After Integration
- [ ] Import added to pipeline.py
- [ ] Engine initialized in `__init__`
- [ ] Processing stage added in `_process_one`
- [ ] Quick test passes
- [ ] Materials v2 logging visible
- [ ] Cache files generated

### After Full Testing
- [ ] Baseline vs enhanced comparison complete
- [ ] Performance overhead < 10%
- [ ] Cache speedup 10-15%
- [ ] Quality metrics acceptable
- [ ] Edge cases handled well

---

## Documentation

- **User Guide:** `docs/MATERIALS_V2_USER_GUIDE.md`
- **Technical Spec:** `docs/MATERIALS_V2_TECHNICAL_SPEC.md`
- **Validation Report:** `MATERIALS_V2_VALIDATION_REPORT.md`
- **Testing Summary:** `MATERIALS_V2_TESTING_SUMMARY.md`

---

## Troubleshooting

### "AttributeError: 'PipelineConfig' object has no attribute 'materials_v2'"
- ✓ **Fixed!** Config updated, CLI integrated

### "Materials v2 not logging"
- Pipeline integration incomplete
- Follow Step 1 above

### "Cache files not created"
- Add `--cache-masks` flag
- Check `--cache-dir` path

### "Processing too slow"
- Reduce `--max-segmentation-side 1024`
- Use `--confidence-threshold 0.7` (less processing)

---

## Need Help?

1. Review integration guide:
   ```bash
   python3 scripts/integrate_materials_v2_pipeline.py
   ```

2. Check validation report:
   ```bash
   cat MATERIALS_V2_VALIDATION_REPORT.md
   ```

3. Analyze pipeline:
   ```bash
   python3 scripts/integrate_materials_v2_pipeline.py --analyze
   ```

---

**Status:** Ready for integration and testing!  
**Estimated Time:** 3-4 hours total  
**Next Step:** Complete pipeline integration (Step 1)
