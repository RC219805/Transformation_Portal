# SAM2 Backend Performance Baselines (Phase 4C)

**Date:** 2026-02-18
**Environment:** macOS, Python 3.11.14, torch 2.5.1, device=mps
**Model:** SAM2.1 Hiera Large (`sam2.1_hiera_large.pt`, size not yet re-verified for the 2.1 release; prior checkpoint was ~856MB)
**Test Suite:** tests/spatial_ai/segmentation/test_sam2_backend_performance.py

## Baseline Metrics

### Auto Mode (Automatic Mask Generation)

#### 512x512 Image
- **Mean latency:** 13.38s
- **P95 latency:** 13.38s
- **Median:** 13.38s
- **Peak memory:** 1673 MB
- **Masks generated:** 1 (simple gradient test image)

**Quality Firewall Thresholds:**
- Mean regression threshold: +15% → 15.4s
- P95 regression threshold: +10% → 14.7s

#### Expected Scaling
- 1024x768 (0.79 MP): ~40s estimated
- 2048x1536 (3.15 MP): ~120s estimated

### Prompted Mode (Points/Bbox)

**TBD** - Requires real model runs (lighter than auto mode)

### Video Mode (Frame Tracking)

**TBD** - Requires real model runs

## Notes

- **First run warm-up:** Model loading takes ~15-20s (one-time cost)
- **Subsequent runs:** Cached model, only inference time counted
- **MPS acceleration:** Tests run on Apple Silicon with MPS backend
- **Memory profile:** Peak RSS includes model weights (~1.6GB) + activations

## Test Execution

```bash
# Run all benchmarks (takes ~10-15 minutes)
pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py -v -s

# Run single test
pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py::TestSAM2AutoModePerformance::test_auto_mode_latency_512x512 -v -s
```

## Regression Detection

Baseline files are stored in `docs/performance/baselines/` and compared automatically by the regression tests.

To establish a new baseline:
1. Delete existing baseline file
2. Run regression test (will skip and create baseline)
3. Commit new baseline

## Known Issues

- Auto mode on simple gradient images generates only 1 mask (expected for uniform content)
- Real-world images with multiple objects will generate more masks
- Latency scales with mask count (more masks = longer processing)

## Future Work

- [ ] Run benchmarks on real luxury real estate images
- [ ] Measure latency vs. mask count correlation
- [ ] Add CPU baseline for comparison
- [ ] Add CUDA baseline (if available)
- [ ] Profile memory by stage (encoder vs decoder vs post-processing)
