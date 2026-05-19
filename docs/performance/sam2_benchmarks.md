# SAM2 Backend Performance Baselines (Phase 4C)

**Last re-measured:** 2026-05-19
**Model:** SAM2.1 Hiera Large (`sam2.1_hiera_large.pt`, ~856MB)
**Test Suite:** tests/spatial_ai/segmentation/test_sam2_backend_performance.py

## Baseline Metrics

### Auto Mode (Automatic Mask Generation)

#### 512x512 Image - MPS

- **Hardware target:** Apple Silicon MPS
- **Last re-measured:** 2026-02-18
- **Environment:** macOS, Python 3.11.14, torch 2.5.1, device=mps
- **Mean latency:** 13.38s
- **P95 latency:** 13.38s
- **Median:** 13.38s
- **Peak memory:** 1673 MB
- **Masks generated:** 1 (simple gradient test image)
- **Assertion threshold:** 20.07s (1.5x mean baseline)

#### 512x512 Image - CPU

- **Hardware target:** Local macOS arm64 Apple Silicon CPU (`TP_SAM2_BENCHMARK_DEVICE=cpu`)
- **Last re-measured:** 2026-05-19
- **Environment:** macOS 26.4.1 arm64, Python 3.12.13, torch 2.8.0, device=cpu
- **Mean latency:** 42.66s
- **P95 latency:** 42.93s
- **Median:** 42.54s
- **Peak memory:** 5617 MB
- **Masks generated:** 1 (simple gradient test image)
- **Assertion threshold:** 63.99s (1.5x mean baseline)

**Bounded Smoke Thresholds:**
- `test_auto_mode_latency_512x512` uses 1.5x the recorded mean baseline for the active device.
- Devices without a recorded baseline skip this threshold assertion instead of using an inherited default.

#### Historical MPS Expected Scaling
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
- **CPU baseline:** Re-measured on the local Apple Silicon CPU path to keep the documented fallback budget explicit
- **Memory profile:** Peak RSS includes model weights (~1.6GB) + activations; the recorded budget is device-specific
  (MPS ~1.7GB, local CPU fallback ~5.6GB), not a single cross-device `<2GB` threshold

## Test Execution

```bash
# Run all benchmarks (takes ~10-15 minutes)
TP_RUN_BENCHMARKS=1 pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py -v -s

# Run single test
TP_RUN_BENCHMARKS=1 pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py::TestSAM2AutoModePerformance::test_auto_mode_latency_512x512 -v -s

# Re-measure the CPU fallback baseline on Apple Silicon
TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=cpu pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py::TestSAM2AutoModePerformance::test_auto_mode_latency_512x512 -v -s
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
- [ ] Re-measure CPU baseline on a CI-like Linux CPU target if CPU enforcement expands beyond local fallback coverage
- [ ] Add CUDA baseline (if available)
- [ ] Profile memory by stage (encoder vs decoder vs post-processing)
