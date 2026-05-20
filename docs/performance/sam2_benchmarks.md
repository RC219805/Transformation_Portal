# SAM2 Backend Performance Baselines (Phase 4C)

**Last re-measured:** 2026-05-20
**Model:** SAM2.1 Hiera Large (`sam2.1_hiera_large.pt`, ~856MB)
**Test Suite:** tests/spatial_ai/segmentation/test_sam2_backend_performance.py

## Threshold Policy

- Latency assertions use `recorded mean latency * 1.5` for the active device.
- Throughput assertions use `recorded FPS / 1.5` for the active device.
- Devices without a recorded baseline for the specific benchmark case skip only that threshold assertion, still pass through the rest of the benchmark, and still write JSON metrics output.
- The new 2026-05-20 CPU rows were measured locally with `TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=cpu`.

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

#### 1024x768 Image - CPU

- **Hardware target:** Local macOS arm64 Apple Silicon CPU (`TP_SAM2_BENCHMARK_DEVICE=cpu`)
- **Last re-measured:** 2026-05-20
- **Environment:** macOS 26.4.1 arm64, Python 3.12.13, torch 2.8.0, device=cpu
- **Mean latency:** 46.924s
- **P95 latency:** 47.26s
- **Masks generated:** 1 (simple gradient test image)
- **Assertion threshold:** 70.386s (1.5x mean baseline)

#### Historical MPS Expected Scaling (Not Assertion Baselines)

- 1024x768 (0.79 MP): previously estimated at ~40s
- 2048x1536 (3.15 MP): previously estimated at ~120s

### Prompted Mode (Points/Bbox)

#### Points Mode 512x512 - CPU

- **Hardware target:** Local macOS arm64 Apple Silicon CPU (`TP_SAM2_BENCHMARK_DEVICE=cpu`)
- **Last re-measured:** 2026-05-20
- **Environment:** macOS 26.4.1 arm64, Python 3.12.13, torch 2.8.0, device=cpu
- **Mean latency:** 1.089s
- **P95 latency:** 1.115s
- **Median:** 1.099s
- **Masks generated:** 3
- **Assertion threshold:** 1.634s (1.5x mean baseline)

#### Bbox Mode 512x512 - CPU

- **Hardware target:** Local macOS arm64 Apple Silicon CPU (`TP_SAM2_BENCHMARK_DEVICE=cpu`)
- **Last re-measured:** 2026-05-20
- **Environment:** macOS 26.4.1 arm64, Python 3.12.13, torch 2.8.0, device=cpu
- **Mean latency:** 1.107s
- **P95 latency:** 1.114s
- **Median:** 1.105s
- **Masks generated:** 3
- **Assertion threshold:** 1.661s (1.5x mean baseline)

### Video Mode (Frame Tracking)

#### 10-Frame 512x512 Tracking - CPU

- **Hardware target:** Local macOS arm64 Apple Silicon CPU (`TP_SAM2_BENCHMARK_DEVICE=cpu`)
- **Last re-measured:** 2026-05-20
- **Environment:** macOS 26.4.1 arm64, Python 3.12.13, torch 2.8.0, device=cpu
- **Total latency:** 13.112s for 10 frames
- **Throughput:** 0.763 FPS
- **Seconds per frame:** 1.311s
- **Peak memory:** 4092 MB
- **Tracked objects:** 10
- **Assertion floor:** 0.509 FPS (mean FPS baseline / 1.5)

## Notes

- **First run warm-up:** Model loading takes ~15-20s (one-time cost)
- **Subsequent runs:** Cached model, only inference time counted
- **MPS acceleration:** The existing 512x512 auto-mode MPS row was measured on Apple Silicon with the MPS backend
- **CPU baseline:** Re-measured on the local Apple Silicon CPU path to keep the documented fallback budget explicit
- **MPS rows:** Only the 512x512 auto-mode MPS row is a recorded assertion baseline. Add MPS rows for the remaining modes only after running those exact benchmark cases under pytest on an MPS-capable environment.
- **Memory profile:** Peak RSS includes model weights (~1.6GB) + activations; the recorded budget is device-specific
  (MPS ~1.7GB, local CPU fallback ~5.6GB), not a single cross-device `<2GB` threshold

## Test Execution

```bash
# Run all CPU benchmarks (takes several minutes)
TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=cpu .venv/bin/pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py -v -s

# Run single test
TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=cpu .venv/bin/pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py::TestSAM2AutoModePerformance::test_auto_mode_latency_512x512 -v -s

# Re-measure MPS rows only on an environment where pytest reports MPS available
TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=mps .venv/bin/pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py -v -s
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
- [ ] Re-measure MPS baselines for 1024x768 auto, points, bbox, and video modes under pytest
- [ ] Add CUDA baseline (if available)
- [ ] Profile memory by stage (encoder vs decoder vs post-processing)
