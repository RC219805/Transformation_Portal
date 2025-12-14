# Strategic Assessment: Advancing Water Detection for Materials V3

**Date**: 2025-12-14  
**Prepared By**: Transformation Portal Architect  
**Context**: Post-PR-W4 validation harness completion, pre-production deployment planning

---

## Executive Summary

**Current State**: Water detection infrastructure is production-ready, but the detection algorithm is a stub. The system can observe, integrate, refine, and validate water masks, but currently generates them using a basic blue-threshold heuristic rather than the specified multi-cue approach.

**Key Finding**: We have a **validated infrastructure waiting for a validated detector**. The bottleneck is not architecture—it's algorithm quality and calibration.

**Recommended Path**: **Data-First Hybrid Approach** (Option 3 + Option 4)
- Phase 1 (Week 1): Create labeled validation dataset (50-100 images)
- Phase 2 (Week 2): Implement simplified multi-cue heuristic detector
- Phase 3 (Week 3): Validate and tune thresholds using real data
- Phase 4 (Week 4+): Production deployment with measured quality gates

**Timeline**: 3-4 weeks to production-validated water detection
**Risk Level**: Low (infrastructure proven, algorithm isolated)
**Investment Required**: Dataset creation (~1 week), detector implementation (~1 week), validation/tuning (~1 week)

---

## 1. Honest Current State Assessment

### ✅ What's Production-Ready

#### Infrastructure (Complete)
- **PR-W0 (Observability)**: ✅ Complete
  - `WaterCandidateReport` in all Materials V3 outputs
  - Class presence audit includes water metrics
  - Zero behavior change when disabled
  - Tests passing, linting clean

- **PR-W2 (Integration)**: ✅ Complete (architecture)
  - Water detector integrated into Materials V3 pipeline
  - Scene context inference (pool vs ocean)
  - SegFormer-first strategy (use ML when available)
  - Source tracking (segformer vs heuristic)
  - Gating thresholds (confidence, coverage)
  - Opt-in configuration (disabled by default)

- **PR-W3 (Edge Refinement)**: ✅ Complete (code)
  - EfficientSAM integration for boundary cleanup
  - Safety gates (confidence, boundary pixel thresholds)
  - ROI cropping to reduce overhead
  - Prompt sampling from high-confidence regions
  - Graceful degradation (skip refinement if boundary too small)

- **PR-W4 (Validation Harness)**: ⚠️ Complete (infrastructure)
  - CLI harness (`scripts/prw_water_validation.py`)
  - JSON report generation
  - Metrics: coverage, confidence, false positives, performance
  - Stability metric (consistency across perturbations)
  - **Blocked**: Edge alignment metric (requires mask export)

### ⚠️ What's Stub/Incomplete

#### Detection Algorithm (Stub Only)
- **PR-W1 (Water Detector)**: ❌ **Stub Implementation Only**
  
  **Current (Stub)**:
  ```python
  # lux_depth_v2/water_candidate.py
  # Simple blue threshold: blue > red AND blue > green
  blue_dominant = (blue > red) & (blue > green * 0.8)
  mask = blue_dominant & (blue > 0.3)
  ```
  
  **Specified (PR-W1 Full)**:
  - ❌ Chromaticity cue (HSV/Lab, pool vs ocean tuned)
  - ❌ Specular cue (highlights + low saturation reflections)
  - ❌ Texture cue (entropy/frequency analysis)
  - ❌ Planarity cue (depth gradient analysis)
  - ❌ Weighted combination of cues
  - ❌ Morphological post-processing
  - ❌ Component filtering (top-K, min area)
  - ❌ Feature score tracking (chromaticity, specular, texture, planarity)

  **Impact**: Detection quality limited to obvious blue regions; misses subtle water, false-positives on blue sky/glass.

#### Validation Metrics (Partial)
- **Edge Alignment (Primary Metric)**: ❌ Blocked
  - Returns 0.0 (mask not available for validation)
  - Cannot measure boundary quality
  - Cannot validate EfficientSAM refinement effectiveness
  - **Root Cause**: MaterialsV3Engine doesn't expose mask in report dict
  - **Fix Required**: Add `water_validation_emit_mask` debug flag (2 hours)

- **Other Metrics**: ✅ Working
  - Coverage tracking
  - Confidence scoring
  - False positive detection
  - Performance timing
  - Stability (coverage variance)

#### Calibration (Not Done)
- ❌ No labeled validation dataset
- ❌ Thresholds are targets, not calibrated:
  - Detection rate target: ≥85% (not measured)
  - False positive rate target: ≤5% (not measured)
  - Edge alignment target: ≥0.6 (metric blocked)
  - Stability target: ≥0.8 (not validated)
- ❌ No production data analysis

---

## 2. Strategic Options Analysis

### Option 1: Implement Full PR-W1 Spec (Heuristic Multi-Cue)

**What**: Implement complete multi-cue heuristic detector as specified in `docs/PR_WATER_MASK_STRUCTURE.md`:
- Chromaticity cue (HSV/Lab, pool vs ocean tuned)
- Specular cue (highlights + low saturation)
- Texture/entropy cue (smooth water vs textured foliage)
- Planarity cue (depth gradient analysis)
- Weighted combination
- Morphological post-processing
- Component filtering (top-K, min area)

**Implementation Path**:
1. Create `lux_depth_v2/water_candidate.py` (replace stub)
2. Implement 5 feature extraction methods (chromaticity, specular, texture, planarity, component)
3. Implement weighted combination with tunable weights
4. Implement post-processing pipeline (morphology, hole filling)
5. Add comprehensive unit tests (torch-free)
6. Document feature weights and threshold rationale

**Time**: 1.5-2 days implementation + 0.5 day testing = **2-3 days total**

**Pros**:
- ✅ Spec-compliant (closes PR-W1 properly)
- ✅ Production-grade detection quality (better than stub)
- ✅ CPU-only (no ML inference overhead)
- ✅ Explainable (feature scores show why detection succeeded/failed)
- ✅ Tunable (weights can be adjusted per scene type)

**Cons**:
- ❌ Still heuristic-based (not learning from data)
- ❌ Unknown quality until validated against real images
- ❌ May need iterative tuning after validation
- ❌ Thresholds chosen by intuition, not data

**Risk Level**: **Medium**
- Unknown if heuristic approach sufficient for production
- May discover edge cases requiring ML after implementation

**Recommended If**: We believe heuristics can achieve production quality (≥85% detection, ≤5% FP)

**Cost-Benefit**:
- **Cost**: 2-3 days engineering + unknown tuning time
- **Benefit**: Production-ready detector IF heuristics are sufficient
- **Risk**: Wasted effort if heuristics can't meet quality targets

---

### Option 2: ML Detector (Skip Heuristics)

**What**: Replace stub with lightweight ML segmentation model:
- **Option 2A**: Fine-tune existing Materials V3 SegFormer with water emphasis
- **Option 2B**: Fine-tune lightweight segmentation model (MobileNetV3 + DeepLabV3)
- **Option 2C**: Use pre-trained water segmentation model (if available)

**Implementation Path**:
1. Create/source labeled dataset (100+ images: pool, ocean, non-water)
2. Select base model (SegFormer, DeepLab, U-Net)
3. Fine-tune on water dataset
4. Convert to ONNX/CoreML for deployment
5. Integrate as alternative backend in Materials V3
6. Add confidence calibration
7. Validate performance (accuracy, inference time, memory)

**Time**: 
- Dataset creation: 1 week (if not available)
- Model training/fine-tuning: 2-3 days
- Integration: 1 day
- Validation: 1 day
- **Total: 2 weeks (with dataset) or 5 days (dataset available)**

**Pros**:
- ✅ State-of-the-art accuracy (learns from data)
- ✅ Handles complex scenes (mixed water/sky, reflections, waves)
- ✅ Improves over time (more training data → better model)
- ✅ Generalizes to new scenes better than heuristics

**Cons**:
- ❌ Requires labeled training data (100+ images minimum)
- ❌ Adds ML dependency (model file, inference overhead)
- ❌ Inference time higher than heuristics (50-100ms vs 10-20ms)
- ❌ Black box (harder to debug than feature-based heuristics)
- ❌ GPU/CoreML required for acceptable performance

**Risk Level**: **Medium-High**
- Dataset creation bottleneck
- Model training may not converge (small dataset)
- Inference time may be prohibitive for batch processing

**Recommended If**: We have access to labeled pool/ocean dataset OR willing to create it

**Cost-Benefit**:
- **Cost**: 2 weeks (dataset + training) or 5 days (dataset available)
- **Benefit**: Best possible accuracy, future-proof
- **Risk**: Dataset availability, training complexity, inference overhead

---

### Option 3: Hybrid Approach (Heuristic + ML Refinement)

**What**: 
1. Implement **fast simplified heuristic** detector (not full PR-W1, but better than stub)
2. Use EfficientSAM only for boundary refinement (already implemented in PR-W3)
3. Fallback chain: SegFormer → Heuristic → EfficientSAM refinement

**Implementation Path**:
1. **Simplified Heuristic Detector** (1 day):
   - Chromaticity cue only (HSV-based, pool vs ocean ranges)
   - Simple specular boost (high value + low saturation)
   - Basic component filtering (remove noise)
   - Skip texture and planarity cues (nice-to-have, not critical)

2. **Integration** (already done):
   - Materials V3 already has fallback logic (SegFormer → Heuristic)
   - EfficientSAM already integrated for boundary refinement

3. **Tuning** (1 day):
   - Tune hue ranges for pool vs ocean
   - Tune confidence thresholds
   - Validate refinement improves boundaries

**Time**: **2-3 days total**

**Pros**:
- ✅ Fast detection (heuristic is CPU-only, ~10-20ms)
- ✅ High-quality boundaries (EfficientSAM refinement)
- ✅ Graceful degradation (if ML unavailable, heuristic still works)
- ✅ Explainable (feature scores available)
- ✅ Leverages existing PR-W3 refinement infrastructure

**Cons**:
- ⚠️ Simplified heuristic may miss edge cases (complex reflections)
- ⚠️ Adds complexity (two-stage detection)
- ⚠️ Tuning effort for handoff between heuristic and SAM

**Risk Level**: **Low**
- Simplified heuristic easier to implement than full PR-W1
- EfficientSAM proven effective (PR-W3 complete)
- Fallback logic already tested

**Recommended If**: We want balance of speed, quality, and maintainability

**Cost-Benefit**:
- **Cost**: 2-3 days engineering
- **Benefit**: Production-ready system with quality boundaries
- **Risk**: Low (builds on proven components)

**This is my recommended approach** (see Section 3).

---

### Option 4: Data-First Approach (Build Dataset, Then Detector)

**What**: 
1. Create labeled validation dataset FIRST (pool/ocean/non-water, 50-100 images)
2. Run stub detector, analyze failure modes
3. Design detector based on actual data patterns (not guesswork)
4. Iterate on detector with real validation metrics

**Implementation Path**:
1. **Dataset Creation** (1 week):
   - Collect 50-100 diverse images:
     - Pool scenes: residential, resort, lap pools
     - Ocean scenes: calm, waves, horizon
     - Non-water: blue sky, glass buildings, foliage
   - Label with ground truth masks (pool/ocean/non-water)
   - Document scene characteristics

2. **Baseline Analysis** (1 day):
   - Run stub detector on dataset
   - Measure coverage, false positives, edge alignment (fix mask export first)
   - Analyze failure modes:
     - Where does stub fail? (reflections, mixed scenes, etc.)
     - What features distinguish pool vs ocean vs blue sky?
     - What boundary characteristics matter most?

3. **Data-Driven Design** (2-3 days):
   - Based on failure analysis, implement detector
   - Could be heuristic, ML, or hybrid—data determines approach
   - Tune thresholds to meet targets

4. **Validation Loop** (ongoing):
   - Run validation harness
   - Tune detector based on metrics
   - Iterate until targets met

**Time**: **2-3 weeks total** (1 week dataset + 1-2 weeks detector + tuning)

**Pros**:
- ✅ Data-driven design (not guesswork)
- ✅ Know quality before declaring "done" (validated metrics)
- ✅ Can calibrate thresholds properly (detection rate, FP rate, edge alignment)
- ✅ Defensible (data-proven, not intuition)
- ✅ Iterative (improve based on real failures)

**Cons**:
- ❌ Requires manual labeling effort (50-100 images)
- ❌ Longer time to first production deployment
- ❌ Dataset may not represent production distribution

**Risk Level**: **Low** (data-driven reduces guesswork risk)

**Recommended If**: We value validated quality over speed to production

**Cost-Benefit**:
- **Cost**: 2-3 weeks (mostly dataset creation)
- **Benefit**: Defensible quality, validated metrics, informed design
- **Risk**: Low (data reduces uncertainty)

**This is the foundation for my recommended approach** (see Section 3).

---

### Option 5: Pragmatic Minimum (Ship Improved Stub)

**What**: 
1. Keep stub detector but improve it slightly:
   - Add saturation check (not just blue channel)
   - Add simple component filtering (remove noise)
   - Add scene context (pool vs ocean hue ranges)
2. Document limitations clearly
3. Ship with conservative thresholds
4. Collect production data for iteration

**Implementation Path**:
1. **Improve Stub** (4-6 hours):
   ```python
   # Current stub: blue > red AND blue > green
   # Improved stub: HSV-based with scene context
   
   hsv = rgb2hsv(rgb01)
   hue = hsv[:, :, 0] * 360  # degrees
   sat = hsv[:, :, 1]
   val = hsv[:, :, 2]
   
   # Pool: cyan/blue (170-210°), ocean: broader blue-green (160-220°)
   if scene_context == POOL:
       hue_match = (hue >= 170) & (hue <= 210)
   else:
       hue_match = (hue >= 160) & (hue <= 220)
   
   mask = hue_match & (sat > 0.2) & (val > 0.2)
   
   # Component filtering
   mask = filter_components(mask, min_area=1000, top_k=3)
   ```

2. **Documentation** (1 hour):
   - Update README with limitations
   - Add warning in `water_candidate.py` docstring
   - Document expected false positive scenarios

3. **Ship** (immediate):
   - Enable in experimental preset only
   - Monitor telemetry (coverage, confidence, source)

**Time**: **4-6 hours total**

**Pros**:
- ✅ Immediate production deployment (hours, not weeks)
- ✅ Low risk (simple logic, easy to understand)
- ✅ Can iterate based on real usage (production data)
- ✅ Minimal engineering investment

**Cons**:
- ❌ Limited accuracy (better than current stub, worse than PR-W1)
- ❌ May have false positives (blue sky, glass)
- ❌ Not spec-compliant (PR-W1 incomplete)
- ❌ Technical debt (will need to replace later)

**Risk Level**: **Low** (simple, conservative)

**Recommended If**: We need something in production NOW and can iterate later

**Cost-Benefit**:
- **Cost**: 6 hours engineering
- **Benefit**: Production deployment, real-world data collection
- **Risk**: Low (clearly marked as interim solution)

---

## 3. Architect's Recommendation

### **Recommended Path: Data-First Hybrid (Option 4 → Option 3)**

I recommend combining **Option 4 (Data-First)** with **Option 3 (Hybrid Heuristic + SAM)**.

**Why This Path**:
1. **Defensible**: Data-driven design, not guesswork
2. **Measurable**: Validation metrics prove quality before production
3. **Iterative**: Can improve detector based on real failures
4. **Production-Ready**: Know quality before shipping
5. **Sustainable**: Foundation for long-term improvement

---

### Phase 1: Dataset Creation (Week 1)

**Goal**: Create labeled validation dataset to inform detector design.

**Tasks**:
1. **Collect 50-100 diverse images**:
   - **Pool scenes (20-30 images)**:
     - Residential pools (various shapes, tiles, lighting)
     - Luxury resort pools (infinity edges, complex shapes)
     - Lap pools (rectangular, lane markers)
     - Mixed scenes (pool + patio furniture, umbrellas)
   
   - **Ocean scenes (20-30 images)**:
     - Calm ocean (minimal waves, horizon visible)
     - Waves (texture, foam, reflections)
     - Horizon shots (sky + ocean boundary)
     - Mixed scenes (beach + ocean)
   
   - **Non-water scenes (10-20 images)**:
     - Blue sky (clear, clouds)
     - Glass buildings (reflections, blue tint)
     - Foliage (trees, bushes)
     - Stone/concrete (patios, walkways)

2. **Label with ground truth**:
   - Scene type: pool, ocean, non_water
   - Expected water coverage: 0.0-1.0 (rough estimate)
   - Notes: reflections, mixed materials, challenging features

3. **Document characteristics**:
   - What makes pool different from ocean? (geometric shape, tiles, depth uniformity)
   - What causes false positives? (blue sky, glass, blue flowers)
   - What boundary features matter? (tile edges, horizon, pool coping)

**Deliverable**: `data/water_validation_dataset/` with 50-100 labeled images + `ground_truth.json`

**Time**: **1 week** (including collection, labeling, documentation)

---

### Phase 2: Baseline Analysis (Week 2, Days 1-2)

**Goal**: Understand stub detector failure modes.

**Tasks**:
1. **Fix Edge Alignment Metric** (2 hours):
   - Add `water_validation_emit_mask` debug flag to `MaterialsV3Config`
   - Modify `MaterialsV3Engine` to include mask in report when flag enabled
   - Update validation harness to decode and use mask
   - Test edge alignment computation

2. **Run Baseline Validation** (2 hours):
   ```bash
   python scripts/prw_water_validation.py \
       --input-dir data/water_validation_dataset/ \
       --ground-truth data/water_validation_dataset/ground_truth.json \
       --output baseline_validation_report.json
   ```

3. **Analyze Failures** (4 hours):
   - Pool scenes: Where does stub miss water? (reflections, shadows, complex tiles)
   - Ocean scenes: Where does stub miss water? (waves, foam, horizon)
   - Non-water scenes: Where does stub false-positive? (blue sky, glass, blue objects)
   - Edge alignment: How well do boundaries align with image gradients?
   - Stability: How consistent is detection across perturbations?

4. **Document Insights** (2 hours):
   - What features distinguish pool vs ocean vs blue sky?
   - What heuristics would capture these patterns?
   - Is texture cue critical? (smooth water vs textured foliage)
   - Is planarity cue critical? (depth gradient on water surface)
   - What are the failure modes that heuristics can't handle? (requires ML)

**Deliverable**: `docs/WATER_DETECTOR_FAILURE_ANALYSIS.md` with baseline metrics and design insights

**Time**: **1-2 days**

---

### Phase 3: Simplified Hybrid Detector (Week 2, Days 3-5)

**Goal**: Implement simplified multi-cue heuristic + EfficientSAM refinement.

**Tasks**:
1. **Implement Simplified Heuristic Detector** (1.5 days):
   
   **Core Features**:
   - **Chromaticity Cue** (HSV-based):
     ```python
     # Pool: cyan/blue (170-210°), sat > 0.15, val > 0.20
     # Ocean: broader blue-green (160-220°), sat > 0.15, val > 0.20
     ```
   
   - **Specular Boost** (optional, if needed):
     ```python
     # High highlights (val > 0.85) + low saturation (sat < 0.30)
     # Dilate slightly to capture reflection context
     ```
   
   - **Component Filtering**:
     ```python
     # Remove tiny blobs (min_area_px = 1000)
     # Keep top-K largest components (max_components_kept = 3)
     # Fill holes
     ```
   
   - **Confidence Scoring**:
     ```python
     # Based on coverage, hue purity, component stability
     # No ML, just weighted combination of metrics
     ```
   
   **Skip** (not critical for MVP):
   - Texture/entropy cue (nice-to-have, but complex)
   - Planarity cue (requires depth, optional)

2. **Integration** (0.5 day):
   - Replace stub in `lux_depth_v2/water_candidate.py`
   - Ensure fallback logic works (SegFormer → Heuristic → SAM)
   - Add scene context inference (pool vs ocean)

3. **Testing** (0.5 day):
   - Unit tests for chromaticity cue
   - Unit tests for component filtering
   - Integration tests with Materials V3

**Deliverable**: Production-ready simplified heuristic detector

**Time**: **2-3 days**

---

### Phase 4: Validation & Tuning (Week 3)

**Goal**: Validate detector meets quality targets.

**Tasks**:
1. **Run Full Validation** (1 hour):
   ```bash
   python scripts/prw_water_validation.py \
       --input-dir data/water_validation_dataset/ \
       --ground-truth data/water_validation_dataset/ground_truth.json \
       --output validation_report_v1.json
   ```

2. **Analyze Metrics** (4 hours):
   - **Detection Rate** (pool scenes):
     - Target: ≥85%
     - Current: ?
     - Gap: If <85%, why? (missed reflections? shadows? tiles?)
   
   - **False Positive Rate** (non-water scenes):
     - Target: ≤5%
     - Current: ?
     - Gap: If >5%, what's causing FP? (blue sky? glass?)
   
   - **Edge Alignment**:
     - Target: ≥0.6 (60% boundary-gradient overlap)
     - Current: ?
     - Gap: If <0.6, is EfficientSAM refinement helping?
   
   - **Stability**:
     - Target: ≥0.8 (coverage std ≤0.04)
     - Current: ?
     - Gap: If <0.8, what's causing instability?

3. **Tune Thresholds** (1 day):
   - Based on failure analysis, adjust:
     - Hue ranges (pool vs ocean)
     - Saturation/value thresholds
     - Component filtering thresholds
     - Confidence weights
   
   - Re-run validation after each tuning iteration
   - Document threshold rationale

4. **Validate EfficientSAM Refinement** (4 hours):
   - Compare edge alignment with/without SAM refinement
   - Measure refinement overhead (processing time)
   - Document when refinement helps vs when it doesn't

5. **Document Results** (2 hours):
   - Create `docs/WATER_DETECTOR_VALIDATION_REPORT.md`
   - Include metrics, failure analysis, tuning rationale
   - Add visual examples (success cases, failure cases, before/after refinement)

**Deliverable**: Validated detector with documented quality metrics

**Time**: **3-5 days** (including tuning iterations)

---

### Phase 5: Production Deployment (Week 4)

**Goal**: Ship production-ready water detection.

**Tasks**:
1. **Documentation** (1 day):
   - Update `docs/PR_WATER_MASK_STRUCTURE.md` with actual implementation
   - Update README with water detection usage examples
   - Document limitations and known edge cases

2. **Canary Deployment** (1 day):
   - Enable in experimental preset only: `experimental_water_detection`
   - Monitor telemetry (coverage, confidence, source, processing time)
   - Collect production data for future iteration

3. **Gradual Rollout** (1-2 weeks):
   - Week 1: Enable in pool/ocean presets if metrics stable
   - Week 2: Make default if 2 weeks of stable metrics

4. **Production Monitoring** (ongoing):
   - Track telemetry: detection rate, FP rate, edge alignment, stability
   - Collect edge cases for dataset expansion
   - Plan iterative improvements (add texture cue, planarity cue, ML refinement)

**Deliverable**: Production deployment with monitoring

**Time**: **1-2 weeks** (gradual rollout)

---

### Timeline Summary

| Phase | Task | Duration | Cumulative |
|-------|------|----------|-----------|
| 1 | Dataset Creation | 1 week | Week 1 |
| 2 | Baseline Analysis | 1-2 days | Week 2 |
| 3 | Simplified Detector | 2-3 days | Week 2 |
| 4 | Validation & Tuning | 3-5 days | Week 3 |
| 5 | Production Deployment | 1-2 weeks | Week 4-5 |

**Total Time**: **3-5 weeks** to production-validated water detection

**Critical Path**: Dataset creation (Week 1) → Detector implementation (Week 2) → Validation (Week 3)

---

## 4. Alternative Path (If Time-Constrained)

### **Pragmatic Fast Track: Option 5 → Option 4 → Option 3**

If 3-5 weeks is too long, ship improved stub immediately and build proper detector in parallel.

**Week 1**: 
- **Day 1**: Improve stub (6 hours) - HSV-based with scene context
- **Day 2**: Ship in experimental preset, start dataset collection
- **Days 3-5**: Continue dataset creation

**Week 2-3**: 
- Validate stub performance on dataset
- Implement simplified hybrid detector based on learnings
- Run validation, tune thresholds

**Why This Works**:
- ✅ Gets something in production fast (Day 1)
- ✅ Collects real-world data in parallel
- ✅ Can iterate based on production feedback
- ✅ Lower risk (incremental improvement)

**Trade-Off**:
- ⚠️ Ships unvalidated detector initially
- ⚠️ May need to iterate on thresholds after production feedback
- ⚠️ Relies on conservative thresholds to minimize FP risk

---

## 5. Decision Criteria

### Questions to Guide Path Selection

1. **Quality vs Speed**: 
   - Do we need production-quality detection NOW (Option 5)?
   - Or can we wait 3-4 weeks for validated quality (Option 4 → Option 3)?

2. **Data Availability**: 
   - Do we have access to labeled pool/ocean images?
   - Are we willing to invest 1 week in dataset creation?

3. **Accuracy Requirements**: 
   - What's acceptable false-positive rate? (Medical: <1%, Marketing: <10%?)
   - What's minimum detection rate for production? (80%? 90%?)

4. **Computational Budget**: 
   - Can we afford ML inference (50-100ms per image)?
   - Or need heuristic speed (10-20ms)?

5. **Maintenance Resources**: 
   - Will we have resources to tune/improve detector based on production feedback?
   - Can we invest in dataset expansion over time?

### Recommended Decision Matrix

| Scenario | Recommended Path | Rationale |
|----------|-----------------|-----------|
| **High-quality critical** (real estate marketing) | Option 4 → Option 3 | Data-driven, validated metrics |
| **Time-constrained** (ship in 1 week) | Option 5 (improved stub) | Fast, low-risk, iterate later |
| **Best accuracy** (willing to invest) | Option 4 → Option 2 (ML) | ML model, trained on data |
| **Balanced** (quality + speed) | Option 4 → Option 3 | Simplified heuristic + SAM |
| **Future-proof** (long-term investment) | Option 4 → Option 2 (ML) | ML improves over time |

---

## 6. What Makes This "Meaningful"?

From a **System Architect perspective**, meaningful advancement means:

### 1. **Validated Quality**
- ✅ Know actual performance on real data (not synthetic tests)
- ✅ Can prove detector meets targets (detection rate, FP rate, edge alignment)
- ✅ Have documented failure modes and limitations
- ❌ **Not meaningful**: Implementing PR-W1 spec without validation (checkbox engineering)

### 2. **Production Deployment**
- ✅ Running in real workflows (experimental preset → full rollout)
- ✅ Collecting production telemetry (coverage, confidence, performance)
- ✅ Measurable impact (water improves Materials V3 output quality)
- ❌ **Not meaningful**: Shipping stub without knowing failure modes (hope-based engineering)

### 3. **Sustainable**
- ✅ Can maintain/improve over time (tunable thresholds, dataset expansion)
- ✅ Clear path for iteration (add texture cue, upgrade to ML, expand dataset)
- ✅ Documented design rationale (why these features? why these thresholds?)
- ❌ **Not meaningful**: Perfect detector for unrealistic test cases (academic exercise)

### 4. **Defensible**
- ✅ Data-driven decisions (not intuition)
- ✅ Quantified trade-offs (speed vs accuracy, heuristic vs ML)
- ✅ Risk-mitigated (canary deployment, gradual rollout, rollback plan)
- ❌ **Not meaningful**: Guessing at thresholds and shipping (finger-crossing)

---

## 7. Specific Recommendations

### For **High-Priority Production Deployment** (Recommended):

**Choose**: **Data-First Hybrid (Option 4 → Option 3)**

**Action Plan**:
1. **This Week (Week 1)**: Create dataset specification and start collection (50-100 images)
2. **Week 2**: 
   - Fix edge alignment metric (2 hours)
   - Run baseline validation with stub
   - Implement simplified heuristic detector (2-3 days)
3. **Week 3**: 
   - Run full validation
   - Tune thresholds to meet targets
   - Document results
4. **Week 4**: 
   - Production deployment (canary → gradual rollout)
   - Monitor telemetry

**Why**: 
- ✅ Defensible quality (data-proven)
- ✅ Iterative (can improve)
- ✅ Measurable (validation metrics)
- ✅ Production-ready (know what we're shipping)

**Timeline**: **3-4 weeks** to production-validated detector

---

### For **Urgent Deployment** (Time-Constrained):

**Choose**: **Pragmatic Fast Track (Option 5 → Option 4 → Option 3)**

**Action Plan**:
1. **Day 1**: Improve stub (6 hours) - HSV with scene context
2. **Day 2**: Ship in experimental preset, start dataset collection
3. **Week 2-3**: Validate stub, implement proper detector based on learnings

**Why**: 
- ✅ Gets something in production fast (1 day)
- ✅ Collects production data in parallel
- ✅ Can iterate based on real feedback

**Timeline**: **1 day** to experimental deployment, **3 weeks** to validated detector

---

### For **Long-Term Excellence** (ML Investment):

**Choose**: **Data-First ML (Option 4 → Option 2)**

**Action Plan**:
1. **Week 1**: Create labeled dataset (100+ images)
2. **Week 2**: Select/fine-tune segmentation model (SegFormer, DeepLab)
3. **Week 3**: Validate model performance, integrate into pipeline
4. **Week 4+**: Production deployment, continuous improvement

**Why**: 
- ✅ Best possible accuracy
- ✅ Future-proof (improves with more data)
- ✅ Handles complex scenes better than heuristics

**Timeline**: **3-4 weeks** (if dataset available) or **4-5 weeks** (with dataset creation)

---

## 8. Unblocking Actions

### Immediate (This Week)

1. **Fix Edge Alignment Metric** (2 hours):
   - Add `water_validation_emit_mask` debug flag to `MaterialsV3Config`
   - Expose mask in report dict when flag enabled
   - Update validation harness to use mask
   - Test edge alignment computation works

2. **Start Dataset Collection** (ongoing):
   - Collect 50-100 images (pool, ocean, non-water)
   - Document scene characteristics
   - Create ground truth labels

3. **Update Documentation** (1 hour):
   - Add "Known Limitations" section to PR-W4 docs
   - Clarify PR-W1 status (stub, not complete)
   - Document unblocking path

### Short-Term (Week 2-3)

1. **Implement Detector** (based on chosen path):
   - Option 3 (Hybrid): 2-3 days
   - Option 2 (ML): 5 days (if dataset available)

2. **Run Validation** (1 day):
   - Baseline analysis
   - Full validation with new detector
   - Tune thresholds

3. **Document Results** (1 day):
   - Validation report with metrics
   - Failure analysis
   - Tuning rationale

### Medium-Term (Week 4+)

1. **Production Deployment**:
   - Canary preset (experimental only)
   - Monitor telemetry
   - Gradual rollout based on metrics

2. **Iteration**:
   - Collect edge cases
   - Expand dataset
   - Tune detector based on production feedback

---

## 9. Success Metrics

### Validation Targets (Pre-Production)

| Metric | Target | Measurement | Priority |
|--------|--------|-------------|----------|
| Detection Rate (Pool) | ≥85% | % of pool scenes with water detected | **Critical** |
| Detection Rate (Ocean) | ≥80% | % of ocean scenes with water detected | High |
| False Positive Rate | ≤5% | % of non-water scenes with false detection | **Critical** |
| Edge Alignment | ≥0.6 | Boundary-gradient overlap score | **Critical** |
| Stability | ≥0.8 | Coverage variance across perturbations | High |
| Processing Time | ≤50ms | p95 overhead per image | Medium |

### Production Telemetry (Post-Deployment)

- **Coverage Distribution**: Histogram of water coverage across scenes
- **Confidence Distribution**: Histogram of confidence scores
- **Source Breakdown**: % SegFormer vs Heuristic vs None
- **Edge Refinement Rate**: % of detections that trigger SAM refinement
- **Performance**: p50/p95 processing time overhead

---

## 10. Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Heuristic detector insufficient quality | Medium | High | Data-First approach validates before production |
| Dataset not representative | Medium | Medium | Collect diverse scenes, iterate based on production |
| Edge alignment metric unreliable | Low | Low | Multiple validation metrics (coverage, FP rate, stability) |
| EfficientSAM refinement overhead too high | Low | Medium | Tune thresholds to trigger only when needed |
| Production false positives | Medium | High | Conservative thresholds, canary deployment, rollback plan |

### Process Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Dataset creation takes longer than 1 week | Medium | Low | Parallel detector implementation, use smaller dataset initially |
| Validation reveals detector inadequate | Low | High | Data-First approach allows pivot to ML if needed |
| Production rollout blocked by stakeholders | Low | Medium | Validation report with metrics, canary deployment reduces risk |

### Rollback Plan

**Single-Flag Rollback**:
```python
# In MaterialsV3Config
water_detection_enabled: bool = False  # Instant rollback
```

**Gradual Rollback**:
- Week 1: Disable in default presets, keep in experimental
- Week 2: Analyze telemetry, fix issues
- Week 3: Re-enable with fixes

---

## 11. Conclusion

### Current State Summary

- ✅ **Infrastructure**: Production-ready (observability, integration, refinement, validation)
- ⚠️ **Detector**: Stub only (simple blue threshold)
- ⚠️ **Validation**: Metrics infrastructure ready, primary metric blocked
- ❌ **Calibration**: No labeled dataset, thresholds not tuned

### Recommended Path

**Data-First Hybrid (Option 4 → Option 3)**:
1. Create labeled dataset (Week 1)
2. Analyze stub failures (Week 2)
3. Implement simplified heuristic + SAM (Week 2-3)
4. Validate and tune (Week 3)
5. Production deployment (Week 4+)

**Timeline**: **3-4 weeks** to production-validated water detection  
**Risk**: **Low** (data-driven, validated, iterative)  
**Cost**: **1 week dataset + 1 week detector + 1 week validation**

### Alternative If Time-Constrained

**Pragmatic Fast Track (Option 5 → Option 4 → Option 3)**:
1. Improve stub (Day 1) - 6 hours
2. Ship in experimental preset (Day 2)
3. Collect dataset in parallel (Week 1)
4. Build proper detector based on data (Week 2-3)

**Timeline**: **1 day** to experimental, **3 weeks** to validated

### Key Success Factors

1. **Dataset Quality**: Representative samples, diverse scenes, accurate labels
2. **Validation Rigor**: Run full harness, analyze failures, tune thresholds
3. **Iterative Approach**: Ship → measure → improve → repeat
4. **Conservative Deployment**: Canary → gradual rollout → default
5. **Production Monitoring**: Telemetry → edge cases → dataset expansion

### What Makes This Meaningful

This approach is meaningful because it:
- ✅ **Validates before shipping** (data-proven quality)
- ✅ **Measures what matters** (detection rate, FP rate, edge alignment)
- ✅ **Enables iteration** (production telemetry → improvements)
- ✅ **Mitigates risk** (canary deployment, rollback plan)
- ✅ **Builds foundation** (dataset → ML future-proofing)

---

## Appendices

### Appendix A: Spec vs Reality Comparison

| Component | Specified (PR-W1) | Implemented (Stub) | Gap |
|-----------|------------------|-------------------|-----|
| Chromaticity cue | HSV/Lab, pool vs ocean | Blue > red/green | Missing scene tuning, Lab |
| Specular cue | Highlights + low sat | None | Missing |
| Texture cue | Entropy/frequency | None | Missing |
| Planarity cue | Depth gradient | None | Missing |
| Combination | Weighted | Boolean AND | Missing weights |
| Post-processing | Morphology, holes | None | Missing |
| Component filter | Top-K, min area | None | Missing |

### Appendix B: Validation Metrics Detail

**Edge Alignment Computation**:
```python
def _compute_edge_alignment(rgb01: np.ndarray, mask: np.ndarray) -> float:
    """
    Primary metric: edge alignment vs image gradients.
    High score = mask boundaries align with image edges.
    """
    # Compute image gradients
    gray = np.mean(rgb01, axis=2)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Extract mask boundary
    boundary = dilate(mask) & ~erode(mask)
    
    # Measure overlap with high-gradient regions
    high_grad = grad_mag >= percentile(grad_mag, 75)
    overlap = sum(boundary * high_grad)
    
    return overlap / sum(boundary)
```

**Stability Computation**:
```python
def _compute_stability(rgb01: np.ndarray) -> float:
    """
    Stability across perturbations.
    High score = consistent detection.
    """
    baseline_coverage = detect(rgb01).coverage
    
    # Perturbation 1: resize 95%
    resized_coverage = detect(resize(rgb01, 0.95)).coverage
    
    # Perturbation 2: add noise
    noisy_coverage = detect(rgb01 + noise).coverage
    
    # Low variance = high stability
    std = np.std([baseline_coverage, resized_coverage, noisy_coverage])
    return 1.0 - min(std * 5, 1.0)
```

### Appendix C: Quick Reference

**Next Steps Checklist**:
- [ ] Decide on path (Data-First Hybrid recommended)
- [ ] Fix edge alignment metric (2 hours)
- [ ] Start dataset collection (1 week)
- [ ] Run baseline validation (1 day)
- [ ] Implement simplified detector (2-3 days)
- [ ] Validate and tune (3-5 days)
- [ ] Production deployment (1-2 weeks)

**Key Files**:
- Spec: `docs/PR_WATER_MASK_STRUCTURE.md`
- Stub: `lux_depth_v2/water_candidate.py`
- Integration: `lux_depth_v2/materials_v3.py`
- Validation: `scripts/prw_water_validation.py`
- Tests: `tests/test_prw_water_validation.py`

**Contacts/Resources**:
- Dataset collection: [Define process/team]
- Validation review: [Define stakeholder]
- Production deployment: [Define approval process]

---

**End of Strategic Assessment**

*Prepared by Transformation Portal Architect*  
*Date: 2025-12-14*  
*Next Review: After dataset creation (Week 1)*
