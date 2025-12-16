# Water Detection - Next Steps Roadmap

**Date**: 2025-12-15  
**Current State**: Baseline governance established, path resolution complete  
**Baseline**: baseline_ci_current_v1.json (83.3% pool recall, 0% FT rate)  
**Status**: ✅ Clean, reproducible state with known limitations

---

## Current Known Limitations

### 1. Pool Recall: 83.3% (5/6 detected)
- **pool_0008**: Missed due to low confidence (0.2546 < 0.4 injection threshold)
- **Root cause**: Low-saturation pool with tile grid patterns
- **Threshold geometry**: Narrow operating window (highest negative 0.375, lowest positive 0.437)

### 2. Single-Scale Grid Detection
- Cannot distinguish high-frequency pool tiles from low-frequency architectural glass
- Current thresholds: alignment=0.15, grid=0.25, penalty=0.6 (safe, conservative)
- Multi-scale fix designed (GLASS_SUPPRESSOR_MULTISCALE_FIX.md) but not implemented

### 3. Limited Telemetry Visibility
- Validation JSON lacks suppressor-level telemetry
- Cannot explain false negatives/near-misses from artifacts alone
- Requires "direct test vs harness" debugging (slow, error-prone)

---

## Recommended Sequencing (Phases A-D)

### Phase A: Lock Observability (Make CI Signal-Rich)

**Goal**: Every false negative/near-miss becomes explainable from one artifact

#### 1. Export Suppressor Telemetry into Validation JSON

**Add to ValidationResult** (scripts/prw_water_validation.py):
- `suppressors_applied: List[str]` - Which suppressors fired (e.g., ["flat_surface", "architectural_glass"])
- `glass_detector: Dict` - Glass suppressor metrics:
  - `alignment_score: float` (0.0-1.0)
  - `grid_score: float` (0.0-1.0)
  - `grid_score_coarse: float` (0.0-1.0, when multi-scale implemented)
  - `grid_persistence_ratio: float` (0.0-2.0+, when multi-scale implemented)
  - `tile_exempted: bool` (when multi-scale implemented)
- `flat_surface_detector: Dict` - Flat surface metrics:
  - `edge_energy: float`
  - `specular_fraction: float`
  - `has_structure: bool`

**Implementation**:
```python
# In prw_water_validation.py, ValidationResult dataclass:
@dataclass
class ValidationResult:
    # ... existing fields ...
    suppressors_applied: List[str] = field(default_factory=list)
    glass_detector: Optional[Dict[str, Any]] = None
    flat_surface_detector: Optional[Dict[str, Any]] = None
```

**Why**: Eliminates "direct test vs harness" divergence - telemetry shows why confidence dropped

**Acceptance**: Regenerate baseline_ci_current_v1.json with telemetry fields, verify pool_0008 shows glass suppressor metrics

---

#### 2. Add Structured "Error Artifact" Path

**Current behavior**: Harness failure → no outputs/water_validation_current.json → regression check warns "missing file"  
**Problem**: Warn-only job becomes "no-signal" on harness failure

**Fix** (in .github/workflows/ci-consolidated.yml):
```yaml
- name: Run Water Validation Harness
  continue-on-error: true
  run: |
    set +e
    python scripts/prw_water_validation.py \
      --ground-truth data/water_v0/ground_truth.json \
      --subset-file data/water_v0/ci_subset.txt \
      --output outputs/water_validation_current.json \
      --seed 42
    ec=$?
    if [ $ec -ne 0 ]; then
      echo "::warning::Harness failed (exit=$ec)"
      # Write structured error artifact
      echo "{\"error\": \"harness_failed\", \"exit_code\": $ec, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > outputs/water_validation_current.json
    fi
    exit 0
```

**Why**: Job remains warn-only but always produces signal (regression check can distinguish "detector regressed" vs "harness didn't run")

**Acceptance**: CI run with intentional harness failure produces structured error artifact

---

### Phase B: Add Real Holdout Set (Calibration Guardrails)

**Goal**: Prevent overfitting to tiny synthetic fixture set (2 negatives)

#### 3. Create Private Holdout Pack (10-20 Negatives)

**Focus on known confusers**:
- Architectural glass façades (3-5 images)
- Blue painted walls (2-3 images)
- Reflective stone/concrete (2-3 images)
- Skylight reflections (1-2 images)
- Pool tiles (close-up, grid-like) (2-3 images)
- Ocean horizon glare (1-2 images)

**Storage strategy**:
- **DO NOT commit images to git** (large binaries)
- Store hash manifest: `data/water_v0/holdout_manifest.json`:
  ```json
  {
    "holdout_version": "v1",
    "images": [
      {
        "filename": "glass_facade_001.jpg",
        "sha256": "abc123...",
        "label": "negative",
        "tags": ["architectural_glass", "real_world"]
      },
      ...
    ]
  }
  ```
- Load via local path or environment variable in calibration runs:
  ```bash
  export WATER_HOLDOUT_DIR=/path/to/private/holdout/images
  python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/holdout_manifest.json \
    --output holdout_validation.json
  ```

**Why**: Prevents threshold tuning to just 2 synthetic negatives (overfitting risk)

**Acceptance**: Holdout validation run produces results, no images committed to repo

---

#### 4. Define Acceptance Gates for Baseline v2

**CI Fixtures** (14 synthetic images):
- Pool recall: ≥83.3% (maintain current, ideally 100%)
- Ocean recall: 100% (maintain)
- False trigger rate: 0% (maintain - critical)

**Holdout Negatives** (10-20 real negatives):
- False trigger rate: ≤5% (at most 1 trigger on 20 negatives)
- Justification: Real-world tolerance for rare false positives

**Positives** (6 pool + 6 ocean fixtures):
- Preserve 6/6 pool + 6/6 ocean on CI fixtures
- Ideal: Recover pool_0008 (100% pool recall)

**Version Promotion Policy**:
- Regenerate baseline_ci_current_v2.json ONLY when:
  - CI fixtures: 100% pool, 100% ocean, 0% FT
  - Holdout: ≤5% FT rate
  - Telemetry shows explainable behavior (no mystery suppressions)
- Update CI to enforce v2, freeze v1 as historical reference

**Why**: Explicit acceptance criteria prevent premature promotion, require proof of improvement

**Acceptance**: ADR updated with baseline v2 promotion policy, CI workflow references v2 only after gates met

---

### Phase C: Implement Multi-Scale Glass Fix (After Telemetry + Holdout)

**Goal**: Recover pool_0008 (100% pool recall) without reintroducing false triggers

#### 5. Implement Multi-Scale Logic Behind Feature Flag

**Add to MaterialsV3Config** (lux_depth_v2/materials_v3.py):
```python
# PR-W5: Multi-scale glass suppressor (opt-in, experimental)
glass_multiscale_enabled: bool = False  # Master gate (default OFF)
glass_multiscale_downsample_factor: int = 4  # 1/4 scale
glass_tile_persistence_threshold: float = 0.8  # <0.8 = high-freq tiles
```

**Implementation** (lux_depth_v2/water_candidate.py):
- Compute grid_score at full resolution (existing)
- If `glass_multiscale_enabled`:
  - Downsample image/mask to 1/4 scale
  - Recompute grid_score_coarse
  - Calculate grid_persistence_ratio = grid_score_coarse / grid_score
  - Set tile_exempted = True if persistence_ratio < 0.8
  - Suppress ONLY if not tile_exempted
- Export telemetry: grid_score_coarse, grid_persistence_ratio, tile_exempted

**Calibration runs**:
```bash
# Test with multi-scale enabled
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --output test_multiscale_on.json \
  --config-override glass_multiscale_enabled=true \
  --seed 42

# Compare before/after
python scripts/check_regression.py \
  --baseline baseline_ci_current_v1.json \
  --current test_multiscale_on.json \
  --mode warning
```

**Proof of improvement** (required before enabling by default):
- CI fixtures: pool_0008 detected (tile_exempted=True in telemetry)
- CI fixtures: negatives still rejected (tile_exempted=False in telemetry)
- Holdout: ≤5% FT rate maintained

**Why**: Feature flag allows controlled testing, telemetry proves behavior, no surprise regressions

**Acceptance**: Multi-scale logic recovers pool_0008, preserves negatives, passes holdout gates

---

#### 6. Regenerate and Promote Baseline v2

**Only after**:
- ✅ Multi-scale fix proven with telemetry
- ✅ CI fixtures pass acceptance gates (100%/100%/0%)
- ✅ Holdout validation passes (≤5% FT rate)

**Promotion workflow**:
```bash
# Enable multi-scale by default in config
# (or create baseline_ci_current_v2.json with glass_multiscale_enabled=true)

# Regenerate baseline
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --subset-file data/water_v0/ci_subset.txt \
  --output data/water_v0/baseline_ci_current_v2.json \
  --config-override glass_multiscale_enabled=true \
  --seed 42

# Verify metrics
jq '.summary | {pool_recall, ocean_recall, false_trigger_rate}' \
  data/water_v0/baseline_ci_current_v2.json
# Expected: 1.0, 1.0, 0.0

# Freeze v1 as historical
git mv data/water_v0/baseline_ci_current_v1.json \
       data/water_v0/baseline_ci_historical_v1.json

# Promote v2 to current
git add data/water_v0/baseline_ci_current_v2.json
git commit -m "feat(water): promote baseline v2 (100% pool recall via multi-scale)"

# Update CI to enforce v2
sed -i 's/baseline_ci_current_v1/baseline_ci_current_v2/' \
  .github/workflows/ci-consolidated.yml
```

**Why**: Explicit promotion ceremony prevents accidental baseline drift

**Acceptance**: CI enforces v2, v1 frozen as historical, multi-scale enabled by default

---

### Phase D: ADE20K Integration (Optional, Strategic Upgrade)

**Goal**: Expand recall robustness beyond synthetic fixtures (real-world edge cases)

#### 7. Integrate ADE20K as Optional Semantic Prior

**Approach** (from docs/PR_W1.3_ADE20K_INTEGRATION.md design):
- **Offline only**: No downloads in CI (keep CI deterministic)
- **Optional backend**: Default disabled, opt-in via config flag
- **Semantic labels**: Union whitelist (water/sea/river/lake/swimming_pool) → unified mask
- **Confidence fusion**: Semantic prior + heuristic → unified confidence
- **Use case**: Offline calibration, real-world robustness testing, failure-mode coverage

**Implementation** (after Phase C complete):
- Add ADE20K SegFormer backend (opt-in)
- Create offline calibration workflow
- Validate with real-world test images (not in CI)
- Document as strategic recall upgrade (not baseline dependency)

**Why**: ADE20K adds robustness for real-world diversity, but AFTER threshold/suppressor problems closed

**Acceptance**: ADE20K backend works offline, CI remains deterministic (no ADE20K dependency)

---

## Phase Dependencies

```
Phase A (Telemetry)
    ↓
Phase B (Holdout Set)
    ↓
Phase C (Multi-Scale Fix)  → Requires A + B complete
    ↓
Phase D (ADE20K)           → Requires C complete (optional)
```

**Critical path**: A → B → C (telemetry, holdout, multi-scale)  
**Optional upgrade**: D (ADE20K after C)

---

## Current Blockers (Ordered by Priority)

1. **No suppressor telemetry in validation JSON** → Phase A.1 (highest priority)
2. **No holdout set** → Phase B.3 (prevents overfitting validation)
3. **Multi-scale not implemented** → Phase C.5 (requires A + B first)

---

## Success Criteria (Overall)

**Baseline v2 Acceptance**:
- ✅ Pool recall: 100% (6/6, including pool_0008)
- ✅ Ocean recall: 100% (6/6)
- ✅ CI false trigger rate: 0% (0/2)
- ✅ Holdout false trigger rate: ≤5% (≤1/20)
- ✅ Telemetry explains all suppressions (no mystery behavior)
- ✅ Safe thresholds maintained (0.15/0.25/0.6 or justified changes with holdout proof)

**Governance Maintained**:
- ✅ Baseline promotion requires ADR approval
- ✅ No threshold tuning without holdout validation
- ✅ Multi-scale changes behind feature flag until proven
- ✅ CI remains deterministic (no external downloads)

---

## Timeline Estimate

**Phase A** (Telemetry): 1-2 days  
**Phase B** (Holdout): 2-3 days (image collection + manifest creation)  
**Phase C** (Multi-Scale): 3-5 days (implementation + validation)  
**Phase D** (ADE20K): 5-7 days (optional, after C)

**Total to baseline v2**: ~7-10 days (A + B + C, assuming no major blockers)

---

## References

- **ADR-001**: Baseline Governance Policy (`docs/architecture/ADR-001-BASELINE-GOVERNANCE.md`)
- **Multi-Scale Design**: Glass Suppressor Fix (`GLASS_SUPPRESSOR_MULTISCALE_FIX.md`)
- **ADE20K Design**: Optional Semantic Prior (`docs/PR_W1.3_ADE20K_INTEGRATION.md` - archived)
- **Current Baseline**: `data/water_v0/baseline_ci_current_v1.json` (83.3% pool recall)

---

**Status**: ✅ Roadmap defined with clear phases, dependencies, and acceptance criteria  
**Next Action**: Begin Phase A.1 (export suppressor telemetry to validation JSON)  
**Date**: 2025-12-15  
**Approved By**: Governance cleanup session
