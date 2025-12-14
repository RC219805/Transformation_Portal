# PR-4D Data Collection Plan: Wood Material Pixel Ops

**Date**: 2025-12-14  
**Purpose**: Collect histogram data from diverse scenes to inform wood pixel ops implementation  
**Previous**: PR-4C merged successfully with schema v3.1 verified

---

## Overview

PR-4C introduced decision separation and reason histograms. Now we collect real-world data to select and implement the next material for pixel ops (likely wood).

---

## Workflow

### Step 1: Data Collection (~5-10 minutes)

Run the batch collection script:

```bash
bash scripts/pr4d_collect_wood_histograms.sh
```

**What it does**:
- Processes 4 scenes from 750 Picacho Lane (Kitchen, PrimaryBedroom, GreatRoom, Pool)
- Uses canary preset: `interior_luxury_apex_quality_materials_v3_glass`
- Smart caching: skips Kitchen if already processed
- Outputs to `outputs/pr4d_wood_data/<scene_name>/`
- Generates `pr4d_collection_summary.txt`

**Expected Processing Time**:
- Kitchen: ~10s (cached if already run)
- PrimaryBedroom: ~10s
- GreatRoom: ~12s (larger)
- Pool: ~10s (exterior)
- **Total**: ~42s + model loading overhead (~30s first run)

---

### Step 2: Histogram Aggregation (~5 seconds)

Run the analysis script:

```bash
python scripts/pr4d_aggregate_histograms.py
```

**What it does**:
- Reads all `*_report.json` files from `outputs/pr4d_wood_data/`
- Validates schema v3.1 presence
- Aggregates `pixel_ops_reasons` and `refinement_reasons` histograms
- Calculates per-material statistics (frequency, coverage, edge signals)
- Ranks materials by implementation priority

**Outputs**:
1. `outputs/pr4d_histogram_aggregate.json` - Raw aggregated data
2. `outputs/pr4d_material_recommendations.md` - Ranked table + analysis

---

### Step 3: Review Recommendations

Read the generated analysis:

```bash
cat outputs/pr4d_material_recommendations.md
```

**Expected Format**:

```markdown
# PR-4D Material Selection Analysis

## Material Statistics

| Material | Frequency | Avg Coverage | Boundary Px | Edge Align | Score | Recommendation |
|----------|-----------|--------------|-------------|------------|-------|----------------|
| wood     | 4/4       | 12.3%        | 1250        | 0.28       | 95    | ⭐ Top Choice  |
| stone    | 4/4       | 8.5%         | 890         | 0.22       | 78    | Alternative    |
| ...      | ...       | ...          | ...         | ...        | ...   | ...            |

## Recommendation: Wood

**Rationale**:
- ✅ High frequency (appears in all scenes)
- ✅ Good coverage (sufficient sample size)
- ✅ Strong edge signals (stable boundaries)
- ✅ Clear need (`no_implementation` in all scenes)
- ✅ Low halo risk (validated in interior scenes)

**Next Steps**: [PR-4D implementation plan]
```

---

## Selected Scenes

### 1. Kitchen (Interior - Controlled Lighting)
**Wood Content**: Cabinets, millwork, island trim  
**Expected**: High wood coverage, stable boundaries  
**Status**: ✅ Already processed in PR-4C validation

### 2. Primary Bedroom (Interior - Warm Lighting)
**Wood Content**: Flooring, furniture, accent wall, built-ins  
**Expected**: Highest wood coverage, mixed lighting  
**Good For**: Testing warm-tone wood response

### 3. Great Room (Interior - Architectural)
**Wood Content**: Millwork, ceiling beams, built-in shelving  
**Expected**: Large architectural wood elements  
**Good For**: Testing structural wood enhancement

### 4. Pool (Exterior - Natural Light)
**Wood Content**: Deck, pergola, soffits  
**Expected**: Exterior wood under full sunlight  
**Good For**: Testing wood in high-contrast outdoor conditions

---

## Excluded Scenes

### Aerial (Low Wood Coverage)
**Reason**: Distant roof details only, not representative  
**Wood Content**: <1% coverage  
**Decision**: Skip - insufficient sample

### Primary Bathroom (OOM Risk)
**Reason**: Large image size (OOM on MPS), stone-dominant  
**Wood Content**: Vanity only (~2% coverage)  
**Decision**: Skip - risk/reward unfavorable

---

## Expected Histogram Patterns

Based on PR-4C Kitchen results, we expect:

### Pixel Ops Reasons (Aggregated)
```json
{
  "no_implementation": 20,  // 5 materials × 4 scenes
  "below_coverage_threshold": 4-8
}
```

### Refinement Reasons (Aggregated)
```json
{
  "below_coverage_threshold": 12-16,
  "not_in_canary_set": 8
}
```

### Per-Material Expectations

| Material | Frequency | no_impl Count | Notes |
|----------|-----------|---------------|-------|
| **wood** | 4/4 | 4 | ⭐ Top candidate |
| **stone** | 4/4 | 4 | Alternative (countertops, tile) |
| **metal** | 3/4 | 3 | Lower priority (appliances, fixtures) |
| **glass** | 2/4 | 0 | Already implemented (PR-4B) |
| **sky** | 2/4 | 0 | `not_in_canary_set` |
| **foliage** | 1/4 | 0 | `not_in_canary_set`, pool only |

---

## Success Criteria

### Data Quality
- ✅ All 4 scenes process successfully
- ✅ Schema v3.1 present in all reports
- ✅ Reason histograms populated (no empty dicts)
- ✅ Edge signals computed for materials with coverage > 1000px

### Material Selection
- ✅ Clear top candidate (frequency 4/4, high coverage)
- ✅ Good edge signals (boundary_pixels >= 250, edge_alignment >= 0.10)
- ✅ High `no_implementation` count (proves need)
- ✅ Not glass (already done), not sky/foliage (canary-only)

### Actionable Output
- ✅ Ranked material table generated
- ✅ Clear recommendation with rationale
- ✅ Next steps documented

---

## Next Steps After Data Collection

### If Wood is Top Candidate (Expected)

**PR-4D Scope** (strict, canary-only):
1. Design wood pixel ops (microcontrast, clarity, warmth preservation)
2. Implement in `lux_depth_v2/materials_v3_pixel_ops.py`
3. Add wood eligibility to `materials_v3_response.py`
4. Create presets:
   - `interior_luxury_apex_quality_materials_v3_wood` (canary)
   - `interior_luxury_apex_quality_materials_v3_wood_validate` (forced-apply)
5. Run two-pass validation (mirror PR-4B):
   - Pass 1: Normal gating (should skip when high-confidence)
   - Pass 2: Forced apply (prove ops correctness + safety)
6. Validate metrics:
   - Halo risk (P95 delta in boundary)
   - Mean delta (overall change magnitude)
   - Gradient change (localized to wood mask)
   - Clamp count (how often safety limits engaged)

**Acceptance**: Same discipline as PR-4B
- ✅ Intelligent skip when confidence high
- ✅ Applied ops pass safety checks (no halos, clamped deltas)
- ✅ CI green
- ✅ Canary-only (not default)

---

## Timeline Estimate

**Data Collection**: 5-10 minutes  
**Analysis**: 5 seconds  
**Review**: 5 minutes  
**Total**: ~20 minutes

**Next Session** (PR-4D Implementation): 2-3 hours
- Wood pixel ops design: 30 min
- Implementation: 1 hour
- Testing + validation: 1-1.5 hours

---

## Files Generated

```
outputs/pr4d_wood_data/
├── Kitchen/
│   └── 750Picacho_Kitchen_UltraQuality_report.json (cached from PR-4C)
├── PrimaryBedroom/
│   └── 750Picacho_PrimaryBedroom_UltraQuality_report.json
├── GreatRoom/
│   └── 750Picacho_GreatRoom_UltraQuality_report.json
├── Pool/
│   └── 750Picacho_Pool_UltraQuality_report.json
└── pr4d_collection_summary.txt

outputs/
├── pr4d_histogram_aggregate.json
└── pr4d_material_recommendations.md
```

---

## Validation Checkpoint

Before proceeding to PR-4D implementation, verify:

```bash
# 1. All scenes processed
ls outputs/pr4d_wood_data/*/750Picacho_*_report.json | wc -l
# Expected: 4

# 2. Schema v3.1 present
jq -r '.materials_v3_response_plan.version' outputs/pr4d_wood_data/*/750Picacho_*_report.json | sort -u
# Expected: v3.1

# 3. Wood detected in all scenes
jq -r '.materials_v3_response_plan.summary.present_classes[] | select(. == "wood")' \
  outputs/pr4d_wood_data/*/750Picacho_*_report.json | wc -l
# Expected: 4 (or close)

# 4. Recommendations generated
test -f outputs/pr4d_material_recommendations.md && echo "✅ Recommendations ready"
```

---

## Session Archive

This plan is part of the PR-4C merge session archive:

```
docs/SESSIONS/2025-12-14_PR4C_MERGE/
├── POST_MERGE_VALIDATION.md       (PR-4C validation results)
└── PR4D_DATA_COLLECTION_PLAN.md   (this file)
```

---

**Status**: ✅ Ready to execute  
**Scripts**: ✅ Production-ready (generated by specialist)  
**Expected Duration**: ~20 minutes total  
**Next Milestone**: PR-4D wood pixel ops implementation
