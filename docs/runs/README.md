# Run Cards: RAG-Powered Learning System

This directory contains **structured run cards** - machine-readable records of every grading experiment that feed the RAG system's memory.

---

## Purpose

Transform manual grading experience into **retrievable knowledge** that the system can consult for future decisions.

### The Learning Loop

```
1. Run experiment → 2. Record structured data → 3. RAG indexes it → 4. System retrieves for similar cases
```

Every time you test a recipe on an image, you're training the system by example.

---

## Run Card Format

Each run card is a YAML file containing:

### Required Fields
```yaml
# Identity
image_id: 750Picacho_Pool
project: 750_picacho_lane
scene_type: pool_exterior
scene_features:
  - swimming_pool
  - water_reflections
  - daylight

# Metrics
source_baseline_score: 50.06
processed_score: 40.53
delta_score: -9.53

# Recipe
recipe: pool_estate
recipe_settings:
  lut_strength: 0.35
  contrast: 0.92
  # ...

# Human Assessment
human_rating: worse_than_source
decision: recipe_avoid_completely

# Learning
lessons:
  - "Pool scenes need extremely light touch"
  - "Current recipe over-processes"

tags:
  - pool
  - high_risk_scene
  - needs_revision
```

---

## Human Rating Scale

Use consistent vocabulary:

| Rating | Meaning | Use When |
|--------|---------|----------|
| `clearly_better` | Obvious improvement | Processed beats baseline significantly |
| `acceptable_but_unnecessary` | OK but not worth it | Slight character added, quality cost unclear |
| `different_not_better` | Lateral move | Film look added but quality degraded equally |
| `worse_than_source` | Quality loss | Baseline was better, ship as-is |
| `significantly_worse` | Major degradation | Recipe failed, needs revision |

---

## Decision Codes

| Code | Meaning |
|------|---------|
| `recipe_strongly_recommended` | Use with confidence |
| `recipe_preferred` | Good default choice |
| `recipe_acceptable` | OK if needed, review results |
| `recipe_not_recommended` | Avoid, use alternative |
| `recipe_avoid_completely` | Do not use, recipe failed |
| `recipe_needs_revision` | Shelve until fixed |

---

## Tags Vocabulary

Use consistent tags for retrievability:

### Scene Types
- `interior`, `exterior`, `aerial`
- `bedroom`, `kitchen`, `great_room`, `bathroom`
- `pool`, `water_features`

### Quality Indicators
- `hero_shot`, `high_baseline`, `low_baseline`
- `pristine_source`, `needs_enhancement`

### Outcomes
- `enhancement_success`, `preservation_success`
- `over_processed`, `under_processed`
- `quality_loss`, `quality_gain`

### Recipe Status
- `proven_recipe`, `experimental`
- `recipe_failed`, `needs_revision`
- `counterintuitive_result`, `critical_learning`

---

## Usage: RAG Recipe Advisor

Query the system for recommendations:

```bash
python scripts/rag/suggest_recipe.py \
    --scene-type interior_bedroom \
    --baseline-score 60.4 \
    --notes "neutral, daylight, high-end staging"
```

The system will:
1. Query RAG for similar past experiments
2. Retrieve relevant run cards
3. Recommend recipe based on historical outcomes
4. Show confidence and expected delta

---

## Best Practices

### After Every Serious Experiment

1. **Create run card immediately** while fresh
2. **Be honest** in human ratings (the system learns from truth)
3. **Document counterintuitive results** (these are valuable)
4. **Tag comprehensively** for future retrievability

### Writing Good Lessons

✅ **Good:**
- "Hero shots (≥55%) resist ALL processing variants"
- "Exterior Enhanced proven for low-baseline aerials (+5-6%)"
- "Pool recipe over-processes - needs LUT 0.25-0.30 max"

❌ **Bad:**
- "This didn't work"
- "Looks weird"
- "Try something else"

### Continuous Improvement

As you process more projects:
- More scene types covered
- More recipe variants tested
- More corner cases documented
- RAG retrieval becomes more accurate

---

## Directory Structure

```
docs/runs/
├── README.md (this file)
├── 750_picacho/
│   ├── 750Picacho_Aerial_exterior_enhanced.yaml
│   ├── 750Picacho_Pool_pool_estate.yaml
│   ├── 750Picacho_PrimaryBedroom_signature_estate_gentle.yaml
│   └── 750Picacho_GreatRoom_interior_warm_minimal.yaml
├── montecito_shores/  (future project)
├── seaview_estate/    (future project)
└── ...
```

Each project gets its own subdirectory with structured run cards.

---

## Integration Points

### Current
- ✅ Run cards indexed by RAG system
- ✅ RAG advisor CLI (`suggest_recipe.py`)
- ✅ Manual query and decision

### Planned
- ⏳ Automatic run card generation from pipeline output
- ⏳ Pipeline integration for auto-suggestion
- ⏳ Comparison report generator
- ⏳ Client preview workflow with RAG insights

---

## Example Query Flow

```
User: Process new bedroom shot (baseline: 58%)

1. Run baseline_quality.yaml → 58% score
2. Call suggest_recipe.py:
   --scene-type interior_bedroom
   --baseline-score 58.0

3. RAG retrieves:
   - 750Picacho_PrimaryBedroom (60.4% → 55.78%, -4.62%)
   - 750Picacho_GreatRoom (57.77% → 51.92%, -5.85%)
   - Human ratings: "worse_than_source", "unnecessary"

4. System recommends:
   → baseline (no processing)
   → Confidence: high
   → Reasoning: "Similar hero shots (≥55%) degraded with ALL recipes"

5. User ships baseline or tests minimal variant
6. Records outcome in new run card
7. RAG learns from this decision
```

---

## Status

**Current Coverage:**
- 4 run cards (750 Picacho project)
- Scene types: interior (bedroom, great room), aerial, pool
- Recipes tested: 5 variants
- Baseline range: 42-60%

**Learning Highlights:**
- Hero shots (≥55%) should ship as baseline
- Exteriors benefit from enhancement (+3-6%)
- Pool scenes high-risk (needs revision)
- Minimal processing ≠ minimal damage

**Next Projects:**
Each new property adds experience and improves RAG recommendations.

---

**Bottom Line:** This directory is the system's **memory**. The more structured run cards you create, the smarter it becomes at routing decisions. You're not just processing images - you're training a grading advisor.
