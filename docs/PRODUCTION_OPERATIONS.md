# Production Operations Manual

**System Status:** Production Standard  
**Effective Date:** 2025-12-04  
**Version:** 1.0

---

## Standard Operating Procedure

This is your **production workflow** for all serious projects. No experimentation - execute the policy.

---

## The Four-Step Process

### Step 1: Ingest & Baseline

**For every new project:**

```bash
cd /Users/rc/Transformation_Portal
source .venv/bin/activate

# Run baseline assessment on all candidate frames
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/{project}/Source_JPEGS/*.jpg" \
  -o "output_{project}_Baseline" \
  -r config/recipes/baseline_quality.yaml
```

**What this tells you:**
- Which images are already hero-grade (≥55%)
- Which need real help (<45%)
- Expected processing time: ~2.5s per image

**Record baseline scores** - these drive all downstream decisions.

---

### Step 2: Consult RAG Advisor

**Before processing ANY frame:**

```bash
python scripts/rag/suggest_recipe.py \
  --scene-type {scene_type} \
  --baseline-score {score} \
  --notes "{short scene description}"
```

**Examples:**

```bash
# Hero bedroom
python scripts/rag/suggest_recipe.py \
  --scene-type interior_bedroom \
  --baseline-score 60.4 \
  --notes "neutral, daylight, premium finishes"

# Weak aerial
python scripts/rag/suggest_recipe.py \
  --scene-type aerial_exterior \
  --baseline-score 42.5 \
  --notes "wide angle, foliage, needs punch"

# Pool scene
python scripts/rag/suggest_recipe.py \
  --scene-type exterior_pool \
  --baseline-score 48.2 \
  --notes "daylight, water reflections"
```

**Decision matrix:**
- **High confidence** → Accept recommendation, proceed
- **Medium confidence** → Test recommended + alternative
- **Low confidence** → Manual decision, document rationale

---

### Step 3: Process with Recommended Recipe

**Execute based on RAG guidance:**

#### For Hero Shots (≥55% baseline)
```bash
# Default: Ship as baseline (no processing)
# Only process if brand consistency absolutely requires it

# If processing required:
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/{project}/{file}.jpg" \
  -o "output_{project}_Processed" \
  -r config/recipes/interior_warm_minimal.yaml

# Then compare and decide
```

**Expected:** Any processing will lose 3-6% quality. Question whether it's worth it.

#### For Good Interiors (45-55% baseline)
```bash
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/{project}/{file}.jpg" \
  -o "output_{project}_Processed" \
  -r config/recipes/signature_estate_gentle.yaml
```

**Expected:** 3-5% quality loss, acceptable for film character.

#### For Weak Exteriors (<45% baseline)
```bash
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/{project}/{file}.jpg" \
  -o "output_{project}_Processed" \
  -r config/recipes/exterior_enhanced.yaml
```

**Expected:** +3-6% quality gain. This is proven.

#### For Pool/Water Scenes
```bash
# Default: Ship as baseline until pool recipe is fixed
# Current pool_estate.yaml is QUARANTINED (-9.5% avg loss)

# If must process:
# - Test signature_estate_gentle first
# - Compare carefully against baseline
# - Document results in run card
```

**Expected:** High-risk. Proceed with extreme caution.

---

### Step 4: Document & Learn

**For any non-trivial decision:**

Create run card in `docs/runs/{project}/`:

```yaml
# Template: docs/runs/{project}/{image_id}_{recipe}.yaml

image_id: ProjectName_SceneName
project: project_name
scene_type: {standardized_type}
scene_features:
  - {feature1}
  - {feature2}

source_baseline_score: {baseline}
processed_score: {processed}
delta_score: {delta}
targets_met: X/4

recipe: {recipe_name}
recipe_settings:
  lut_strength: {value}
  contrast: {value}
  # ...

human_rating: {clearly_better|acceptable_but_unnecessary|worse_than_source|...}
decision: {recipe_recommended|recipe_avoid|...}

notes:
  - "{observation1}"
  - "{observation2}"

lessons:
  - "{learning1}"
  - "{learning2}"

tags:
  - {tag1}
  - {tag2}
```

**When to create run card:**
- ✅ New scene type
- ✅ New recipe variant
- ✅ Counterintuitive result
- ✅ Different outcome than RAG predicted
- ✅ Client feedback that changes your assessment

**When NOT to create:**
- ❌ Routine processing with expected results
- ❌ Quick tests with no clear outcome

**Time investment:** 60 seconds of judgment after automation (see roadmap).

---

## Default Recipes by Scene Type

### Interiors

| Baseline Score | Recipe | Expected Δ | Confidence |
|----------------|--------|------------|------------|
| ≥60% | baseline (no processing) | 0% | very_high |
| 55-60% | baseline OR interior_warm_minimal | -2 to -6% | high |
| 50-55% | signature_estate_gentle | -3 to -5% | medium |
| 45-50% | signature_estate_gentle | -3 to -5% | medium |
| <45% | signature_estate | +2 to +4% | low |

**Key insight:** Hero interiors resist ALL processing. Ship them as-is.

### Exteriors / Aerials

| Baseline Score | Recipe | Expected Δ | Confidence |
|----------------|--------|------------|------------|
| ≥55% | exterior_enhanced OR signature_estate | +0 to +3% | medium |
| 45-55% | exterior_enhanced | +3 to +5% | high |
| <45% | exterior_enhanced | +5 to +7% | very_high |

**Key insight:** Low-baseline exteriors benefit significantly from enhancement.

### Pool / Water

| Baseline Score | Recipe | Expected Δ | Confidence |
|----------------|--------|------------|------------|
| Any | baseline (no processing) | 0% | high |
| Experimental | signature_estate_gentle | -3 to -5% | low |

**Status:** ⚠️ Pool-specific recipe QUARANTINED. Default to baseline.

---

## Hard Gates (Never Override)

### 1. Quarantined Recipes

**Current Status:**
- `pool_estate.yaml` - QUARANTINED (avg -9.5% on tested cases)

**Policy:**
- Never auto-recommend quarantined recipes
- Manual use requires:
  - Explicit acknowledgment of risk
  - Side-by-side comparison with baseline
  - Run card documentation
  - Visual review before delivery

**Exit criteria:**
- 3+ successful cases (positive human rating)
- Average Δ ≥ -2% vs baseline
- No catastrophic failures

### 2. Hero Shot Protection

**If baseline ≥ 55%:**
- Default recommendation: baseline (no processing)
- Any processing requires:
  - Explicit justification (brand consistency, client request)
  - Visual comparison mandatory
  - Accept that quality loss (3-6%) is inevitable

**Rationale:** Hero shots degrade with ALL recipe variants (measured).

### 3. Pool/Water Scene Caution

**Any pool/water scene:**
- Flag as high-risk
- Show historical failure case (750Picacho_Pool: -9.5%)
- Recommend baseline as safe default
- If processing: test gently, compare carefully

---

## Quality Gates

### Acceptable Quality Loss Thresholds

| Scene Type | Baseline Range | Max Acceptable Loss | Requires Review |
|------------|----------------|---------------------|-----------------|
| Interior hero | ≥55% | 2% | Any processing |
| Interior good | 45-55% | 5% | >4% loss |
| Exterior weak | <45% | N/A (expect gain) | Negative gain |
| Pool/water | Any | 0% | Any processing |

**Review process:**
1. Side-by-side comparison
2. Check against baseline
3. Question: "Is processed clearly better, or just different?"
4. If doubt exists: ship baseline

---

## Scene Type Taxonomy (STANDARDIZED)

**Use exactly these labels** for consistency:

### Interiors
- `interior_bedroom`
- `interior_great_room`
- `interior_kitchen`
- `interior_bathroom`
- `interior_living_room`
- `interior_dining_room`
- `interior_office`
- `interior_closet`

### Exteriors
- `exterior_pool`
- `exterior_garden`
- `exterior_courtyard`
- `exterior_terrace`
- `exterior_facade`
- `aerial_exterior`

### Special
- `twilight_exterior`
- `night_interior`

**Rationale:** Consistent labels enable RAG clustering. Drift = noise.

---

## When to Update This Manual

### Add new recipe
1. Test thoroughly (5+ images, variety of baselines)
2. Document in run cards
3. Add to recipe table with expected Δ
4. Update RAG advisor if needed

### Discover new pattern
1. Document in run cards first (3+ cases)
2. If pattern holds, add to policy
3. Update decision matrix
4. Communicate to team

### Client feedback changes assessment
1. Update relevant run card with client verdict
2. If pattern emerges across clients, update policy
3. RAG will incorporate automatically

---

## Troubleshooting

### RAG recommendation doesn't make sense
1. Check baseline score accuracy
2. Verify scene type label is standardized
3. Look at retrieved historical cases
4. If still unclear: manual decision + document

### Processed result worse than expected
1. Check if baseline was hero-grade (≥55%)
2. Review recipe settings
3. Compare to similar historical cases
4. Create run card documenting outcome
5. Consider quarantining recipe if pattern emerges

### No confidence in any recipe
1. Check if it's a new scene type (no historical data)
2. Run 2-3 recipe variants
3. Compare all vs baseline
4. Document extensively in run cards
5. Next similar case will have guidance

---

## Metrics to Track

### Per Project
- Total images processed
- Baseline score distribution
- Recipe usage frequency
- Quality delta distribution
- Delivery rate (processed vs baseline)

### Quarterly
- Run cards created
- New scene types covered
- Recipe success rates
- Quarantined recipes
- RAG advisor accuracy

### Annually
- Total institutional knowledge
- Coverage completeness
- Team adherence to process
- Client satisfaction correlation

---

## Red Flags

**Stop and review if:**
- ❌ Processed result significantly worse than baseline (>5% loss)
- ❌ Recipe recommendation ignored without documentation
- ❌ Hero shot (≥55%) processed without explicit justification
- ❌ Pool scene processed without caution protocol
- ❌ Run card not created for non-trivial decision
- ❌ Team member doesn't understand the policy

**Each red flag requires:**
1. Immediate comparison with baseline
2. Root cause analysis
3. Documentation in run card
4. Process improvement if needed

---

## Communication

### To Clients
"We use a data-driven grading framework that measures quality at every step. Your images are assessed against 100+ prior cases to ensure consistent, high-quality results. We preserve excellence and enhance only where needed."

### To Team
"This is not experimentation - this is execution. The RAG advisor tells you what worked in similar cases. Follow it, or document why you didn't. Every decision teaches the system."

### To Vendors/Partners
"Our grading pipeline is measurable and traceable. Here's the baseline score, the recipe used, and the measured outcome. This is the institutional standard."

---

## Next Production Project: Validation Test

**Before deploying to paying client:**

1. **Select test project:**
   - Ideally: one you already delivered
   - Provides: ground truth comparison

2. **Run the workflow:**
   - Baseline assessment
   - RAG recommendations
   - Process according to policy
   - Compare to what you actually shipped

3. **Answer:**
   - Did RAG recommend what you did?
   - Did metrics align with your judgment?
   - Would you ship the policy output?

4. **Document:**
   - Alignment rate (% matching your decisions)
   - Disagreements and why
   - Adjustments needed
   - Confidence in production deployment

**This validates the system reflects your taste accurately.**

---

## Support

**Questions about:**
- Baseline interpretation → Check `PRODUCTION_POLICY_750_PICACHO.md`
- Recipe selection → Run RAG advisor
- Run card format → See `docs/runs/README.md`
- System architecture → See `FRAMEWORK_SUMMARY.md`

**Issues:**
- System not working → Check `INSTALLATION_COMPLETE.md`
- Unexpected results → Create run card, flag for review
- Policy unclear → Update this document

---

**Status:** This is your production standard. Execute it. Document deviations. The system learns. 📋✨
