# Quick Start Guide - Daily Operations

**For:** Getting back into production after closing terminal  
**Time:** < 30 seconds to working state

---

## One-Command Startup

```bash
/Users/rc/Transformation_Portal/tp_start.sh
```

This gets you:
- ✅ Right directory
- ✅ Virtual environment activated
- ✅ Quick command reference
- ✅ Ready to work

---

## Manual Startup (If Preferred)

```bash
# 1. Navigate to repo
cd /Users/rc/Transformation_Portal

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Sanity check (optional)
python --version  # Should show 3.11.14
which python      # Should be in .venv/
```

---

## The Four-Step Workflow

### 1️⃣ Baseline Assessment (Always First)

```bash
# Replace PROJECT with actual project name
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/Source_JPEGS/*.jpg" \
  -o "output_PROJECT_Baseline" \
  -r config/recipes/baseline_quality.yaml
```

**Output:** Quality scores for all images (~2.5s per image)

**Record scores** - these drive all decisions.

---

### 2️⃣ Consult RAG Advisor

```bash
# For each key frame, ask what to do
python scripts/rag/suggest_recipe.py \
  --scene-type {TYPE} \
  --baseline-score {SCORE} \
  --notes "{description}"
```

**Scene types:**
- `interior_bedroom`
- `interior_great_room`
- `interior_kitchen`
- `interior_bathroom`
- `exterior_pool`
- `aerial_exterior`

**Example:**
```bash
python scripts/rag/suggest_recipe.py \
  --scene-type interior_bedroom \
  --baseline-score 60.4 \
  --notes "neutral, daylight, premium finishes"
```

**Decision:**
- High confidence → Use suggested recipe
- Medium confidence → Test + compare
- Low confidence → Manual judgment + document

---

### 3️⃣ Process with Recipe

#### Hero Shots (Baseline ≥55%)
```bash
# DEFAULT: Ship as baseline (no processing)
# Only process if brand consistency absolutely requires it

# If must process:
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/hero_shot.jpg" \
  -o "output_PROJECT_Test" \
  -r config/recipes/interior_warm_minimal.yaml

# Then compare vs baseline and decide
```

#### Good Interiors (45-55%)
```bash
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/interior.jpg" \
  -o "output_PROJECT_Processed" \
  -r config/recipes/signature_estate_gentle.yaml
```

#### Weak Exteriors (<45%)
```bash
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/aerial.jpg" \
  -o "output_PROJECT_Processed" \
  -r config/recipes/exterior_enhanced.yaml
```

#### Batch Processing (Same Recipe)
```bash
# Process multiple files with same recipe
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/Source_JPEGS/*Bedroom*.jpg" \
  -o "output_PROJECT_Bedrooms" \
  -r config/recipes/signature_estate_gentle.yaml
```

---

### 4️⃣ Document Results (System Learns)

**For any non-routine outcome:**

Create run card: `docs/runs/PROJECT/ImageID_recipe.yaml`

```yaml
image_id: ProjectName_SceneName
project: project_name
scene_type: interior_bedroom
scene_features:
  - daylight
  - premium_finishes

source_baseline_score: 60.4
processed_score: 55.8
delta_score: -4.6
targets_met: 3/4

recipe: signature_estate_gentle
human_rating: acceptable_but_unnecessary
decision: recipe_not_recommended_for_hero_shots

notes:
  - "Baseline already pristine"
  - "Processing added film character but cost quality"

lessons:
  - "Hero shots (≥60%) should ship as baseline"
  - "Gentle recipe still loses 4-5% on excellent sources"

tags:
  - interior
  - bedroom
  - hero_shot
  - high_baseline
```

**When to document:**
- ✅ New scene type
- ✅ Counterintuitive result
- ✅ RAG prediction wrong
- ✅ Client feedback
- ❌ Routine/expected results

---

## Common Patterns

### New Project Startup
```bash
# 1. One-time setup
mkdir -p input_images/PROJECT_NAME/Source_JPEGS
mkdir -p docs/runs/PROJECT_NAME

# 2. Copy source images
cp /path/to/renders/* input_images/PROJECT_NAME/Source_JPEGS/

# 3. Baseline assessment
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT_NAME/Source_JPEGS/*.jpg" \
  -o "output_PROJECT_NAME_Baseline" \
  -r config/recipes/baseline_quality.yaml

# 4. Review baseline scores, consult RAG for each key frame
# 5. Process according to recommendations
# 6. Document significant decisions
```

### Health Check (After Git Pull or Dependency Change)
```bash
source .venv/bin/activate
make test-fast
```

### List Available Recipes
```bash
python -c "from transformation_portal.cli import app; app()" pipeline list-recipes
```

### View Recipe Details
```bash
cat config/recipes/signature_estate_gentle.yaml
```

---

## Copy-Paste Templates

### Template: New Project - Full Workflow

Replace `PROJECT` with actual name:

```bash
# Setup
mkdir -p input_images/PROJECT/Source_JPEGS
mkdir -p docs/runs/PROJECT

# Baseline
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/Source_JPEGS/*.jpg" \
  -o "output_PROJECT_Baseline" \
  -r config/recipes/baseline_quality.yaml

# Get recommendations for key frames
python scripts/rag/suggest_recipe.py \
  --scene-type interior_bedroom \
  --baseline-score XX.X

# Process (example: gentle interior)
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/Source_JPEGS/*Bedroom*.jpg" \
  -o "output_PROJECT_Processed" \
  -r config/recipes/signature_estate_gentle.yaml

# Process (example: exterior)
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/PROJECT/Source_JPEGS/*Aerial*.jpg" \
  -o "output_PROJECT_Processed" \
  -r config/recipes/exterior_enhanced.yaml
```

---

## Troubleshooting

### "Command not found"
```bash
# Virtual environment not activated
source .venv/bin/activate
```

### "Module not found"
```bash
# Wrong Python or venv not activated
which python  # Should show .venv path
source .venv/bin/activate
```

### "Recipe not found"
```bash
# Check recipe name
ls config/recipes/*.yaml

# List available
python -c "from transformation_portal.cli import app; app()" pipeline list-recipes
```

### "Out of memory" (MPS warnings)
```bash
# Normal on M4 Max with large images
# System falls back gracefully
# Warnings can be ignored
```

---

## Quick Reference: Recipe Selection

| Baseline Score | Scene Type | Recipe | Expected Δ |
|----------------|------------|--------|------------|
| ≥60% | Any interior | **baseline** | 0% |
| 55-60% | Interior | baseline OR minimal | -2 to -6% |
| 45-55% | Interior | gentle | -3 to -5% |
| 45-55% | Exterior | enhanced | +3 to +5% |
| <45% | Exterior | enhanced | +5 to +7% |
| Any | Pool | **baseline** | 0% |

**Default rules:**
- Hero (≥55%): Ship as baseline
- Weak exterior (<45%): Use exterior_enhanced  
- Pool: Baseline (quarantined recipe)

---

## File Locations

**Recipes:** `config/recipes/`  
**Run Cards:** `docs/runs/{project}/`  
**Documentation:** `docs/PRODUCTION_OPERATIONS.md`  
**RAG Advisor:** `scripts/rag/suggest_recipe.py`  
**Startup Script:** `tp_start.sh`

---

## Support

**Questions:**
- Workflow → This file
- Policies → `docs/PRODUCTION_OPERATIONS.md`
- Run cards → `docs/runs/README.md`
- System → `docs/guides/RAG_LEARNING_SYSTEM.md`

**Issues:**
- Environment → `docs/guides/INSTALLATION_COMPLETE.md`
- Fixes → `docs/guides/FIXES_APPLIED.md`

---

**Last Updated:** 2025-12-04  
**System Status:** Production Ready ✅

Start with: `/Users/rc/Transformation_Portal/tp_start.sh`
