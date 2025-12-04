# Next Three Improvements (Priority Order)

**Status:** Post-Production Framework  
**Focus:** Operational efficiency, not features  
**Timeline:** Implement incrementally as projects demand

---

## Why These Three

The foundation is solid. These aren't "nice to haves" - they're **force multipliers** that:
1. Reduce admin friction from 5 min → 60 sec per decision
2. Keep RAG retrieval accurate through consistency
3. Prevent bad recipes from polluting production

**Implementation principle:** Only build when the manual process becomes painful.

---

## 1. Automate Run Card Generation 🎯 **HIGH PRIORITY**

### Current State
- Manual YAML creation: ~5 minutes
- Requires discipline to maintain
- Fragile - easy to skip when busy

### Target State
- Pipeline writes draft run card automatically
- Human fills in judgment fields only: ~60 seconds
- Impossible to skip - always generated

### Implementation

#### Phase 1: Post-Processing Script
```python
# scripts/generate_run_card.py

import sys
from pathlib import Path
import yaml
from datetime import datetime

def generate_run_card(
    image_path: str,
    baseline_score: float,
    processed_score: float,
    recipe_name: str,
    recipe_settings: dict,
    project_name: str,
    output_dir: str = "docs/runs"
):
    """
    Generate draft run card from pipeline output.
    Human fills in: human_rating, decision, notes, lessons, tags
    """
    
    image_id = Path(image_path).stem
    scene_type = infer_scene_type(image_path)  # From filename/folder
    
    run_card = {
        "image_id": image_id,
        "project": project_name,
        "scene_type": scene_type,
        "scene_features": ["TODO: Review and add"],
        
        "source_baseline_score": round(baseline_score, 2),
        "processed_score": round(processed_score, 2),
        "delta_score": round(processed_score - baseline_score, 2),
        "targets_met": "TODO: Review quality report",
        
        "recipe": recipe_name,
        "recipe_path": f"config/recipes/{recipe_name}.yaml",
        "recipe_settings": recipe_settings,
        
        "processing_time_seconds": "TODO: From pipeline log",
        "date": datetime.now().strftime("%Y-%m-%d"),
        
        "# HUMAN REVIEW REQUIRED": {
            "human_rating": "TODO: [clearly_better|acceptable_but_unnecessary|worse_than_source|significantly_worse]",
            "decision": "TODO: [recipe_recommended|recipe_acceptable|recipe_avoid]",
            "notes": ["TODO: Add observations"],
            "lessons": ["TODO: Add learnings"],
            "tags": ["TODO: Add tags"]
        }
    }
    
    # Write to project subdirectory
    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = project_dir / f"{image_id}_{recipe_name}.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(run_card, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Draft run card: {output_file}")
    print(f"⚠️  Review and complete human assessment fields")
    
    return output_file

def infer_scene_type(image_path: str) -> str:
    """Infer scene type from filename or folder structure."""
    path_lower = str(image_path).lower()
    
    # Interior detection
    if any(x in path_lower for x in ["bedroom", "bed"]):
        return "interior_bedroom"
    elif any(x in path_lower for x in ["kitchen", "kit"]):
        return "interior_kitchen"
    elif any(x in path_lower for x in ["bathroom", "bath"]):
        return "interior_bathroom"
    elif any(x in path_lower for x in ["great", "living", "family"]):
        return "interior_great_room"
    
    # Exterior detection
    elif any(x in path_lower for x in ["pool", "water"]):
        return "exterior_pool"
    elif any(x in path_lower for x in ["aerial", "drone"]):
        return "aerial_exterior"
    elif any(x in path_lower for x in ["garden", "yard"]):
        return "exterior_garden"
    
    # Default
    return "TODO: Specify scene type"

if __name__ == "__main__":
    # Example usage
    generate_run_card(
        image_path=sys.argv[1],
        baseline_score=float(sys.argv[2]),
        processed_score=float(sys.argv[3]),
        recipe_name=sys.argv[4],
        recipe_settings={},  # TODO: Extract from recipe
        project_name=sys.argv[5]
    )
```

Usage after processing:
```bash
python scripts/generate_run_card.py \
  "input_images/project/image.jpg" \
  58.3 \
  54.1 \
  signature_estate_gentle \
  project_name
```

#### Phase 2: Pipeline Integration
```python
# In src/transformation_portal/pipeline_unified.py

def process_batch(...):
    for image_path in images:
        # ... existing processing ...
        
        # Generate draft run card
        if config.get('generate_run_cards', False):
            from scripts.generate_run_card import generate_run_card
            run_card_path = generate_run_card(
                image_path=image_path,
                baseline_score=baseline_result.score,
                processed_score=processed_result.score,
                recipe_name=recipe.name,
                recipe_settings=recipe.settings,
                project_name=config['project_name']
            )
            logger.info(f"Draft run card: {run_card_path}")
```

#### Success Criteria
- ✅ Draft generated automatically
- ✅ Human review takes <2 minutes
- ✅ All technical fields pre-filled
- ✅ Run card creation no longer skipped

---

## 2. Standardize Scene Type Taxonomy 📋 **MEDIUM PRIORITY**

### Current State
- Ad-hoc scene type labels
- Potential drift: "bedroom" vs "interior_bedroom" vs "master_bedroom"
- RAG retrieval gets noisy with inconsistent labels

### Target State
- Fixed vocabulary of ~20 scene types
- Validation on run card creation
- Consistent clustering in RAG

### Implementation

#### Phase 1: Define Taxonomy
```python
# src/transformation_portal/scene_types.py

SCENE_TYPES = {
    # Interiors
    "interior_bedroom": {
        "aliases": ["bedroom", "bed", "master", "guest_room"],
        "description": "Bedrooms, master suites, guest rooms"
    },
    "interior_great_room": {
        "aliases": ["great", "living", "family", "lounge"],
        "description": "Great rooms, living rooms, family rooms"
    },
    "interior_kitchen": {
        "aliases": ["kitchen", "kit", "pantry"],
        "description": "Kitchens, butler's pantry"
    },
    "interior_bathroom": {
        "aliases": ["bathroom", "bath", "powder"],
        "description": "Bathrooms, powder rooms"
    },
    "interior_dining_room": {
        "aliases": ["dining", "breakfast"],
        "description": "Dining rooms, breakfast nooks"
    },
    "interior_office": {
        "aliases": ["office", "study", "library"],
        "description": "Offices, studies, libraries"
    },
    "interior_closet": {
        "aliases": ["closet", "wardrobe", "dressing"],
        "description": "Walk-in closets, dressing rooms"
    },
    
    # Exteriors
    "exterior_pool": {
        "aliases": ["pool", "spa", "water"],
        "description": "Pools, spas, water features"
    },
    "exterior_garden": {
        "aliases": ["garden", "yard", "landscape"],
        "description": "Gardens, yards, landscaping"
    },
    "exterior_courtyard": {
        "aliases": ["courtyard", "patio", "terrace"],
        "description": "Courtyards, patios, terraces"
    },
    "exterior_facade": {
        "aliases": ["facade", "front", "entry"],
        "description": "Building facades, entries"
    },
    "aerial_exterior": {
        "aliases": ["aerial", "drone", "overhead"],
        "description": "Aerial views, drone shots"
    },
    
    # Special
    "twilight_exterior": {
        "aliases": ["twilight", "dusk", "blue_hour"],
        "description": "Twilight exterior shots"
    },
    "night_interior": {
        "aliases": ["night", "evening_interior"],
        "description": "Night interior shots"
    }
}

def normalize_scene_type(raw_input: str) -> str:
    """Convert any alias to canonical scene type."""
    raw_lower = raw_input.lower()
    
    for canonical, config in SCENE_TYPES.items():
        if raw_lower == canonical:
            return canonical
        if any(alias in raw_lower for alias in config["aliases"]):
            return canonical
    
    raise ValueError(
        f"Unknown scene type: {raw_input}. "
        f"Valid types: {', '.join(SCENE_TYPES.keys())}"
    )

def validate_scene_type(scene_type: str) -> bool:
    """Check if scene type is in canonical taxonomy."""
    return scene_type in SCENE_TYPES
```

#### Phase 2: Validation
```python
# In generate_run_card.py

from transformation_portal.scene_types import normalize_scene_type, validate_scene_type

def generate_run_card(...):
    # Infer from path
    raw_scene_type = infer_scene_type(image_path)
    
    # Normalize to canonical
    try:
        scene_type = normalize_scene_type(raw_scene_type)
    except ValueError as e:
        print(f"⚠️  {e}")
        scene_type = "TODO: Specify valid scene type"
    
    # ... rest of run card generation
```

#### Phase 3: Migration
```bash
# scripts/migrate_scene_types.py
# Convert existing run cards to canonical taxonomy
# One-time operation
```

#### Success Criteria
- ✅ Fixed vocabulary of ~20 types
- ✅ Automatic normalization of aliases
- ✅ Validation on run card creation
- ✅ RAG retrieval clusters accurately

---

## 3. Hard Gate on Risky Recipes 🚫 **LOW PRIORITY**

### Current State
- RAG advisor uses rule-based suggestions
- No enforcement of quarantined recipes
- Manual vigilance required

### Target State
- Recipes can be marked as "quarantined"
- RAG advisor never auto-recommends quarantined recipes
- Clear exit criteria for un-quarantine

### Implementation

#### Phase 1: Recipe Metadata
```yaml
# config/recipes/pool_estate.yaml

name: "Pool Estate"
description: "Pool/water exterior recipe"

# Metadata
status: quarantined
quarantine_reason: "Over-processes: avg -9.5% quality loss (750 Picacho)"
quarantine_date: "2025-12-04"
exit_criteria:
  - "3+ successful cases (positive human rating)"
  - "Average Δ ≥ -2% vs baseline"
  - "No catastrophic failures (>-7% loss)"
test_cases: []

# ... recipe settings
```

#### Phase 2: RAG Advisor Integration
```python
# In scripts/rag/suggest_recipe.py

def get_recipe_status(recipe_name: str) -> dict:
    """Load recipe metadata including quarantine status."""
    recipe_path = Path(f"config/recipes/{recipe_name}.yaml")
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)
    return {
        "status": recipe.get("status", "active"),
        "quarantine_reason": recipe.get("quarantine_reason"),
        "exit_criteria": recipe.get("exit_criteria", [])
    }

def rule_based_suggestion(...):
    # ... existing logic ...
    
    # Check if recommended recipe is quarantined
    recipe_status = get_recipe_status(suggestion["recipe"])
    
    if recipe_status["status"] == "quarantined":
        # Override recommendation
        original_recipe = suggestion["recipe"]
        suggestion["recipe"] = find_safe_alternative(scene_type, baseline_score)
        suggestion["warning"] = (
            f"⚠️  {original_recipe} is QUARANTINED: "
            f"{recipe_status['quarantine_reason']}"
        )
        suggestion["alternative_if_desperate"] = original_recipe
    
    return suggestion
```

#### Phase 3: Exit Criteria Tracking
```python
# scripts/check_quarantine_exit.py

def check_exit_criteria(recipe_name: str):
    """Check if quarantined recipe meets exit criteria."""
    recipe_status = get_recipe_status(recipe_name)
    if recipe_status["status"] != "quarantined":
        print(f"✅ {recipe_name} is not quarantined")
        return
    
    # Load all run cards using this recipe
    run_cards = find_run_cards_by_recipe(recipe_name)
    
    # Check criteria
    successful_cases = [
        rc for rc in run_cards 
        if rc["human_rating"] in ["clearly_better", "acceptable_but_unnecessary"]
    ]
    avg_delta = statistics.mean([rc["delta_score"] for rc in run_cards])
    catastrophic_failures = [rc for rc in run_cards if rc["delta_score"] < -7.0]
    
    print(f"\n{recipe_name} Quarantine Status:")
    print(f"  Successful cases: {len(successful_cases)}/3 required")
    print(f"  Average Δ: {avg_delta:.1f}% (≥-2% required)")
    print(f"  Catastrophic failures: {len(catastrophic_failures)} (0 required)")
    
    if (len(successful_cases) >= 3 and 
        avg_delta >= -2.0 and 
        len(catastrophic_failures) == 0):
        print(f"\n✅ {recipe_name} meets exit criteria!")
        print(f"   Ready to un-quarantine")
    else:
        print(f"\n⚠️  {recipe_name} still quarantined")
```

#### Success Criteria
- ✅ Quarantined recipes never auto-recommended
- ✅ Clear warning if user asks for quarantined recipe
- ✅ Automatic tracking of exit criteria
- ✅ Systematic path to un-quarantine

---

## Implementation Priority

### Do Immediately (This Week)
**None** - System is production-ready as-is.

### Do When Pain Emerges
1. **Run card automation** - after 2nd project when manual process feels tedious
2. **Scene type taxonomy** - after seeing drift/confusion in RAG retrieval
3. **Recipe quarantine** - after 2nd bad recipe discovered

### Don't Do Yet
- Automatic recipe selection (needs more data)
- Machine learning scoring (current heuristics sufficient)
- Client preview UI (manual comparison works)
- Multi-user collaboration (single operator for now)

---

## Validation Checkpoint

**Before any of these:** Run one more project through current system.

1. Pick an older shoot you already delivered
2. Run baseline → RAG → process workflow
3. Compare:
   - Did RAG recommend what you actually did?
   - Do metrics align with your judgment?
   - Would you ship the policy output?
4. Measure:
   - Alignment rate (% matching decisions)
   - Time saved vs manual approach
   - Confidence in system

**If alignment <70%:** Adjust policies before automation.  
**If alignment >70%:** Proceed with automation.

This validates the system reflects your taste before building on top of it.

---

## Anti-Roadmap (Don't Build)

**Things that sound good but aren't worth it:**

❌ **Real-time web dashboard** - CLI + filesystem is fine  
❌ **Complex UI** - Text output works, don't gold-plate it  
❌ **ML model training** - Rule-based + RAG is sufficient  
❌ **Multi-tenant architecture** - Single operator doesn't need it  
❌ **API endpoints** - No external consumers  
❌ **Database** - YAML + filesystem scales to 1000s of cards  

**Keep it simple. Build when pain is real, not anticipated.**

---

## Success Metrics

### After Implementing All Three

**Time savings:**
- Run card creation: 5 min → 60 sec (80% reduction)
- RAG accuracy: +15% from consistent taxonomy
- Recipe failures: 0 quarantined recipes in production

**System health:**
- Run card creation rate: >90% (automation makes it impossible to skip)
- Scene type consistency: 100% (validation enforces)
- Bad recipe exposure: 0% (hard gate prevents)

**ROI:**
- Per-project admin time: -30 minutes
- Institutional knowledge quality: +measurably better retrieval
- Production risk: -eliminated bad recipe exposure

---

**Next Action:** Pick test project, run validation workflow, measure alignment. Build automation only when manual process hurts. 🎯
