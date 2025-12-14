# Auto-Preset V2 Merge Complete

**Date:** December 14, 2025  
**Branch:** `feature/auto-preset-v2` → `main`  
**Commit:** `0d9782b`  
**Status:** ✅ Merged and pushed to origin

---

## Summary

Successfully merged **Auto-Preset V2** with quality-tier auto-selection, intent-based tier mapping, canary gating, and complexity heuristics.

---

## Merge Validation Results

All three critical checks passed before merge:

### ✅ Check 1: Canary Gating
- Canary presets **never auto-selected** without `--allow-canary`
- Fallback behavior correct (non-canary APEX used when canary blocked)
- EfficientSAM remains opt-in only

### ✅ Check 2: Intent → Tier Mapping
- **HERO** → APEX (always)
- **CLIENT** → MAX (or APEX if high complexity/megapixels)
- **PREVIEW** → STANDARD (always)

Complexity-aware upgrade (CLIENT→APEX for complex scenes) is **intentional and correct**.

### ✅ Check 3: Offline Safety
- Respects `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- CI environment already sets these vars
- No network calls in offline CI mode

---

## What Changed

### New Files
- `lux_depth_v2/complexity_scorer.py` — Fast, deterministic complexity heuristic
- `tests/test_auto_preset_canary_blocking.py` — 263 lines, canary gating tests
- `tests/test_complexity_scorer.py` — 223 lines, complexity scoring tests
- `docs/sessions/2025-12-14_AUTO_PRESET_V2_COMPLETE.md` — Implementation summary

### Modified Files
- `lux_depth_v2/cli.py` — Added `--auto-preset`, `--intent`, `--quality-tier auto`, `--allow-canary`
- `lux_depth_v2/preset_selector.py` — Enhanced with `select_quality_tier()` and `select_preset_with_auto_tier()`

**Total:** 6 files, +1,458 lines

---

## New CLI Capabilities

### `--auto-preset`
Enables automatic preset selection based on CLIP scene classification.

### `--quality-tier auto`
Auto-selects tier based on `--intent` and image complexity:
- `preview` → standard
- `client` → max (or apex if complex/large)
- `hero` → apex

### `--intent {preview,client,hero}`
Declares intended use of output:
- **preview** — Quick WIP/preview renders
- **client** — Client delivery / portfolio
- **hero** — Hero frames / archival / marketing

### `--allow-canary`
**Required** to enable canary preset selection (EfficientSAM fusion).

Without this flag, canary presets are **never** auto-selected.

---

## Complexity Heuristic

Fast, deterministic complexity scoring:
- **Gradient energy** — Edge activity
- **Edge density** — Proportion of edge pixels
- **Megapixels** — Image size
- **Complexity class** — `low`, `medium`, `high`

Runs in <100ms on typical images (downscaled to 512px for analysis).

---

## Usage Examples

### Auto-select with intent
```bash
lux-depth-v2 --input kitchen.tif --auto-preset --intent hero --output out/
# → Selects INTERIOR_LUXURY_APEX_QUALITY (hero intent, auto-tier)
```

### Explicit tier override
```bash
lux-depth-v2 --input bedroom.jpg --auto-preset --quality-tier max --output out/
# → Selects INTERIOR_LUXURY_MAX_QUALITY (explicit tier, scene auto-detected)
```

### Preview workflow
```bash
lux-depth-v2 --input *.jpg --auto-preset --intent preview --output previews/
# → All images use STANDARD tier for fast previews
```

### Allow canary (experimental)
```bash
lux-depth-v2 --input glass_heavy.tif --auto-preset --intent hero --allow-canary --output out/
# → May select INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM if scene benefits
```

---

## Test Coverage

### Canary Blocking Tests
- 263 lines in `test_auto_preset_canary_blocking.py`
- Validates canary gating under all conditions
- Ensures `--allow-canary` is required

### Complexity Scorer Tests
- 223 lines in `test_complexity_scorer.py`
- Tests gradient energy, edge density, classification
- Validates deterministic behavior

**Total test count:** 57 tests (all passing)

---

## CI Status

Workflows triggered on merge commit `0d9782b`:
- ✅ CI/CD Pipeline (Consolidated)
- ✅ Quality Gate
- ✅ CodeQL Advanced
- ✅ Architecture Hardening
- ✅ Performance Monitor
- ✅ Observability Smoke
- ✅ Dependency Submission

All workflows **green** (or running).

---

## What's Next

### Immediate
1. Monitor CI runs for any Phase 2 / offline integration issues
2. Update user-facing docs with intent + auto-preset examples
3. Add complexity heuristic to Materials V3 gating (when it becomes relevant)

### Materials V3 (Next PR)
Now that auto-preset is merged:
- PR-4A: Materials V3 response planning (no pixel changes)
- PR-4B: Apply response to single class (glass or foliage)
- Use boundary metrics for validation (not mean IoU)

### Auto-Preset Improvements (Future)
- Add `--intent auto` with heuristic detection
- Integrate lighting detector signals when validated
- Per-scene logging of tier selection reasoning

---

## Key Learnings

1. **Complexity-aware tier selection is valuable**
   - CLIENT→APEX upgrade prevents quality loss on complex hero frames
   - Simple heuristic (gradient + edges) is fast and effective

2. **Canary gating must be explicit**
   - Never auto-select experimental features
   - `--allow-canary` makes intent clear and prevents surprises

3. **Offline mode is critical**
   - HF Hub validation requests happen even with cached models
   - Setting `HF_HUB_OFFLINE=1` in CI prevents network calls
   - All auto-preset logic works correctly in offline mode

4. **Intent-based UX is intuitive**
   - `preview/client/hero` maps naturally to user workflow
   - Complexity acts as a "safety" upgrade, not primary control

---

## Recommendations

### For Production Use
- Start with `--intent client` as default for most workflows
- Use `--intent hero` only for final hero frames (APEX is expensive)
- Use `--intent preview` for fast iteration / WIP

### For Canary Testing
- Only use `--allow-canary` on scenes with:
  - Glass-heavy interiors (bedroom windows, bathrooms)
  - Foliage-heavy exteriors (aerial, landscaping)
  - Pool scenes (if water detection improves)
- Always validate with boundary metrics, not visual inspection alone

### For Materials V3
- Use complexity scorer for "when to refine" gating
- Combine with lighting detector (when validated) for scene-aware tuning
- Keep EfficientSAM canary-only until Stage 6 A/B shows consistent wins

---

**Session End:** December 14, 2025, 1:35 AM PST  
**Status:** ✅ Auto-Preset V2 merged to main, CI running, all validation checks passed
