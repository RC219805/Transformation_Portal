# Holdout Image Quick Reference Card

**Goal**: 13 real-world architectural glass negatives (NO water visible)  
**Acceptance Gate**: ≤1 false positive (max 1 of 13 images triggers detector)

## The 13 Images (Priority Order)

### 🔴 CRITICAL - Must Have (Pool Tiles)
11. **pool_tiles_closeup_001.jpg** - Pool tiles, grid pattern, BONE DRY
12. **pool_tiles_closeup_002.jpg** - Different tile style/size, also DRY
    - *Why Critical*: Core test of multi-scale grid suppressor
    - *Where*: Empty pools, tile showrooms, pool supply stores

### 🟡 High Priority (Glass Facades - 3 variations)
1. **glass_facade_001.jpg** - Modern office tower, reflective glass
2. **glass_facade_002.jpg** - Curtain wall, seamless glass, sky reflections
3. **glass_facade_003.jpg** - Commercial building, tinted glass panels
   - *Where*: Downtown, corporate campuses, modern buildings
   - *When*: Clear/partly cloudy day (need sky reflections)

### 🟢 Medium Priority (Reflective Surfaces)
6. **reflective_stone_001.jpg** - Polished granite/marble, specular highlights
7. **reflective_concrete_001.jpg** - Wet concrete pavement (no standing water)
8. **reflective_stone_002.jpg** - Polished floor, interior lighting
   - *Where*: Building lobbies, malls, outdoor plazas after rain
   - *When*: Anytime (interior), or after light rain (exterior)

### 🟢 Medium Priority (Skylights)
9. **skylight_reflection_001.jpg** - Glass skylight, strong sky blue
10. **skylight_reflection_002.jpg** - Interior skylight, diffused light
    - *Where*: Malls, atriums, modern buildings
    - *When*: Clear day for strong blue tint

### 🟢 Medium Priority (Blue Surfaces)
4. **blue_wall_painted_001.jpg** - Flat blue painted exterior
5. **blue_wall_painted_002.jpg** - Interior blue wall, soft lighting
13. **concrete_blue_tint_001.jpg** - Concrete with sky reflection blue tint
    - *Where*: Buildings, homes, sidewalks/parking lots
    - *When*: Clear blue sky day for #13

## Smartphone Capture Checklist

### Before Leaving Location
- ✅ Check image in camera roll (not blurry)
- ✅ Verify resolution (3MP+ / 2048×1536 minimum)
- ✅ Confirm NO water visible (critical!)
- ✅ Key confuser visible (blue tint, grid, reflection)

### Camera Settings (Smartphone)
- **Mode**: Auto or Photo mode
- **HDR**: OFF (avoid AI processing)
- **Flash**: OFF (use natural light)
- **Zoom**: Digital zoom OFF (move closer instead)
- **Focus**: Tap subject to lock focus

### Composition Tips
- **Distance**: 1-3m for tiles/walls, 10-30m for buildings
- **Angle**: Slight upward (10-20°) for facades, straight-on for tiles
- **Framing**: Fill 60-80% of frame with subject
- **Context**: Include edges/boundaries (not just infinite surface)

## 7-Day Quick Plan

**Day 1-2**: Glass facades (#1-3) - downtown/business district trip  
**Day 3**: Blue walls + concrete (#4-5, #13) - walk neighborhood  
**Day 4**: Reflective surfaces (#6-8) - indoor/outdoor after rain  
**Day 5**: Skylights (#9-10) - mall/atrium visit  
**Day 6-7**: **CRITICAL** Pool tiles (#11-12) - tile stores/empty pools

## Emergency Substitutions

**Can't find empty pool tiles?**
→ Home Depot/Lowe's tile section (pool tile samples)  
→ Pool waterline tiles above water (must be DRY)

**No modern glass facades nearby?**
→ Glass bus shelters, large storefront windows  
→ Greenhouse/conservatory glass walls

**Weather not cooperating?**
→ Indoor shots first (#5, #6, #8, #9, #10)  
→ Postpone outdoor blue sky shots (#1, #2, #7, #13)

## Integration Steps (After Capture)

1. Transfer 13 JPGs to `/Users/rc/Transformation_Portal/data/water_v0/holdout_images/`
2. Rename to exact manifest filenames
3. Generate SHA256: `shasum -a 256 *.jpg > checksums.txt`
4. Update `holdout_manifest.json` with real hashes (replace placeholders)
5. Set env: `export WATER_HOLDOUT_DIR="$(pwd)/data/water_v0/holdout_images"`
6. Run: `./scripts/validate_holdout.sh`
7. Archive: `mv holdout_validation_v1.json holdout_results/run_$(date +%Y%m%d_%H%M%S)/`

## Success = Phase C Complete

✅ All 13 images captured  
✅ Zero actual water visible  
✅ Detector triggers ≤1 false positive  
✅ Results archived  

**Then**: Proceed to Phase D (threshold tuning with holdout evidence)
