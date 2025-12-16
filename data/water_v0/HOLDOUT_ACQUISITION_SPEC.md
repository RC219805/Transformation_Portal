# Water Detection Holdout Acquisition Specification

**Version**: v1  
**Date**: 2025-12-16  
**Purpose**: Phase C completion gate - validate multi-scale glass suppressor on real-world negatives

## Selection Requirements

### Hard Requirements
- ✅ Real photographs (not synthetic CI fixtures)
- ✅ Negative ground truth: `should_detect = false` (no actual water visible)
- ✅ Architectural glass heavy: curtain walls, window grids, facades, skylights, balconies, greenhouses
- ✅ Variety: lighting conditions, reflections, perspectives, distances, grid frequencies
- ✅ Resolution: ≥1024px short edge (no upsampling)
- ✅ No near-duplicates: different buildings/scenes/angles

### Avoid
- ❌ Images containing actual water (ocean/pool/lake/river) - even small amounts
- ❌ Heavy HDR artifacts / AI-generated / stylized renders
- ❌ Copyright violations (use own photos, licensed stock, or explicit permission)

## 15-Image Breakdown (Orthogonal Coverage)

### Bucket A: True Architectural Glass Grids (6 images)
**Tags**: `glass_grid`, `curtain_wall`, `facade`

1. **glass_curtain_wall_001.jpg** - Office tower, dense repetitive grid
2. **glass_curtain_wall_002.jpg** - Commercial building, vertical mullions
3. **glass_curtain_wall_003.jpg** - High-rise, horizontal/vertical grid pattern
4. **glass_residential_001.jpg** - Apartment facade with balcony glass
5. **glass_residential_002.jpg** - Residential with window mullions
6. **glass_greenhouse_001.jpg** - Greenhouse/conservatory glass structure

**Difficulty**: Hard  
**Why**: Primary glass suppressor target - tests grid detection logic

### Bucket B: Reflection Traps (5 images)
**Tags**: `reflection`, `sky_blue`, `specular`, `landscape_reflection`

7. **glass_sky_reflection_001.jpg** - Strong sky reflection (blue gradient)
8. **glass_sky_reflection_002.jpg** - Sky + clouds reflection (color/brightness)
9. **glass_landscape_reflection_001.jpg** - Trees/greenery reflection (texture)
10. **glass_landscape_reflection_002.jpg** - Building/urban reflection
11. **glass_interior_reflection_001.jpg** - Interior glass with lights/reflections

**Difficulty**: Hard  
**Why**: Tests color/brightness heuristics against blue reflections

### Bucket C: Grid-But-Not-Glass Confounders (4 images)
**Tags**: `grid_pattern`, `non_glass`, `confounder`

12. **metal_fence_grid_001.jpg** - Metal fence or railing grid
13. **solar_panels_001.jpg** - Solar panel array (grid pattern)
14. **tiled_facade_001.jpg** - Patterned wall tiles (not water-related)
15. **window_blinds_001.jpg** - Blinds/shutters pattern through window

**Difficulty**: Medium  
**Why**: Catch over-broad grid heuristics that don't verify glass material

## Storage Structure

```
data/water_v0/
├── holdout_manifest.json        # Tracked in git (metadata only)
├── images/                       # GITIGNORED (large binaries)
│   ├── glass_curtain_wall_001.jpg
│   ├── glass_curtain_wall_002.jpg
│   ├── ...
│   └── window_blinds_001.jpg
└── HOLDOUT_ACQUISITION_SPEC.md  # This file (tracked)
```

### .gitignore Entry
```gitignore
# Holdout images (private dataset, large binaries)
data/water_v0/images/
```

## Manifest Schema (Per Image)

```json
{
  "filename": "glass_curtain_wall_001.jpg",
  "sha256": "a1b2c3d4e5f6...",
  "label": "negative",
  "should_detect": false,
  "tags": ["glass_grid", "curtain_wall", "office"],
  "bucket": "A",
  "difficulty": "hard",
  "source": "self-captured | licensed | public-domain",
  "permission_note": "Own photo, smartphone, 2025-12-16",
  "scene_description": "Modern office tower with dense glass curtain wall grid"
}
```

### Required Fields
- `filename`: Relative path in `images/` dir
- `sha256`: SHA-256 checksum for integrity
- `should_detect`: Always `false` for negatives
- `tags`: Scene characteristics (list)
- `source`: Provenance category
- `permission_note`: One-line license/permission justification

### Optional Fields
- `bucket`: A/B/C (for coverage analysis)
- `difficulty`: hard/medium/easy
- `scene_description`: Human-readable context

## Acquisition Checklist

1. ✅ Collect 15 images matching bucket breakdown
2. ✅ Place in `data/water_v0/images/` (create if doesn't exist)
3. ✅ Generate SHA256 hashes: `sha256sum images/*.jpg > checksums.txt`
4. ✅ Update `holdout_manifest.json` with actual filenames + hashes
5. ✅ Set environment: `export WATER_HOLDOUT_DIR=/abs/path/to/data/water_v0/images`
6. ✅ Run validation: `./scripts/validate_holdout.sh`
7. ✅ Archive results: `mv holdout_validation_v1.json data/water_v0/results/run_$(date +%Y%m%d_%H%M%S)/`

## Acceptance Gates

- **False Trigger Rate**: ≤5% (at most 1 false trigger on 15 images)
- **Telemetry Required**: For any false triggers, review `grid_score_coarse`, `grid_persistence_ratio`, `tile_exempted`
- **Reproducibility**: SHA256 manifest ensures deterministic validation

## Phase C Completion Criteria

Phase C is **COMPLETE** when:
1. ✅ 15 real-world images acquired
2. ✅ Manifest updated with actual SHA256 hashes
3. ✅ First holdout validation run completed
4. ✅ Results archived with timestamp
5. ✅ False trigger rate ≤5% (or documented justification for exceptions)

**Only then** proceed to Phase D (threshold tuning, flag enablement).

## Legal/Licensing Notes

- **Self-captured photos**: Cleanest option (own copyright)
- **Licensed stock**: Verify commercial/public repo usage rights
- **Client photos**: Get explicit written permission
- **Public domain**: Verify provenance (Wikimedia Commons, Unsplash, etc.)

If repo is public, commit only manifest (metadata). Never commit third-party images without unambiguous license.

## Reference

- Validation script: `scripts/validate_holdout.sh`
- Validation harness: `scripts/prw_water_validation.py`
- Ground truth schema: `data/water_v0/ground_truth.schema.json`
- Phase C docs: `docs/guides/PHASE_C_IMPLEMENTATION_SUMMARY.md`
