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

## 13-Image Breakdown (Orthogonal Coverage)

**Note**: Reduced from 15 to 13 images after removing ocean entries (real water violates negative-only purity)

### Bucket A: True Architectural Glass (3 images)
**Tags**: `architectural_glass`, `facade`, `curtain_wall`, `skylight`

1. **glass_facade_001.jpg** - Modern office building with reflective glass façade
2. **glass_facade_002.jpg** - Glass curtain wall with sky reflections (blue tint)
3. **glass_facade_003.jpg** - Commercial building with tinted glass panels

**Difficulty**: Hard  
**Why**: Primary glass suppressor target - tests grid detection and reflectivity

### Bucket B: Reflective/Painted Surfaces (5 images)
**Tags**: `blue_painted`, `reflective_stone`, `reflective_concrete`, `skylight`

4. **blue_wall_painted_001.jpg** - Flat blue painted exterior wall
5. **blue_wall_painted_002.jpg** - Interior blue wall with soft lighting
6. **reflective_stone_001.jpg** - Polished granite/marble surface
7. **reflective_concrete_001.jpg** - Wet concrete pavement with reflections
8. **reflective_stone_002.jpg** - Polished stone floor in luxury interior

**Difficulty**: Medium-Hard  
**Why**: Tests color/brightness heuristics, specular highlights, transient water-like appearance

### Bucket C: Grid/Tile Confounders (5 images)
**Tags**: `skylight`, `pool_tiles`, `grid_pattern`, `no_water`, `concrete`

9. **skylight_reflection_001.jpg** - Glass skylight with sky reflection
10. **skylight_reflection_002.jpg** - Interior skylight with diffused blue light
11. **pool_tiles_closeup_001.jpg** - Pool tiles with grid pattern, no water visible
12. **pool_tiles_closeup_002.jpg** - Blue ceramic pool tiles without water
13. **concrete_blue_tint_001.jpg** - Concrete surface with blue color cast

**Difficulty**: Hard-Medium  
**Why**: Tests grid alignment, blue hue confusers, and pool context without actual water

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

1. ✅ Collect 13 images matching bucket breakdown (A: 3, B: 5, C: 5)
2. ✅ Place in `data/water_v0/images/` (create if doesn't exist)
3. ✅ Generate SHA256 hashes: `sha256sum images/*.jpg > checksums.txt`
4. ✅ Update `holdout_manifest.json` with actual filenames + hashes
5. ✅ Set environment: `export WATER_HOLDOUT_DIR=/abs/path/to/data/water_v0/images`
6. ✅ Run validation: `./scripts/validate_holdout.sh`
7. ✅ Archive results: `mv holdout_validation_v1.json data/water_v0/results/run_$(date +%Y%m%d_%H%M%S)/`

## Acceptance Gates

- **Max False Triggers**: ≤1 (absolute cap, avoids percentage rounding on small datasets)
  - Effective rate: 1/13 = 7.7% (acceptable for real-world architectural confusers)
- **Telemetry Required**: For any false triggers, review `grid_score_coarse`, `grid_persistence_ratio`, `tile_exempted`
- **Reproducibility**: SHA256 manifest ensures deterministic validation

## Phase C Completion Criteria

Phase C is **COMPLETE** when:
1. ✅ 13 real-world negative images acquired (no actual water present)
2. ✅ Manifest updated with actual SHA256 hashes
3. ✅ First holdout validation run completed
4. ✅ Results archived with timestamp
5. ✅ False trigger count ≤1 (or documented justification for exceptions)

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
