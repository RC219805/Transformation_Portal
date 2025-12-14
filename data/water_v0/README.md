# Water Detection Dataset v0

This dataset contains labeled images for validating water detection in pool and ocean scenes.

## Structure

```
water_v0/
├── images/
│   ├── pool/           # Pool scenes (20 water + 2 hard negatives)
│   └── ocean/          # Ocean scenes (20 water + 2 hard negatives)
├── ground_truth.json   # Ground truth labels (v0 schema)
├── ci_subset.txt       # 14 images for fast CI validation
├── LABELING_GUIDE.md   # Labeling instructions
└── README.md           # This file
```

## Schema Version

**v0** - Two-label schema (pool, ocean) with negative controls

See `docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md` for complete schema documentation.

## Key Fields

- `label`: `pool` or `ocean` (folder organization)
- `should_detect`: `true` for water, `false` for hard negatives
- `difficulty`: `easy` | `medium` | `hard`
- `tags`: Track failure modes (e.g., `low-light`, `reflection`, `waves`)

## Hard Negatives

Hard negatives (`should_detect: false`) are critical for measuring false trigger rate:

**Pool hard negatives**:
- Blue painted walls
- Blue sky through windows
- Blue fabric/umbrellas

**Ocean hard negatives**:
- Reflective glass buildings
- Blue painted surfaces
- Sky reflections

## Validation Harness

Run validation:
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --output water_validation_report.json \
    --seed 42
```

Run CI subset only:
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --subset-file data/water_v0/ci_subset.txt \
    --output ci_report.json \
    --seed 42
```

## Status

**Dataset**: Scaffolding created; images pending (store privately or commit thumbnails)

**Validation harness**: Complete and deterministic (PR-W4)

**Detector**: Stub implementation (PR-W1 pending)

## Next Steps

1. Collect 44 labeled images (20 pool + 20 ocean water, 2+2 hard negatives)
2. Run validation to establish baseline
3. Store baseline report as `data/water_v0/baseline_v0.json`
4. Implement real detector (PR-W1)
5. Re-run validation and check for regression
