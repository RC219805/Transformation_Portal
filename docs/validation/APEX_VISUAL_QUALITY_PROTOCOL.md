# APEX Visual Quality Protocol

This protocol makes APEX quality changes evidence-led before adding new visual effects.

## Corpus

The seed corpus lives in `evalsets/picacho_apex/evalset.json`. It stores referenced assets only:

- `asset_ref`: source image path, relative to the repository root unless absolute
- `sha256`: exact source image digest
- `scene_type`: architectural scene category
- `expected_materials`: materials expected to be present
- `risk_zones`: boundaries or surfaces likely to show artifacts
- `reject_if`: visual defects that disqualify a candidate
- `manual_quality_score`: nullable human APEX score placeholder

Large source images are not committed as eval artifacts. The runner reports missing assets as `missing_asset` and checksum drift as `checksum_mismatch`.

## Report

Run:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/picacho_apex \
  --output-dir output/apex_eval \
  --emit-report on
```

The output is `apex_eval_report.json`. It records corpus readiness and, when candidate outputs are supplied, visible-delta metrics:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/picacho_apex \
  --output-dir output/apex_eval \
  --candidate-output depth_pro_materials:picacho_pool_750_v2_enhanced=output/candidate_pool.tif
```

## Quality Trajectory

APEX promotion requires a report-backed improvement path:

- Depth Pro is the research-quality yardstick.
- DA3 metric is the commercial-safe baseline.
- Materials V3 pixel operations require calibrated material confidence.
- APEX runs fail closed when masks exist, implemented operations exist, and zero Materials V3 pixel operations apply.

The long-term APEX score should combine depth edge fidelity, material precision, pixel-op false positive risk, visible delta metrics, and manual APEX score. A missed enhancement is acceptable; a wrong material edit is not.
