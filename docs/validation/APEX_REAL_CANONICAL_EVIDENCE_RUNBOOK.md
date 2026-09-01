# APEX Real Canonical Evidence Runbook

This runbook describes the first non-synthetic APEX evidence workflow for the
metadata-only `apex_real_estate_v1` canonical manifest.

The expected outcome is an evidence bundle with real canonical inputs and
fail-closed verdicts. A failed promotion verdict is a valid real evidence result
when candidate outputs, telemetry, dimensions, bit depth, metrics, or Materials
V3 gates fail closed.

## Prerequisites

- External canonical assets are mounted outside git.
- `evalsets/apex_real_estate_v1/evalset.json` is present.
- Candidate outputs are externally stored 16-bit TIFF/TIF files.
- Candidate output dimensions match each asset's `reference_path`.
- Materials V3 telemetry JSON files are stored outside git.
- Generated `output/` reports remain uncommitted.

Set the local asset root:

```bash
export APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets"
```

## Canonical Asset Audit

Run the audit before producing evidence:

```bash
export APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets"
.venv/bin/python tools/audit_apex_assets.py \
  --evalset evalsets/apex_real_estate_v1/evalset.json \
  --require-canonical on \
  --output-dir output/apex_asset_audit
```

The audit should exit `0` when all mounted canonical references are present,
checksum-valid, TIFF/TIF, and at least 16-bit.

## Materials V3 Telemetry

Each `materials_v3` candidate needs telemetry evidence. Minimal telemetry shape:

```json
{
  "materials_v3_enabled": true,
  "pixel_ops_enabled": true,
  "masks_exist": true,
  "implemented_ops_exist": true,
  "applied_ops_count": 1,
  "blocked_reason_counts": {
    "below_confidence_threshold": 0
  },
  "confidence_authority": {
    "raw_clip_similarity_authorized_pixel_ops": false,
    "calibrated_score_type": "clip_softmax_margin_v1"
  }
}
```

For APEX promotion, `raw_clip_similarity_authorized_pixel_ops` must be `false`.
Raw CLIP similarity may appear as evidence, but it must not authorize pixel
operations.

## Candidate Generation / Extraction

Candidate generation, candidate extraction, and candidate scoring are separate
steps:

- Candidate generation runs Lux Depth V3 with Materials V3 enabled.
- Candidate extraction copies the generated 16-bit Materials V3 candidate and
  telemetry into stable external baseline paths.
- Candidate scoring runs `tools/run_apex_eval.py` against those extracted
  external files.

For manual Lux Depth V3 -> Materials V3 candidate extraction, the generation
command must include `--keep-intermediates`. Without it, the pipeline may remove
`<output-dir>/temp/*_materials_v3_enhanced.*` before the operator can copy the
16-bit Materials V3 candidate to the stable external baseline path.

Use the same governed APEX settings that will be scored later:

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir "$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/reference_16bit" \
  --output-dir output/apex_candidate_generation \
  --quality-tier apex \
  --materials-v3 on \
  --enable-segmentation on \
  --strict-segmentation \
  --output-bit-depth 16 \
  --emit-run-card on \
  --keep-intermediates \
  --overwrite
```

Copy extracted artifacts to stable external baseline paths under
`baselines/materials_v3`:

```text
$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/<asset_id>_materials_v3_16bit.tif
$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/<asset_id>_materials_v3_evidence.json
```

## Scoped Evidence Run

Use an explicit run scope for a one-asset real evidence run:

```bash
export APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets"
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/apex_real_estate_v1/evalset.json \
  --asset-root "$APEX_EVAL_ASSET_ROOT" \
  --output-dir output/apex_eval_real \
  --candidate-output materials_v3:pool_water_stone_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/pool_water_stone_001_materials_v3_16bit.tif" \
  --candidate-evidence materials_v3:pool_water_stone_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/pool_water_stone_001_materials_v3_evidence.json" \
  --run-scope-asset-id pool_water_stone_001 \
  --emit-evidence-bundle on \
  --synthetic-data off
```

Candidate-output scoring compares the candidate against `reference_path`, not
`asset_ref`. For canonical assets, the reference and candidate must be readable
16-bit TIFF/TIF files with matching dimensions. Scoring does not resize.

Portable report fields must not expose full local resolved reference paths.

## Full Manifest Evidence Run

If `--run-scope-asset-id` is omitted, promotion eligibility is evaluated across
all canonical-eligible assets in the manifest. A single candidate output against
the three-asset manifest is not promotion-eligible unless the run is explicitly
scoped to that asset.

Full-manifest promotion requires candidate output and Materials V3 telemetry for
all current canonical assets:

```bash
export APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets"
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/apex_real_estate_v1/evalset.json \
  --asset-root "$APEX_EVAL_ASSET_ROOT" \
  --output-dir output/apex_eval_real \
  --candidate-output materials_v3:pool_water_stone_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/pool_water_stone_001_materials_v3_16bit.tif" \
  --candidate-evidence materials_v3:pool_water_stone_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/pool_water_stone_001_materials_v3_evidence.json" \
  --candidate-output materials_v3:kitchen_glass_metal_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/kitchen_glass_metal_001_materials_v3_16bit.tif" \
  --candidate-evidence materials_v3:kitchen_glass_metal_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/kitchen_glass_metal_001_materials_v3_evidence.json" \
  --candidate-output materials_v3:exterior_foliage_sky_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/exterior_foliage_sky_001_materials_v3_16bit.tif" \
  --candidate-evidence materials_v3:exterior_foliage_sky_001="$APEX_EVAL_ASSET_ROOT/apex_real_estate_v1/baselines/materials_v3/exterior_foliage_sky_001_materials_v3_evidence.json" \
  --emit-evidence-bundle on \
  --synthetic-data off
```

## Evidence Statuses And Promotion Reasons

Asset audit and candidate comparison entries can report per-asset or
per-candidate statuses such as:

- `missing_asset`
- `path_escapes_asset_root`
- `missing_reference`
- `missing_candidate`
- `unreadable_reference`
- `unreadable_candidate`
- `unsupported_reference_bit_depth`
- `unsupported_candidate_bit_depth`
- `dimension_mismatch`
- `metrics_not_computed`

These statuses are useful for troubleshooting the underlying asset or candidate
failure. They are not all emitted directly as run-level
`promotion_blocked_reasons`.

Evidence bundle `promotion_blocked_reasons` currently include:

- `zero_canonical_eligible_assets`
- `missing_candidate_output`
- `invalid_metrics`
- `missing_materials_v3_evidence`
- `synthetic_data`
- `APEX_MATERIALS_PIXEL_OPS_EMPTY`
- `raw_clip_similarity_authorized_pixel_ops`

Generated reports and evidence bundles under `output/` are local validation
artifacts. Do not commit them unless they satisfy the redacted fixture policy:

`docs/validation/APEX_REDACTED_EVIDENCE_FIXTURE_POLICY.md`
