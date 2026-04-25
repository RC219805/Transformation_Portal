# APEX Visual Quality Protocol

This protocol makes APEX quality changes evidence-led before adding new visual effects.

## Corpus

The seed corpus lives in `evalsets/picacho_apex/evalset.json`. It stores referenced assets only. The current
Picacho corpus is a smoke/readiness corpus, not canonical APEX quality evidence, because it does not include
validated 16-bit source references.

- `asset_ref`: source image path, relative to the repository root unless absolute
- `sha256`: exact source image digest
- `dataset_tier`: corpus class, such as `canonical_apex`, `smoke_or_readiness`, `synthetic_smoke`, or
  `delivery_preview`
- `asset_role`: asset class, such as `canonical_apex_reference`, `delivery_preview`, `synthetic_smoke`, or
  `compatibility_fixture`
- `reference_path`: canonical reference path when available; defaults to `asset_ref`
- `delivery_path`: optional delivery derivative path
- `canonical_bit_depth`: canonical reference bit depth when known
- `canonical_format`: canonical reference format when known
- `canonical_color_space`: documented source profile or color space
- `evaluate_at_native_resolution`: whether final scoring must run at native reference resolution
- `allow_downsampled_model_inference`: whether model-specific tensor inputs may be downsampled or normalized
- `preserve_16bit_intermediates`: whether pixel operations must preserve 16-bit working data
- `canonical_scoring_eligible`: explicit manifest assertion; the report still verifies the fields fail closed
- `canonical_scoring_blocked_reason`: reason a ready asset is not canonical scoring evidence
- `scene_type`: architectural scene category
- `expected_materials`: materials expected to be present
- `risk_zones`: boundaries or surfaces likely to show artifacts
- `reject_if`: visual defects that disqualify a candidate
- `manual_quality_score`: nullable human APEX score placeholder

Large source images are not committed as eval artifacts. The runner reports missing assets as `missing_asset` and
checksum drift as `checksum_mismatch`.

Canonical APEX scoring requires all of the following:

- `dataset_tier=canonical_apex`
- `asset_role=canonical_apex_reference`
- `canonical_bit_depth >= 16`
- `canonical_format` is `tif` or `tiff`
- `evaluate_at_native_resolution=true`
- `preserve_16bit_intermediates=true`

8-bit JPEGs, delivery derivatives, renderings, and synthetic fixtures may be ready and useful for smoke coverage,
UI/report checks, compatibility checks, or edge-case tests. They must not count as canonical APEX quality-scoring
references.

## Report

Run:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/picacho_apex \
  --output-dir output/apex_eval \
  --emit-report on
```

The output is `apex_eval_report.json`. It records corpus readiness and, when candidate outputs are supplied,
visible-delta metrics:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/picacho_apex \
  --output-dir output/apex_eval \
  --candidate-output depth_pro_materials:picacho_pool_750_v2_enhanced=output/candidate_pool.tif
```

Readiness and canonical scoring eligibility are separate report concepts. A ready asset can still be noncanonical:

- `ready_asset_count`: assets with valid checksums
- `canonical_scoring_eligible_count`: ready assets that pass the canonical 16-bit contract
- `missing_asset_count`: unresolved referenced assets
- `noncanonical_asset_count`: ready assets blocked from canonical APEX scoring
- `canonical_scoring_blocked_reason_counts`: blocked reasons for ready noncanonical assets

Each asset report records `asset_role`, `reference_bit_depth`, `reference_format`, `reference_color_space`,
`canonical_scoring_eligible`, and `canonical_scoring_blocked_reason`.

Depth benchmark reports also separate model input derivation from the evaluation target. Depth Pro, DA3, and other
depth backends may consume normalized 8-bit or downsampled tensors, but benchmark cases must still record the
16-bit source path as `evaluation_target` when one exists.

## 16-Bit Working Path

Canonical APEX scoring follows this path:

```text
16-bit source
  -> model-specific normalized inference copy
  -> masks / depth / material decisions
  -> Materials V3 pixel operations against the 16-bit working image
  -> final 16-bit QC metrics
  -> 8-bit delivery derivative
```

Model inference resolution is not the same thing as evaluation or output resolution. A backend may run on a normalized
copy, but Materials V3 edits and APEX quality metrics should operate on the 16-bit working image when a canonical
reference exists.

Expected artifacts for canonical APEX runs:

- `*_materials_v3_master16.tif`
- `*_v2_master16.tif`
- `*_delivery_srgb8.jpg`
- `*_diff_heatmap.png`
- `*_mask_overlay.png`

## Quality Trajectory

APEX promotion requires a report-backed improvement path:

- Depth Pro is the research-quality yardstick.
- DA3 metric is the commercial-safe baseline.
- Materials V3 pixel operations require calibrated material confidence.
- APEX runs fail closed when masks exist, implemented operations exist, and zero Materials V3 pixel operations apply.

The long-term APEX score should combine depth edge fidelity, material precision, pixel-op false positive risk,
visible delta metrics, and manual APEX score. A missed enhancement is acceptable; a wrong material edit is not.
