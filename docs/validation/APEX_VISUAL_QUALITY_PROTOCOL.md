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
- `source_raw_path`: optional camera-original RAW provenance path; metadata only
- `source_raw_format`: optional RAW provenance format, such as `dng`, `cr2`, `cr3`, `nef`, `arw`, `raf`, `orf`, or `rw2`
- `source_raw_sha256`: optional camera-original RAW SHA-256 provenance digest
- `raw_development_profile`: optional metadata label or settings reference for the RAW development profile
- `raw_development_settings_sha256`: optional SHA-256 digest for pinned RAW development settings
- `canonical_icc_profile_name`: optional embedded or declared canonical ICC/profile name
- `canonical_icc_profile_sha256`: optional ICC/profile SHA-256 provenance digest
- `working_color_space`: optional color space used for scoring/editing intermediates
- `working_transfer_function`: optional transfer function used for scoring/editing intermediates
- `evaluate_at_native_resolution`: whether final scoring must run at native reference resolution
- `allow_downsampled_model_inference`: whether model-specific tensor inputs may be downsampled or normalized
- `preserve_16bit_intermediates`: whether pixel operations must preserve 16-bit working data
- `canonical_scoring_eligible`: explicit manifest assertion; the report still verifies the fields fail closed
- `canonical_scoring_blocked_reason`: reason a ready asset is not canonical scoring evidence
- `scene_type`: architectural scene category
- `expected_materials`: materials expected to be present
- `risk_zones`: boundaries or surfaces likely to show artifacts
- `reject_if`: visual defects that disqualify a candidate
- `manual_quality_score`: nullable normalized human APEX score in `[0.0, 1.0]`, where `1.0` indicates no
  visible regression and APEX-quality improvement

Large source images are not committed as eval artifacts. The runner reports missing assets as `missing_asset` and
checksum drift as `checksum_mismatch`.

Canonical assets may be resolved from an external asset root using `--asset-root` or `APEX_EVAL_ASSET_ROOT`.
Resolution order is absolute manifest path, explicit CLI asset root, environment asset root, then repository-relative
behavior. Reports include portable `asset_resolution` metadata by default and do not emit full local resolved paths.
Relative `asset_ref` and `reference_path` values that escape the selected asset root fail closed with
`path_escapes_asset_root`.

Camera-original RAW files such as DNG, CR2/CR3, NEF, ARW, RAF, ORF, and RW2 may be recorded as provenance assets.
They are not canonical scoring targets. `source_raw_path`, `raw_development_profile`, ICC/profile fields, and
working-color fields are metadata-only in the v1 provenance contract; they are not resolved, loaded, or
existence-checked, and they do not affect asset readiness or canonical scoring eligibility.

Canonical APEX scoring is performed against a deterministic developed 16-bit RGB reference, preferably
`*_master16.tif`, with pinned color-management and development settings.

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

For external canonical assets:

```bash
APEX_EVAL_ASSET_ROOT=/Volumes/apex_eval_assets \
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/apex_real_estate_v1/evalset.example.json \
  --output-dir output/apex_eval \
  --emit-report on
```

The output is `apex_eval_report.json`. It records corpus readiness and, when candidate outputs are supplied,
authoritative `apex_metrics.v1` candidate metrics:

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
`canonical_scoring_eligible`, and `canonical_scoring_blocked_reason`. Optional RAW, ICC/profile, and working-color
provenance fields are emitted only when present.

Candidate-output scoring compares against `reference_path`, not `asset_ref`. For canonical assets, the reference and
candidate output must be readable 16-bit TIFF/TIF images with matching dimensions; mismatches fail closed and are not
resized. Candidate entries use `metric_contract=apex_metrics.v1` and `metrics_authoritative=true` only when
`compute_apex_metrics` produced valid metric-status evidence. Legacy visible-delta-only metrics are compatibility
evidence and are not promotion-authoritative.

Depth benchmark reports also separate model input derivation from the evaluation target. Depth Pro, DA3, and other
depth backends may consume normalized 8-bit or downsampled tensors, but benchmark cases must still record the
16-bit source path as `evaluation_target` when one exists. RAW and working-color provenance belongs under
`evaluation_target`, not `model_input`.

APEX evidence bundles may attach existing candidate outputs and explicit Materials V3 telemetry:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/apex_real_estate_v1/evalset.example.json \
  --asset-root /Volumes/apex_eval_assets \
  --output-dir output/apex_eval \
  --candidate-output materials_v3:pool_water_stone_001=output/pool_materials_v3_master16.tif \
  --candidate-evidence materials_v3:pool_water_stone_001=output/pool_materials_v3_evidence.json \
  --run-scope-asset-id pool_water_stone_001 \
  --emit-evidence-bundle on \
  --synthetic-data off
```

When a run scope is provided, every scoped canonical asset must have candidate output evidence before promotion can be
eligible. When run scope is omitted, candidate outputs and required candidate telemetry must cover every canonical asset
in the manifest. Missing Materials V3 telemetry, invalid `apex_metrics.v1` statuses, unsupported bit depth, dimension
mismatch, `APEX_MATERIALS_PIXEL_OPS_EMPTY`, raw CLIP pixel-op authority, or `synthetic_data=true` blocks promotion.

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
  The strict APEX Materials V3 no-op failure code is `APEX_MATERIALS_PIXEL_OPS_EMPTY`.
- Synthetic APEX performance reports are plumbing and regression evidence only. They are not real APEX image-quality
  evidence.

The long-term APEX score should combine depth edge fidelity, material precision, pixel-op false positive risk,
visible delta metrics, and manual APEX score. A missed enhancement is acceptable; a wrong material edit is not.
