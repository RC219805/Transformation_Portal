# APEX Model-Family Characterization Protocol

This protocol defines the offline model-family characterization matrix for APEX
quality work. The report compares explicit depth and segmentation families
without running models, reading raw artifacts, or changing existing evidence
schemas.

The characterization report is not a promotion verdict. It sits above local
evidence such as `apex_evidence_bundle.v1` and records which families are
declared, governed, observed, and comparable.

## Scope

`tools/characterize_apex_model_families.py` emits
`apex_model_family_characterization_report.v1`.

The tool does not:

- run model inference
- call `tools/run_apex_eval.py`
- read run cards, batch files, combined manifests, provenance files, TIFFs,
  NPZ masks, NPY arrays, or PNG debug artifacts
- tune Materials V3 thresholds or depth gates
- mutate `apex_evidence_bundle.v1` or `apex_eval_report.v1`

Observed local evidence must be reduced to an `apex_redacted_summary.v1` file
before it is attached to a family row.

## Family Names

Canonical family names always include tier:

```text
materials_v{N}_{depth}_{seg}_{quality_tier}[_pbr][_v2]
```

Examples:

```text
materials_v3_da3_efficientsam_apex
materials_v3_depthpro_sam2_premium_pbr
```

`materials_version` defaults to `3`. If `candidate_family` is omitted from a
family spec, the tool derives it. If it is supplied and does not match the
canonical name, the row is marked `spec_validation.status: name_mismatch`.

## Command

Use the checked-in default matrix:

```bash
.venv/bin/python tools/characterize_apex_model_families.py \
  --family-file config/apex_family_matrix.json \
  --non-commercial-ok off \
  --accept-depth-pro-license off \
  --output output/apex_model_family_characterization_report.json \
  --now 2026-04-25T00:00:00Z
```

Use a matrix expression for local exploration:

```bash
.venv/bin/python tools/characterize_apex_model_families.py \
  --matrix "depth_backend=da3,depth_pro;segmentation_backend=sam2,efficientsam;quality_tier=apex;pbr_enabled=false;v2_enabled=false" \
  --output output/apex_model_family_characterization_report.json \
  --now 2026-04-25T00:00:00Z
```

Attach mocked observations only by explicit family binding:

```bash
.venv/bin/python tools/characterize_apex_model_families.py \
  --family "depth_backend=da3,segmentation_backend=efficientsam,quality_tier=apex,pbr_enabled=false,v2_enabled=false" \
  --observation "candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked,fallback_used=false,runtime_ms=1234,promotion_verdict=eligible,metric_contract=apex_metrics.v1,mask_evidence_status=ok" \
  --output output/apex_model_family_characterization_report.json \
  --now 2026-04-25T00:00:00Z
```

## Governance

DA3 rows are `commercial_ready`.

Depth Pro rows are `license_blocked` unless both acknowledgements are supplied:

```text
--non-commercial-ok on
--accept-depth-pro-license on
```

With both acknowledgements, Depth Pro rows are `research_only`, not
commercial-ready.

## Redacted Summaries

`observed_local` rows must use `apex_redacted_summary.v1`. Required fields are:

```text
schema_version
source_evidence_sha256
candidate_family
```

Only aggregate fields are allowed. Path-like values and metadata-bearing keys
are rejected, including EXIF/IPTC/XMP/GPS, camera/lens/serial/creator/copyright,
run cards, batch files, combined/provenance JSON, TIFF, NPZ, NPY, and PNG
references.

The controlled local `materials_v3` evidence maps externally to:

```text
materials_v3_da3_efficientsam_apex
```

The exploratory Depth Pro/SAM2/PBR premium evidence maps separately to:

```text
materials_v3_depthpro_sam2_premium_pbr
```

Do not rewrite existing evidence bundle candidate IDs to encode this mapping.

## Artifact Guardrail

Do not commit real local evidence or provenance artifacts:

```text
output/apex_eval_real*
output/lux_depth_v3_apex/
run_card_*.json
batch_*.json
*_combined.json
*_combined_provenance.json
*.npz
*.npy
*.tif
*.tiff
*.png
```

Real provenance and debug artifacts can include creator/copyright fields, camera
serials, GPS/drone metadata, local paths, EXIF/IPTC/XMP history, and source file
names. Keep them local unless explicitly redacted under fixture policy.
