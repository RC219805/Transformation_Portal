# APEX Canonical Corpus Bootstrap

APEX canonical assets are large production images and are not committed to the
repository. Manifests stay deterministic in git; image, RAW, ICC, annotation,
and baseline binaries live under a local or mounted asset root.

## Asset Root

Use either the CLI flag or environment variable:

```bash
.venv/bin/python tools/run_apex_eval.py \
  --evalset evalsets/apex_real_estate_v1/evalset.example.json \
  --asset-root /Volumes/apex_eval_assets \
  --output-dir output/apex_eval
```

```bash
APEX_EVAL_ASSET_ROOT=/Volumes/apex_eval_assets \
.venv/bin/python tools/audit_apex_assets.py \
  --evalset evalsets/apex_real_estate_v1/evalset.example.json \
  --require-canonical on
```

Resolution order is:

1. absolute manifest path
2. explicit `--asset-root`
3. `APEX_EVAL_ASSET_ROOT`
4. repository-relative behavior

Relative `asset_ref` and `reference_path` values resolved under an asset root
must remain inside that root. Traversal outside the root fails closed with
`path_escapes_asset_root`.

## External Directory Shape

```text
apex_real_estate_v1/
  source_raw/
  reference_16bit/
  delivery_8bit/
  annotations/
  baselines/
```

`source_raw_path`, RAW-development references, ICC/profile fields, and
working-color fields are provenance metadata only in the v1 contract. They are
not resolved, loaded, existence-checked, or allowed to affect readiness or
canonical scoring eligibility.

## Audit Behavior

`tools/audit_apex_assets.py` writes `apex_asset_audit_report.json` and exits:

- `0`: audit complete; missing assets are report-only in default mode
- `1`: checksum mismatch on a present asset
- `2`: `--require-canonical on` failed because canonical assets were missing,
  invalid, or zero assets were canonical-scoring eligible
- `3`: malformed evalset or input validation error

In default mode, missing external canonical assets are reported but do not fail
the process. Use `--require-canonical on` for local canonical validation when
the external asset root is mounted.

## First Real Manifest

`evalsets/apex_real_estate_v1/evalset.example.json` is metadata-only and uses
placeholder checksums. The real `evalset.json` should be added only after assets
are selected and checksums are known.

Initial minimum:

- 1 pool/water/stone asset
- 1 kitchen/glass/metal asset
- 1 exterior foliage/sky asset

Target corpus size is 8-12 real, high-quality 16-bit TIFF/TIF references.
