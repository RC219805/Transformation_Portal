# APEX Real Estate v1

This evalset is the initial real canonical APEX real-estate corpus.

The repository commits metadata only. Canonical 16-bit TIFF/TIF references, RAW files, ICC/profile assets, delivery derivatives, candidate outputs, and generated evidence bundles must remain outside git.

`evalset_id` and top-level `version` identify the `apex_real_estate_v1` corpus family. `metadata.manifest_revision` tracks the initial metadata-only real manifest revision.

## External asset root

Expected local layout:

```text
$APEX_EVAL_ASSET_ROOT/
  apex_real_estate_v1/
    reference_16bit/
      pool_water_stone_001_master16.tif
      kitchen_glass_metal_001_master16.tif
      exterior_foliage_sky_001_master16.tif
    delivery_8bit/
    source_raw/
    annotations/
    baselines/
```

Recommended local root:

```bash
export APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets"
```

## Canonical evidence rule

Canonical APEX quality evidence requires externally resolved, deterministic developed 16-bit TIFF/TIF references.

RAW files, ICC/profile fields, and working-color fields are provenance only.

8-bit delivery derivatives, renderings, and synthetic fixtures are NOT canonical APEX quality evidence.

For this first manifest, `asset_ref` and `reference_path` intentionally point to the same canonical 16-bit TIFF/TIF master.

## Local audit

```bash
APEX_EVAL_ASSET_ROOT="$HOME/apex_eval_assets" \
.venv/bin/python tools/audit_apex_assets.py \
  --evalset evalsets/apex_real_estate_v1/evalset.json \
  --require-canonical on \
  --output-dir output/apex_asset_audit
```

Expected:

```text
exit 0
canonical_scoring_eligible_count >= 3
no checksum_mismatch
no missing canonical references
```

## Non-binary policy

Do NOT commit:

```text
*.tif
*.tiff
*.dng
*.cr2
*.cr3
*.nef
*.arw
*.raf
*.orf
*.rw2
output/
```

Large delivery JPEGs, including files under `delivery_8bit/`, must stay under the external asset root.

## Next step

After this manifest lands, the next PR documents the first real evidence run using:

```text
--candidate-output
--candidate-evidence
--emit-evidence-bundle on
--synthetic-data off
--run-scope-asset-id
```

Generated outputs must remain uncommitted.
