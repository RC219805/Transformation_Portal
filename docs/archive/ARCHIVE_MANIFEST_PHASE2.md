# Archive Manifest Phase 2 Reports

This package turns an ExifTool CSV **Archive_Manifest** into deterministic, audit-oriented reports aligned with the
Transformation Portal governance model (determinism + provenance + enforceable policy).

## Inputs

- ExifTool CSV (example: `archive_manifest.csv`) with at least:
  `SourceFile, FileName, FileSize, Model, DateTimeOriginal, ImageSize, Quality, FocalLength, ShutterSpeed, Aperture, ISO, WhiteBalance, Flash`

## Outputs

Generated into an output directory:

- `archive_index_normalized.csv.gz`
  Per-file normalized index: role/category tagging + path canonicalization + approx bytes.

- `asset_grouping_report.csv.gz`
  One row per **basekey** = `(directory + basename)`. Includes anomaly flags:
  - `flag_xmp_orphan_no_raw_jpeg` (matches exec-summary definition)
  - `flag_sidecar_only`
  - `flag_video_still_collision`
  - `flag_container_imovie`
  - `flag_still_missing_datetime`

- `anomaly_hotspots.csv`
  Aggregations by `(origin_drive, partition)` for triage.

- `summary.json`
  High-signal rollups for dashboards / procurement packets.

## Usage

```bash
python tools/archive_manifest_reports.py \
  --input /path/to/archive_manifest.csv \
  --outdir /path/to/out_reports \
  --chunk-mb 50
```

### Recombine chunked outputs

If `--chunk-mb` produced `.partNNN` files:

```bash
cat archive_index_normalized.csv.gz.part* > archive_index_normalized.csv.gz
gunzip -k archive_index_normalized.csv.gz
```

On Windows PowerShell:

```powershell
Get-Content archive_index_normalized.csv.gz.part* -Encoding Byte | Set-Content archive_index_normalized.csv.gz -Encoding Byte
gzip -d archive_index_normalized.csv.gz
```

## Notes

- These reports are **no-hash** by design. Hash-first identity (SHA-256) belongs in the vault ingest layer.
- The schemas for each output are in `docs/archive/schemas/`.
