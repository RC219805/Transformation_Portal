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
  - `flag_xmp_orphan_no_image_any_raster`
  - `flag_sidecar_only`
  - `flag_video_still_collision`
  - `flag_container_imovie`
  - `flag_still_missing_datetime`

- `anomaly_hotspots.csv`
  Aggregations by `(origin_drive, partition)` for triage.

- `summary.json`
  High-signal rollups for dashboards / procurement packets.

## Dependencies

This tool requires optional Python dependencies that may not be present in minimal environments.

**Recommended (pinned versions for reproducibility):**

```bash
pip install -r requirements/tools-archive.txt
```

**Alternatively (latest versions):**

```bash
pip install numpy pandas
```

The pinned versions in `requirements/tools-archive.txt` are tested for deterministic behavior across platforms.

## Usage

```bash
python tools/archive_manifest_reports.py \
  --input /path/to/archive_manifest.csv \
  --outdir /path/to/out_reports \
  --chunk-mb 50
```

For fail-closed reproducibility enforcement (recommended in CI/governed runs):

```bash
python tools/archive_manifest_reports.py \
  --input /path/to/archive_manifest.csv \
  --outdir /path/to/out_reports \
  --validate-schemas \
  --strict-root-marker \
  --min-root-marker-coverage 0.95
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

### Verify Determinism (Recommended)

After recombining (or on any run), quickly confirm byte-for-byte reproducibility:

```bash
# Linux / macOS
sha256sum archive_index_normalized.csv.gz \
         asset_grouping_report.csv.gz \
         anomaly_hotspots.csv \
         summary.json

# Windows PowerShell
Get-FileHash archive_index_normalized.csv.gz, asset_grouping_report.csv.gz, anomaly_hotspots.csv, summary.json -Algorithm SHA256 | Format-Table Hash, Path
```

> Note: `.csv.gz` outputs are written deterministically with fixed gzip `mtime=0` and no embedded source filename.

## Notes

- These reports are **no-hash** by design. Hash-first identity (SHA-256) belongs in the vault ingest layer.
- The schemas for each output are in `docs/archive/schemas/` (including `summary.schema.json`).
- Phase 3 hash-first follow-on is documented in `docs/archive/ARCHIVE_MANIFEST_PHASE3.md`.
