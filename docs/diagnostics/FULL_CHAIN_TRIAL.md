# Full-Chain Determinism Trial

The determinism harness is [`scripts/diagnostics/full_chain_determinism_trial.sh`](/Users/rc/Projects/Transformation_Portal/scripts/diagnostics/full_chain_determinism_trial.sh).

It validates the full Phase 4 chain:

1. 4C `extract_capture_metadata.py`
2. 4D `build_metadata_manifest.py`
3. 4E `build_provenance_manifest.py`
4. 4E `build_provenance_merkle.py`

It records per-run artifacts, per-step logs, and hash ledgers.

## Quick Start

### RAW mode (real files)

```bash
scripts/diagnostics/full_chain_determinism_trial.sh \
  --input-root ./trial_dataset/input_raw \
  --runs 2
```

### Artifact mode (CI-friendly)

```bash
scripts/diagnostics/full_chain_determinism_trial.sh \
  --capture-metadata tests/golden/phase4/expected_capture_metadata.tp.meta.capture.v1.json \
  --runs 2 \
  --no-tmp \
  --compare both
```

## Useful Flags

- `--verbose`: stream step logs to stdout while still writing log files.
- `--no-input-hash`: skip expensive input SHA-256 generation.
- `--compare raw|canonical|both`: comparison gate mode.
- `--strip-json-keys k1,k2`: remove keys before canonical JSON hashing.
- `--tool-4c/--tool-4d/--tool-4e-prov/--tool-4e-merkle`: override tool paths.

## Output Layout

Each run writes to `trial_runs/full_chain_determinism_<timestamp>/` (or `--out`):

- `trial_meta.txt`
- `input.files.txt`
- `input.sizes.txt`
- `input.sha256.txt` (unless `--no-input-hash`)
- `run_XX/artifacts/`
- `run_XX/logs/4C.log`, `4D.log`, `4E_prov.log`, `4E_merkle.log`
- `run_XX/artifacts.sha256` (byte hashes)
- `run_XX/artifacts.canon.sha256` (canonical JSON hashes)
- `run_XX/artifacts.sizes`

If `/tmp` mode is enabled, matching `tmp_run_XX/` directories are generated.

## Triage Guidance

- `raw mismatch` + `canonical match`: likely serialization/formatting drift.
- `canonical mismatch`: likely semantic drift.
- `primary vs /tmp mismatch`: CWD/relocatability defect.

Attach these files when reporting failures:

1. `trial_meta.txt`
2. offending `artifacts*.sha256` ledgers
3. offending step logs under `run_XX/logs/`
