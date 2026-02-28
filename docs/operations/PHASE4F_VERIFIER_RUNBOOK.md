# Phase 4F Verifier Runbook

## Purpose

Operational runbook for externally verifying the Phase 4 capture provenance chain
(`tp.meta.capture.v1` -> `tp.meta.capture_manifest.v1` ->
`tp.meta.provenance.v1` -> `tp.meta.provenance_merkle.v1`) using the standalone
Phase 4F verifier.

## Inputs

Required artifacts:

- `capture_metadata.tp.meta.capture.v1.json`
- `metadata_manifest.tp.meta.capture_manifest.v1.json`
- `provenance_manifest.tp.meta.provenance.v1.json`
- `provenance_merkle.tp.meta.provenance_merkle.v1.json`

## Standard Verify Command

```bash
python tools/verify_phase4_chain.py \
  --capture-metadata artifacts/capture_metadata.tp.meta.capture.v1.json \
  --metadata-manifest artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json \
  --provenance-manifest artifacts/provenance_manifest.tp.meta.provenance.v1.json \
  --provenance-merkle artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json
```

## Deterministic Report (Success Only by Default)

```bash
python tools/verify_phase4_chain.py \
  --capture-metadata artifacts/capture_metadata.tp.meta.capture.v1.json \
  --metadata-manifest artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json \
  --provenance-manifest artifacts/provenance_manifest.tp.meta.provenance.v1.json \
  --provenance-merkle artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json \
  --out-report artifacts/verification_report.tp.meta.verification_report.v1.json
```

## Failure Report (Opt-In)

```bash
python tools/verify_phase4_chain.py \
  --capture-metadata artifacts/capture_metadata.tp.meta.capture.v1.json \
  --metadata-manifest artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json \
  --provenance-manifest artifacts/provenance_manifest.tp.meta.provenance.v1.json \
  --provenance-merkle artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json \
  --out-report artifacts/verification_report.tp.meta.verification_report.v1.json \
  --write-report-on-failure
```

## Strict vs Non-Strict

- Default is strict ordering: `--strict-input-order` (enabled by default).
- Strict mode requires arrays already sorted by `relative_path`.
- Non-strict mode (`--no-strict-input-order`) relaxes ordering checks only; it
  does not relax schema validity, hash integrity, or Merkle integrity checks.

## Exit Code Contract

- `0`: verification passed
- `31`: malformed input / invalid arguments / unreadable files
- `32`: schema validation failure
- `33`: alignment failure (set mismatch, duplicates, ordering, version mismatch)
- `34`: metadata hash mismatch
- `35`: provenance entry hash mismatch
- `36`: Merkle mismatch
- `37`: report write failure

See `docs/contracts/exit_codes.md` for normative contract details.

## Remediation Map

- `31`: verify file paths, file readability, and CLI flags.
- `32`: validate artifact JSON shape against Phase 4 schemas.
- `33`: fix `relative_path` set/order consistency and contract-version drift.
- `34`: regenerate metadata manifest from canonical capture metadata.
- `35`: regenerate provenance manifest from aligned capture+metadata manifest.
- `36`: regenerate provenance merkle from validated provenance manifest.
- `37`: fix output destination permissions/path and retry report emission.

## Cross-Runtime Parity Gate

Cross-runtime parity is enforced by CI in
`.github/workflows/determinism-gate.yml` at step
`Phase 4F verifier cross-runtime parity gate (3.11 vs 3.12)`.

Expected behavior:

- CI step green: parity passed (identical outcomes and report bytes)
- CI step failure: inspect determinism-gate logs for parity mismatch or runtime
  invocation failure details.
