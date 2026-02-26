# ADR-038 Operational Determinism Enforcement Layer

## Status
Accepted

## Date
2026-02-26

## Executive Summary

This ADR establishes an operational determinism enforcement layer for the
Phase 4 metadata/provenance chain. The repository now requires repeatable
full-chain replay diagnostics and CI gating so determinism is validated as a
runtime property, not only as a contract/property-test assumption.

This decision is recorded as the governance anchor for merge commit:
`862ee60d0d53c24ee471d7e30311e103cbc1397d`.

## Context

Prior phases locked deterministic contracts for:

- deterministic identity and hashing at ingest/integrity boundaries,
- provenance binding surfaces,
- Merkle tree contract behavior,
- bundle root anchoring and verification.

Those controls were necessary but incomplete: they did not prove that the
entire Phase 4 chain remained deterministic under repeated execution and
invocation-context variance (for example, different working directories).

Operational risk before this decision:

- byte drift escaping unit/fixture scope,
- CWD-sensitive behavior hidden until integration execution,
- slower triage when failures required interactive reruns for context.

## Decision

### D1. Full-Chain Diagnostic Harness Is a Governance Surface

The repository adopts `scripts/diagnostics/full_chain_determinism_trial.sh`
as the operational replay harness for the Phase 4 chain:

1. 4C `extract_capture_metadata.py`
2. 4D `build_metadata_manifest.py`
3. 4E `build_provenance_manifest.py`
4. 4E `build_provenance_merkle.py`

The harness supports:

- repeated deterministic replay,
- per-step log capture,
- artifact hash ledgers,
- input-manifest recording,
- relocatability checks (`/tmp` execution mode),
- RAW and artifact-driven execution modes.

### D2. Byte-Level Determinism Remains the Primary Gate

Raw artifact SHA-256 ledger equality is the normative determinism gate.
Canonical JSON hashing is supplementary classifier evidence for triage and must
not silently override raw mismatch failures unless explicitly requested by run
mode.

### D3. CI Enforces Operational Replay

`.github/workflows/diagnostic-trial.yml` is accepted as the CI enforcement
workflow for:

- deterministic full-chain replay in CI-friendly artifact mode,
- evidence bundle manifest generation and verification checks,
- bundle root compute/write/verify checks.

### D4. Execution Context Independence Is Required

The determinism harness must validate context independence for local diagnostics
through primary-vs-`/tmp` replay comparison. Divergence is treated as a defect
in path or environment invariants.

### D5. Trial Artifacts Are Operational Byproducts

`trial_runs/` outputs are classified as local operational artifacts and are
excluded from VCS tracking via `.gitignore` to reduce accidental commit risk.

## Consequences

Positive:

- determinism is enforced under real invocation variability,
- CI catches operational drift before downstream verification/signing,
- forensic triage is faster with per-step logs and hash ledgers.

Trade-off:

- CI runtime and diagnostic artifact volume increase modestly,
- governance discipline must maintain script/workflow compatibility with
  evolving Phase 4 tooling.

## Enforcement

Enforcement occurs via:

- `scripts/diagnostics/full_chain_determinism_trial.sh`,
- `.github/workflows/diagnostic-trial.yml`,
- associated docs under `docs/operations/FULL_CHAIN_TRIAL.md`.

Any change that weakens replay scope, hash-ledger requirements, or CI
enforcement semantics requires a superseding ADR.

## References

- Merge anchor:
  `862ee60d0d53c24ee471d7e30311e103cbc1397d`
- `scripts/diagnostics/full_chain_determinism_trial.sh`
- `.github/workflows/diagnostic-trial.yml`
- `docs/operations/FULL_CHAIN_TRIAL.md`
- `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`
- `docs/architecture/ADR-037-repo-root-contract.md`
