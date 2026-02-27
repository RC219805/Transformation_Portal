# ADR-041 Phase 4F External Verification and Trust Export for Phase 4 Capture Provenance

## Status
Proposed

## Date
2026-02-27

## Executive Summary

Phase 4F introduces an isolated, dependency-minimal verification surface for the
Phase 4 capture provenance chain. The verifier recomputes and validates all
Phase 4C/4D/4E invariants (schema validity, object hashes, provenance entry hashes,
Merkle root) deterministically and produces a stable, machine-readable verification
report suitable for external auditors and downstream attestation workflows.

This phase does not change extraction, canonicalization, or existing artifact formats.
It adds a verifier and an optional deterministic report artifact that binds the
verification result to the exact inputs and contract versions.

## Context

Phase 4 currently provides deterministic production of:

- Phase 4C: capture metadata objects (`tp.meta.capture.v1`)
- Phase 4D: metadata manifest with object hashes (`tp.meta.capture_manifest.v1`)
- Phase 4E: provenance manifest + Merkle root (`tp.meta.provenance.v1`, `tp.meta.provenance_merkle.v1`)

Determinism is enforced by CI parity and canonical serialization rules. However,
external validation currently depends on executing the repository toolchain
and/or trusting the producing environment.

The repository needs an explicit, minimal, and reproducible verifier surface
that enables third parties to validate the Phase 4 chain from artifacts alone,
without relying on the producer runtime or broader pipeline.

## Decision

### D1. Introduce a Standalone Phase 4 Verifier CLI

Add a Phase 4F verifier tool:

- `tools/verify_phase4_chain.py`

The verifier accepts Phase 4 artifact inputs (paths) and validates:

- schema validity for each artifact,
- record alignment by `relative_path`,
- recomputation of `metadata_sha256` from canonical object bytes (Phase 4D rules),
- recomputation of `provenance_entry_sha256` preimages (Phase 4E rules),
- recomputation of Merkle root from provenance leaves (Phase 4E rules),
- invariant checks: uniqueness, ordering requirements under strict mode,
  and contract version consistency.

The verifier MUST operate offline and MUST NOT read system time, network state,
host metadata, environment-dependent paths, or non-input files in verification logic.

### D2. Verifier Dependency Boundary Is Explicit and Minimal

The verifier implementation MUST NOT import heavy/optional pipeline dependencies
(e.g., ML stacks). It should depend only on:

- Python standard library
- schema validation dependency (if used) MUST be explicit and pinned

If JSON Schema validation is implemented via `jsonschema`, it MUST be treated as a
required verifier dependency (not optional) for deterministic audit behavior.

### D3. Strict Mode Semantics Are Frozen for Verification

The verifier supports `--strict` with the following frozen semantics:

- inputs MUST already be sorted by `relative_path` when ordering is defined,
- aligned inputs MUST match 1:1 by `relative_path`,
- any mismatch in recomputed hashes, Merkle root, or schema validity is fatal,
- strict mode does not reinterpret Phase 4C warnings; it verifies what was emitted.

Phase 4F MUST NOT change Phase 4C strict-mode extraction policy.

### D4. Deterministic Verification Report Artifact (Optional Output)

Phase 4F introduces one additive, deterministic artifact surface:

- `artifacts/verification_report.tp.meta.verification_report.v1.json`

Produced only on successful verification unless explicitly requested otherwise.

The report MUST be deterministic given identical inputs, and MUST contain no
runtime timestamps or environment-dependent fields.

The report MUST include:

- `verification_contract_version` (const)
- `inputs` digests (sha256 of each input file, plus contract versions)
- `computed` values (recomputed metadata hashes digest summary, Merkle root)
- `verifier` identity block (`name`, `version`, and deterministic build identifier)

### D5. Verification Report Schema Is Versioned and Additive

Add a Phase 4 schema:

- `schemas/phase4/verification_report.schema.json`

This schema is authoritative for the report object and follows Phase 4 conventions:

- `additionalProperties: false`
- explicit `required` lists
- canonical version constants

### D6. Canonical Serialization Rules for Report Are Frozen

When a report is emitted, it MUST be serialized canonically:

- UTF-8
- `sort_keys=True`
- compact separators `(",", ":")`
- `ensure_ascii=False`
- `allow_nan=False`
- single trailing LF (`\n`) required

No non-deterministic fields (timestamps, absolute paths, host IDs) are allowed.

### D7. Exit Code Contract Is Frozen

`tools/verify_phase4_chain.py` exit codes are frozen:

- `0`: success (verification passed)
- `31`: malformed input / invalid arguments / unreadable files
- `32`: schema validation failure
- `33`: alignment failure (path sets mismatch, duplicates, ordering violation in strict)
- `34`: metadata hash mismatch (Phase 4D recomputation mismatch)
- `35`: provenance entry hash mismatch (Phase 4E recomputation mismatch)
- `36`: Merkle root mismatch (Phase 4E recomputation mismatch)
- `37`: report write failure

Exit-code assignments MUST be recorded in `docs/contracts/exit_codes.md`.

### D8. Cross-Runtime Parity Requirement

Verification results MUST be stable across supported Python runtimes (3.11, 3.12):

- identical PASS/FAIL outcomes,
- identical recomputed digests and Merkle root,
- byte-identical report output when emitted.

A CI parity gate MUST enforce this requirement using golden Phase 4 fixtures.

### D9. Phase 4 Contracts Remain Immutable; Changes Are Versioned

Phase 4F MUST NOT mutate:

- `tp.meta.capture.v1`
- `tp.meta.capture_manifest.v1`
- `tp.meta.provenance.v1`
- `tp.meta.provenance_merkle.v1`
- Phase 4D object-hash canonicalization semantics
- Phase 4E preimage and Merkle semantics
- Evidence Bundle / Phase 3 contracts

Any incompatible change requires a version bump and a new ADR.

## Non-goals

- This ADR does not introduce new capture fields or extraction behavior.
- This ADR does not change canonicalization policy or rounding rules.
- This ADR does not implement network notarization, timestamping, or signing.
- This ADR does not replace existing Phase 3 evidence bundle verification tools.

## Alternatives Considered

- Rely on internal CI determinism gates only:
  rejected because external auditors cannot reproduce verification independently.
- Encode precision constraints in schema using float arithmetic (`multipleOf`):
  rejected as brittle under IEEE-754 representation (already addressed in ADR-040).
- Emit verification output with timestamps and host metadata:
  rejected because it breaks deterministic report reproducibility.

## Implementation Plan

- Add verifier core logic in a new Phase 4 module (library-first).
- Implement CLI wrapper with frozen exit codes.
- Add a deterministic verification report schema and canonical writer.
- Add golden fixtures and tests that validate parity across runtimes.
- Add CI job to run verifier under Python 3.11 and 3.12 and compare outputs.

## Success Metrics

- Verifier passes on existing golden Phase 4 fixtures.
- Verifier fails deterministically (stable exit code) on:
  - schema violations,
  - path alignment mismatches,
  - metadata hash mismatches,
  - provenance entry mismatches,
  - Merkle root mismatches.
- Verification report bytes are identical across runs and across Python 3.11/3.12.
- CI parity gate fails on any drift in recomputation semantics or report serialization.

## Enforcement

Enforcement occurs through:

- verifier unit tests and golden fixtures,
- cross-runtime parity CI workflow,
- schema validation in CI (`Draft202012`),
- determinism gates (existing) plus Phase 4F-specific parity checks,
- ADR governance requirements for any semantic change.

## Consequences

Positive:

- enables third-party verification of Phase 4 artifacts without trusting producers,
- reduces audit friction and increases defensibility,
- isolates verification logic from pipeline complexity and dependencies,
- creates an explicit trust-export surface (deterministic report) for attestation.

Trade-offs:

- adds another contract surface (verification report schema) that must be versioned,
- requires ongoing maintenance of verifier parity across supported runtimes.

## References

- `schemas/phase4/metadata.schema.json`
- `schemas/phase4/metadata_manifest.schema.json`
- `schemas/phase4/provenance_manifest.schema.json`
- `schemas/phase4/provenance_merkle.schema.json`
- `schemas/phase4/verification_report.schema.json`
- `docs/contracts/phase4d_metadata_hash_canonicalization.md`
- `docs/contracts/exit_codes.md`
- `tools/extract_capture_metadata.py`
- `tools/build_metadata_manifest.py`
- `tools/build_provenance_manifest.py`
- `tools/build_provenance_merkle.py`
- `tools/verify_phase4_chain.py`
- `tools/check_phase4f_verifier_cross_runtime.py`
- `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`
- `docs/architecture/ADR-040-remove-multipleof-floats-tp-meta-capture-v1.md`
