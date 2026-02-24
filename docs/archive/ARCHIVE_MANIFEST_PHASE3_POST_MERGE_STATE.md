# Archive Manifest Phase 3.1 Post-Merge Architectural State Assessment

## Status
Adopted (PR #1008 merged to `main`)

## Current Layer Model

```text
Phase 2   -> Deterministic Identity
Phase 3   -> Deterministic Integrity (hash + Merkle + verify)
Phase 3.1 -> Detached Attestation (sign + verify)
```

Each layer is explicitly versioned, contract-bounded, and CI-gated.

Phase 3 deterministic integrity and Phase 3.1 detached attestation are now
operational production infrastructure.

## What This Now Provides

### 1. Deterministic archive integrity

- Deterministic content hashing and Merkle aggregation are enforced by contract.
- Schema-locked artifacts and golden regression checks protect output stability.
- Cross-ISA determinism parity remains part of the integrity surface.

### 2. Cryptographically attested integrity

- Detached Ed25519 signatures bind exact artifact bytes.
- Envelope metadata binds artifact filename and SHA-256 digest.
- Trust semantics are explicit and separated from deterministic integrity logic.

### 3. Governance-grade operational discipline

- Determinism harness, schema guards, and dependency controls are active gates.
- Artifact boundaries are maintained across integrity and attestation layers.
- Enforcement posture is reproducible and reviewable.

## Confirmed Architectural State

### Deterministic archive integrity (Phase 3)

- Content hashing is deterministic and schema-governed.
- Merkle aggregation and verification are stable across host/ISA parity checks.
- Golden regression enforcement protects output surface and method metadata.

### Cryptographic detached attestation (Phase 3.1)

- Detached Ed25519 signatures bind to exact artifact bytes.
- Envelope binds filename and SHA-256 digest to the signed payload.
- Envelope contract is versioned and malformed input paths are explicit.
- Optional crypto import handling preserves deterministic execution semantics.

### Governance and enforcement discipline

- Artifact boundary enforcement is active.
- Schema drift and dependency drift checks are active.
- Determinism harness remains the authority for reproducible output.

## Architectural Quality Snapshot

| Dimension | Rating |
| --- | --- |
| Determinism discipline | Excellent |
| Contract clarity | Excellent |
| Forward compatibility | Strong |
| Crypto correctness | Strong |
| CI integration | Strong |
| Scope containment | Excellent |
| Long-term evolvability | Strong |

No structural regressions were identified in the merge outcome.

## Guardrails: What Not To Do Next

- Do not add timestamping fields into Phase 3 artifacts.
- Do not auto-sign in CI as part of the deterministic integrity path.
- Do not introduce non-versioned envelope extensions.
- Do not mix signing logic into Phase 3 artifact generation.
- Do not broaden signing scope without an explicit envelope version bump.

## Recommended Next Increments

### Phase 3.2 - Detached timestamp anchoring

Goal: add existence-at-time evidence without modifying Phase 3 or Phase 3.1 artifacts.

- Implement detached timestamp CLI (for example `timestamp_merkle_signature.py`).
- Anchor either artifact digest or detached signature bytes via RFC 3161 TSA.
- Emit detached `.tsr` artifact only.
- Keep networked timestamping outside default CI deterministic gates.

Outcome: integrity + identity + time, while preserving layering.

### Phase 3.3 - Multi-signature envelope

Evolve to an envelope with a signatures array and explicit `envelope_version` increment.

Use cases:
- Institutional co-signing
- Vendor/internal dual attestation
- Separation-of-duties workflows

### Phase 3.4 - Evidence bundle format

Define canonical compliance packet content:

- `merkle_roots.json`
- `hash_manifest.csv.gz` digest
- `hash_summary.json`
- Detached signature envelope(s)
- Tool/version metadata

Sign canonical manifest JSON for bundle members, not container bytes.

### Phase 3.5 - Merkle method hardening (optional)

Future versioned method upgrade candidates:

- Domain-separated leaf/internal hashing
- Duplicate-last ambiguity removal
- `tree_method_version` increment and migration plan

## Strategic Recommendation

- Freeze Phase 3 and 3.1 for one sprint.
- Tag release line for attested integrity milestone (for example `v0.4.0-attested-integrity`).
- Start Phase 3.2 on a clean branch with detached timestamp scope only.

This preserves deterministic integrity as core infrastructure while extending auditability through detached layers.

## Strategic Positioning

The current state supports:

- Dataset defensibility and external verification workflows
- Production of structured, cryptographically verifiable integrity evidence
  appropriate for regulatory or disclosure workflows
- Cryptographically verifiable archive attestations suitable for audit,
  disclosure, and adversarial review contexts
