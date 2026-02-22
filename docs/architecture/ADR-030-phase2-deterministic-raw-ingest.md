# ADR-030: Phase II Deterministic RAW Ingest

**Status:** Implemented
**Date:** 2026-02-20
**Updated:** 2026-02-22
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-021, ADR-023, ADR-027, ADR-029
**Enforcement:** Determinism harness conformance (`SPEC-DH-001`)

---

## Executive Summary

**Decision:** Define a deterministic ingest boundary for all RAW-origin inputs and certify output parity against a bounded numeric envelope.

**Boundary Contract:** Ingest MUST emit canonical linear color tensors in `xyz_d50_linear_fp32` with stable geometry metadata and provenance hashes.

**Phase II Principle:** Determinism is a first-class interface contract, not a best-effort implementation detail.

---

## Implementation Status

**Phase II Exit Criteria (Met):**

1. ✅ **Harness passes on target runners** - Cross-ISA parity verified on linux-x86_64 and arm64 via `determinism-cross-isa.yml`
2. ✅ **CI blocks merges on ingest-bound violations** - `determinism-gate.yml` and `ingest_contract_validation.yml` enforce merge gates
3. ✅ **Ingest produces canonical payload** - `camera_native_linear` contract emits `xyz_d50_linear_fp32` + geometry + provenance
4. ✅ **Baseline update mechanics are auditable** - Manifest schema v3 includes probe_version, probe_policy, and environment fingerprints

**Implementation Artifacts:**

| Component | Location | Description |
|-----------|----------|-------------|
| Canonical Ingest | `src/transformation_portal/spatial_ai/ingest/phase2_camera_native_linear.py` | Phase II certified RAW → xyz_d50_linear_fp32 |
| Contract Dispatcher | `src/transformation_portal/spatial_ai/ingest/contracts.py` | IngestOptions + decode_contract |
| Determinism CLI | `src/transformation_portal/determinism/cli.py` | run, verify, info commands |
| Environment Fingerprint | `src/transformation_portal/determinism/environment.py` | SPEC-DH-001 Section 5 compliance |
| FP-State Probe | `src/transformation_portal/determinism/fp_probe.py` | Cross-ISA FTZ/DAZ behavioral probe |
| Policy | `policy/adr030_v1.json` | Acceptance bounds (max_abs_error ≤ 5e-6, mean_abs_error ≤ 5e-7) |
| CI Gate | `.github/workflows/determinism-gate.yml` | PR-blocking determinism checks |
| Cross-ISA CI | `.github/workflows/determinism-cross-isa.yml` | linux-x86_64 + arm64 parity verification |

---

## Context

Phase II expands model and orchestration complexity (ADR-027, ADR-029). Without a strict ingest contract, small decode and color-processing drifts propagate into segmentation, materials, and 3D reconstruction and create non-reproducible outputs across CPU/ISA variants.

Current risk channels:

1. Vendor/library decode differences for RAW files.
2. Implicit color transforms and metadata ambiguity.
3. Unbounded floating-point variance across hardware targets.

To keep "same input -> same result" meaningful, deterministic controls must start at ingest.

---

## Decision

### 1. Canonical Ingest Output

Ingest MUST output:

1. `image`: float32 tensor in canonical `xyz_d50_linear_fp32`.
2. `geometry`: normalized dimensions/orientation metadata.
3. `provenance`: content hash, decoder identity, transform chain, and config hash.

### 2. Policy Requirements

1. Inputs violating ingest policy (including unsupported 8-bit RAW pathways) MUST fail closed.
2. Transform order and numeric operations MUST remain explicit and fixed.
3. Optional fast paths MUST produce bounded-equivalent output or be disabled.

### 3. Bounded Determinism

Cross-ISA output parity is certified by harness bounds defined in `SPEC-DH-001`. Bound compliance is required for release candidates that touch ingest behavior.

---

## Consequences

### Positive

1. Stable upstream contract for segmentation, materials, and reconstruction.
2. Reproducible baselines across supported hardware classes.
3. Faster incident triage via deterministic provenance.

### Tradeoffs

1. Added CI/runtime checks and test vector maintenance.
2. Stricter rejection path for ambiguous media inputs.
3. Potential short-term throughput impact where deterministic mode disables non-equivalent fast paths.

---

## Implementation Plan

1. ✅ Land canonical ingest contract adapters and provenance schema.
2. ✅ Introduce deterministic harness vectors and certification gates (`SPEC-DH-001`).
3. ✅ Publish validation analysis and drift envelope (`ANALYSIS-DH-001`).
4. ✅ Require harness pass for ingest-affecting pull requests.

---

## Success Metrics

1. ✅ Harness certification pass rate is 100% on target runners.
2. ✅ No unresolved ingest-related nondeterminism incidents in release branch.
3. ✅ Golden baselines remain stable for ingest-dependent pipelines.

---

## Next Steps (Post-Implementation)

Per `ANALYSIS-DH-001`, the following are planned follow-ups:

1. **Expand harness coverage**: Add more camera/RAW format profiles and larger vector sets.
2. **GPU path maturity**: Keep GPU certification observational until ready for promotion to blocking status.
3. **Reference artifact versioning**: Content-addressed artifact storage is in place; consider tagging reference baselines for release milestones.
