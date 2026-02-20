# ADR-030: Phase II Deterministic RAW Ingest

**Status:** Proposed
**Date:** 2026-02-20
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

1. Land canonical ingest contract adapters and provenance schema.
2. Introduce deterministic harness vectors and certification gates (`SPEC-DH-001`).
3. Publish validation analysis and drift envelope (`ANALYSIS-DH-001`).
4. Require harness pass for ingest-affecting pull requests.

---

## Success Metrics

1. Harness certification pass rate is 100% on target runners.
2. No unresolved ingest-related nondeterminism incidents in release branch.
3. Golden baselines remain stable for ingest-dependent pipelines.
