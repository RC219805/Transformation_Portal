# SPEC-DH-001: Determinism Harness for RAW Ingest Boundary

**Status:** LOCKED
**Version:** 1.0.0
**Date:** 2026-02-20
**Owner:** Transformation Portal Architect
**Scope:** Certification harness for `ADR-030` ingest determinism

---

## 1. Purpose

Define normative acceptance criteria for bounded determinism at the ingest boundary so that supported execution targets remain reproducible for equivalent inputs.

---

## 2. Normative Requirements

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" are to be interpreted as normative requirements.

1. Harness input vectors MUST be immutable and content-hashed.
2. Candidate ingest implementations MUST emit canonical `xyz_d50_linear_fp32`.
3. Candidate outputs MUST be compared against locked reference artifacts.
4. Certification MUST fail when any bound in Section 4 is violated.

---

## 3. Test Matrix

Harness runs across at least:

1. `linux-x86_64` CPU reference runner.
2. `macos-arm64` CPU runner.
3. Optional GPU runner for regression visibility (non-blocking unless promoted).

Each run uses identical input vectors and locked configuration.

---

## 4. Acceptance Bounds

For each image tensor channel:

1. `max_abs_error <= 5e-6`
2. `mean_abs_error <= 5e-7`
3. Geometry metadata equality: exact string/shape match required.
4. Provenance fields: hash and transform chain MUST match exactly.

Any bound violation is a hard failure for certification.

---

## 5. Required Outputs

Each harness run MUST publish:

1. Per-vector metric report.
2. Aggregate pass/fail summary.
3. Artifact manifest with content hashes.
4. Environment fingerprint (OS, ISA, runtime, dependency lock IDs).

---

## 6. CI Gate Policy

1. Ingest-affecting pull requests MUST run this harness.
2. Failing results MUST block merge.
3. Bound changes require explicit architecture approval and version bump of this spec.

---

## 7. Change Control

Because this spec is LOCKED:

1. Compatible clarifications MAY be added without changing bounds.
2. Any bound, matrix, or artifact schema change requires `SPEC-DH-001` version increment and review sign-off.
