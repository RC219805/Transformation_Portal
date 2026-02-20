# ANALYSIS-DH-001: Certified Bounded Determinism at RAW Ingest

**Status:** Informative
**Date:** 2026-02-20
**Related:** ADR-030, SPEC-DH-001

---

## Summary

This analysis documents why bounded determinism is required at the ingest boundary and explains expected cross-ISA variance behavior under the `xyz_d50_linear_fp32` contract.

Primary conclusion:

1. Bit-identical parity is not required for trustworthy reproducibility.
2. Certified bounded parity is sufficient when strict envelopes are enforced and monitored.

---

## Problem Framing

RAW ingest combines decode, color adaptation, and geometric normalization. These steps are sensitive to library/runtime details and floating-point execution order. If unconstrained, downstream models observe effectively different data distributions for the same source frame.

Observed risk categories:

1. Decode-path divergence due to backend/library differences.
2. Metadata interpretation drift (orientation, white balance assumptions).
3. Floating-point accumulation order differences across ISA targets.

---

## Why Bounded Determinism

Bounded determinism provides a tractable contract:

1. Tight numeric envelopes preserve model behavior and baseline comparability.
2. Exact-match requirements remain for metadata and provenance where practical.
3. CI gates convert hidden drift into visible merge-time failures.

This approach balances scientific reproducibility with realistic cross-platform execution constraints.

---

## Operational Implications

1. Ingest changes require harness evidence before merge.
2. Baseline updates require explicit rationale and artifact traceability.
3. Incident response can isolate nondeterminism regressions quickly via manifests and environment fingerprints.

---

## Limitations and Follow-Up

1. Bound choices are currently conservative and may need recalibration with larger vector sets.
2. GPU-path certification remains observational until promoted to blocking status.
3. Harness coverage should expand with new camera and RAW format profiles.
