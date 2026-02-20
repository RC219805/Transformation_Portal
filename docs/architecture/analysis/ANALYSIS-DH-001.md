# Architectural Certification of Bounded Determinism
## Phase II RAW Ingest Pipelines — Systems Analysis and Implementation Blueprint

| Field | Value |
| --- | --- |
| Document ID | ANALYSIS-DH-001 |
| Status | Informative (Non-normative) |
| Repository | RC219805/Transformation_Portal |
| Verification Policy Version Referenced | adr030-v1 |
| Aligned With | ADR-030, SPEC-DH-001 |
| Date | 2026-02-20 |

> Conventions Note (Normative Safety): This document is informative. Any uppercase BCP 14 keywords (RFC 2119 / RFC 8174) appearing here are descriptive restatements of requirements defined in ADR-030 and SPEC-DH-001. Only ADR-030, SPEC-DH-001, and the executable policy artifact are normative authorities.

---

## 1. Executive Overview

Phase II upgrades RAW ingest from heuristic, perceptual preprocessing into mathematically verifiable compute infrastructure. The goal is to provide downstream spatial pipelines (depth routing, materials, reconstruction, rendering) an ingest boundary whose outputs are **measurably stable** and **cryptographically auditable**.

Legacy RAW handling frequently traded determinism for visually pleasing output via:

- sRGB-oriented colorimetry choices
- auto-brightness / auto-scale heuristics
- dynamic decoder behaviors dependent on platform, CPU features, and threading

Within this repository, those behaviors historically appeared in modules such as:

- `lux_depth_v3/raw_loader.py`
- `spatial_ai/ingest/linear_decoder.py`

Under Phase II, these legacy behaviors are treated as **deprecated for certified ingest boundary roles**.

The architectural requirement is a dedicated certified contract path producing the ingest boundary tensor:

- `tensor_role = "xyz_d50_linear_fp32"`

This tensor becomes the deterministic trust anchor upstream of all high-tier deliverables.

---

## 2. Operational Definition of Certified Bounded Determinism

**Certified Bounded Determinism (CBD)** is satisfied when:

- For identical RAW input **and**
- identical executable policy version (`adr030-v1`) **and**
- certified execution constraints (ADR-030 + SPEC-DH-001)

the emitted `xyz_d50_linear_fp32` tensor satisfies ADR-030's dual parity gates under deterministic reduction semantics, independent of ISA (x86_64 vs arm64).

**Bit-for-bit identity is a best-case outcome, not the certification requirement.**
Certification requires bounded parity.

---

## 3. Authority Segregation and Governance

### 3.1 Normative Authorities (Binding)

Only the following artifacts define binding requirements:

- ADR-030 (verification policy: float model, gates, certified tensor semantics, baseline authority)
- SPEC-DH-001 (mechanism: execution semantics, CAS layout, hashing rules, telemetry/runs)
- `policy/adr030_vX.(json|py)` (single executable policy representation)

Everything else (including this document) is analysis and implementation commentary.

### 3.2 ADRs as Governance Ledger

Architectural records are treated as immutable ledgers to prevent:

- silent threshold drift
- undocumented behavioral changes
- regressions masked by refactors

This is necessary because even micro-scale numerical drift can cascade into high-variance downstream model behavior.

### 3.3 Governance Vectors

| Governance vector | Standard | Rationale |
| --- | --- | --- |
| Record immutability | Version-controlled ADR/SPEC | Prevent silent policy mutation |
| Pipeline isolation | Pinned CI substrate | Make drift distinguishable from defects |
| Policy versioning | Executable policy file (`adr030_v1`) | Eliminate transcription drift |
| Baseline authority | Signed baseline artifacts | Establish canonical truth for cross-ISA parity |

---

## 4. Executable Policy as Single Source of Truth

Textual ADRs are not executable. To eliminate the "human transcription gap," Phase II requires a single machine-readable policy representation.

Example policy structure (illustrative):

```json
{
  "verification_policy_version": "adr030-v1",
  "float_model": "IEEE754_binary32",
  "epsilon": "2^-23",
  "pixel_parity_multiplier": 128,
  "mae_threshold": 1e-7,
  "rmse_threshold": 5e-7,
  "nan_policy": "fail_closed",
  "inf_policy": "fail_closed",
  "subnormal_policy": "preserve",
  "reduction_mode": "single_thread_float64_c_order",
  "matrix_backend": "explicit_f32_no_blas"
}
```

**Design intent:**

* Verification code loads thresholds from this policy artifact.
* No duplicate threshold constants exist elsewhere in code.
* Evidence captures the policy version and policy digest for forensic traceability.

---

## 5. Cross-ISA Floating-Point Determinism

Cross-ISA parity is achievable when floating-point hazards are converted into enforceable constraints.

### 5.1 IEEE 754 binary32 and epsilon binding

Phase II certification binds epsilon to IEEE-754 binary32 machine epsilon:

* `ε = 2^-23 ≈ 1.1920929e-7`
* Code alias: `FLOAT32_EPS`

All tolerance constants derive from this value.

### 5.2 Floating-point hazard mitigation matrix

| Hazard | Failure mode | Mitigation | Certification value |
| --- | --- | --- | --- |
| x87 80-bit promotion | double rounding | enforce SSE2/no x87 reliance | avoids hidden drift |
| FMA contraction | single-rounding vs double | disable contraction (`-ffp-contract=off`) | algebraic equivalence |
| fast-math | reassociation, contraction | prohibit `-ffast-math` | prevents uncontrolled transforms |
| FTZ/DAZ subnormal modes | silent zeroing near 0 | runtime probe + fail closed | protects convergence stability |
| parallel reduction | non-associativity drift | single-thread deterministic reductions | stabilizes MAE/RMSE |
| BLAS GEMM for micro-matrices | reordering / vectorized kernels | explicit multiply-add loops | freezes operation order |

### 5.3 Subnormals and FTZ/DAZ

Performance-driven systems often enable FTZ/DAZ to avoid subnormal slow paths. That is incompatible with certified determinism.

Phase II requires:

* subnormals preserved
* FTZ/DAZ disabled
* runtime detection and fail-closed behavior if FTZ/DAZ is enabled

(Implementations may require a small platform-specific shim to read MXCSR/FPCR state.)

---

## 6. Deterministic Verification Metrics

### 6.1 Why verification must be deterministic

MAE and RMSE are reduction metrics. Floating-point addition is non-associative; parallel reductions and varying reduction trees cause run-to-run drift even on identical hardware.

Verification logic is certification logic. It is not performance code.

### 6.2 Thread neutralization

Thread pools must be collapsed **before** importing numerical C-extensions (NumPy, SciPy, etc.). Setting these variables after library initialization has no effect because thread pools are configured at import time.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

### 6.3 Reduction order and pinned NumPy semantics

NumPy reduction behavior is a library detail. Phase II defines the canonical reduction order as:

* `.ravel(order="C")` on inputs
* single-threaded execution
* pinned NumPy build/version recorded in baseline manifest
* explicit float64 accumulation

Reference computation (policy-aligned):

```python
diff = reference.astype(np.float64).ravel(order="C") - candidate.astype(np.float64).ravel(order="C")
abs_diff = np.abs(diff)

mae = np.sum(abs_diff, dtype=np.float64) / abs_diff.size
rmse = np.sqrt(np.sum(diff * diff, dtype=np.float64) / diff.size)
max_abs_diff = np.max(abs_diff)
```

---

## 7. Dual Gate Certification Semantics

### 7.1 Preconditions

* identical shape
* dtype float32
* no NaN/Inf allowed (fail closed)

### 7.2 Pixel parity gate

Passes iff:

* `max_abs_diff <= 128 * ε`

This gate isolates structural divergence.

### 7.3 Global drift gate

Passes iff:

* `MAE < 1e-7`
* `RMSE < 5e-7`

Note: NaN/Inf are **not allowed** (automatic failure), consistent with ADR-030.

---

## 8. Certified Linear Contract: `xyz_d50_linear_fp32`

Certified tensor invariants:

* dtype: float32
* layout: HWC, C-contiguous
* channel order: XYZ
* white point: D50
* gamma: linear (1.0)
* no perceptual companding
* decoder: deterministic parameters (no auto-bright, no auto-scale)
* demosaic: explicitly selected and pinned
* white balance: explicit semantics, multipliers captured when applied

Auto white balance is permitted only when:

* multipliers are captured deterministically
* multipliers are fingerprinted / hashed into ingest fingerprint evidence

---

## 9. Bradford D65 → D50 Adaptation: Hard-Frozen Matrix

Phase II requires a version-controlled float32 Bradford matrix embedded as constants.

Example constants (float32):

```text
[[ 1.0478112,  0.0228866, -0.0501270],
 [ 0.0295424,  0.9904844, -0.0170491],
 [-0.0092319,  0.0150436,  0.7521316]]
```

Certification constraint:

* prohibit BLAS/GEMM kernels for these 3×3 transforms
* implement as explicit float32 multiply/add sequences with fixed left-to-right evaluation order

This freezes execution order across ISA and avoids hidden vectorized reorderings.

---

## 10. Cryptographic Content Addressing

### 10.1 Headerless canonical bytes

`.npy` headers can drift across NumPy versions; hashing `.npy` is structurally unsafe.

Certified roles require:

* canonical little-endian float32 bytes (`<f4`)
* C-order contiguous layout
* headerless serialization: `output_tensor.bin`

### 10.2 Artifact hash preimage

Canonical preimage (normative in SPEC-DH-001):

```text
b"tensor_role=xyz_d50_linear_fp32\n"
b"dtype=float32\n"
b"order=C\n"
b"shape=H,W,3\n"
+ raw canonical bytes
```

Endianness is normalized prior to hashing.

---

## 11. JSON Canonicalization (RFC 8785 JCS)

JSON evidence and hashed metadata must be canonicalized to eliminate drift from:

* key ordering
* whitespace
* Unicode normalization differences
* number string formatting differences

RFC 8785 mandates ECMAScript-aligned number rendering.

Non-finite numbers (NaN/Infinity) are prohibited in hashed JSON.

Python `json.dumps()` must not be assumed compliant without strict verification. Implementations should use a dedicated RFC 8785 library (e.g., `canonicaljson`) or a verified wrapper that enforces deterministic key ordering, UTF-8 encoding, and ECMAScript number formatting.

---

## 12. Trace Context Isolation (W3C Trace Context)

Trace context is used for observability and correlation:

* trace IDs propagate across services and are logged into run evidence
* trace IDs and execution IDs are excluded from artifact identity (`artifact_id`)

Observability must not contaminate content identity.

---

## 13. CAS Atomicity and Mutability Semantics

The CAS uses two mutability models:

* tensor root: immutable once committed
* run evidence: append-only under `runs/<execution_id>/`

Atomicity requirement:

* temporary staging directory must be on the same filesystem as CAS root
* commit sequence: write → fsync → atomic rename
* no cross-device rename fallback behavior (EXDEV copy/delete is forbidden for certified commits)

---

## 14. Run-Card Provenance (Reproducibility Certificate)

Each execution emits:

* `runs/<execution_id>/reproducibility_certificate.json`

This run card binds:

* artifact_id and tensor hash
* raw hash
* policy version and policy digest
* metrics (MAE/RMSE/max_abs_diff) and pass/fail
* ISA, OS, libc, BLAS linkage
* FTZ/DAZ state
* trace context identifiers

Run evidence is append-only; CAS tensor roots remain immutable.

---

## 15. Implementation Mapping: Phase II Scaffold (Repository)

To implement Phase II determinism without breaking legacy pipelines:

1. Introduce a new certified contract path producing `xyz_d50_linear_fp32` without altering legacy sRGB heuristics used elsewhere.
2. Add a determinism harness module that:

   * bootstraps deterministic runtime controls (thread neutralization, PYTHONHASHSEED re-exec)
   * materializes CAS artifacts and `.bin` bytes for certified roles
   * runs verification against baseline artifacts when supplied
3. Add an executable policy representation (`policy/adr030_v1.(json|py)`) consumed by verification code.
4. Add CI parity gates for x86_64 vs arm64 using baseline artifacts.

---

## 16. Strategic Synthesis

Phase II CBD transforms ingest into a computational trust anchor:

* drift becomes measurable and bounded
* CI gates become enforceable and audit-friendly
* artifacts become content-addressed and forensically reconstructible over time
* policy becomes versioned, executable, and non-transcribable

This is a rare level of rigor in ML-adjacent pipelines and directly supports long-lived scientific reproducibility.

---

## References

* ADR-030: Phase II Deterministic RAW Ingest (policy)
* SPEC-DH-001: Determinism Harness MVP Specification (mechanism)
* RFC 2119 / RFC 8174: BCP 14 normative language
* RFC 8785: JSON Canonicalization Scheme
* W3C Trace Context specification
* IEEE 754 floating-point standard
