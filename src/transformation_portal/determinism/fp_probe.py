"""Cross-ISA FTZ/DAZ behavioral probe with normalization layer.

This module provides a versioned, policy-driven FP-state probe that tests both
scalar and vector operations to detect cross-ISA divergence in subnormal handling.

Architecture: ADR-030 determinism harness extension.
Probe Version: 1

The probe tests:
- Scalar multiplication of smallest subnormal × 1.0
- Scalar addition of smallest subnormal + 0.0
- Vectorized multiplication (SIMD kernels) of subnormal array × 1.0
- Vectorized addition (SIMD kernels) of subnormal array + 0.0

This dual-path approach reduces false confidence from scalar-only probes,
which may pass on one ISA while vectorized ufunc kernels behave differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


ProbePolicy = Literal["strict", "relaxed", "scalar_only", "vector_only"]


@dataclass(frozen=True)
class FPProbeRaw:
    """Raw FP-state probe results.

    This captures the detailed probe outcomes before policy normalization.
    All boolean fields are Python primitives (not np.bool_) for JCS/JSON safety.
    """

    probe_version: int
    smallest_subnormal_nonzero: bool
    scalar_mul_preserved: bool
    scalar_add_preserved: bool
    vector_mul_preserved: bool
    vector_add_preserved: bool
    note: Optional[str] = None


@dataclass(frozen=True)
class FPProbeNormalized:
    """Policy-normalized FP probe result.

    This is the gating artifact for determinism evidence. The `subnormals_preserved`
    boolean is derived from raw probe results under an explicit policy.
    """

    probe_version: int
    policy: ProbePolicy
    subnormals_preserved: bool
    reason: Optional[str] = None


def _pybool(x: object) -> bool:
    """Normalize numpy scalar bools to Python bool (JCS/JSON-friendly)."""
    return bool(x)


def probe_fpstate_raw() -> FPProbeRaw:
    """Best-effort FTZ/DAZ behavioral probe.

    We test both scalar-like ops and vectorized ufunc kernels to reduce
    cross-ISA false positives. This does NOT read hardware registers; it
    detects whether subnormal values survive basic arithmetic operations.

    Returns:
        FPProbeRaw with probe_version=1 and scalar/vector preservation flags.
    """
    probe_version = 1

    # Smallest positive float32 subnormal.
    x = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)

    if x == np.float32(0.0):
        return FPProbeRaw(
            probe_version=probe_version,
            smallest_subnormal_nonzero=False,
            scalar_mul_preserved=False,
            scalar_add_preserved=False,
            vector_mul_preserved=False,
            vector_add_preserved=False,
            note="nextafter_returned_zero",
        )

    # --- Scalar-ish checks ---
    y_mul = x * np.float32(1.0)
    y_add = x + np.float32(0.0)

    scalar_mul_ok = _pybool(y_mul != np.float32(0.0))
    scalar_add_ok = _pybool(y_add != np.float32(0.0))

    # --- Vectorized checks (force ufunc kernels) ---
    # Use a small but non-trivial vector length to prefer SIMD kernels.
    v = np.full((1024,), x, dtype=np.float32)

    v_mul = v * np.float32(1.0)
    v_add = v + np.float32(0.0)

    vector_mul_ok = _pybool(np.any(v_mul != np.float32(0.0)))
    vector_add_ok = _pybool(np.any(v_add != np.float32(0.0)))

    return FPProbeRaw(
        probe_version=probe_version,
        smallest_subnormal_nonzero=True,
        scalar_mul_preserved=scalar_mul_ok,
        scalar_add_preserved=scalar_add_ok,
        vector_mul_preserved=vector_mul_ok,
        vector_add_preserved=vector_add_ok,
    )


def normalize_fp_probe(raw: FPProbeRaw, *, policy: ProbePolicy = "strict") -> FPProbeNormalized:
    """Convert raw probe results into a single boolean under an explicit policy.

    Policies:
    - strict: require scalar AND vector preservation (recommended for CI/certification)
    - relaxed: scalar OR vector preservation
    - scalar_only: scalar preservation only
    - vector_only: vector preservation only

    Args:
        raw: Raw probe results from probe_fpstate_raw().
        policy: The normalization policy to apply.

    Returns:
        FPProbeNormalized with policy-derived subnormals_preserved flag.

    Raises:
        ValueError: If policy is unrecognized.
    """
    if not raw.smallest_subnormal_nonzero:
        return FPProbeNormalized(
            probe_version=raw.probe_version,
            policy=policy,
            subnormals_preserved=False,
            reason=raw.note or "no_subnormal",
        )

    scalar_ok = raw.scalar_mul_preserved and raw.scalar_add_preserved
    vector_ok = raw.vector_mul_preserved and raw.vector_add_preserved

    if policy == "strict":
        ok = scalar_ok and vector_ok
        reason = None if ok else "strict_requires_scalar_and_vector"
    elif policy == "relaxed":
        ok = scalar_ok or vector_ok
        reason = None if ok else "relaxed_requires_scalar_or_vector"
    elif policy == "scalar_only":
        ok = scalar_ok
        reason = None if ok else "scalar_only_failed"
    elif policy == "vector_only":
        ok = vector_ok
        reason = None if ok else "vector_only_failed"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return FPProbeNormalized(
        probe_version=raw.probe_version,
        policy=policy,
        subnormals_preserved=bool(ok),
        reason=reason,
    )


def probe_fpstate_normalized(*, policy: ProbePolicy = "strict") -> FPProbeNormalized:
    """Run FP-state probe and return policy-normalized result.

    This is the primary entry point for cross-ISA determinism gating.

    Args:
        policy: Normalization policy (default: "strict").

    Returns:
        FPProbeNormalized with probe_version, policy, and subnormals_preserved.
    """
    raw = probe_fpstate_raw()
    return normalize_fp_probe(raw, policy=policy)
