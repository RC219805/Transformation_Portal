from __future__ import annotations

"""FP-state enforcement/probe helpers for ADR-030 determinism evidence.

This module is used by the deterministic CLI runner to capture whether FTZ/DAZ
enforcement was attempted and whether subnormal behavior is preserved.

The FPStateReport captures:
- enforced: whether hardware enforcement was attempted and succeeded
- backend: which enforcement method was used
- probe_version: version of the probe algorithm (for cross-ISA audit)
- probe_policy: policy used to derive subnormals_preserved (strict, relaxed, etc.)
- subnormals_preserved: behavior-level probe result AFTER enforcement attempt
- note: optional diagnostic note (no timestamps / no host IDs)

Architecture: ADR-030 determinism harness extension.
"""

from dataclasses import dataclass
from typing import Optional

from .fp_probe import ProbePolicy, probe_fpstate_normalized


@dataclass(frozen=True)
class FPStateReport:
    """Deterministic FP-state report.

    All fields are Python primitives for JCS/JSON serialization safety.
    No timestamps or host identifiers are included.
    """

    enforced: bool
    backend: str
    probe_version: int
    probe_policy: str
    subnormals_preserved: bool
    note: Optional[str] = None


def enforce_fpstate_and_probe(
    *,
    require_subnormals: bool = False,
    probe_policy: ProbePolicy = "strict",
) -> FPStateReport:
    """Enforce FTZ/DAZ-disabled (where supported) and then probe behavior.

    Args:
        require_subnormals: If True, raises RuntimeError when subnormals are
            not preserved AFTER enforcement. Keep False in minimal CLI; enable
            in certification contexts/CI as preferred.
        probe_policy: Policy for normalizing probe results (default: "strict").
            - "strict": require scalar AND vector preservation
            - "relaxed": scalar OR vector preservation
            - "scalar_only": scalar preservation only
            - "vector_only": vector preservation only

    Returns:
        FPStateReport with enforcement status and probe results.

    Raises:
        RuntimeError: If require_subnormals=True and subnormals are not preserved.
    """
    enforced = False
    backend = "probe_only"
    note: Optional[str] = None

    # Prefer the repo's established hardware-enforcement primitive when present.
    try:
        from transformation_portal.determinism.fpstate import FPStateError, enforce_ftz_daz_disabled
    except (ImportError, AttributeError) as e:
        # Do not fail here; report and rely on probe.
        note = f"enforce_unavailable:{type(e).__name__}"
        enforced = False
        backend = "probe_only"
    else:
        try:
            enforce_ftz_daz_disabled()
        except FPStateError as e:
            # Enforcement path ran and found FTZ/DAZ policy violation.
            note = f"enforce_failed:{type(e).__name__}:{str(e)}"
            enforced = False
            backend = "fpstate.enforce_ftz_daz_disabled"
        except (RuntimeError, OSError, AttributeError) as e:
            # Known enforcement-path failures are non-fatal in minimal mode.
            note = f"enforce_unavailable:{type(e).__name__}"
            enforced = False
            backend = "probe_only"
        else:
            enforced = True
            backend = "fpstate.enforce_ftz_daz_disabled"

    # Run the normalized probe with the specified policy.
    probe = probe_fpstate_normalized(policy=probe_policy)
    subnormals_ok = probe.subnormals_preserved

    # Append probe reason to note if present (for diagnostics).
    if probe.reason:
        if note is None:
            note = f"probe:{probe.probe_version}:{probe.policy}:{probe.reason}"
        else:
            note = f"{note}|probe:{probe.probe_version}:{probe.policy}:{probe.reason}"

    if require_subnormals and not subnormals_ok:
        raise RuntimeError(
            f"FP-state invariance failure: subnormals are not preserved after enforcement "
            f"(policy={probe.policy}, reason={probe.reason})."
        )

    return FPStateReport(
        enforced=enforced,
        backend=backend,
        probe_version=probe.probe_version,
        probe_policy=probe.policy,
        subnormals_preserved=subnormals_ok,
        note=note,
    )
