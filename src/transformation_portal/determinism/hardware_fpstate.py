from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FPStateReport:
    """
    Deterministic FP-state report.

    - enforced: whether we attempted a hardware state enforcement action
    - backend: which method was used (best-effort)
    - subnormals_preserved: behavior-level probe result AFTER enforcement attempt
    - note: optional short note for diagnostics (no timestamps / no host IDs)
    """

    enforced: bool
    backend: str
    subnormals_preserved: bool
    note: Optional[str] = None


def enforce_fpstate_and_probe(*, require_subnormals: bool = False) -> FPStateReport:
    """
    Enforce FTZ/DAZ-disabled (where supported) and then probe behavior.

    require_subnormals:
      - if True, raises RuntimeError when subnormals are not preserved AFTER enforcement.
      - keep False in minimal CLI; enable in certification contexts/CI as you prefer.
    """
    # Local import to avoid import-time side effects.
    from transformation_portal.determinism.ingest import probe_subnormals_preserved

    enforced = False
    backend = "probe_only"
    note: Optional[str] = None

    # Prefer the repo's established hardware-enforcement primitive when present.
    try:
        from transformation_portal.determinism.fpstate import enforce_ftz_daz_disabled  # type: ignore

        enforce_ftz_daz_disabled()
        enforced = True
        backend = "fpstate.enforce_ftz_daz_disabled"
    except Exception as e:
        # Do not fail here; report and rely on probe.
        note = f"enforce_unavailable:{type(e).__name__}"
        enforced = False
        backend = "probe_only"

    subnormals_ok = bool(probe_subnormals_preserved())

    if require_subnormals and not subnormals_ok:
        raise RuntimeError("FP-state invariance failure: subnormals are not preserved after enforcement.")

    return FPStateReport(
        enforced=enforced,
        backend=backend,
        subnormals_preserved=subnormals_ok,
        note=note,
    )
