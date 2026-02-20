from __future__ import annotations

from typing import Any, Dict


class FPStateError(RuntimeError):
    """Raised when FP control state cannot be probed or violates policy."""


def read_fp_state() -> Dict[str, Any]:
    try:
        from . import _fpstate  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FPStateError(
            "Unable to import compiled fpstate probe (transformation_portal.determinism._fpstate). "
            "Reinstall from source with a working C compiler/toolchain."
        ) from e

    try:
        state = _fpstate.get_fp_state()
    except Exception as e:
        raise FPStateError("Unable to read floating-point state from compiled fpstate probe.") from e

    if not isinstance(state, dict):
        raise FPStateError(f"Invalid fpstate response type: {type(state)}")
    return state


def enforce_ftz_daz_disabled() -> None:
    state = read_fp_state()
    ftz = bool(state.get("ftz"))
    daz = bool(state.get("daz"))

    if ftz or daz:
        raise FPStateError(
            "FTZ/DAZ enabled; Phase II Certified Bounded Determinism requires subnormal preservation. " f"state={state}"
        )
