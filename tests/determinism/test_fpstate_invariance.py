from __future__ import annotations

import pytest

from transformation_portal.determinism import fp_probe, hardware_fpstate
from transformation_portal.determinism.fpstate import FPStateError
from transformation_portal.determinism.hardware_fpstate import enforce_fpstate_and_probe

pytestmark = pytest.mark.unit


def test_fpstate_enforcement_is_idempotent_and_preserves_subnormals():
    r1 = enforce_fpstate_and_probe(require_subnormals=True)
    r2 = enforce_fpstate_and_probe(require_subnormals=True)

    # Must remain stable across repeated enforcement calls
    assert r1.subnormals_preserved is True
    assert r2.subnormals_preserved is True


def test_fpstate_enforcement_raises_when_subnormals_not_preserved(monkeypatch):
    monkeypatch.setattr(
        hardware_fpstate,
        "probe_fpstate_normalized",
        lambda policy="strict": fp_probe.FPProbeNormalized(
            probe_version=1,
            policy=policy,
            subnormals_preserved=False,
            reason="strict_requires_scalar_and_vector",
        ),
    )

    with pytest.raises(RuntimeError, match="subnormals are not preserved"):
        enforce_fpstate_and_probe(require_subnormals=True)


def test_fpstate_enforcement_failure_captured_in_report_note(monkeypatch):
    def _raise_fpstate_error() -> None:
        raise FPStateError("FTZ/DAZ enabled for this environment")

    monkeypatch.setattr("transformation_portal.determinism.fpstate.enforce_ftz_daz_disabled", _raise_fpstate_error)
    monkeypatch.setattr(
        hardware_fpstate,
        "probe_fpstate_normalized",
        lambda policy="strict": fp_probe.FPProbeNormalized(
            probe_version=1,
            policy=policy,
            subnormals_preserved=True,
            reason=None,
        ),
    )

    report = enforce_fpstate_and_probe(require_subnormals=False)

    assert report.enforced is False
    assert report.backend == "fpstate.enforce_ftz_daz_disabled"
    assert report.note is not None
    assert report.note.startswith("enforce_failed:FPStateError:")
