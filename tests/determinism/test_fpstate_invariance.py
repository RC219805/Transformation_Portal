from __future__ import annotations

from transformation_portal.determinism.hardware_fpstate import enforce_fpstate_and_probe


def test_fpstate_enforcement_is_idempotent_and_preserves_subnormals():
    r1 = enforce_fpstate_and_probe(require_subnormals=True)
    r2 = enforce_fpstate_and_probe(require_subnormals=True)

    # Must remain stable across repeated enforcement calls
    assert r1.subnormals_preserved is True
    assert r2.subnormals_preserved is True
