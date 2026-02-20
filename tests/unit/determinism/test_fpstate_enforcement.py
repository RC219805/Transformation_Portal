"""Unit tests for deterministic FP state probing and bootstrap enforcement."""

from __future__ import annotations

import os
import sys
import types

import pytest

from transformation_portal.determinism import bootstrap as bootstrap_mod
from transformation_portal.determinism.fpstate import FPStateError, enforce_ftz_daz_disabled, read_fp_state

pytestmark = [pytest.mark.unit]


def test_read_fp_state_uses_compiled_module(monkeypatch):
    fake_mod = types.SimpleNamespace(get_fp_state=lambda: {"arch": "x86", "ftz": False, "daz": False})
    monkeypatch.setitem(sys.modules, "transformation_portal.determinism._fpstate", fake_mod)

    state = read_fp_state()
    assert state["arch"] == "x86"
    assert state["ftz"] is False
    assert state["daz"] is False


def test_read_fp_state_raises_when_extension_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformation_portal.determinism._fpstate", None)
    with pytest.raises(FPStateError, match="Unable to import compiled fpstate probe"):
        read_fp_state()


def test_enforce_ftz_daz_disabled_raises_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "transformation_portal.determinism.fpstate.read_fp_state",
        lambda: {"arch": "x86", "ftz": True, "daz": False},
    )
    with pytest.raises(FPStateError, match="FTZ/DAZ enabled"):
        enforce_ftz_daz_disabled()


def test_enforce_ftz_daz_disabled_raises_when_daz_enabled(monkeypatch):
    monkeypatch.setattr(
        "transformation_portal.determinism.fpstate.read_fp_state",
        lambda: {"arch": "x86", "ftz": False, "daz": True},
    )
    with pytest.raises(FPStateError, match="FTZ/DAZ enabled"):
        enforce_ftz_daz_disabled()


def test_enforce_ftz_daz_disabled_passes_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "transformation_portal.determinism.fpstate.read_fp_state",
        lambda: {"arch": "x86", "ftz": False, "daz": False},
    )
    enforce_ftz_daz_disabled()


def test_bootstrap_calls_fpstate_enforcement(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    called = {"value": False}

    def _fake_enforce() -> None:
        called["value"] = True

    monkeypatch.setattr(bootstrap_mod, "enforce_ftz_daz_disabled", _fake_enforce)
    bootstrap_mod.bootstrap()

    assert called["value"] is True
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
