#!/usr/bin/env python3
"""Coverage for under-tested torch_security helpers.

`tests/security/test_platform_security_profile.py` already covers the canonical
profile, install idempotency, and the torch-required block path. This file
covers the remaining surface that runs without requiring torch:

- safe_load FileNotFoundError / ImportError
- _enforced_torch_load when enforcement is not installed
- _unsafe_torch_load_bypass approval gate
- uninstall_global_enforcement / assert_enforcement_installed
- warn_if_vulnerable behavior
- pickle_module DeprecationWarning
"""

from __future__ import annotations

import builtins
import sys
import types
import warnings
from pathlib import Path

import pytest

from transformation_portal.core.security import torch_security as ts

pytestmark = [pytest.mark.unit, pytest.mark.security]


@pytest.fixture(autouse=True)
def _reset_enforcement_state():
    """Snapshot and restore module-level enforcement state for each test."""
    saved_installed = ts._enforcement_installed
    saved_original = ts._original_torch_load
    yield
    ts._enforcement_installed = saved_installed
    ts._original_torch_load = saved_original


# ---------------------------------------------------------------------------
# safe_load
# ---------------------------------------------------------------------------


def test_safe_load_raises_file_not_found(tmp_path, monkeypatch):
    # Inject a fake torch so the existence check (which runs after `import torch`)
    # is reachable without a real torch install.
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *a, **kw: pytest.fail("torch.load should not be called")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError) as exc_info:
        ts.safe_load(missing)
    assert "nope.pt" in str(exc_info.value)


def test_safe_load_passes_weights_only_true(tmp_path, monkeypatch):
    fake_torch = types.ModuleType("torch")
    captured = {}

    def fake_load(f, *, map_location=None, weights_only=False, **kwargs):
        captured["f"] = f
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return "loaded"

    fake_torch.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    target = tmp_path / "model.pt"
    target.write_bytes(b"\x00")
    result = ts.safe_load(target, map_location="cpu")
    assert result == "loaded"
    assert captured["weights_only"] is True
    assert captured["map_location"] == "cpu"


def test_safe_load_raises_import_error_when_torch_missing(tmp_path, monkeypatch):
    # Force `import torch` inside safe_load to fail. ImportError fires before
    # the file existence check is reached.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    target = tmp_path / "anything.pt"
    target.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(ImportError):
        ts.safe_load(target)


# ---------------------------------------------------------------------------
# _enforced_torch_load
# ---------------------------------------------------------------------------


def test_enforced_torch_load_raises_runtime_error_when_not_installed():
    ts._enforcement_installed = False
    ts._original_torch_load = None
    with pytest.raises(RuntimeError, match="enforcement not installed"):
        ts._enforced_torch_load("dummy.pt")


def test_enforced_torch_load_blocks_explicit_weights_only_false():
    # Pretend enforcement is installed by wiring up a fake original loader.
    calls = []

    def fake_loader(*args, **kwargs):
        calls.append(kwargs)
        return "loaded"

    ts._original_torch_load = fake_loader
    ts._enforcement_installed = True

    with pytest.raises(ts.SecurityError, match="weights_only=False"):
        ts._enforced_torch_load("any.pt", weights_only=False)
    assert not calls


def test_enforced_torch_load_passes_through_with_weights_only_true():
    received = {}

    def fake_loader(f, *, map_location=None, weights_only=False, **kwargs):
        received["f"] = f
        received["map_location"] = map_location
        received["weights_only"] = weights_only
        return "ok"

    ts._original_torch_load = fake_loader
    ts._enforcement_installed = True

    result = ts._enforced_torch_load("model.pt", map_location="cpu")
    assert result == "ok"
    assert received["weights_only"] is True
    assert received["map_location"] == "cpu"


def test_enforced_torch_load_warns_on_pickle_module():
    def fake_loader(*args, **kwargs):
        return "ok"

    ts._original_torch_load = fake_loader
    ts._enforcement_installed = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts._enforced_torch_load("m.pt", pickle_module=object())
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "expected DeprecationWarning when pickle_module is supplied"


def test_enforced_torch_load_overrides_unspecified_weights_only_to_true():
    received = {}

    def fake_loader(f, *, weights_only=False, **kwargs):
        received["weights_only"] = weights_only
        return "ok"

    ts._original_torch_load = fake_loader
    ts._enforcement_installed = True

    # Caller omits weights_only entirely; enforcement defaults it to True.
    ts._enforced_torch_load("m.pt")
    assert received["weights_only"] is True


# ---------------------------------------------------------------------------
# _unsafe_torch_load_bypass
# ---------------------------------------------------------------------------


def test_unsafe_torch_load_bypass_requires_security_review_flag():
    with pytest.raises(ts.SecurityError, match="Security review approval required"):
        ts._unsafe_torch_load_bypass("anything.pt")


def test_unsafe_torch_load_bypass_uses_original_loader_when_available():
    captured = {}

    def fake_loader(f, map_location=None, **kwargs):
        captured["f"] = f
        captured["map_location"] = map_location
        captured["kwargs"] = kwargs
        return "raw-checkpoint"

    ts._original_torch_load = fake_loader
    ts._enforcement_installed = True

    result = ts._unsafe_torch_load_bypass(
        "trusted.pt",
        map_location="cuda:0",
        _security_review_approved=True,
        weights_only=False,
    )
    assert result == "raw-checkpoint"
    assert captured["f"] == "trusted.pt"
    assert captured["map_location"] == "cuda:0"
    # weights_only is forwarded as-is in bypass mode (the bypass exists for
    # exactly this case after security review).
    assert captured["kwargs"]["weights_only"] is False


def test_unsafe_torch_load_bypass_falls_back_to_torch_load_when_no_original(monkeypatch):
    ts._enforcement_installed = False
    ts._original_torch_load = None

    fake_torch = types.ModuleType("torch")
    captured = {}

    def fake_load(f, map_location=None, **kwargs):
        captured["f"] = f
        captured["map_location"] = map_location
        captured["kwargs"] = kwargs
        return "fresh-load"

    fake_torch.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = ts._unsafe_torch_load_bypass(
        "trusted.pt",
        _security_review_approved=True,
        weights_only=False,
    )
    assert result == "fresh-load"
    assert captured["f"] == "trusted.pt"


# ---------------------------------------------------------------------------
# uninstall_global_enforcement / assert_enforcement_installed
# ---------------------------------------------------------------------------


def test_uninstall_returns_false_when_not_installed():
    ts._enforcement_installed = False
    ts._original_torch_load = None
    assert ts.uninstall_global_enforcement() is False


def test_uninstall_restores_original_loader(monkeypatch):
    fake_torch = types.ModuleType("torch")

    def original_load(*args, **kwargs):
        return "original"

    fake_torch.load = ts._enforced_torch_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ts._enforcement_installed = True
    ts._original_torch_load = original_load

    assert ts.uninstall_global_enforcement() is True
    # `fake_torch.load` is set above as a dynamic ModuleType attribute, which
    # pylint's static analysis can't see. The attribute is real at runtime.
    assert fake_torch.load is original_load  # pylint: disable=no-member
    assert ts._enforcement_installed is False
    assert ts._original_torch_load is None


def test_assert_enforcement_installed_raises_when_missing():
    ts._enforcement_installed = False
    with pytest.raises(RuntimeError, match="enforcement not installed"):
        ts.assert_enforcement_installed()


def test_assert_enforcement_installed_passes_when_installed():
    ts._enforcement_installed = True
    ts.assert_enforcement_installed()  # Must not raise


# ---------------------------------------------------------------------------
# warn_if_vulnerable
# ---------------------------------------------------------------------------


def test_warn_if_vulnerable_emits_when_below_baseline(monkeypatch):
    monkeypatch.setattr(
        ts,
        "check_torch_security_compliance",
        lambda: {
            "torch_version": "2.0.0",
            "supported_security_baseline_met": False,
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts.warn_if_vulnerable()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "expected UserWarning for outdated torch"
    assert "2.0.0" in str(user_warnings[0].message)


def test_warn_if_vulnerable_silent_when_baseline_met(monkeypatch):
    monkeypatch.setattr(
        ts,
        "check_torch_security_compliance",
        lambda: {
            "torch_version": "2.12.0",
            "supported_security_baseline_met": True,
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts.warn_if_vulnerable()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []


def test_warn_if_vulnerable_silent_when_torch_missing(monkeypatch):
    # When torch isn't installed compliance returns torch_version=None; in that
    # case we must not emit a noisy UserWarning.
    monkeypatch.setattr(
        ts,
        "check_torch_security_compliance",
        lambda: {
            "torch_version": None,
            "supported_security_baseline_met": False,
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts.warn_if_vulnerable()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []


# ---------------------------------------------------------------------------
# install_global_enforcement: torch-missing branch
# ---------------------------------------------------------------------------


def test_install_global_enforcement_returns_false_when_torch_missing(monkeypatch):
    ts._enforcement_installed = False
    ts._original_torch_load = None

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ts.install_global_enforcement() is False
    assert ts._enforcement_installed is False
