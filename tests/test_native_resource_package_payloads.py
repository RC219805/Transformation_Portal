"""Resource capsules bind actual SDK code/models, never package versions alone."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformation_portal.photographic_delivery import native_resources as resources

pytestmark = pytest.mark.unit


def _record(path: str, value: bytes) -> importlib.metadata.PackagePath:
    entry = importlib.metadata.PackagePath(path)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode("ascii")
    entry.hash = importlib.metadata.FileHash("sha256=" + encoded)
    entry.size = len(value)
    return entry


@pytest.fixture(name="installed")
def fixture_installed(tmp_path, monkeypatch):
    payloads = {"closed_sdk/__init__.py": b"CLOSED = True\n", "closed_sdk/data/_retry.json": b'{"attempts":5}'}
    for name, value in payloads.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    entries = [_record(name, value) for name, value in payloads.items()]
    distribution = SimpleNamespace(files=entries, locate_file=lambda relative: tmp_path / relative)
    monkeypatch.setattr(resources.importlib.metadata, "distribution", lambda name: distribution)
    return tmp_path, distribution, payloads


def _identity():
    return resources._distribution_payload_identity("closed-sdk", ("closed_sdk",))


def test_actual_sdk_source_and_model_bytes_have_stable_bounded_identity(installed):
    root, distribution, payloads = installed
    result = _identity()
    rows = [[name, len(value), hashlib.sha256(value).hexdigest()] for name, value in sorted(payloads.items())]
    assert result == {
        "sha256": hashlib.sha256(resources._canonical(rows)).hexdigest(),
        "files": 2,
        "bytes": sum(map(len, payloads.values())),
    }
    distribution.files.reverse()
    assert _identity() == result
    cache = root / "closed_sdk/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"generated cache is not wheel authority")
    assert _identity() == result


@pytest.mark.parametrize("target", ["closed_sdk/__init__.py", "closed_sdk/data/_retry.json"])
def test_changed_actual_code_or_model_with_unchanged_record_is_rejected(installed, target):
    root, _, payloads = installed
    original = payloads[target]
    (root / target).write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="bytes differ from RECORD"):
        _identity()


@pytest.mark.parametrize("change", ["missing", "extra", "unhashed", "wrong_size", "wrong_algorithm", "symlink", "duplicate"])
def test_record_and_actual_tree_must_agree_exactly(installed, change):
    root, distribution, _ = installed
    entry = distribution.files[0]
    if change == "missing":
        (root / str(entry)).unlink()
    elif change == "extra":
        (root / "closed_sdk/unrecorded.py").write_bytes(b"unreviewed executable")
    elif change == "unhashed":
        entry.hash = None
    elif change == "wrong_size":
        entry.size += 1
    elif change == "wrong_algorithm":
        entry.hash = importlib.metadata.FileHash("md5=arbitrary")
    elif change == "symlink":
        original = root / str(entry)
        saved = root / "saved.py"
        original.rename(saved)
        original.symlink_to(saved)
    else:
        distribution.files.append(entry)
    with pytest.raises(ValueError):
        _identity()


def test_missing_record_and_bounded_bytes_files_fail_closed(installed, monkeypatch):
    _, distribution, _ = installed
    monkeypatch.setattr(resources, "_MAX_PACKAGE_PAYLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="unverifiable"):
        _identity()
    monkeypatch.setattr(resources, "_MAX_PACKAGE_PAYLOAD_BYTES", 1000)
    monkeypatch.setattr(resources, "_MAX_PACKAGE_PAYLOAD_FILES", 1)
    with pytest.raises(ValueError, match="bounded RECORD"):
        _identity()
    distribution.files = None
    with pytest.raises(ValueError, match="bounded RECORD"):
        _identity()


def test_source_only_bytecode_cannot_enter_unrecorded_package_tree(installed):
    root, _, _ = installed
    (root / "closed_sdk/hidden.pyc").write_bytes(b"unreviewed sourceless code")
    with pytest.raises(ValueError, match="tree differs from RECORD"):
        _identity()


def test_loaded_package_from_another_environment_cannot_borrow_record(installed, tmp_path, monkeypatch):
    alternate = tmp_path / "alternate.py"
    alternate.write_bytes(b"different implementation")
    monkeypatch.setitem(resources.sys.modules, "closed_sdk", SimpleNamespace(__file__=str(alternate)))
    with pytest.raises(ValueError, match="another tree"):
        _identity()


def test_loaded_verified_package_and_namespace_modules_are_supported(installed, monkeypatch):
    root, _, _ = installed
    monkeypatch.setitem(resources.sys.modules, "closed_sdk", SimpleNamespace(__file__=str(root / "closed_sdk/__init__.py")))
    monkeypatch.setitem(resources.sys.modules, "closed_sdk.namespace", SimpleNamespace())
    assert _identity()["files"] == 2


def test_runtime_identity_includes_payloads_without_importing_application(monkeypatch, tmp_path):
    source = tmp_path / "http_client.py"
    source.write_bytes(b"closed standard library transport")
    monkeypatch.setattr(resources, "_PAYLOAD_PACKAGES", {"closed-sdk": ("closed_sdk",)})
    monkeypatch.setattr(resources, "_distribution_payload_identity", lambda name, roots: {"sha256": "a" * 64})
    monkeypatch.setattr(resources, "_STDLIB_PAYLOADS", ("closed_transport",))
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(source)))
    monkeypatch.delitem(resources.sys.modules, "app", raising=False)
    result = resources._runtime_package_payloads()
    assert result["closed-sdk"]["sha256"] == "a" * 64
    assert result["stdlib"]["closed_transport"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "app" not in resources.sys.modules


def test_missing_or_mixed_standard_library_payload_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_PAYLOAD_PACKAGES", {})
    monkeypatch.setattr(resources, "_STDLIB_PAYLOADS", ("closed_transport",))
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="missing"):
        resources._runtime_package_payloads()
    source = tmp_path / "transport.py"
    source.write_bytes(b"transport")
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(source)))
    monkeypatch.setitem(resources.sys.modules, "closed_transport", SimpleNamespace(__file__=str(tmp_path / "other.py")))
    with pytest.raises(ValueError, match="standard library differs"):
        resources._runtime_package_payloads()
