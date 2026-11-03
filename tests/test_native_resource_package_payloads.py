"""Resource capsules bind actual SDK code/models, never package versions alone."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import marshal
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformation_portal.photographic_delivery import native_resources as resources

pytestmark = pytest.mark.unit


def _record(path: str, value: bytes) -> list[str]:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode("ascii")
    return [path, "sha256=" + encoded, str(len(value))]


def _read_record(distribution) -> list[list[str]]:
    return list(csv.reader(io.StringIO((distribution._path / "RECORD").read_text())))


def _write_record(distribution, rows: list[list[str]]) -> None:
    raw = io.StringIO(newline="")
    csv.writer(raw, lineterminator="\n").writerows(rows)
    (distribution._path / "RECORD").write_text(raw.getvalue())


@pytest.fixture(name="installed")
def fixture_installed(tmp_path, monkeypatch):
    payloads = {"closed_sdk/__init__.py": b"CLOSED = True\n", "closed_sdk/data/_retry.json": b'{"attempts":5}'}
    for name, value in payloads.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    metadata_root = tmp_path / "closed_sdk-1.0.dist-info"
    metadata_root.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: closed-sdk\nVersion: 1.0\n"
    (metadata_root / "METADATA").write_bytes(metadata)
    distribution = importlib.metadata.PathDistribution(metadata_root)
    entries = [_record(name, value) for name, value in payloads.items()]
    entries.extend([_record(metadata_root.name + "/METADATA", metadata), [metadata_root.name + "/RECORD", "", ""]])
    _write_record(distribution, entries)
    monkeypatch.setattr(resources.importlib.metadata, "distribution", lambda name: distribution)
    return tmp_path, distribution, payloads


def _identity():
    return resources._distribution_payload_identity("closed-sdk", ("closed_sdk",))


def _cache(source: Path, *, replacement: bytes | None = None, optimization: int = 0, flags: int = 0) -> Path:
    """Construct a selectable test cache without executing its code."""
    value = source.read_bytes()
    code = compile(
        value if replacement is None else replacement, str(source), "exec", dont_inherit=True, optimize=optimization
    )
    header = importlib.util.MAGIC_NUMBER + flags.to_bytes(4, "little")
    if flags:
        header += importlib.util.source_hash(value)
    else:
        header += (int(source.stat().st_mtime) & 0xFFFFFFFF).to_bytes(4, "little") + len(value).to_bytes(4, "little")
    path = Path(importlib.util.cache_from_source(str(source), optimization="" if optimization == 0 else str(optimization)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + marshal.dumps(code))
    return path


def test_actual_sdk_source_and_model_bytes_have_stable_bounded_identity(installed):
    root, distribution, payloads = installed
    result = _identity()
    rows = [[name, len(value), hashlib.sha256(value).hexdigest()] for name, value in sorted(payloads.items())]
    assert result == {
        "sha256": hashlib.sha256(resources._canonical(rows)).hexdigest(),
        "files": 2,
        "bytes": sum(map(len, payloads.values())),
    }
    _write_record(distribution, list(reversed(_read_record(distribution))))
    assert _identity() == result
    _cache(root / "closed_sdk/__init__.py")
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
    entries = _read_record(distribution)
    entry = entries[0]
    if change == "missing":
        (root / entry[0]).unlink()
    elif change == "extra":
        (root / "closed_sdk/unrecorded.py").write_bytes(b"unreviewed executable")
    elif change == "unhashed":
        entry[1], entry[2] = "", ""
    elif change == "wrong_size":
        entry[2] = str(int(entry[2]) + 1)
    elif change == "wrong_algorithm":
        entry[1] = "md5=arbitrary"
    elif change == "symlink":
        original = root / entry[0]
        saved = root / "saved.py"
        original.rename(saved)
        original.symlink_to(saved)
    else:
        entries.append(entry.copy())
    _write_record(distribution, entries)
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
    (distribution._path / "RECORD").unlink()
    with pytest.raises(ValueError, match="bounded RECORD"):
        _identity()


@pytest.mark.parametrize("change", ["oversized", "too_many_missing_rows"])
def test_record_is_bounded_before_importlib_can_materialize_file_entries(installed, monkeypatch, change):
    _, distribution, _ = installed

    def forbidden(_self):
        pytest.fail("Unbounded importlib Distribution.files must never be accessed")

    monkeypatch.setattr(importlib.metadata.PathDistribution, "files", property(forbidden))
    record = distribution._path / "RECORD"
    if change == "oversized":
        with record.open("wb") as stream:
            stream.truncate(resources._MAX_PACKAGE_RECORD_BYTES + 1)
    else:
        _write_record(
            distribution,
            [
                _record(f"unrelated/missing-{index}.py", b"missing")
                for index in range(resources._MAX_PACKAGE_PAYLOAD_FILES + 1)
            ],
        )
        assert record.stat().st_size < resources._MAX_PACKAGE_RECORD_BYTES
    with pytest.raises(ValueError, match="bounded RECORD"):
        _identity()


@pytest.mark.parametrize("raw", [b"one,two\n", b'"unfinished', b"bad,sha256=invalid,1\n", b"bad,,1\n", b"bad,,\xff\n"])
def test_malformed_record_rows_fail_closed(installed, raw):
    _, distribution, _ = installed
    (distribution._path / "RECORD").write_bytes(raw)
    with pytest.raises(ValueError, match="RECORD"):
        _identity()


@pytest.mark.parametrize("alias", ["metadata", "record", "root"])
def test_record_metadata_and_root_must_be_canonical_local_paths(installed, alias, monkeypatch):
    root, distribution, _ = installed
    if alias == "metadata":
        path = distribution._path
        saved = path.with_name("preserved.dist-info")
        path.rename(saved)
        path.symlink_to(saved, target_is_directory=True)
    elif alias == "record":
        path = distribution._path / "RECORD"
        saved = distribution._path / "preserved-record"
        path.rename(saved)
        path.symlink_to(saved)
    else:
        linked = root / "alias"
        linked.symlink_to(root, target_is_directory=True)
        distribution = importlib.metadata.PathDistribution(linked / distribution._path.name)
        monkeypatch.setattr(resources.importlib.metadata, "distribution", lambda name: distribution)
    with pytest.raises(ValueError, match="RECORD"):
        _identity()


def test_custom_distribution_cannot_supply_unbounded_file_metadata(installed, monkeypatch):
    monkeypatch.setattr(resources.importlib.metadata, "distribution", lambda name: SimpleNamespace())
    with pytest.raises(ValueError, match="bounded local RECORD"):
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
    source = root / "closed_sdk/__init__.py"
    cache = _cache(source)
    spec = importlib.util.spec_from_file_location("closed_sdk", source)
    monkeypatch.setitem(
        resources.sys.modules,
        "closed_sdk",
        SimpleNamespace(__file__=str(source), __cached__=str(cache), __spec__=spec, __loader__=spec.loader),
    )
    monkeypatch.setitem(resources.sys.modules, "closed_sdk.namespace", SimpleNamespace())
    assert _identity()["files"] == 2


@pytest.mark.parametrize("optimization", [0, 1, 2])
@pytest.mark.parametrize("flags", [0, 1, 3])
def test_verified_warm_timestamp_hash_and_optimized_caches_preserve_identity(installed, optimization, flags):
    root, _, _ = installed
    baseline = _identity()
    cache = _cache(root / "closed_sdk/__init__.py", optimization=optimization, flags=flags)
    assert _identity() == baseline
    cache.write_bytes(cache.read_bytes())
    assert _identity() == baseline


@pytest.mark.parametrize("already_loaded", [False, True])
def test_restored_reviewed_source_cannot_authorize_a_timestamp_valid_forged_cache(installed, monkeypatch, already_loaded):
    root, _, _ = installed
    source = root / "closed_sdk/__init__.py"
    original = source.read_bytes()
    replacement = original.replace(b"True", b"None")
    assert len(replacement) == len(original)
    cache = _cache(source, replacement=replacement)
    assert source.read_bytes() == original
    if already_loaded:
        spec = importlib.util.spec_from_file_location("closed_sdk", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.CLOSED is None
        monkeypatch.setitem(resources.sys.modules, "closed_sdk", module)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Native resource validation must never unmarshal untrusted code")

    monkeypatch.setattr(marshal, "loads", forbidden)
    with pytest.raises(ValueError, match="cache verification failed"):
        _identity()
    assert cache.exists() and source.read_bytes() == original


@pytest.mark.parametrize("prefix", [False, True])
def test_selected_prefix_and_single_file_package_caches_are_verified(tmp_path, monkeypatch, prefix):
    source = tmp_path / "six.py"
    source.write_bytes(b"VALUE = 1\n")
    if prefix:
        monkeypatch.setattr(resources.sys, "pycache_prefix", str(tmp_path / "external-prefix"))
    _cache(source)
    resources._verify_executable_caches((source,))
    _cache(source, replacement=b"VALUE = 2\n")
    with pytest.raises(ValueError, match="cache verification failed"):
        resources._verify_executable_caches((source,))


@pytest.mark.parametrize("change", ["missing", "foreign", "wrong_optimization", "sourceless"])
def test_loaded_python_modules_require_selected_cache_evidence(installed, monkeypatch, change):
    root, _, _ = installed
    source = root / "closed_sdk/__init__.py"
    cached = _cache(source, optimization=1 if change == "wrong_optimization" else 0)
    if change == "missing":
        cached.unlink()
    elif change == "foreign":
        cached = root / "foreign.pyc"
    elif change == "sourceless":
        cached = None
    module = SimpleNamespace(__file__=str(source), __cached__=None if cached is None else str(cached))
    module.__spec__ = importlib.util.spec_from_file_location("closed_sdk", source)
    module.__loader__ = module.__spec__.loader
    monkeypatch.setitem(resources.sys.modules, "closed_sdk", module)
    with pytest.raises(ValueError, match="cache"):
        _identity()


@pytest.mark.parametrize("change", ["custom", "subclass", "foreign_spec", "missing_spec"])
def test_loaded_custom_or_mixed_loaders_cannot_borrow_a_verified_cache(installed, monkeypatch, change):
    root, _, _ = installed
    source = root / "closed_sdk/__init__.py"
    cache = _cache(source)
    spec = importlib.util.spec_from_file_location("closed_sdk", source)
    module = SimpleNamespace(__file__=str(source), __cached__=str(cache), __spec__=spec, __loader__=spec.loader)
    if change == "custom":
        module.__loader__ = object()
    elif change == "subclass":

        class AlternateLoader(importlib.machinery.SourceFileLoader):
            pass

        module.__loader__ = AlternateLoader("closed_sdk", str(source))
        spec.loader = module.__loader__
    elif change == "foreign_spec":
        spec.origin = str(root / "other.py")
    else:
        module.__spec__ = None
    monkeypatch.setitem(resources.sys.modules, "closed_sdk", module)
    with pytest.raises(ValueError, match="source loader"):
        _identity()


@pytest.mark.parametrize("change", ["truncated", "magic", "flags", "timestamp", "source_hash", "oversized"])
def test_cache_headers_and_file_bounds_fail_closed_before_loading(installed, monkeypatch, change):
    from transformation_portal.lux_depth_v4 import runtime as bytecode

    root, _, _ = installed
    cache = _cache(root / "closed_sdk/__init__.py", flags=3 if change == "source_hash" else 0)
    raw = cache.read_bytes()
    if change == "truncated":
        raw = raw[:10]
    elif change == "magic":
        raw = b"bad!" + raw[4:]
    elif change == "flags":
        raw = raw[:4] + (7).to_bytes(4, "little") + raw[8:]
    elif change == "timestamp":
        raw = raw[:8] + b"\0" * 8 + raw[16:]
    elif change == "source_hash":
        raw = raw[:8] + b"\0" * 8 + raw[16:]
    else:
        monkeypatch.setattr(bytecode, "_MAX_BYTECODE_BYTES", len(raw) - 1)
    cache.write_bytes(raw)
    with pytest.raises(ValueError, match="cache"):
        _identity()


def test_cache_compilation_source_and_total_budget_are_bounded(installed, monkeypatch):
    from transformation_portal.lux_depth_v4 import runtime as bytecode

    root, _, _ = installed
    source = root / "closed_sdk/__init__.py"
    _cache(source)
    monkeypatch.setattr(bytecode, "_MAX_BYTECODE_SOURCE_BYTES", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="cache verification failed"):
        _identity()
    monkeypatch.setattr(bytecode, "_MAX_BYTECODE_SOURCE_BYTES", 4 * 1024**2)
    monkeypatch.setattr(bytecode, "_MAX_BYTECODE_VALIDATION_BYTES", 1)
    with pytest.raises(ValueError, match="validation budget"):
        _identity()


def test_cache_source_must_remain_the_verified_package_bytes(installed, monkeypatch):
    root, _, _ = installed
    source = root / "closed_sdk/__init__.py"
    _cache(source)
    original = resources._verify_executable_caches

    def changed(sources, loaded=(), expected_hashes=None):
        source.write_bytes(source.read_bytes().replace(b"True", b"None"))
        return original(sources, loaded, expected_hashes)

    monkeypatch.setattr(resources, "_verify_executable_caches", changed)
    with pytest.raises(ValueError, match="verified bytes"):
        _identity()


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
