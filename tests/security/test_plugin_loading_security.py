"""Security tests for plugin loading trust boundaries."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from transformation_portal.plugins.loader import PluginLoader
from transformation_portal.plugins.registry import PluginRegistry
from transformation_portal.plugins.signing import PLUGIN_SIGNATURE_ALGORITHM, sign_manifest

_TRUST_SECRET = "operator-secret"
_TEST_MODULE_PREFIXES = ("sec_plugin_", "plugin_sec_plugin_")


@pytest.fixture(autouse=True)
def _forget_test_plugin_modules() -> Iterator[None]:
    """Keep modules imported from temp plugin directories out of later tests."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith(_TEST_MODULE_PREFIXES):
            sys.modules.pop(name, None)


def _write_marked_plugin(directory: Path, module_name: str, *, marker: Path) -> Path:
    """Write a plugin module named ``module_name`` that creates ``marker`` when its code runs."""
    directory.mkdir(parents=True, exist_ok=True)
    module_path = directory / f"{module_name}.py"
    module_path.write_text(
        f"""from pathlib import Path

from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

Path({str(marker)!r}).write_text("executed", encoding="utf-8")


class MarkedPlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(name={module_name!r}, version="1.0.0", plugin_type=PluginType.CUSTOM)

    def initialize(self, config=None):
        self._initialized = True

    def execute(self, *args, **kwargs):
        return {module_name!r}
""",
        encoding="utf-8",
    )
    return module_path


def _write_package(root: Path, module_name: str, *, marker: Path, signed: bool) -> Path:
    """Write ``root/<module_name>_pkg`` with a plugin.json that is optionally signed."""
    package_dir = root / f"{module_name}_pkg"
    _write_marked_plugin(package_dir, module_name, marker=marker)
    manifest: dict[str, object] = {
        "name": module_name,
        "version": "1.0.0",
        "plugin_type": "custom",
        "entry_point": f"{module_name}:MarkedPlugin",
    }
    if signed:
        manifest["signature_algorithm"] = PLUGIN_SIGNATURE_ALGORITHM
        manifest["signature_key_id"] = "ops"
        manifest["signature"] = sign_manifest(manifest, secret=_TRUST_SECRET)
    manifest_path = package_dir / "plugin.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_trust_store(tmp_path: Path) -> Path:
    trust_store = tmp_path / "plugin-trust.json"
    trust_store.write_text(json.dumps({"keys": {"ops": _TRUST_SECRET}}) + "\n", encoding="utf-8")
    return trust_store


def _builtin_loader_path() -> Path:
    """Return builtin plugin path used by PluginLoader defaults."""
    import transformation_portal.plugins.loader as loader_module

    return (Path(loader_module.__file__).resolve().parent / "builtin").resolve()


def _builtin_registry_path() -> Path:
    """Return builtin plugin path used by PluginRegistry defaults."""
    import transformation_portal.plugins.registry as registry_module

    return (Path(registry_module.__file__).resolve().parent / "builtin").resolve()


def test_loader_defaults_to_builtin_only(monkeypatch, tmp_path: Path):
    """PluginLoader should not load user/env plugin paths unless explicitly enabled."""
    env_plugin_dir = tmp_path / "external_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_plugin_dir))
    monkeypatch.delenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", raising=False)

    loader = PluginLoader()
    paths = [path.resolve() for path in loader.get_search_paths()]

    assert paths == [_builtin_loader_path()]
    assert env_plugin_dir.resolve() not in paths


def test_loader_rejects_programmatic_external_paths_when_disabled(tmp_path: Path):
    """Programmatic path additions should honor secure-by-default policy."""
    loader = PluginLoader(allow_external_plugins=False)
    with pytest.raises(ValueError, match="External plugin paths are disabled"):
        loader.add_search_path(tmp_path / "external_plugins")

    paths = [path.resolve() for path in loader.get_search_paths()]
    assert paths == [_builtin_loader_path()]


def test_loader_external_paths_enabled_via_env(monkeypatch, tmp_path: Path):
    """PluginLoader should include user/env plugin paths when opt-in flag is set."""
    env_plugin_dir = tmp_path / "external_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "1")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_plugin_dir))

    loader = PluginLoader()
    paths = [path.resolve() for path in loader.get_search_paths()]

    assert _builtin_loader_path() in paths
    assert (Path.home() / ".transformation_portal" / "plugins").resolve() in paths
    assert env_plugin_dir.resolve() in paths


def test_registry_defaults_to_builtin_only(monkeypatch, tmp_path: Path):
    """PluginRegistry default discovery paths should be builtin-only."""
    env_plugin_dir = tmp_path / "external_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_plugin_dir))
    monkeypatch.delenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", raising=False)

    registry = PluginRegistry()
    paths = [path.resolve() for path in registry._get_default_plugin_paths()]  # noqa: SLF001 - security contract test

    assert paths == [_builtin_registry_path()]
    assert env_plugin_dir.resolve() not in paths


def test_registry_external_paths_enabled_via_env(monkeypatch, tmp_path: Path):
    """PluginRegistry should include user/env paths only when explicitly enabled."""
    env_plugin_dir = tmp_path / "external_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "true")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_plugin_dir))

    registry = PluginRegistry()
    paths = [path.resolve() for path in registry._get_default_plugin_paths()]  # noqa: SLF001 - security contract test

    assert _builtin_registry_path() in paths
    assert (Path.home() / ".transformation_portal" / "plugins").resolve() in paths
    assert env_plugin_dir.resolve() in paths


def test_registry_refuses_explicit_external_path_when_external_plugins_disabled(tmp_path: Path):
    """An explicit search path must not bypass the external-plugin opt-in."""
    marker = tmp_path / "executed.txt"
    external = tmp_path / "external"
    _write_marked_plugin(external, "sec_plugin_disabled_loose", marker=marker)
    registry = PluginRegistry(allow_external_plugins=False)

    with pytest.raises(ValueError, match="External plugin paths are disabled"):
        registry.discover_plugins([external])

    assert not marker.exists()
    assert registry.list_plugins() == {}


def test_registry_loads_nothing_when_one_requested_path_is_external_and_disabled(tmp_path: Path):
    """The refusal happens before any path is searched, built-in root included."""
    marker = tmp_path / "executed.txt"
    external = tmp_path / "external"
    _write_marked_plugin(external, "sec_plugin_mixed_loose", marker=marker)
    registry = PluginRegistry(allow_external_plugins=False)

    with pytest.raises(ValueError, match="External plugin paths are disabled"):
        registry.discover_plugins([_builtin_registry_path(), external])

    assert not marker.exists()
    assert registry.list_plugins() == {}


def test_registry_default_discovery_honours_env_trust_store(monkeypatch, tmp_path: Path):
    """ADR-049: with a trust store, loose files and unsigned packages never execute."""
    external = tmp_path / "external"
    loose_marker = tmp_path / "loose.txt"
    unsigned_marker = tmp_path / "unsigned.txt"
    nested_marker = tmp_path / "nested.txt"
    _write_marked_plugin(external, "sec_plugin_env_loose", marker=loose_marker)
    _write_package(external, "sec_plugin_env_unsigned", marker=unsigned_marker, signed=False)
    _write_marked_plugin(external / "nested" / "deeper", "sec_plugin_env_nested", marker=nested_marker)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "1")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(external))
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", str(_write_trust_store(tmp_path)))

    registry = PluginRegistry()
    with pytest.warns(RuntimeWarning, match="signature verification failed"):
        registry.discover_plugins()

    assert not loose_marker.exists()
    assert not unsigned_marker.exists()
    assert not nested_marker.exists()
    assert registry.list_plugins().get("custom", []) == []


def test_registry_with_trust_store_loads_only_signed_external_packages(tmp_path: Path):
    external = tmp_path / "external"
    signed_marker = tmp_path / "signed.txt"
    unsigned_marker = tmp_path / "unsigned.txt"
    loose_marker = tmp_path / "loose.txt"
    _write_package(external, "sec_plugin_signed", marker=signed_marker, signed=True)
    _write_package(external, "sec_plugin_unsigned", marker=unsigned_marker, signed=False)
    _write_marked_plugin(external, "sec_plugin_loose", marker=loose_marker)
    registry = PluginRegistry(allow_external_plugins=True, plugin_trust_store_path=_write_trust_store(tmp_path))

    with pytest.warns(RuntimeWarning, match="signature verification failed"):
        discovered = registry.discover_plugins([external])

    assert discovered == 1
    assert registry.list_plugins() == {"custom": ["sec_plugin_signed"]}
    assert signed_marker.exists()
    assert not unsigned_marker.exists()
    assert not loose_marker.exists()


def test_registry_rejects_tampered_signed_manifest_before_import(tmp_path: Path):
    external = tmp_path / "external"
    marker = tmp_path / "tampered.txt"
    manifest_path = _write_package(external, "sec_plugin_tampered", marker=marker, signed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "9.9.9"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    registry = PluginRegistry(allow_external_plugins=True, plugin_trust_store_path=_write_trust_store(tmp_path))

    with pytest.warns(RuntimeWarning, match="does not match trusted key"):
        discovered = registry.discover_plugins([external])

    assert discovered == 0
    assert not marker.exists()
    assert registry.list_plugins() == {}


def test_registry_honours_trust_store_configured_after_construction(monkeypatch, tmp_path: Path):
    """The global registry is usually created at import time, before operator configuration."""
    external = tmp_path / "external"
    marker = tmp_path / "late.txt"
    _write_marked_plugin(external, "sec_plugin_late_loose", marker=marker)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", raising=False)
    registry = PluginRegistry(allow_external_plugins=True)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", str(_write_trust_store(tmp_path)))

    assert registry.discover_plugins([external]) == 0
    assert not marker.exists()


def test_registry_without_trust_store_keeps_loading_opted_in_external_plugins(monkeypatch, tmp_path: Path):
    """ADR-049 leaves unsigned external plugins unchanged when no trust store is configured."""
    external = tmp_path / "external"
    marker = tmp_path / "opted-in.txt"
    _write_marked_plugin(external, "sec_plugin_opted_in", marker=marker)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", raising=False)
    registry = PluginRegistry(allow_external_plugins=True)

    assert registry.discover_plugins([external]) == 1
    assert marker.exists()
    assert registry.list_plugins() == {"custom": ["sec_plugin_opted_in"]}


def test_loader_discover_path_refuses_external_directory_when_disabled(tmp_path: Path):
    marker = tmp_path / "executed.txt"
    external = tmp_path / "external"
    _write_marked_plugin(external, "sec_plugin_loader_disabled", marker=marker)
    loader = PluginLoader(allow_external_plugins=False)

    with pytest.raises(ValueError, match="External plugin paths are disabled"):
        loader.discover_path(external)

    assert not marker.exists()


def test_loader_discover_path_applies_trust_store_to_the_given_directory_only(monkeypatch, tmp_path: Path):
    external = tmp_path / "external"
    other_default = tmp_path / "env-default"
    signed_marker = tmp_path / "signed.txt"
    loose_marker = tmp_path / "loose.txt"
    other_marker = tmp_path / "other.txt"
    _write_package(external, "sec_plugin_path_signed", marker=signed_marker, signed=True)
    _write_marked_plugin(external, "sec_plugin_path_loose", marker=loose_marker)
    _write_package(other_default, "sec_plugin_path_other", marker=other_marker, signed=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(other_default))
    loader = PluginLoader(allow_external_plugins=True, plugin_trust_store_path=_write_trust_store(tmp_path))

    discovered = loader.discover_path(external)

    assert [loaded.manifest.name for loaded in discovered if loaded.is_valid] == ["sec_plugin_path_signed"]
    assert signed_marker.exists()
    assert not loose_marker.exists()
    assert not other_marker.exists()
    assert loader.discover_path(tmp_path / "does-not-exist") == []


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
    pytest.mark.security,
]
