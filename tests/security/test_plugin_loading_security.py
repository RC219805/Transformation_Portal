"""Security tests for plugin loading trust boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.plugins.loader import PluginLoader
from transformation_portal.plugins.registry import PluginRegistry


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


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
    pytest.mark.security,
]
