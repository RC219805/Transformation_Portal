"""PluginLoader external path trust-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.plugins.loader import PluginLoader

pytestmark = [pytest.mark.unit, pytest.mark.security]


def _builtin_loader_path() -> Path:
    import transformation_portal.plugins.loader as loader_module

    return (Path(loader_module.__file__).resolve().parent / "builtin").resolve()


def test_env_plugin_path_ignored_when_external_loading_disabled(monkeypatch, tmp_path: Path):
    env_path = tmp_path / "env_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_path))
    monkeypatch.delenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", raising=False)

    loader = PluginLoader()

    assert [path.resolve() for path in loader.get_search_paths()] == [_builtin_loader_path()]


def test_direct_external_path_rejected_when_external_loading_disabled(tmp_path: Path):
    loader = PluginLoader(allow_external_plugins=False)

    with pytest.raises(ValueError, match="External plugin paths are disabled"):
        loader.add_search_path(tmp_path / "external")

    assert [path.resolve() for path in loader.get_search_paths()] == [_builtin_loader_path()]


def test_env_opt_in_adds_env_plugin_path(monkeypatch, tmp_path: Path):
    env_path = tmp_path / "env_plugins"
    monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "yes")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(env_path))

    loader = PluginLoader()
    paths = [path.resolve() for path in loader.get_search_paths()]

    assert _builtin_loader_path() in paths
    assert env_path.resolve() in paths
    assert (Path.home() / ".transformation_portal" / "plugins").resolve() in paths


def test_constructor_opt_in_allows_programmatic_external_path(tmp_path: Path):
    plugin_path = tmp_path / "programmatic_plugins"

    loader = PluginLoader(allow_external_plugins=True)
    loader.add_search_path(plugin_path)

    assert plugin_path.resolve() in [path.resolve() for path in loader.get_search_paths()]
