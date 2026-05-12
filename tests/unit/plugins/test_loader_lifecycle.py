"""PluginLoader lifecycle and singleton tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.unit.plugins.loader_test_helpers import (
    isolated_loader,
    write_plugin_json,
    write_plugin_module,
)

pytestmark = [pytest.mark.unit]


def test_unload_plugin_calls_cleanup_and_removes_module_cache(tmp_path: Path):
    package_dir = tmp_path / "lifecycle_package"
    write_plugin_module(
        package_dir,
        "lifecycle_plugin",
        class_name="LifecyclePlugin",
        plugin_name="lifecycle_plugin",
        cleanup_counter=True,
    )
    write_plugin_json(
        package_dir,
        name="lifecycle_plugin",
        module_name="lifecycle_plugin",
        class_name="LifecyclePlugin",
    )
    loader = isolated_loader(tmp_path)
    loaded = loader.discover_all()[0]
    module = sys.modules["lifecycle_plugin"]

    assert loaded.module_name == "lifecycle_plugin"
    assert loader.load_plugin("lifecycle_plugin") is loaded

    assert loader.unload_plugin("lifecycle_plugin") is True

    assert module.cleanup_calls == 1
    assert loader.load_plugin("lifecycle_plugin") is None
    assert "lifecycle_plugin" not in loader._module_cache  # noqa: SLF001 - lifecycle cache contract
    assert "lifecycle_plugin" not in sys.modules


def test_unload_missing_plugin_returns_false(tmp_path: Path):
    assert isolated_loader(tmp_path).unload_plugin("missing_plugin") is False


def test_reload_plugin_reloads_manifest_plugin(tmp_path: Path):
    package_dir = tmp_path / "reload_package"
    write_plugin_module(
        package_dir,
        "reload_plugin",
        class_name="ReloadPlugin",
        plugin_name="reload_plugin",
        execute_result="first",
    )
    write_plugin_json(
        package_dir,
        name="reload_plugin",
        module_name="reload_plugin",
        class_name="ReloadPlugin",
    )
    loader = isolated_loader(tmp_path)
    first = loader.discover_all()[0]

    reloaded = loader.reload_plugin("reload_plugin")

    assert reloaded is not None
    assert reloaded is not first
    assert reloaded.plugin.metadata.name == "reload_plugin"


def test_get_global_loader_returns_stable_singleton(monkeypatch):
    import transformation_portal.plugins.loader as loader_module

    monkeypatch.setattr(loader_module, "_global_loader", None)

    first = loader_module.get_global_loader()
    second = loader_module.get_global_loader()

    assert first is second
