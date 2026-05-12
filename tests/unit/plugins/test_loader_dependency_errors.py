"""PluginLoader dependency and entry-point failure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.plugins.loader_test_helpers import (
    isolated_loader,
    write_plugin_json,
    write_plugin_module,
)

pytestmark = [pytest.mark.unit]


def _discover_single(tmp_path: Path):
    discovered = isolated_loader(tmp_path).discover_all()
    assert len(discovered) == 1
    return discovered[0]


def test_missing_dependency_is_recorded_as_load_error(tmp_path: Path):
    package_dir = tmp_path / "missing_dependency_package"
    write_plugin_module(
        package_dir,
        "missing_dependency_plugin",
        class_name="MissingDependencyPlugin",
        plugin_name="missing_dependency_plugin",
    )
    write_plugin_json(
        package_dir,
        name="missing_dependency_plugin",
        module_name="missing_dependency_plugin",
        class_name="MissingDependencyPlugin",
        dependencies=["definitely_missing_package_for_loader_tests>=99"],
    )

    loaded = _discover_single(tmp_path)

    assert loaded.plugin.metadata.name == "missing_dependency_plugin"
    assert loaded.is_valid is False
    assert loaded.load_errors == ["Missing dependency: definitely_missing_package_for_loader_tests>=99"]


def test_installed_dependency_does_not_create_load_error(tmp_path: Path):
    package_dir = tmp_path / "installed_dependency_package"
    write_plugin_module(
        package_dir,
        "installed_dependency_plugin",
        class_name="InstalledDependencyPlugin",
        plugin_name="installed_dependency_plugin",
    )
    write_plugin_json(
        package_dir,
        name="installed_dependency_plugin",
        module_name="installed_dependency_plugin",
        class_name="InstalledDependencyPlugin",
        dependencies=["sys>=0"],
    )

    loaded = _discover_single(tmp_path)

    assert loaded.plugin.metadata.name == "installed_dependency_plugin"
    assert loaded.is_valid is True
    assert loaded.load_errors == []


def test_invalid_entry_point_format_returns_load_error(tmp_path: Path):
    package_dir = tmp_path / "invalid_entry_point_package"
    write_plugin_json(
        package_dir,
        name="invalid_entry_point_plugin",
        module_name="unused",
        extra={"entry_point": "module.without.class"},
    )

    loaded = _discover_single(tmp_path)

    assert loaded.plugin is None
    assert loaded.module_name == ""
    assert loaded.load_errors == ["Invalid entry_point format: module.without.class"]


def test_missing_entry_point_class_returns_load_error(tmp_path: Path):
    package_dir = tmp_path / "missing_class_package"
    write_plugin_module(
        package_dir,
        "missing_class_plugin",
        class_name="PresentPlugin",
        plugin_name="missing_class_plugin",
    )
    write_plugin_json(
        package_dir,
        name="missing_class_plugin",
        module_name="missing_class_plugin",
        class_name="MissingPlugin",
    )

    loaded = _discover_single(tmp_path)

    assert loaded.plugin is None
    assert loaded.module_name == "missing_class_plugin"
    assert loaded.load_errors == ["Class MissingPlugin not found in module missing_class_plugin"]


def test_entry_point_class_must_inherit_plugin_interface(tmp_path: Path):
    package_dir = tmp_path / "plain_class_package"
    package_dir.mkdir()
    (package_dir / "plain_class_plugin.py").write_text(
        """class PlainClass:
    pass
""",
        encoding="utf-8",
    )
    write_plugin_json(
        package_dir,
        name="plain_class_plugin",
        module_name="plain_class_plugin",
        class_name="PlainClass",
    )

    loaded = _discover_single(tmp_path)

    assert loaded.plugin is None
    assert loaded.load_errors == ["PlainClass does not inherit from PluginInterface"]
