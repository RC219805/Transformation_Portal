"""Helpers for isolated PluginLoader unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from transformation_portal.plugins.loader import PluginLoader


def isolated_loader(
    search_path: Path,
    *,
    auto_resolve_dependencies: bool = True,
    plugin_trust_store_path: Path | None = None,
) -> PluginLoader:
    """Return a loader whose discovery paths are limited to one temp path."""
    loader = PluginLoader(
        allow_external_plugins=True,
        auto_resolve_dependencies=auto_resolve_dependencies,
        plugin_trust_store_path=plugin_trust_store_path,
    )
    loader._search_paths.clear()  # noqa: SLF001 - tests need hermetic paths.
    loader.add_search_path(search_path)
    return loader


def write_plugin_module(
    directory: Path,
    module_name: str,
    *,
    class_name: str = "DemoPlugin",
    plugin_name: str = "demo_plugin",
    plugin_type: str = "CUSTOM",
    execute_result: str = "ok",
    cleanup_counter: bool = False,
) -> Path:
    """Write a concrete PluginInterface implementation to a temp module."""
    directory.mkdir(parents=True, exist_ok=True)
    cleanup_method = (
        """
    def cleanup(self):
        global cleanup_calls
        cleanup_calls += 1
        super().cleanup()
"""
        if cleanup_counter
        else ""
    )
    module_path = directory / f"{module_name}.py"
    module_path.write_text(
        f"""from __future__ import annotations

from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

cleanup_calls = 0


class {class_name}(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name={plugin_name!r},
            version="1.0.0",
            plugin_type=PluginType.{plugin_type},
            description="test plugin",
            author="tests",
        )

    def initialize(self, config=None):
        self._initialized = True
        self._config = config or {{}}

    def execute(self, *args, **kwargs):
        return {execute_result!r}
{cleanup_method}
""",
        encoding="utf-8",
    )
    return module_path


def write_plugin_json(
    package_dir: Path,
    *,
    name: str = "demo_plugin",
    module_name: str = "demo_plugin",
    class_name: str = "DemoPlugin",
    plugin_type: str = "custom",
    dependencies: Iterable[str] = (),
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write a plugin.json manifest for a temp plugin package."""
    package_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "name": name,
        "version": "1.0.0",
        "plugin_type": plugin_type,
        "entry_point": f"{module_name}:{class_name}",
        "description": "test manifest",
        "author": "tests",
        "dependencies": list(dependencies),
    }
    if extra:
        payload.update(extra)
    manifest_path = package_dir / "plugin.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_pyproject_manifest(
    package_dir: Path,
    *,
    name: str = "demo_plugin",
    module_name: str = "demo_plugin",
    class_name: str = "DemoPlugin",
    plugin_type: str = "custom",
    dependencies: Iterable[str] = (),
) -> Path:
    """Write a pyproject.toml plugin manifest for a temp plugin package."""
    package_dir.mkdir(parents=True, exist_ok=True)
    dependency_list = json.dumps(list(dependencies))
    pyproject_path = package_dir / "pyproject.toml"
    pyproject_path.write_text(
        f"""[tool.transformation_portal.plugin]
name = "{name}"
version = "1.0.0"
plugin_type = "{plugin_type}"
entry_point = "{module_name}:{class_name}"
description = "test manifest"
author = "tests"
dependencies = {dependency_list}
""",
        encoding="utf-8",
    )
    return pyproject_path
