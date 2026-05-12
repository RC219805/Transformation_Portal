"""PluginLoader single-file plugin discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.plugins.loader_test_helpers import isolated_loader, write_plugin_module

pytestmark = [pytest.mark.unit]


def test_file_plugin_discovery_skips_private_modules(tmp_path: Path):
    write_plugin_module(
        tmp_path,
        "public_file_plugin",
        class_name="PublicFilePlugin",
        plugin_name="public_file_plugin",
    )
    (tmp_path / "_private.py").write_text(
        """raise AssertionError("private plugin modules must not be imported")
""",
        encoding="utf-8",
    )

    discovered = isolated_loader(tmp_path).discover_all()

    assert [plugin.manifest.name for plugin in discovered if plugin.manifest] == ["public_file_plugin"]
    assert discovered[0].source_path == tmp_path / "public_file_plugin.py"
    assert discovered[0].module_name == "plugin_public_file_plugin"


def test_file_plugin_discovery_loads_only_concrete_plugin_classes(tmp_path: Path):
    (tmp_path / "mixed_file_plugin.py").write_text(
        """from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType


class AbstractOnly(PluginInterface):
    pass


class ConcretePlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name="concrete_file_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )

    def initialize(self, config=None):
        self._initialized = True

    def execute(self, *args, **kwargs):
        return "ok"
""",
        encoding="utf-8",
    )

    discovered = isolated_loader(tmp_path).discover_all()

    assert len(discovered) == 1
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.name == "concrete_file_plugin"
    assert discovered[0].plugin.execute() == "ok"
