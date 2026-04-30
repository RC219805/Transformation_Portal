"""Unit tests for plugins.registry.PluginRegistry.

Covers registration, retrieval, duplicate rejection, listing, unregistration,
metadata caching, and the external-plugin opt-in gate — without requiring any
real plugin implementations or filesystem discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Test helpers: minimal concrete PluginInterface implementations
# ---------------------------------------------------------------------------


def _make_plugin(name: str = "test_plugin", plugin_type_value: str = "processor"):
    """Create a minimal, concrete plugin for testing."""
    from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

    plugin_type = PluginType(plugin_type_value)

    class _TestPlugin(PluginInterface):
        def _create_metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name=name,
                version="1.0.0",
                plugin_type=plugin_type,
                description="Test plugin",
            )

        def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
            self._initialized = True

        def execute(self, *args, **kwargs) -> Any:
            return "executed"

    return _TestPlugin()


def _make_registry(*, allow_external: bool = False):
    from transformation_portal.plugins.registry import PluginRegistry

    return PluginRegistry(allow_external_plugins=allow_external)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestRegistryInit:
    def test_registry_initializes_all_plugin_type_categories(self):
        from transformation_portal.plugins.interface import PluginType
        from transformation_portal.plugins.registry import PluginRegistry

        registry = PluginRegistry(allow_external_plugins=False)
        for pt in PluginType:
            assert pt.value in registry._plugins

    def test_registry_starts_empty(self):
        registry = _make_registry()
        for plugins in registry._plugins.values():
            assert len(plugins) == 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_adds_plugin(self):
        registry = _make_registry()
        plugin = _make_plugin("my_proc", "processor")
        registry.register(plugin)
        retrieved = registry.get_plugin("processor", "my_proc")
        assert retrieved is plugin

    def test_register_duplicate_raises_value_error(self):
        registry = _make_registry()
        plugin = _make_plugin("dup", "processor")
        registry.register(plugin)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_plugin("dup", "processor"))

    def test_register_duplicate_with_replace_succeeds(self):
        registry = _make_registry()
        p1 = _make_plugin("replaceable", "processor")
        p2 = _make_plugin("replaceable", "processor")
        registry.register(p1)
        registry.register(p2, replace_existing=True)
        assert registry.get_plugin("processor", "replaceable") is p2

    def test_register_non_plugin_interface_raises_type_error(self):
        registry = _make_registry()
        with pytest.raises(TypeError):
            registry.register(object())  # type: ignore[arg-type]

    def test_register_populates_metadata_cache(self):
        registry = _make_registry()
        plugin = _make_plugin("cached_plugin", "enhancer")
        registry.register(plugin)
        meta = registry.get_metadata("enhancer", "cached_plugin")
        assert meta is not None
        assert meta.name == "cached_plugin"

    def test_register_deprecated_plugin_emits_warning(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _DepPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="old_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                    deprecated=True,
                    replacement="new_plugin",
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        registry = _make_registry()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            registry.register(_DepPlugin())


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieval:
    def test_get_plugin_returns_none_for_unknown(self):
        registry = _make_registry()
        assert registry.get_plugin("processor", "no_such_plugin") is None

    def test_get_plugin_returns_none_for_unknown_type(self):
        registry = _make_registry()
        assert registry.get_plugin("nonexistent_type", "anything") is None

    def test_get_metadata_returns_none_for_unknown(self):
        registry = _make_registry()
        assert registry.get_metadata("processor", "missing") is None


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:
    def test_list_plugins_all_types(self):
        registry = _make_registry()
        registry.register(_make_plugin("proc_a", "processor"))
        registry.register(_make_plugin("depth_a", "depth_model"))
        result = registry.list_plugins()
        assert "processor" in result
        assert "depth_model" in result

    def test_list_plugins_filter_by_type(self):
        registry = _make_registry()
        registry.register(_make_plugin("proc_a", "processor"))
        registry.register(_make_plugin("depth_a", "depth_model"))
        result = registry.list_plugins(plugin_type="processor")
        assert "processor" in result
        assert "depth_model" not in result

    def test_list_plugins_excludes_deprecated_by_default(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _DepPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="dep_proc",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                    deprecated=True,
                    replacement="new_proc",
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        registry = _make_registry()
        with pytest.warns(DeprecationWarning):
            registry.register(_DepPlugin())

        result = registry.list_plugins(plugin_type="processor")
        # deprecated plugin should not appear in default listing
        assert "dep_proc" not in result.get("processor", [])

    def test_list_plugins_includes_deprecated_when_requested(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _DepPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="dep_proc2",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                    deprecated=True,
                    replacement="new_proc",
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        registry = _make_registry()
        with pytest.warns(DeprecationWarning):
            registry.register(_DepPlugin())

        result = registry.list_plugins(plugin_type="processor", include_deprecated=True)
        assert "dep_proc2" in result.get("processor", [])

    def test_list_plugins_names_are_sorted(self):
        registry = _make_registry()
        registry.register(_make_plugin("zzz_proc", "processor"))
        registry.register(_make_plugin("aaa_proc", "processor"))
        names = registry.list_plugins(plugin_type="processor").get("processor", [])
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Unregistration
# ---------------------------------------------------------------------------


class TestUnregistration:
    def test_unregister_existing_returns_true(self):
        registry = _make_registry()
        registry.register(_make_plugin("to_remove", "processor"))
        result = registry.unregister("processor", "to_remove")
        assert result is True

    def test_unregister_existing_removes_from_registry(self):
        registry = _make_registry()
        registry.register(_make_plugin("gone", "processor"))
        registry.unregister("processor", "gone")
        assert registry.get_plugin("processor", "gone") is None

    def test_unregister_existing_removes_from_metadata_cache(self):
        registry = _make_registry()
        registry.register(_make_plugin("cached_gone", "processor"))
        registry.unregister("processor", "cached_gone")
        assert registry.get_metadata("processor", "cached_gone") is None

    def test_unregister_nonexistent_returns_false(self):
        registry = _make_registry()
        assert registry.unregister("processor", "ghost") is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all_plugins(self):
        registry = _make_registry()
        registry.register(_make_plugin("p1", "processor"))
        registry.register(_make_plugin("p2", "enhancer"))
        registry.clear()
        assert not registry.list_plugins()
        assert len(registry._metadata_cache) == 0

    def test_clear_reinitialises_plugin_type_keys(self):
        from transformation_portal.plugins.interface import PluginType

        registry = _make_registry()
        registry.clear()
        for pt in PluginType:
            assert pt.value in registry._plugins


# ---------------------------------------------------------------------------
# External plugin gating
# ---------------------------------------------------------------------------


class TestExternalPluginGating:
    def test_default_paths_do_not_include_home_dir(self):
        registry = _make_registry(allow_external=False)
        paths = registry._get_default_plugin_paths()
        expected = (Path.home() / ".transformation_portal" / "plugins").resolve()
        assert not any(p.resolve() == expected for p in paths)

    def test_external_enabled_includes_home_plugins_dir(self):
        registry = _make_registry(allow_external=True)
        paths = registry._get_default_plugin_paths()
        expected = (Path.home() / ".transformation_portal" / "plugins").resolve()
        assert any(p.resolve() == expected for p in paths)

    def test_env_variable_controls_external_discovery(self, monkeypatch):
        monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "true")
        from transformation_portal.plugins.registry import _external_plugins_enabled_from_env

        assert _external_plugins_enabled_from_env() is True

    def test_env_variable_false_disables_external_discovery(self, monkeypatch):
        monkeypatch.setenv("TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS", "false")
        from transformation_portal.plugins.registry import _external_plugins_enabled_from_env

        assert _external_plugins_enabled_from_env() is False

    def test_env_path_added_when_external_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", str(tmp_path))
        registry = _make_registry(allow_external=True)
        paths = registry._get_default_plugin_paths()
        expected = tmp_path.resolve()
        assert any(p.resolve() == expected for p in paths)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalRegistrySingleton:
    def test_get_global_registry_returns_same_instance(self):
        from transformation_portal.plugins.registry import get_global_registry

        r1 = get_global_registry()
        r2 = get_global_registry()
        assert r1 is r2
