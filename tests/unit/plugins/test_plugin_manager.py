"""Unit tests for plugins.manager.PluginManager.

Tests plugin lifecycle state tracking, initialize_plugin, execute with
fallback, plugin_session context manager, set_default_config, and
list_plugins — using isolated PluginLoader/PluginRegistry instances
backed entirely by in-process mock plugins (no filesystem discovery).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_plugin(
    name: str = "test_plugin",
    plugin_type_str: str = "processor",
    execute_result: Any = "result",
    init_raises: Optional[Exception] = None,
    execute_raises: Optional[Exception] = None,
):
    from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

    pt = PluginType(plugin_type_str)
    _exec_result = execute_result
    _init_raises = init_raises
    _execute_raises = execute_raises

    class _Plugin(PluginInterface):
        def _create_metadata(self):
            return PluginMetadata(name=name, version="1.0.0", plugin_type=pt)

        def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
            if _init_raises:
                raise _init_raises
            self._initialized = True

        def execute(self, *args, **kwargs) -> Any:
            if _execute_raises:
                raise _execute_raises
            return _exec_result

    return _Plugin()


def _make_loaded_plugin(plugin, source_path: str = "/fake/plugin.py"):
    """Wrap a plugin in a LoadedPlugin container."""
    from transformation_portal.plugins.loader import LoadedPlugin, PluginManifest

    manifest = PluginManifest(
        name=plugin.metadata.name,
        version=plugin.metadata.version,
        plugin_type=plugin.metadata.plugin_type.value,
        entry_point=f"fake_module:{plugin.__class__.__name__}",
    )
    return LoadedPlugin(
        plugin=plugin,
        manifest=manifest,
        source_path=Path(source_path),
        module_name=f"fake_module_{plugin.metadata.name}",
        load_errors=[],
    )


def _make_manager_with_plugin(plugin):
    """Return a PluginManager whose loader has the given plugin pre-loaded."""
    from transformation_portal.plugins.loader import PluginLoader
    from transformation_portal.plugins.manager import PluginManager
    from transformation_portal.plugins.registry import PluginRegistry

    loaded = _make_loaded_plugin(plugin)

    loader = MagicMock(spec=PluginLoader)
    loader.load_plugin.return_value = loaded
    loader.discover_all.return_value = [loaded]
    loader.get_loaded_plugins.return_value = {plugin.metadata.name: loaded}
    loader.get_plugins_by_type.return_value = [loaded]
    loader.add_search_path.return_value = None
    loader.unload_plugin.return_value = True

    registry = PluginRegistry(allow_external_plugins=False)

    manager = PluginManager(loader=loader, registry=registry)
    return manager


# ---------------------------------------------------------------------------
# PluginState enum
# ---------------------------------------------------------------------------


class TestPluginStateEnum:
    def test_all_states_exist(self):
        from transformation_portal.plugins.manager import PluginState

        values = {s.value for s in PluginState}
        assert "discovered" in values
        assert "loaded" in values
        assert "initialized" in values
        assert "active" in values
        assert "error" in values
        assert "unloaded" in values


# ---------------------------------------------------------------------------
# PluginContext dataclass
# ---------------------------------------------------------------------------


class TestPluginContextDataclass:
    def test_default_state_is_discovered(self):
        from transformation_portal.plugins.manager import PluginContext, PluginState

        ctx = PluginContext()
        assert ctx.state == PluginState.DISCOVERED

    def test_execution_count_starts_at_zero(self):
        from transformation_portal.plugins.manager import PluginContext

        assert PluginContext().execution_count == 0

    def test_error_message_defaults_to_none(self):
        from transformation_portal.plugins.manager import PluginContext

        assert PluginContext().error_message is None


# ---------------------------------------------------------------------------
# ExecutionResult dataclass
# ---------------------------------------------------------------------------


class TestExecutionResultDataclass:
    def test_success_true(self):
        from transformation_portal.plugins.manager import ExecutionResult

        r = ExecutionResult(success=True, result=42, plugin_name="p", execution_time_ms=1.5)
        assert r.success is True
        assert r.result == 42

    def test_failure_carries_error(self):
        from transformation_portal.plugins.manager import ExecutionResult

        r = ExecutionResult(success=False, error="boom", plugin_name="p")
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# initialize_plugin
# ---------------------------------------------------------------------------


class TestInitializePlugin:
    def test_initialize_sets_state_to_initialized(self):
        plugin = _make_plugin("my_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("my_proc")
        from transformation_portal.plugins.manager import PluginState

        assert manager.get_plugin_state("my_proc") == PluginState.INITIALIZED

    def test_initialize_increments_initialization_count(self):
        plugin = _make_plugin("count_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("count_proc")
        manager.initialize_plugin("count_proc")
        ctx = manager.get_plugin_context("count_proc")
        assert ctx.initialization_count == 2

    def test_initialize_merges_default_config(self):
        plugin = _make_plugin("cfg_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.set_default_config("cfg_proc", {"device": "cpu"})
        manager.initialize_plugin("cfg_proc", {"batch_size": 4})
        ctx = manager.get_plugin_context("cfg_proc")
        assert ctx.config.get("device") == "cpu"
        assert ctx.config.get("batch_size") == 4

    def test_initialize_raises_when_plugin_not_found(self):
        from transformation_portal.plugins.interface import PluginInitializationError
        from transformation_portal.plugins.loader import PluginLoader
        from transformation_portal.plugins.manager import PluginManager
        from transformation_portal.plugins.registry import PluginRegistry

        loader = MagicMock(spec=PluginLoader)
        loader.load_plugin.return_value = None

        manager = PluginManager(
            loader=loader,
            registry=PluginRegistry(allow_external_plugins=False),
        )
        with pytest.raises(PluginInitializationError, match="not found"):
            manager.initialize_plugin("ghost_plugin")

    def test_initialize_sets_error_state_on_failure_if_context_exists(self):
        # The manager only updates ERROR state when a context already exists.
        # Seed it by initialising a success variant first, then swap in a failing plugin.
        success_plugin = _make_plugin("error_proc")
        manager = _make_manager_with_plugin(success_plugin)
        manager.initialize_plugin("error_proc")

        # Now make the loader return a failing plugin for the same name
        fail_plugin = _make_plugin("error_proc", init_raises=RuntimeError("init boom"))
        loaded_fail = _make_loaded_plugin(fail_plugin, source_path="/fake/error_proc.py")
        manager._loader.load_plugin.return_value = loaded_fail

        from transformation_portal.plugins.interface import PluginInitializationError
        from transformation_portal.plugins.manager import PluginState

        with pytest.raises(PluginInitializationError):
            manager.initialize_plugin("error_proc")

        assert manager.get_plugin_state("error_proc") == PluginState.ERROR

    def test_initialize_stores_error_message_on_failure_if_context_exists(self):
        success_plugin = _make_plugin("err_msg_proc")
        manager = _make_manager_with_plugin(success_plugin)
        manager.initialize_plugin("err_msg_proc")

        fail_plugin = _make_plugin("err_msg_proc", init_raises=ValueError("specific error"))
        loaded_fail = _make_loaded_plugin(fail_plugin, source_path="/fake/err_msg_proc.py")
        manager._loader.load_plugin.return_value = loaded_fail

        from transformation_portal.plugins.interface import PluginInitializationError

        with pytest.raises(PluginInitializationError):
            manager.initialize_plugin("err_msg_proc")

        ctx = manager.get_plugin_context("err_msg_proc")
        assert ctx is not None
        assert ctx.error_message is not None
        assert "specific error" in ctx.error_message


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_returns_success_result(self):
        plugin = _make_plugin("exec_proc", execute_result="hello")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("exec_proc")
        result = manager.execute("exec_proc")
        assert result.success is True
        assert result.result == "hello"

    def test_execute_records_plugin_name(self):
        plugin = _make_plugin("named_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("named_proc")
        result = manager.execute("named_proc")
        assert result.plugin_name == "named_proc"

    def test_execute_records_timing(self):
        plugin = _make_plugin("timed_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("timed_proc")
        result = manager.execute("timed_proc")
        assert result.execution_time_ms >= 0.0

    def test_execute_increments_execution_count(self):
        plugin = _make_plugin("cnt_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("cnt_proc")
        manager.execute("cnt_proc")
        manager.execute("cnt_proc")
        ctx = manager.get_plugin_context("cnt_proc")
        assert ctx.execution_count == 2

    def test_execute_sets_state_to_active(self):
        from transformation_portal.plugins.manager import PluginState

        plugin = _make_plugin("active_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("active_proc")
        manager.execute("active_proc")
        assert manager.get_plugin_state("active_proc") == PluginState.ACTIVE

    def test_execute_failure_returns_failure_result(self):
        plugin = _make_plugin("fail_proc", execute_raises=RuntimeError("boom"))
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("fail_proc")
        result = manager.execute("fail_proc")
        assert result.success is False
        assert "boom" in (result.error or "")

    def test_execute_no_plugin_returns_failure(self):
        from transformation_portal.plugins.loader import PluginLoader
        from transformation_portal.plugins.manager import PluginManager
        from transformation_portal.plugins.registry import PluginRegistry

        loader = MagicMock(spec=PluginLoader)
        loader.load_plugin.return_value = None

        manager = PluginManager(
            loader=loader,
            registry=PluginRegistry(allow_external_plugins=False),
        )
        result = manager.execute("ghost")
        assert result.success is False

    def test_execute_with_fallback_tries_next_on_init_failure(self):
        """If primary plugin fails init, fallback plugin should be tried."""
        from transformation_portal.plugins.loader import PluginLoader
        from transformation_portal.plugins.manager import PluginManager
        from transformation_portal.plugins.registry import PluginRegistry

        primary = _make_plugin("primary", init_raises=RuntimeError("init fail"))
        fallback = _make_plugin("fallback", execute_result="fallback_result")

        loaded_primary = _make_loaded_plugin(primary)
        loaded_fallback = _make_loaded_plugin(fallback)

        def load_plugin_side_effect(name):
            if name == "primary":
                return loaded_primary
            if name == "fallback":
                return loaded_fallback
            return None

        loader = MagicMock(spec=PluginLoader)
        loader.load_plugin.side_effect = load_plugin_side_effect
        loader.get_loaded_plugins.return_value = {}

        manager = PluginManager(
            loader=loader,
            registry=PluginRegistry(allow_external_plugins=False),
        )
        result = manager.execute("primary", fallback_plugins=["fallback"])
        assert result.success is True
        assert result.result == "fallback_result"
        assert result.plugin_name == "fallback"


# ---------------------------------------------------------------------------
# plugin_session context manager
# ---------------------------------------------------------------------------


class TestPluginSession:
    def test_plugin_session_yields_initialized_plugin(self):
        plugin = _make_plugin("session_proc")
        manager = _make_manager_with_plugin(plugin)

        with manager.plugin_session("session_proc") as p:
            assert p._initialized is True

    def test_plugin_session_resets_state_on_exit(self):
        from transformation_portal.plugins.manager import PluginState

        plugin = _make_plugin("reset_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("reset_proc")

        # Force state to ACTIVE before entering session
        with manager._lock:
            manager._contexts["reset_proc"].state = PluginState.ACTIVE

        with manager.plugin_session("reset_proc"):
            pass

        # After session exits, state should drop back to INITIALIZED
        assert manager.get_plugin_state("reset_proc") == PluginState.INITIALIZED

    def test_plugin_session_raises_when_plugin_not_found(self):
        from transformation_portal.plugins.interface import PluginInitializationError
        from transformation_portal.plugins.loader import PluginLoader
        from transformation_portal.plugins.manager import PluginManager
        from transformation_portal.plugins.registry import PluginRegistry

        loader = MagicMock(spec=PluginLoader)
        loader.load_plugin.return_value = None

        manager = PluginManager(
            loader=loader,
            registry=PluginRegistry(allow_external_plugins=False),
        )
        with pytest.raises(PluginInitializationError, match="not found"):
            with manager.plugin_session("nonexistent"):
                pass


# ---------------------------------------------------------------------------
# get_plugin_state / get_plugin_context
# ---------------------------------------------------------------------------


class TestStateAndContextQueries:
    def test_get_plugin_state_returns_none_for_unknown(self):
        plugin = _make_plugin("known_proc")
        manager = _make_manager_with_plugin(plugin)
        assert manager.get_plugin_state("unknown_proc") is None

    def test_get_plugin_context_returns_none_for_unknown(self):
        plugin = _make_plugin("known_proc")
        manager = _make_manager_with_plugin(plugin)
        assert manager.get_plugin_context("unknown_proc") is None

    def test_get_plugin_context_returns_context_after_init(self):
        from transformation_portal.plugins.manager import PluginContext

        plugin = _make_plugin("ctx_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.initialize_plugin("ctx_proc")
        ctx = manager.get_plugin_context("ctx_proc")
        assert isinstance(ctx, PluginContext)


# ---------------------------------------------------------------------------
# set_default_config
# ---------------------------------------------------------------------------


class TestSetDefaultConfig:
    def test_default_config_stored(self):
        plugin = _make_plugin("cfg_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.set_default_config("cfg_proc", {"key": "value"})
        assert manager._default_configs["cfg_proc"]["key"] == "value"

    def test_default_config_overridden_by_init_config(self):
        plugin = _make_plugin("override_proc")
        manager = _make_manager_with_plugin(plugin)
        manager.set_default_config("override_proc", {"key": "default"})
        manager.initialize_plugin("override_proc", {"key": "override"})
        ctx = manager.get_plugin_context("override_proc")
        assert ctx.config["key"] == "override"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalManagerSingleton:
    def test_get_global_manager_returns_same_instance(self):
        from transformation_portal.plugins.manager import get_global_manager

        m1 = get_global_manager()
        m2 = get_global_manager()
        assert m1 is m2
