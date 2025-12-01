"""Plugin manager for high-level plugin lifecycle management."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

from .interface import (
    PluginInitializationError,
    PluginInterface,
    PluginType,
)
from .loader import LoadedPlugin, PluginLoader, get_global_loader
from .registry import PluginRegistry, get_global_registry

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """State of a plugin in its lifecycle."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    VALIDATED = "validated"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class PluginContext:
    """Context for plugin execution with configuration and state."""
    config: Dict[str, Any] = field(default_factory=dict)
    state: PluginState = PluginState.DISCOVERED
    error_message: Optional[str] = None
    initialization_count: int = 0
    execution_count: int = 0
    last_result: Any = None


@dataclass
class ExecutionResult:
    """Result of plugin execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    plugin_name: str = ""
    execution_time_ms: float = 0.0


class PluginManager:
    """High-level manager for plugin lifecycle and execution.

    Provides a unified interface for:
    - Plugin discovery and loading
    - Lifecycle management (initialize, execute, cleanup)
    - Context-aware plugin selection
    - Execution with fallback strategies
    - Plugin configuration management

    Example:
        >>> manager = PluginManager()
        >>> manager.discover_plugins()
        >>> manager.initialize_plugin("my_depth_model", {"device": "cuda"})
        >>>
        >>> # Execute with automatic fallback
        >>> result = manager.execute(
        ...     plugin_type=PluginType.DEPTH_MODEL,
        ...     input_data=image,
        ...     fallback_plugins=["depth_anything", "midas"]
        ... )
        >>>
        >>> # Context manager for automatic cleanup
        >>> with manager.plugin_session("my_enhancer") as enhancer:
        ...     result = enhancer.enhance(image)
    """

    def __init__(
        self,
        loader: Optional[PluginLoader] = None,
        registry: Optional[PluginRegistry] = None,
        auto_discover: bool = False,
    ):
        """Initialize plugin manager.

        Args:
            loader: Plugin loader instance (uses global if not provided)
            registry: Plugin registry instance (uses global if not provided)
            auto_discover: Automatically discover plugins on initialization
        """
        self._loader = loader or get_global_loader()
        self._registry = registry or get_global_registry()
        self._contexts: Dict[str, PluginContext] = {}
        self._lock = Lock()
        self._default_configs: Dict[str, Dict[str, Any]] = {}

        if auto_discover:
            self.discover_plugins()

    def discover_plugins(
        self,
        search_paths: Optional[List[Path]] = None
    ) -> Dict[str, LoadedPlugin]:
        """Discover and load all plugins.

        Args:
            search_paths: Additional paths to search

        Returns:
            Dictionary of discovered plugins
        """
        if search_paths:
            for path in search_paths:
                self._loader.add_search_path(path)

        loaded = self._loader.discover_all()

        # Register discovered plugins with registry
        for loaded_plugin in loaded:
            if loaded_plugin.is_valid and loaded_plugin.plugin:
                try:
                    self._registry.register(
                        loaded_plugin.plugin,
                        replace_existing=True
                    )

                    # Initialize context for plugin
                    with self._lock:
                        self._contexts[loaded_plugin.manifest.name] = PluginContext(
                            state=PluginState.LOADED
                        )

                except Exception as e:
                    logger.error(f"Failed to register plugin {loaded_plugin.manifest.name}: {e}")

        return self._loader.get_loaded_plugins()

    def get_plugin(
        self,
        name: str,
        initialize: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[PluginInterface]:
        """Get a plugin by name.

        Args:
            name: Plugin name
            initialize: Initialize plugin if not already initialized
            config: Configuration for initialization

        Returns:
            Plugin instance or None if not found
        """
        loaded = self._loader.load_plugin(name)
        if not loaded or not loaded.plugin:
            return None

        if initialize:
            self.initialize_plugin(name, config)

        return loaded.plugin

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of plugin instances
        """
        loaded = self._loader.get_plugins_by_type(plugin_type)
        return [lp.plugin for lp in loaded if lp.plugin]

    def initialize_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Initialize a plugin with configuration.

        Args:
            name: Plugin name
            config: Configuration dictionary

        Returns:
            True if initialization succeeded

        Raises:
            PluginInitializationError: If initialization fails
        """
        loaded = self._loader.load_plugin(name)
        if not loaded or not loaded.plugin:
            raise PluginInitializationError(f"Plugin '{name}' not found")

        plugin = loaded.plugin

        # Merge with default config
        merged_config = self._default_configs.get(name, {}).copy()
        if config:
            merged_config.update(config)

        try:
            plugin.initialize(merged_config)

            with self._lock:
                if name not in self._contexts:
                    self._contexts[name] = PluginContext()

                self._contexts[name].state = PluginState.INITIALIZED
                self._contexts[name].config = merged_config
                self._contexts[name].initialization_count += 1
                self._contexts[name].error_message = None

            logger.info(f"Initialized plugin: {name}")
            return True

        except Exception as e:
            with self._lock:
                if name in self._contexts:
                    self._contexts[name].state = PluginState.ERROR
                    self._contexts[name].error_message = str(e)

            raise PluginInitializationError(f"Failed to initialize '{name}': {e}") from e

    def execute(
        self,
        name: str,
        *args,
        fallback_plugins: Optional[List[str]] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute a plugin's main function.

        Args:
            name: Plugin name
            *args: Arguments for execution
            fallback_plugins: List of fallback plugin names if primary fails
            **kwargs: Keyword arguments for execution

        Returns:
            ExecutionResult with success status and result/error
        """
        import time

        plugins_to_try = [name] + (fallback_plugins or [])

        for plugin_name in plugins_to_try:
            loaded = self._loader.load_plugin(plugin_name)
            if not loaded or not loaded.plugin:
                continue

            plugin = loaded.plugin

            # Auto-initialize if needed
            if not plugin._initialized:
                try:
                    self.initialize_plugin(plugin_name)
                except PluginInitializationError as e:
                    logger.warning(f"Failed to initialize {plugin_name}: {e}")
                    continue

            # Execute
            start_time = time.perf_counter()
            try:
                result = plugin.execute(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000

                with self._lock:
                    if plugin_name in self._contexts:
                        self._contexts[plugin_name].execution_count += 1
                        self._contexts[plugin_name].last_result = result
                        self._contexts[plugin_name].state = PluginState.ACTIVE

                return ExecutionResult(
                    success=True,
                    result=result,
                    plugin_name=plugin_name,
                    execution_time_ms=elapsed,
                )

            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.warning(f"Plugin {plugin_name} execution failed: {e}")

                with self._lock:
                    if plugin_name in self._contexts:
                        self._contexts[plugin_name].state = PluginState.ERROR
                        self._contexts[plugin_name].error_message = str(e)

                if plugin_name == plugins_to_try[-1]:
                    # Last plugin, return error
                    return ExecutionResult(
                        success=False,
                        error=str(e),
                        plugin_name=plugin_name,
                        execution_time_ms=elapsed,
                    )

        return ExecutionResult(
            success=False,
            error=f"No plugins available for execution: {plugins_to_try}",
        )

    def execute_by_type(
        self,
        plugin_type: PluginType,
        *args,
        prefer_plugin: Optional[str] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute a plugin by type with automatic selection.

        Args:
            plugin_type: Type of plugin to execute
            *args: Arguments for execution
            prefer_plugin: Preferred plugin name (used first if available)
            **kwargs: Keyword arguments for execution

        Returns:
            ExecutionResult with success status and result/error
        """
        plugins = self.get_plugins_by_type(plugin_type)

        if not plugins:
            return ExecutionResult(
                success=False,
                error=f"No plugins available for type: {plugin_type.value}",
            )

        # Build priority list
        plugin_names = [p.metadata.name for p in plugins]

        if prefer_plugin and prefer_plugin in plugin_names:
            # Move preferred plugin to front
            plugin_names.remove(prefer_plugin)
            plugin_names.insert(0, prefer_plugin)

        return self.execute(
            plugin_names[0],
            *args,
            fallback_plugins=plugin_names[1:],
            **kwargs
        )

    @contextmanager
    def plugin_session(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Iterator[PluginInterface]:
        """Context manager for plugin session with automatic cleanup.

        Args:
            name: Plugin name
            config: Configuration for initialization

        Yields:
            Initialized plugin instance

        Example:
            >>> with manager.plugin_session("my_enhancer") as enhancer:
            ...     result = enhancer.enhance(image, strength=0.8)
        """
        loaded = self._loader.load_plugin(name)
        if not loaded or not loaded.plugin:
            raise PluginInitializationError(f"Plugin '{name}' not found")

        plugin = loaded.plugin

        try:
            # Initialize if needed
            if not plugin._initialized:
                self.initialize_plugin(name, config)

            yield plugin

        finally:
            # Mark as no longer active
            with self._lock:
                if name in self._contexts:
                    self._contexts[name].state = PluginState.INITIALIZED

    def set_default_config(self, name: str, config: Dict[str, Any]) -> None:
        """Set default configuration for a plugin.

        Args:
            name: Plugin name
            config: Default configuration
        """
        self._default_configs[name] = config

    def get_plugin_state(self, name: str) -> Optional[PluginState]:
        """Get the current state of a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin state or None if not found
        """
        with self._lock:
            if name in self._contexts:
                return self._contexts[name].state
        return None

    def get_plugin_context(self, name: str) -> Optional[PluginContext]:
        """Get full context for a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin context or None if not found
        """
        with self._lock:
            return self._contexts.get(name)

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin and clean up resources.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unloaded
        """
        # Unregister from registry
        loaded = self._loader.load_plugin(name)
        if loaded and loaded.plugin:
            plugin_type = loaded.plugin.metadata.plugin_type.value
            self._registry.unregister(plugin_type, name)

        # Remove context
        with self._lock:
            if name in self._contexts:
                self._contexts[name].state = PluginState.UNLOADED
                del self._contexts[name]

        # Unload from loader
        return self._loader.unload_plugin(name)

    def reload_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Hot-reload a plugin.

        Args:
            name: Plugin name
            config: New configuration

        Returns:
            True if reload succeeded
        """
        # Reload via loader
        reloaded = self._loader.reload_plugin(name)
        if not reloaded or not reloaded.plugin:
            return False

        # Re-register
        try:
            self._registry.register(reloaded.plugin, replace_existing=True)
        except Exception as e:
            logger.error(f"Failed to re-register plugin {name}: {e}")
            return False

        # Re-initialize if config provided
        if config:
            try:
                self.initialize_plugin(name, config)
            except PluginInitializationError:
                return False

        return True

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        state: Optional[PluginState] = None
    ) -> List[Dict[str, Any]]:
        """List plugins with optional filtering.

        Args:
            plugin_type: Filter by plugin type
            state: Filter by plugin state

        Returns:
            List of plugin info dictionaries
        """
        result: List[Dict[str, Any]] = []
        loaded_plugins = self._loader.get_loaded_plugins()

        for name, loaded in loaded_plugins.items():
            if not loaded.plugin:
                continue

            # Filter by type
            if plugin_type and loaded.plugin.metadata.plugin_type != plugin_type:
                continue

            # Get context
            context = self._contexts.get(name)

            # Filter by state
            if state and context and context.state != state:
                continue

            info = loaded.plugin.get_info()
            info["source_path"] = str(loaded.source_path)
            info["state"] = context.state.value if context else "unknown"
            info["execution_count"] = context.execution_count if context else 0

            result.append(info)

        return result

    def cleanup_all(self) -> None:
        """Clean up all plugins and release resources."""
        loaded_plugins = self._loader.get_loaded_plugins()

        for name in list(loaded_plugins.keys()):
            try:
                self.unload_plugin(name)
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {name}: {e}")

        with self._lock:
            self._contexts.clear()

        logger.info("All plugins cleaned up")


# Global manager instance
_global_manager: Optional[PluginManager] = None
_manager_lock = Lock()


def get_global_manager() -> PluginManager:
    """Get the global plugin manager singleton.

    Returns:
        Global PluginManager instance
    """
    global _global_manager

    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = PluginManager()

    return _global_manager
