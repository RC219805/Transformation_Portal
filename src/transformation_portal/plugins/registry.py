"""Plugin registry for managing and discovering plugins."""

import importlib
import importlib.util
import inspect
import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .interface import PluginInterface, PluginMetadata, PluginType
from .loader import _PLUGIN_TRUST_STORE_ENV, PluginLoader, builtin_module_name

_ENABLE_EXTERNAL_PLUGINS_ENV = "TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _external_plugins_enabled_from_env() -> bool:
    """Return True when external plugin loading is explicitly enabled."""
    value = os.environ.get(_ENABLE_EXTERNAL_PLUGINS_ENV, "")
    return value.strip().lower() in _TRUTHY_VALUES


class PluginRegistry:
    """Central registry for managing plugins.

    Supports plugin discovery, registration, retrieval, and hot-swapping.
    Thread-safe for concurrent access.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.discover_plugins('~/.transformation_portal/plugins')
        >>> depth_model = registry.get_plugin('depth', 'depth_anything_v2')
        >>> depth_model.initialize({'device': 'cuda'})
        >>> result = depth_model.execute(image)
    """

    def __init__(
        self,
        allow_external_plugins: Optional[bool] = None,
        plugin_trust_store_path: Optional[Path] = None,
    ):
        """Initialize plugin registry.

        Args:
            allow_external_plugins: Enable plugin discovery outside the built-in
                plugin root. If None, reads TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS.
            plugin_trust_store_path: Optional JSON trust store. When set (or when
                TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE is), external plugins are
                discovered only through the signed-manifest gate of PluginLoader.
        """
        self._plugins: Dict[str, Dict[str, PluginInterface]] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._lock = Lock()
        self._plugin_paths: List[Path] = []
        self._allow_external_plugins = (
            allow_external_plugins if allow_external_plugins is not None else _external_plugins_enabled_from_env()
        )
        self._plugin_trust_store_path = (
            Path(plugin_trust_store_path).expanduser().resolve() if plugin_trust_store_path is not None else None
        )

        # Initialize plugin type categories
        for plugin_type in PluginType:
            self._plugins[plugin_type.value] = {}

    def register(self, plugin: PluginInterface, replace_existing: bool = False) -> None:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance to register
            replace_existing: Allow replacing existing plugin with same name

        Raises:
            ValueError: If plugin already registered and replace_existing=False
            PluginValidationError: If plugin fails validation
        """
        if not isinstance(plugin, PluginInterface):
            raise TypeError(f"Plugin must implement PluginInterface, got {type(plugin)}")

        plugin_type = plugin.metadata.plugin_type.value
        plugin_name = plugin.metadata.name

        with self._lock:
            if plugin_name in self._plugins[plugin_type] and not replace_existing:
                raise ValueError(
                    f"Plugin '{plugin_name}' already registered in category '{plugin_type}'. "
                    "Use replace_existing=True to override."
                )

            # Validate plugin
            if not plugin.validate():
                warnings.warn(f"Plugin '{plugin_name}' validation returned False. " "Plugin may not be properly initialized.")

            self._plugins[plugin_type][plugin_name] = plugin
            self._metadata_cache[f"{plugin_type}:{plugin_name}"] = plugin.metadata

            if plugin.metadata.deprecated:
                warnings.warn(
                    f"Plugin '{plugin_name}' is deprecated. " f"Consider using '{plugin.metadata.replacement}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    def get_plugin(
        self,
        plugin_type: str,
        plugin_name: str,
        initialize: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[PluginInterface]:
        """Retrieve a registered plugin.

        Args:
            plugin_type: Type of plugin (e.g., 'depth_model', 'processor')
            plugin_name: Name of specific plugin
            initialize: Whether to initialize plugin if not already initialized
            config: Configuration for initialization

        Returns:
            Plugin instance or None if not found
        """
        with self._lock:
            plugin = self._plugins.get(plugin_type, {}).get(plugin_name)

            if plugin and initialize and not plugin._initialized:
                plugin.initialize(config)

            return plugin

    def list_plugins(self, plugin_type: Optional[str] = None, include_deprecated: bool = False) -> Dict[str, List[str]]:
        """List all registered plugins.

        Args:
            plugin_type: Filter by plugin type (None for all)
            include_deprecated: Include deprecated plugins

        Returns:
            Dictionary mapping plugin types to lists of plugin names
        """
        result = {}

        with self._lock:
            types_to_list = [plugin_type] if plugin_type else self._plugins.keys()

            for ptype in types_to_list:
                plugins = []
                for name, plugin in self._plugins.get(ptype, {}).items():
                    if include_deprecated or not plugin.metadata.deprecated:
                        plugins.append(name)

                if plugins:
                    result[ptype] = sorted(plugins)

        return result

    def unregister(self, plugin_type: str, plugin_name: str) -> bool:
        """Unregister a plugin.

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin

        Returns:
            True if plugin was unregistered, False if not found
        """
        with self._lock:
            if plugin_name in self._plugins.get(plugin_type, {}):
                plugin = self._plugins[plugin_type][plugin_name]
                plugin.cleanup()
                del self._plugins[plugin_type][plugin_name]
                del self._metadata_cache[f"{plugin_type}:{plugin_name}"]
                return True
            return False

    def discover_plugins(self, search_paths: Optional[List[Path]] = None) -> int:
        """Discover and register plugins from filesystem.

        Args:
            search_paths: Paths to search for plugins (defaults to standard locations)

        Returns:
            Number of plugins discovered and registered

        Raises:
            ValueError: If a path lies outside the built-in plugin root while
                external plugin loading is disabled. Nothing is loaded then.
        """
        if search_paths is None:
            search_paths = self._get_default_plugin_paths()

        resolved_paths = [Path(path).expanduser().resolve() for path in search_paths]
        external_paths = {path for path in resolved_paths if not self._is_builtin_path(path)}
        if external_paths and not self._allow_external_plugins:
            raise ValueError(
                "External plugin paths are disabled. "
                "Set TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS=1 "
                "or pass allow_external_plugins=True to PluginRegistry."
            )
        trust_store_path = self._resolve_trust_store_path()

        discovered = 0

        for path in resolved_paths:
            if trust_store_path is not None and path in external_paths:
                # ADR-049: with a trust store, external code runs only behind the signed-manifest gate.
                discovered += self._discover_signed_external_plugins(path, trust_store_path)
                continue

            if not path.exists():
                continue

            # Find all Python files in plugin directory
            for plugin_file in path.rglob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    # Load module dynamically
                    builtin_module = builtin_module_name(plugin_file)
                    if builtin_module is not None:
                        # Built-in plugins use package-relative imports: import them as package members.
                        module = importlib.import_module(builtin_module)
                    else:
                        spec = importlib.util.spec_from_file_location(f"plugin_{plugin_file.stem}", plugin_file)
                        if not spec or not spec.loader:
                            continue
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                    # Find plugin classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, PluginInterface) and obj is not PluginInterface and not inspect.isabstract(obj):

                            # Instantiate and register plugin
                            plugin_instance = obj()
                            self.register(plugin_instance, replace_existing=True)
                            discovered += 1

                except Exception as e:
                    warnings.warn(f"Failed to load plugin from {plugin_file}: {e}", RuntimeWarning)

        return discovered

    def _is_builtin_path(self, path: Path) -> bool:
        """Return True when a resolved path is the built-in plugin root or inside it."""
        builtin_root = (Path(__file__).resolve().parent / "builtin").resolve()
        try:
            path.relative_to(builtin_root)
        except ValueError:
            return False
        return True

    def _resolve_trust_store_path(self) -> Optional[Path]:
        """Return the trust store that gates external plugins, if one is configured.

        The environment is read at discovery time: the global registry is often
        created at import time, before an operator configures the trust store.
        """
        if self._plugin_trust_store_path is not None:
            return self._plugin_trust_store_path
        env_trust_store = os.environ.get(_PLUGIN_TRUST_STORE_ENV)
        return Path(env_trust_store).expanduser().resolve() if env_trust_store else None

    def _discover_signed_external_plugins(self, path: Path, trust_store_path: Path) -> int:
        """Register external plugins that pass the PluginLoader signed-manifest gate.

        With a trust store configured, external code runs only after its
        plugin.json signature verifies; loose files and unsigned packages are
        never executed.
        """
        loader = PluginLoader(allow_external_plugins=True, plugin_trust_store_path=trust_store_path)

        discovered = 0
        for loaded in loader.discover_path(path):
            if loaded.is_valid and loaded.plugin is not None:
                self.register(loaded.plugin, replace_existing=True)
                discovered += 1
            else:
                warnings.warn(
                    f"Rejected plugin from {loaded.source_path}: {'; '.join(loaded.load_errors)}",
                    RuntimeWarning,
                )
        return discovered

    def get_metadata(self, plugin_type: str, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin.

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin

        Returns:
            Plugin metadata or None if not found
        """
        return self._metadata_cache.get(f"{plugin_type}:{plugin_name}")

    def _get_default_plugin_paths(self) -> List[Path]:
        """Get default plugin search paths.

        Returns:
            List of default plugin directories
        """
        paths = [
            Path(__file__).parent / "builtin",
        ]

        if not self._allow_external_plugins:
            return paths

        paths.append(Path.home() / ".transformation_portal" / "plugins")

        # Add environment variable path if set
        if "TRANSFORMATION_PORTAL_PLUGINS" in os.environ:
            paths.append(Path(os.environ["TRANSFORMATION_PORTAL_PLUGINS"]))

        return paths

    def reload_plugin(
        self,
        plugin_type: str,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Hot-reload a plugin (unregister, reload module, re-register).

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin
            config: New configuration for initialization

        Returns:
            True if reload successful, False otherwise
        """
        with self._lock:
            plugin = self._plugins.get(plugin_type, {}).get(plugin_name)
            if not plugin:
                return False

            # Clean up existing plugin
            plugin.cleanup()

            # Re-initialize with new config
            try:
                plugin.initialize(config)
                return True
            except Exception as e:
                warnings.warn(f"Failed to reload plugin '{plugin_name}': {e}")
                return False

    def clear(self) -> None:
        """Clear all registered plugins."""
        with self._lock:
            for plugin_type in self._plugins.values():
                for plugin in plugin_type.values():
                    plugin.cleanup()

            self._plugins.clear()
            self._metadata_cache.clear()

            # Re-initialize plugin type categories
            for plugin_type in PluginType:
                self._plugins[plugin_type.value] = {}


# Global registry instance
_global_registry: Optional[PluginRegistry] = None
_registry_lock = Lock()


def get_global_registry() -> PluginRegistry:
    """Get the global plugin registry singleton.

    Returns:
        Global PluginRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = PluginRegistry()

    return _global_registry
