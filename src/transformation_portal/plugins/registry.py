"""Plugin registry for managing and discovering plugins."""

import importlib
import importlib.util
import inspect
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from threading import Lock

from .interface import (
    PluginInterface,
    PluginType,
    PluginMetadata,
    PluginValidationError,
)


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
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, Dict[str, PluginInterface]] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._lock = Lock()
        self._plugin_paths: List[Path] = []
        
        # Initialize plugin type categories
        for plugin_type in PluginType:
            self._plugins[plugin_type.value] = {}
    
    def register(
        self,
        plugin: PluginInterface,
        replace_existing: bool = False
    ) -> None:
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
                    f"Use replace_existing=True to override."
                )
            
            # Validate plugin
            if not plugin.validate():
                warnings.warn(
                    f"Plugin '{plugin_name}' validation returned False. "
                    f"Plugin may not be properly initialized."
                )
            
            self._plugins[plugin_type][plugin_name] = plugin
            self._metadata_cache[f"{plugin_type}:{plugin_name}"] = plugin.metadata
            
            if plugin.metadata.deprecated:
                warnings.warn(
                    f"Plugin '{plugin_name}' is deprecated. "
                    f"Consider using '{plugin.metadata.replacement}' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
    
    def get_plugin(
        self,
        plugin_type: str,
        plugin_name: str,
        initialize: bool = False,
        config: Optional[Dict[str, Any]] = None
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
    
    def list_plugins(
        self,
        plugin_type: Optional[str] = None,
        include_deprecated: bool = False
    ) -> Dict[str, List[str]]:
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
        """
        if search_paths is None:
            search_paths = self._get_default_plugin_paths()
        
        discovered = 0
        
        for path in search_paths:
            path = Path(path).expanduser().resolve()
            if not path.exists():
                continue
            
            # Find all Python files in plugin directory
            for plugin_file in path.rglob('*.py'):
                if plugin_file.name.startswith('_'):
                    continue
                
                try:
                    # Load module dynamically
                    spec = importlib.util.spec_from_file_location(
                        f"plugin_{plugin_file.stem}",
                        plugin_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find plugin classes in module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, PluginInterface) and
                                obj is not PluginInterface and
                                not inspect.isabstract(obj)):
                                
                                # Instantiate and register plugin
                                plugin_instance = obj()
                                self.register(plugin_instance, replace_existing=True)
                                discovered += 1
                
                except Exception as e:
                    warnings.warn(
                        f"Failed to load plugin from {plugin_file}: {e}",
                        RuntimeWarning
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
            Path.home() / '.transformation_portal' / 'plugins',
            Path(__file__).parent / 'builtin',
        ]
        
        # Add environment variable path if set
        import os
        if 'TRANSFORMATION_PORTAL_PLUGINS' in os.environ:
            paths.append(Path(os.environ['TRANSFORMATION_PORTAL_PLUGINS']))
        
        return paths
    
    def reload_plugin(
        self,
        plugin_type: str,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None
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
