"""Decorators for plugin development."""

import functools
import warnings
from typing import Callable, Optional, List
from packaging import version as pkg_version

from .interface import PluginInterface, PluginMetadata, PluginType
from .registry import get_global_registry


def plugin(
    name: str,
    plugin_type: PluginType,
    version: str = "1.0.0",
    description: str = "",
    auto_register: bool = True,
    **metadata_kwargs
):
    """Decorator to mark a class as a plugin and automatically register it.
    
    Args:
        name: Plugin name
        plugin_type: Type of plugin
        version: Plugin version
        description: Plugin description
        auto_register: Automatically register with global registry
        **metadata_kwargs: Additional metadata fields
        
    Example:
        >>> @plugin(
        ...     name="my_depth_model",
        ...     plugin_type=PluginType.DEPTH_MODEL,
        ...     version="1.0.0",
        ...     description="Custom depth model"
        ... )
        ... class MyDepthModel(DepthModelPlugin):
        ...     def _create_metadata(self):
        ...         return self._decorator_metadata
        ...     
        ...     def initialize(self, config=None):
        ...         self._initialized = True
        ...     
        ...     def estimate_depth(self, image):
        ...         return process_image(image)
    """
    def decorator(cls):
        # Store metadata for the plugin class
        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type=plugin_type,
            description=description,
            **metadata_kwargs
        )
        
        # Add metadata as class attribute
        cls._decorator_metadata = metadata
        
        # Auto-register if enabled
        if auto_register:
            try:
                instance = cls()
                get_global_registry().register(instance, replace_existing=True)
            except Exception as e:
                warnings.warn(
                    f"Failed to auto-register plugin {name}: {e}",
                    RuntimeWarning
                )
        
        return cls
    
    return decorator


def requires_version(
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
):
    """Decorator to enforce Transformation Portal version requirements.
    
    Args:
        min_version: Minimum required portal version
        max_version: Maximum supported portal version
        
    Example:
        >>> @requires_version(min_version="0.1.0", max_version="0.2.0")
        ... class MyPlugin(PluginInterface):
        ...     pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Check version compatibility
            from transformation_portal import __version__ as portal_version
            
            if min_version:
                if pkg_version.parse(portal_version) < pkg_version.parse(min_version):
                    raise RuntimeError(
                        f"Plugin requires Transformation Portal >= {min_version}, "
                        f"but current version is {portal_version}"
                    )
            
            if max_version:
                if pkg_version.parse(portal_version) > pkg_version.parse(max_version):
                    warnings.warn(
                        f"Plugin may not be compatible with Transformation Portal > {max_version}. "
                        f"Current version is {portal_version}.",
                        RuntimeWarning
                    )
            
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def deprecated_plugin(
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
    message: Optional[str] = None
):
    """Decorator to mark a plugin as deprecated.
    
    Args:
        replacement: Name of replacement plugin
        removal_version: Version when plugin will be removed
        message: Custom deprecation message
        
    Example:
        >>> @deprecated_plugin(
        ...     replacement="new_depth_model",
        ...     removal_version="2.0.0"
        ... )
        ... class OldDepthModel(DepthModelPlugin):
        ...     pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Show deprecation warning
            warning_msg = message or (
                f"Plugin {cls.__name__} is deprecated"
            )
            
            if replacement:
                warning_msg += f" and will be replaced by '{replacement}'"
            
            if removal_version:
                warning_msg += f". It will be removed in version {removal_version}"
            
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            
            original_init(self, *args, **kwargs)
            
            # Mark metadata as deprecated
            if hasattr(self, 'metadata'):
                self.metadata.deprecated = True
                self.metadata.replacement = replacement
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def cached_execution(maxsize: int = 128):
    """Decorator to cache plugin execution results (LRU cache).
    
    Args:
        maxsize: Maximum cache size
        
    Example:
        >>> class MyPlugin(PluginInterface):
        ...     @cached_execution(maxsize=256)
        ...     def execute(self, image_hash):
        ...         return expensive_computation(image_hash)
    """
    def decorator(func: Callable) -> Callable:
        return functools.lru_cache(maxsize=maxsize)(func)
    
    return decorator


def measure_performance(func: Callable) -> Callable:
    """Decorator to measure and log plugin execution performance.
    
    Example:
        >>> class MyPlugin(PluginInterface):
        ...     @measure_performance
        ...     def execute(self, image):
        ...         return process(image)
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        import time
        start = time.perf_counter()
        
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if hasattr(self, 'metadata'):
                plugin_name = self.metadata.name
            else:
                plugin_name = self.__class__.__name__
            
            # Log performance (can be extended to send to monitoring)
            print(f"[Performance] {plugin_name}.{func.__name__}: {elapsed*1000:.2f}ms")
            
            return result
        
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"[Performance] {func.__name__} failed after {elapsed*1000:.2f}ms: {e}")
            raise
    
    return wrapper
