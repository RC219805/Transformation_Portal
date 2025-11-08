"""Plugin Architecture for Transformation Portal.

Provides extensible, hot-swappable components for depth models, processors,
and enhancement pipelines. Enables community plugins and future model support.

Example:
    >>> from transformation_portal.plugins import PluginRegistry, PluginInterface
    >>> registry = PluginRegistry()
    >>> registry.discover_plugins()
    >>> depth_model = registry.get_plugin('depth', 'depth_anything_v2')
"""

from .decorators import deprecated_plugin, plugin, requires_version
from .interface import PluginInterface, PluginMetadata, PluginType
from .registry import PluginRegistry, get_global_registry

__all__ = [
    'PluginInterface',
    'PluginMetadata',
    'PluginType',
    'PluginRegistry',
    'get_global_registry',
    'plugin',
    'requires_version',
    'deprecated_plugin',
]

__version__ = '1.0.0'
