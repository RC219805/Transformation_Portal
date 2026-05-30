"""Plugin Architecture for Transformation Portal.

Provides extensible, hot-swappable components for depth models, processors,
and enhancement pipelines. Enables community plugins and future model support.

Components:
- PluginInterface: Base interface for all plugins
- PluginLoader: Dynamic plugin discovery and loading
- PluginManager: High-level lifecycle management
- PluginValidator: Dependency and compatibility validation
- PluginRegistry: Plugin registration and retrieval

Example:
    >>> from transformation_portal.plugins import PluginManager
    >>> manager = PluginManager(auto_discover=True)
    >>> result = manager.execute_by_type(PluginType.DEPTH_MODEL, image)

    >>> # Or use the registry directly
    >>> from transformation_portal.plugins import get_global_registry
    >>> registry = get_global_registry()
    >>> depth_model = registry.get_plugin('depth_model', 'edge_depth_estimator')
    >>> depth_model.initialize({'edge_threshold': 50})
    >>> depth_map = depth_model.execute(image)
"""

from .decorators import cached_execution, deprecated_plugin, measure_performance, plugin, requires_version
from .interface import (
    DepthModelPlugin,
    EnhancerPlugin,
    PluginExecutionError,
    PluginInitializationError,
    PluginInterface,
    PluginMetadata,
    PluginType,
    PluginValidationError,
    ProcessorPlugin,
)
from .loader import LoadedPlugin, PluginLoader, PluginManifest, get_global_loader
from .manager import ExecutionResult, PluginContext, PluginManager, PluginState, get_global_manager
from .registry import PluginRegistry, get_global_registry
from .signing import PluginSignatureError, sign_manifest, verify_manifest_signature
from .validator import PluginValidator, ValidationIssue, ValidationResult, ValidationSeverity, quick_validate, validate_plugin

__all__ = [
    # Core interfaces
    "PluginInterface",
    "PluginMetadata",
    "PluginType",
    "DepthModelPlugin",
    "ProcessorPlugin",
    "EnhancerPlugin",
    # Exceptions
    "PluginInitializationError",
    "PluginExecutionError",
    "PluginValidationError",
    # Loader
    "PluginLoader",
    "PluginManifest",
    "LoadedPlugin",
    "get_global_loader",
    "PluginSignatureError",
    "sign_manifest",
    "verify_manifest_signature",
    # Manager
    "PluginManager",
    "PluginState",
    "PluginContext",
    "ExecutionResult",
    "get_global_manager",
    # Registry
    "PluginRegistry",
    "get_global_registry",
    # Validator
    "PluginValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_plugin",
    "quick_validate",
    # Decorators
    "plugin",
    "requires_version",
    "deprecated_plugin",
    "cached_execution",
    "measure_performance",
]

__version__ = "1.0.0"
