"""Plugin loader for dynamic discovery and loading of plugins."""

import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .interface import PluginInterface, PluginMetadata, PluginType
from .signing import PluginSignatureError, verify_manifest_signature

logger = logging.getLogger(__name__)

_ENABLE_EXTERNAL_PLUGINS_ENV = "TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS"
_PLUGIN_TRUST_STORE_ENV = "TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _external_plugins_enabled_from_env() -> bool:
    """Return True when external plugin loading is explicitly enabled."""
    value = os.environ.get(_ENABLE_EXTERNAL_PLUGINS_ENV, "")
    return value.strip().lower() in _TRUTHY_VALUES


@dataclass
class PluginManifest:
    """Manifest describing a plugin package.

    Loaded from plugin.json or pyproject.toml [tool.transformation_portal.plugin]
    """

    name: str
    version: str
    plugin_type: str
    entry_point: str  # Module path to plugin class, e.g., "my_plugin:MyPlugin"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    min_portal_version: str = "0.1.0"
    max_portal_version: Optional[str] = None
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None
    signature_key_id: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            plugin_type=data.get("plugin_type", "custom"),
            entry_point=data.get("entry_point", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            dependencies=data.get("dependencies", []),
            min_portal_version=data.get("min_portal_version", "0.1.0"),
            max_portal_version=data.get("max_portal_version"),
            homepage=data.get("homepage", ""),
            tags=data.get("tags", []),
            config_schema=data.get("config_schema", {}),
            signature=data.get("signature"),
            signature_algorithm=data.get("signature_algorithm"),
            signature_key_id=data.get("signature_key_id"),
            raw_data=dict(data),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "PluginManifest":
        """Load manifest from plugin.json file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_pyproject(cls, path: Path) -> Optional["PluginManifest"]:
        """Load manifest from pyproject.toml [tool.transformation_portal.plugin]."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                logger.warning("tomllib/tomli not available for pyproject.toml parsing")
                return None

        with open(path, "rb") as f:
            data = tomllib.load(f)

        plugin_data = data.get("tool", {}).get("transformation_portal", {}).get("plugin")
        if plugin_data:
            return cls.from_dict(plugin_data)
        return None


@dataclass
class LoadedPlugin:
    """Container for a loaded plugin with its metadata and state."""

    plugin: PluginInterface
    manifest: Optional[PluginManifest]
    source_path: Path
    module_name: str
    load_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if plugin loaded without errors."""
        return len(self.load_errors) == 0


class PluginLoader:
    """Advanced plugin loader with manifest support and dependency resolution.

    Features:
    - Directory scanning for plugins
    - Manifest-based plugin packages (plugin.json / pyproject.toml)
    - Dependency resolution and validation
    - Safe module loading with isolation
    - Hot-reload support
    - External plugin search paths are opt-in (secure-by-default)

    Example:
        >>> loader = PluginLoader(allow_external_plugins=True)
        >>> loader.add_search_path("~/.transformation_portal/plugins")
        >>> plugins = loader.discover_all()
        >>> for plugin in plugins:
        ...     print(f"Found: {plugin.manifest.name} v{plugin.manifest.version}")
    """

    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        auto_resolve_dependencies: bool = True,
        allow_external_plugins: Optional[bool] = None,
        plugin_trust_store_path: Optional[Path] = None,
    ):
        """Initialize plugin loader.

        Args:
            search_paths: Initial paths to search for plugins
            auto_resolve_dependencies: Automatically check dependencies on load
            allow_external_plugins: Enable user/env plugin discovery paths.
                If None, reads TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS.
            plugin_trust_store_path: Optional JSON trust store for verifying
                external plugin.json manifests before importing plugin code.
        """
        self._search_paths: List[Path] = []
        self._loaded_plugins: Dict[str, LoadedPlugin] = {}
        self._module_cache: Dict[str, Any] = {}
        self._lock = Lock()
        self._auto_resolve_deps = auto_resolve_dependencies
        self._allow_external_plugins = (
            allow_external_plugins if allow_external_plugins is not None else _external_plugins_enabled_from_env()
        )
        env_trust_store = os.environ.get(_PLUGIN_TRUST_STORE_ENV)
        self._plugin_trust_store_path = (
            Path(plugin_trust_store_path).expanduser().resolve()
            if plugin_trust_store_path is not None
            else Path(env_trust_store).expanduser().resolve() if env_trust_store else None
        )

        # Add default paths
        self._add_default_paths()

        # Add custom paths
        if search_paths:
            for path in search_paths:
                self.add_search_path(path)

    def _add_default_paths(self) -> None:
        """Add default plugin search paths."""
        # Builtin plugins directory
        builtin_plugins = (Path(__file__).resolve().parent / "builtin").resolve()
        self._search_paths.append(builtin_plugins)

        if not self._allow_external_plugins:
            env_path = os.environ.get("TRANSFORMATION_PORTAL_PLUGINS")
            if env_path:
                logger.warning(
                    "Ignoring TRANSFORMATION_PORTAL_PLUGINS because external plugin loading is disabled. "
                    "Set %s=1 to opt in.",
                    _ENABLE_EXTERNAL_PLUGINS_ENV,
                )
            return

        # User plugins directory
        user_plugins = Path.home() / ".transformation_portal" / "plugins"
        self._search_paths.append(user_plugins)

        # Environment variable path
        env_path = os.environ.get("TRANSFORMATION_PORTAL_PLUGINS")
        if env_path:
            self._search_paths.append(Path(env_path))

    def add_search_path(self, path: Path) -> None:
        """Add a path to search for plugins.

        Args:
            path: Directory path to add
        """
        path = Path(path).expanduser().resolve()
        if not self._allow_external_plugins:
            builtin_plugins = (Path(__file__).resolve().parent / "builtin").resolve()
            if path != builtin_plugins:
                raise ValueError(
                    "External plugin paths are disabled. "
                    "Set TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS=1 "
                    "or pass allow_external_plugins=True to PluginLoader."
                )
        if path not in self._search_paths:
            self._search_paths.append(path)

    def remove_search_path(self, path: Path) -> bool:
        """Remove a path from search paths.

        Args:
            path: Directory path to remove

        Returns:
            True if path was removed, False if not found
        """
        path = Path(path).expanduser().resolve()
        if path in self._search_paths:
            self._search_paths.remove(path)
            return True
        return False

    def get_search_paths(self) -> List[Path]:
        """Get list of current search paths."""
        return self._search_paths.copy()

    def discover_all(self) -> List[LoadedPlugin]:
        """Discover and load all plugins from search paths.

        Returns:
            List of loaded plugins
        """
        discovered: List[LoadedPlugin] = []

        for search_path in self._search_paths:
            if not search_path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue

            # Discover package-based plugins (with manifest)
            package_plugins = self._discover_package_plugins(search_path)
            discovered.extend(package_plugins)

            # Discover single-file plugins
            file_plugins = self._discover_file_plugins(search_path)
            discovered.extend(file_plugins)

        return discovered

    def _discover_package_plugins(self, search_path: Path) -> List[LoadedPlugin]:
        """Discover plugin packages with manifests.

        Args:
            search_path: Directory to search

        Returns:
            List of loaded plugin packages
        """
        discovered: List[LoadedPlugin] = []

        for item in search_path.iterdir():
            if not item.is_dir():
                continue

            # Skip hidden directories and __pycache__
            if item.name.startswith((".", "_")):
                continue

            # Look for manifest files
            manifest = None
            manifest_path = None

            # Check plugin.json first
            plugin_json = item / "plugin.json"
            if plugin_json.exists():
                try:
                    manifest = PluginManifest.from_json_file(plugin_json)
                    manifest_path = plugin_json  # noqa: F841
                except Exception as e:
                    logger.warning(f"Failed to parse {plugin_json}: {e}")

            # Check pyproject.toml as fallback
            if manifest is None:
                pyproject = item / "pyproject.toml"
                if pyproject.exists():
                    try:
                        manifest = PluginManifest.from_pyproject(pyproject)
                        _manifest_path = pyproject  # noqa: F841
                    except Exception as e:
                        logger.warning(f"Failed to parse {pyproject}: {e}")

            if manifest and manifest.entry_point:
                signature_error = self._verify_manifest_trust(item, manifest)
                if signature_error:
                    discovered.append(
                        LoadedPlugin(
                            plugin=None,  # type: ignore
                            manifest=manifest,
                            source_path=item,
                            module_name="",
                            load_errors=[signature_error],
                        )
                    )
                    continue

                # Load the plugin from manifest
                loaded = self._load_from_manifest(item, manifest)
                if loaded:
                    discovered.append(loaded)

        return discovered

    def _discover_file_plugins(self, search_path: Path) -> List[LoadedPlugin]:
        """Discover single-file plugins.

        Args:
            search_path: Directory to search

        Returns:
            List of loaded single-file plugins
        """
        discovered: List[LoadedPlugin] = []

        # Find all .py files (non-recursive for file plugins)
        for py_file in search_path.glob("*.py"):
            if self._requires_manifest_signature(py_file):
                logger.warning(
                    "Skipping external single-file plugin %s because %s requires signed plugin.json manifests",
                    py_file,
                    _PLUGIN_TRUST_STORE_ENV,
                )
                continue

            # Skip private/special files
            if py_file.name.startswith("_"):
                continue

            loaded = self._load_from_file(py_file)
            discovered.extend(loaded)

        return discovered

    def _builtin_plugins_root(self) -> Path:
        """Return the resolved built-in plugin package root."""
        return (Path(__file__).resolve().parent / "builtin").resolve()

    def _is_builtin_path(self, path: Path) -> bool:
        """Return True when a plugin path is under the built-in plugin root."""
        try:
            Path(path).resolve().relative_to(self._builtin_plugins_root())
        except ValueError:
            return False
        return True

    def _requires_manifest_signature(self, path: Path) -> bool:
        """Return True when an external plugin must pass signed-manifest trust."""
        return self._plugin_trust_store_path is not None and not self._is_builtin_path(path)

    def _verify_manifest_trust(self, package_dir: Path, manifest: PluginManifest) -> Optional[str]:
        """Validate external package manifest trust before importing code."""
        if not self._requires_manifest_signature(package_dir):
            return None
        if not manifest.raw_data:
            return "External plugin packages require a signed plugin.json manifest when plugin trust is configured"
        try:
            verify_manifest_signature(
                manifest.raw_data,
                trust_store_path=self._plugin_trust_store_path,  # type: ignore[arg-type]
            )
        except (OSError, PluginSignatureError, json.JSONDecodeError) as exc:
            return f"Plugin manifest signature verification failed: {exc}"
        return None

    def _load_from_manifest(self, package_dir: Path, manifest: PluginManifest) -> Optional[LoadedPlugin]:
        """Load a plugin from its manifest.

        Args:
            package_dir: Plugin package directory
            manifest: Plugin manifest

        Returns:
            Loaded plugin or None if loading failed
        """
        errors: List[str] = []

        # Validate dependencies if enabled
        if self._auto_resolve_deps:
            dep_errors = self._check_dependencies(manifest.dependencies)
            errors.extend(dep_errors)

        # Check portal version compatibility
        from transformation_portal import __version__ as portal_version

        metadata = PluginMetadata(
            name=manifest.name,
            version=manifest.version,
            plugin_type=PluginType(manifest.plugin_type),
            min_portal_version=manifest.min_portal_version,
            max_portal_version=manifest.max_portal_version,
        )

        if not metadata.is_compatible(portal_version):
            errors.append(f"Plugin {manifest.name} is not compatible with portal version {portal_version}")

        # Parse entry point (format: "module:ClassName" or "module.submodule:ClassName")
        try:
            module_path, class_name = manifest.entry_point.split(":")
        except ValueError:
            errors.append(f"Invalid entry_point format: {manifest.entry_point}")
            return LoadedPlugin(
                plugin=None,  # type: ignore
                manifest=manifest,
                source_path=package_dir,
                module_name="",
                load_errors=errors,
            )

        # Add package directory to sys.path temporarily
        sys_path_modified = False
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
            sys_path_modified = True

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the plugin class
            if not hasattr(module, class_name):
                errors.append(f"Class {class_name} not found in module {module_path}")
                return LoadedPlugin(
                    plugin=None,  # type: ignore
                    manifest=manifest,
                    source_path=package_dir,
                    module_name=module_path,
                    load_errors=errors,
                )

            plugin_class = getattr(module, class_name)

            # Validate it's a proper plugin
            if not issubclass(plugin_class, PluginInterface):
                errors.append(f"{class_name} does not inherit from PluginInterface")
                return LoadedPlugin(
                    plugin=None,  # type: ignore
                    manifest=manifest,
                    source_path=package_dir,
                    module_name=module_path,
                    load_errors=errors,
                )

            # Instantiate the plugin
            plugin_instance = plugin_class()

            with self._lock:
                self._loaded_plugins[manifest.name] = LoadedPlugin(
                    plugin=plugin_instance,
                    manifest=manifest,
                    source_path=package_dir,
                    module_name=module_path,
                    load_errors=errors,
                )
                self._module_cache[module_path] = module

            logger.info(f"Loaded plugin: {manifest.name} v{manifest.version}")
            return self._loaded_plugins[manifest.name]

        except Exception as e:
            errors.append(f"Failed to load plugin: {e}")
            logger.error(f"Failed to load plugin from {package_dir}: {e}")
            return LoadedPlugin(
                plugin=None,  # type: ignore
                manifest=manifest,
                source_path=package_dir,
                module_name=module_path if "module_path" in dir() else "",
                load_errors=errors,
            )

        finally:
            # Clean up sys.path if we modified it
            if sys_path_modified and str(package_dir) in sys.path:
                sys.path.remove(str(package_dir))

    def _load_from_file(self, file_path: Path) -> List[LoadedPlugin]:
        """Load plugins from a single Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of loaded plugins found in file
        """
        loaded: List[LoadedPlugin] = []
        module_name = f"plugin_{file_path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not create spec for {file_path}")
                return loaded

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find all plugin classes in module
            import inspect

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginInterface) and obj is not PluginInterface and not inspect.isabstract(obj):

                    try:
                        plugin_instance = obj()

                        # Create a basic manifest from metadata
                        manifest = PluginManifest(
                            name=plugin_instance.metadata.name,
                            version=plugin_instance.metadata.version,
                            plugin_type=plugin_instance.metadata.plugin_type.value,
                            entry_point=f"{module_name}:{name}",
                            description=plugin_instance.metadata.description,
                            author=plugin_instance.metadata.author,
                        )

                        loaded_plugin = LoadedPlugin(
                            plugin=plugin_instance,
                            manifest=manifest,
                            source_path=file_path,
                            module_name=module_name,
                        )

                        with self._lock:
                            self._loaded_plugins[manifest.name] = loaded_plugin
                            self._module_cache[module_name] = module

                        loaded.append(loaded_plugin)
                        logger.info(f"Loaded plugin from file: {manifest.name}")

                    except Exception as e:
                        logger.warning(f"Failed to instantiate {name} from {file_path}: {e}")

        except Exception as e:
            logger.error(f"Failed to load plugins from {file_path}: {e}")

        return loaded

    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if plugin dependencies are satisfied.

        Args:
            dependencies: List of dependency strings (pip format)

        Returns:
            List of error messages for missing dependencies
        """
        errors: List[str] = []

        for dep in dependencies:
            # Parse dependency string (e.g., "numpy>=1.20.0")
            import re

            match = re.match(r"^([a-zA-Z0-9_-]+)(.*)$", dep)
            if not match:
                continue

            package_name = match.group(1)
            _version_spec = match.group(2)  # noqa: F841

            try:
                # Try to import the package
                importlib.import_module(package_name.replace("-", "_"))
            except ImportError:
                errors.append(f"Missing dependency: {dep}")

        return errors

    def load_plugin(self, name: str) -> Optional[LoadedPlugin]:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            Loaded plugin or None if not found
        """
        with self._lock:
            return self._loaded_plugins.get(name)

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin and clean up resources.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unloaded, False if not found
        """
        with self._lock:
            if name not in self._loaded_plugins:
                return False

            loaded = self._loaded_plugins[name]

            # Cleanup plugin
            if loaded.plugin:
                try:
                    loaded.plugin.cleanup()
                except Exception as e:
                    logger.warning(f"Error during plugin cleanup: {e}")

            # Remove from cache
            if loaded.module_name in self._module_cache:
                del self._module_cache[loaded.module_name]

            # Remove from sys.modules
            if loaded.module_name in sys.modules:
                del sys.modules[loaded.module_name]

            del self._loaded_plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True

    def reload_plugin(self, name: str) -> Optional[LoadedPlugin]:
        """Hot-reload a plugin.

        Args:
            name: Plugin name

        Returns:
            Reloaded plugin or None if reload failed
        """
        with self._lock:
            if name not in self._loaded_plugins:
                logger.warning(f"Plugin {name} not found for reload")
                return None

            loaded = self._loaded_plugins[name]
            source_path = loaded.source_path
            manifest = loaded.manifest

        # Unload first
        self.unload_plugin(name)

        # Reload based on source type
        if source_path.is_dir() and manifest:
            return self._load_from_manifest(source_path, manifest)
        elif source_path.is_file():
            plugins = self._load_from_file(source_path)
            for p in plugins:
                if p.manifest and p.manifest.name == name:
                    return p

        return None

    def get_loaded_plugins(self) -> Dict[str, LoadedPlugin]:
        """Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to loaded plugins
        """
        with self._lock:
            return self._loaded_plugins.copy()

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[LoadedPlugin]:
        """Get all loaded plugins of a specific type.

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of loaded plugins matching the type
        """
        result: List[LoadedPlugin] = []

        with self._lock:
            for loaded in self._loaded_plugins.values():
                if loaded.manifest and loaded.manifest.plugin_type == plugin_type.value:
                    result.append(loaded)
                elif loaded.plugin and loaded.plugin.metadata.plugin_type == plugin_type:
                    result.append(loaded)

        return result


# Global loader instance
_global_loader: Optional[PluginLoader] = None
_loader_lock = Lock()


def get_global_loader() -> PluginLoader:
    """Get the global plugin loader singleton.

    Returns:
        Global PluginLoader instance
    """
    global _global_loader

    if _global_loader is None:
        with _loader_lock:
            if _global_loader is None:
                _global_loader = PluginLoader()

    return _global_loader
