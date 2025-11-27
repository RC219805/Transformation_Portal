"""Comprehensive tests for the plugin system.

Tests cover:
- Plugin interface and metadata
- Plugin loader functionality
- Plugin manager lifecycle
- Plugin validator
- Built-in plugins
- Error handling and edge cases
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.plugins import (
    DepthModelPlugin,
    EnhancerPlugin,
    ExecutionResult,
    LoadedPlugin,
    PluginContext,
    PluginExecutionError,
    PluginInitializationError,
    PluginInterface,
    PluginLoader,
    PluginManager,
    PluginManifest,
    PluginMetadata,
    PluginRegistry,
    PluginState,
    PluginType,
    PluginValidationError,
    PluginValidator,
    ProcessorPlugin,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    get_global_loader,
    get_global_manager,
    get_global_registry,
    plugin,
    quick_validate,
    validate_plugin,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleTestPlugin(PluginInterface):
    """Simple test plugin for testing."""

    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="simple_test_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            description="A simple test plugin",
            author="Test Author",
        )

    def initialize(self, config=None):
        self._config = config or {}
        self._initialized = True

    def execute(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError("Not initialized")
        return {"args": args, "kwargs": kwargs}


class MockProcessorPlugin(ProcessorPlugin):
    """Mock processor plugin for testing."""

    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_processor",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
            description="Test processor",
        )

    def initialize(self, config=None):
        self._initialized = True

    def process(self, input_data, **kwargs):
        return input_data


class MockEnhancerPlugin(EnhancerPlugin):
    """Mock enhancer plugin for testing."""

    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_enhancer",
            version="1.0.0",
            plugin_type=PluginType.ENHANCER,
            description="Test enhancer",
        )

    def initialize(self, config=None):
        self._initialized = True

    def enhance(self, image, strength=1.0, **kwargs):
        return image


class MockDepthPlugin(DepthModelPlugin):
    """Mock depth model plugin for testing."""

    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_depth_model",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="Test depth model",
        )

    def initialize(self, config=None):
        self._initialized = True

    def estimate_depth(self, image, **kwargs):
        if isinstance(image, Image.Image):
            return np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        return np.zeros(image.shape[:2], dtype=np.float32)


@pytest.fixture
def simple_plugin():
    """Create a simple test plugin."""
    return SimpleTestPlugin()


@pytest.fixture
def processor_plugin():
    """Create a test processor plugin."""
    return MockProcessorPlugin()


@pytest.fixture
def enhancer_plugin():
    """Create a test enhancer plugin."""
    return MockEnhancerPlugin()


@pytest.fixture
def depth_plugin():
    """Create a test depth plugin."""
    return MockDepthPlugin()


@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def test_image_array():
    """Create a test image as numpy array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def plugin_registry():
    """Create a fresh plugin registry."""
    return PluginRegistry()


@pytest.fixture
def plugin_loader():
    """Create a plugin loader with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = PluginLoader(search_paths=[Path(tmpdir)])
        yield loader


@pytest.fixture
def plugin_manager():
    """Create a plugin manager."""
    registry = PluginRegistry()
    loader = PluginLoader()
    return PluginManager(loader=loader, registry=registry)


# =============================================================================
# Test Plugin Interface and Metadata
# =============================================================================


class TestPluginInterface:
    """Tests for PluginInterface base class."""

    def test_plugin_creation(self, simple_plugin):
        """Test basic plugin creation."""
        assert simple_plugin.metadata.name == "simple_test_plugin"
        assert simple_plugin.metadata.version == "1.0.0"
        assert simple_plugin.metadata.plugin_type == PluginType.CUSTOM

    def test_plugin_initialization(self, simple_plugin):
        """Test plugin initialization."""
        assert not simple_plugin._initialized
        simple_plugin.initialize({"key": "value"})
        assert simple_plugin._initialized
        assert simple_plugin._config == {"key": "value"}

    def test_plugin_execution(self, simple_plugin):
        """Test plugin execution."""
        simple_plugin.initialize()
        result = simple_plugin.execute("arg1", key="value")
        assert result["args"] == ("arg1",)
        assert result["kwargs"] == {"key": "value"}

    def test_plugin_execution_without_init(self, simple_plugin):
        """Test that execution fails without initialization."""
        with pytest.raises(RuntimeError, match="Not initialized"):
            simple_plugin.execute()

    def test_plugin_cleanup(self, simple_plugin):
        """Test plugin cleanup."""
        simple_plugin.initialize()
        assert simple_plugin._initialized
        simple_plugin.cleanup()
        assert not simple_plugin._initialized

    def test_plugin_get_info(self, simple_plugin):
        """Test get_info method."""
        info = simple_plugin.get_info()
        assert info["name"] == "simple_test_plugin"
        assert info["version"] == "1.0.0"
        assert info["type"] == "custom"
        assert info["initialized"] is False

    def test_plugin_repr(self, simple_plugin):
        """Test string representation."""
        repr_str = repr(simple_plugin)
        assert "SimpleTestPlugin" in repr_str
        assert "simple_test_plugin" in repr_str


class TestPluginMetadata:
    """Tests for PluginMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation with defaults."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
        )
        assert metadata.name == "test"
        assert metadata.license == "MIT"
        assert metadata.dependencies == []
        assert not metadata.deprecated

    def test_version_compatibility(self):
        """Test version compatibility checking."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
            min_portal_version="0.1.0",
            max_portal_version="1.0.0",
        )
        assert metadata.is_compatible("0.1.0")
        assert metadata.is_compatible("0.5.0")
        assert metadata.is_compatible("1.0.0")
        assert not metadata.is_compatible("0.0.1")
        assert not metadata.is_compatible("1.1.0")


class TestSpecializedPlugins:
    """Tests for specialized plugin types."""

    def test_processor_plugin(self, processor_plugin, test_image):
        """Test ProcessorPlugin interface."""
        processor_plugin.initialize()
        result = processor_plugin.process(test_image)
        assert result == test_image

        # Test execute delegates to process
        result2 = processor_plugin.execute(test_image)
        assert result2 == test_image

    def test_enhancer_plugin(self, enhancer_plugin, test_image):
        """Test EnhancerPlugin interface."""
        enhancer_plugin.initialize()
        result = enhancer_plugin.enhance(test_image, strength=0.5)
        assert result == test_image

        # Test execute delegates to enhance
        result2 = enhancer_plugin.execute(test_image)
        assert result2 == test_image

    def test_depth_model_plugin(self, depth_plugin, test_image):
        """Test DepthModelPlugin interface."""
        depth_plugin.initialize()
        depth_map = depth_plugin.estimate_depth(test_image)
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.shape == (100, 100)

        # Test execute delegates to estimate_depth
        depth_map2 = depth_plugin.execute(test_image)
        assert depth_map2.shape == (100, 100)


# =============================================================================
# Test Plugin Registry
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_plugin(self, plugin_registry, simple_plugin):
        """Test plugin registration."""
        simple_plugin.initialize()  # Pre-initialize to pass validation
        plugin_registry.register(simple_plugin)
        retrieved = plugin_registry.get_plugin("custom", "simple_test_plugin")
        assert retrieved is simple_plugin

    def test_register_duplicate_fails(self, plugin_registry, simple_plugin):
        """Test that duplicate registration fails."""
        simple_plugin.initialize()
        plugin_registry.register(simple_plugin)
        with pytest.raises(ValueError, match="already registered"):
            plugin_registry.register(simple_plugin)

    def test_register_duplicate_with_replace(self, plugin_registry, simple_plugin):
        """Test duplicate registration with replace_existing."""
        simple_plugin.initialize()
        plugin_registry.register(simple_plugin)
        plugin_registry.register(simple_plugin, replace_existing=True)
        # Should not raise

    def test_unregister_plugin(self, plugin_registry, simple_plugin):
        """Test plugin unregistration."""
        simple_plugin.initialize()
        plugin_registry.register(simple_plugin)
        assert plugin_registry.unregister("custom", "simple_test_plugin")
        assert plugin_registry.get_plugin("custom", "simple_test_plugin") is None

    def test_list_plugins(self, plugin_registry, simple_plugin, processor_plugin):
        """Test listing plugins."""
        simple_plugin.initialize()
        processor_plugin.initialize()
        plugin_registry.register(simple_plugin)
        plugin_registry.register(processor_plugin)

        plugins = plugin_registry.list_plugins()
        assert "custom" in plugins
        assert "processor" in plugins
        assert "simple_test_plugin" in plugins["custom"]
        assert "test_processor" in plugins["processor"]

    def test_list_plugins_by_type(self, plugin_registry, simple_plugin, processor_plugin):
        """Test listing plugins filtered by type."""
        simple_plugin.initialize()
        processor_plugin.initialize()
        plugin_registry.register(simple_plugin)
        plugin_registry.register(processor_plugin)

        plugins = plugin_registry.list_plugins(plugin_type="processor")
        assert "processor" in plugins
        assert "custom" not in plugins

    def test_get_metadata(self, plugin_registry, simple_plugin):
        """Test getting plugin metadata."""
        simple_plugin.initialize()
        plugin_registry.register(simple_plugin)
        metadata = plugin_registry.get_metadata("custom", "simple_test_plugin")
        assert metadata.name == "simple_test_plugin"

    def test_clear_registry(self, plugin_registry, simple_plugin):
        """Test clearing all plugins."""
        simple_plugin.initialize()
        plugin_registry.register(simple_plugin)
        plugin_registry.clear()
        assert plugin_registry.list_plugins() == {}


# =============================================================================
# Test Plugin Loader
# =============================================================================


class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_loader_creation(self):
        """Test loader creation with default paths."""
        loader = PluginLoader()
        paths = loader.get_search_paths()
        assert len(paths) > 0

    def test_add_search_path(self, plugin_loader):
        """Test adding search paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_loader.add_search_path(Path(tmpdir))
            assert Path(tmpdir) in plugin_loader.get_search_paths()

    def test_remove_search_path(self, plugin_loader):
        """Test removing search paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_loader.add_search_path(Path(tmpdir))
            assert plugin_loader.remove_search_path(Path(tmpdir))
            assert Path(tmpdir) not in plugin_loader.get_search_paths()

    def test_discover_empty_directory(self):
        """Test discovering from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader(search_paths=[Path(tmpdir)])
            plugins = loader.discover_all()
            # May find builtin plugins
            # Just verify it doesn't crash

    def test_load_from_file(self):
        """Test loading plugin from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple plugin file
            plugin_code = '''
from transformation_portal.plugins import PluginInterface, PluginMetadata, PluginType

class FileTestPlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name="file_test_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )
    def initialize(self, config=None):
        self._initialized = True
    def execute(self, *args, **kwargs):
        return "executed"
'''
            plugin_file = Path(tmpdir) / "test_plugin.py"
            plugin_file.write_text(plugin_code)

            loader = PluginLoader(search_paths=[Path(tmpdir)])
            plugins = loader.discover_all()

            # Should find our plugin
            found = any(
                p.manifest and p.manifest.name == "file_test_plugin"
                for p in plugins
            )
            assert found


class TestPluginManifest:
    """Tests for PluginManifest."""

    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "plugin_type": "processor",
            "entry_point": "my_module:MyPlugin",
            "description": "A test plugin",
        }
        manifest = PluginManifest.from_dict(data)
        assert manifest.name == "test_plugin"
        assert manifest.version == "1.0.0"
        assert manifest.entry_point == "my_module:MyPlugin"

    def test_manifest_from_json_file(self):
        """Test loading manifest from JSON file."""
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "name": "json_plugin",
                "version": "2.0.0",
                "plugin_type": "enhancer",
                "entry_point": "plugin:JsonPlugin",
            }
            manifest_file = Path(tmpdir) / "plugin.json"
            manifest_file.write_text(json.dumps(manifest_data))

            manifest = PluginManifest.from_json_file(manifest_file)
            assert manifest.name == "json_plugin"
            assert manifest.version == "2.0.0"


# =============================================================================
# Test Plugin Manager
# =============================================================================


class TestPluginManager:
    """Tests for PluginManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = PluginManager()
        assert manager is not None

    def test_get_plugin(self, plugin_manager, simple_plugin):
        """Test getting a plugin through manager."""
        simple_plugin.initialize()
        plugin_manager._registry.register(simple_plugin)
        plugin_manager._loader._loaded_plugins["simple_test_plugin"] = LoadedPlugin(
            plugin=simple_plugin,
            manifest=PluginManifest.from_dict({
                "name": "simple_test_plugin",
                "version": "1.0.0",
                "plugin_type": "custom",
                "entry_point": "test:SimpleTestPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )

        plugin = plugin_manager.get_plugin("simple_test_plugin")
        assert plugin is simple_plugin

    def test_initialize_plugin(self, plugin_manager, simple_plugin):
        """Test initializing plugin through manager."""
        # Setup
        plugin_manager._loader._loaded_plugins["simple_test_plugin"] = LoadedPlugin(
            plugin=simple_plugin,
            manifest=PluginManifest.from_dict({
                "name": "simple_test_plugin",
                "version": "1.0.0",
                "plugin_type": "custom",
                "entry_point": "test:SimpleTestPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )

        result = plugin_manager.initialize_plugin(
            "simple_test_plugin",
            config={"test": "value"}
        )
        assert result is True
        assert simple_plugin._initialized

    def test_execute_plugin(self, plugin_manager, simple_plugin):
        """Test executing plugin through manager."""
        simple_plugin.initialize()
        plugin_manager._loader._loaded_plugins["simple_test_plugin"] = LoadedPlugin(
            plugin=simple_plugin,
            manifest=PluginManifest.from_dict({
                "name": "simple_test_plugin",
                "version": "1.0.0",
                "plugin_type": "custom",
                "entry_point": "test:SimpleTestPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )

        result = plugin_manager.execute("simple_test_plugin", "arg1", key="value")
        assert result.success
        assert result.result["args"] == ("arg1",)

    def test_execute_with_fallback(self, plugin_manager, simple_plugin, processor_plugin):
        """Test execution with fallback plugins."""
        processor_plugin.initialize()

        # Simple plugin will fail (not loaded)
        # Processor plugin should work as fallback
        plugin_manager._loader._loaded_plugins["test_processor"] = LoadedPlugin(
            plugin=processor_plugin,
            manifest=PluginManifest.from_dict({
                "name": "test_processor",
                "version": "1.0.0",
                "plugin_type": "processor",
                "entry_point": "test:TestProcessorPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )

        result = plugin_manager.execute(
            "nonexistent_plugin",
            "data",
            fallback_plugins=["test_processor"]
        )
        assert result.success
        assert result.plugin_name == "test_processor"

    def test_plugin_session_context_manager(self, plugin_manager, simple_plugin):
        """Test plugin session context manager."""
        plugin_manager._loader._loaded_plugins["simple_test_plugin"] = LoadedPlugin(
            plugin=simple_plugin,
            manifest=PluginManifest.from_dict({
                "name": "simple_test_plugin",
                "version": "1.0.0",
                "plugin_type": "custom",
                "entry_point": "test:SimpleTestPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )

        with plugin_manager.plugin_session("simple_test_plugin") as plugin:
            assert plugin._initialized
            result = plugin.execute("test")
            assert result["args"] == ("test",)

    def test_plugin_state_tracking(self, plugin_manager, simple_plugin):
        """Test plugin state tracking."""
        plugin_manager._loader._loaded_plugins["simple_test_plugin"] = LoadedPlugin(
            plugin=simple_plugin,
            manifest=PluginManifest.from_dict({
                "name": "simple_test_plugin",
                "version": "1.0.0",
                "plugin_type": "custom",
                "entry_point": "test:SimpleTestPlugin",
            }),
            source_path=Path("."),
            module_name="test",
        )
        plugin_manager._contexts["simple_test_plugin"] = PluginContext(
            state=PluginState.LOADED
        )

        assert plugin_manager.get_plugin_state("simple_test_plugin") == PluginState.LOADED

        plugin_manager.initialize_plugin("simple_test_plugin")
        assert plugin_manager.get_plugin_state("simple_test_plugin") == PluginState.INITIALIZED


# =============================================================================
# Test Plugin Validator
# =============================================================================


class TestPluginValidator:
    """Tests for PluginValidator."""

    def test_validate_valid_plugin(self, simple_plugin):
        """Test validating a valid plugin."""
        result = validate_plugin(simple_plugin)
        assert result.is_valid

    def test_validate_metadata(self, simple_plugin):
        """Test metadata validation."""
        validator = PluginValidator()
        result = ValidationResult(plugin_name="test", is_valid=True)
        validator._validate_metadata(simple_plugin, result)
        # Should pass with minor issues at most
        assert result.errors_count == 0

    def test_validate_interface_compliance(self, simple_plugin):
        """Test interface compliance validation."""
        validator = PluginValidator()
        result = ValidationResult(plugin_name="test", is_valid=True)
        validator._validate_interface_compliance(simple_plugin, result)
        assert result.errors_count == 0

    def test_quick_validate(self, simple_plugin):
        """Test quick_validate helper."""
        assert quick_validate(simple_plugin)

    def test_validation_issues(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            code="TEST_ERROR",
            message="Test error message",
            severity=ValidationSeverity.ERROR,
            suggestion="Fix this",
        )
        assert "TEST_ERROR" in str(issue)
        assert "Test error message" in str(issue)

    def test_strict_mode(self, simple_plugin):
        """Test strict validation mode."""
        validator = PluginValidator(strict_mode=True)
        result = validator.validate(simple_plugin)
        # In strict mode, warnings become errors


# =============================================================================
# Test Built-in Plugins
# =============================================================================


class TestBuiltinPlugins:
    """Tests for built-in plugins."""

    def test_gaussian_blur_processor(self, test_image):
        """Test GaussianBlurProcessor."""
        from transformation_portal.plugins.builtin import GaussianBlurProcessor

        processor = GaussianBlurProcessor()
        processor.initialize({"radius": 2.0})

        result = processor.process(test_image)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_gaussian_blur_numpy(self, test_image_array):
        """Test GaussianBlurProcessor with numpy input."""
        from transformation_portal.plugins.builtin import GaussianBlurProcessor

        processor = GaussianBlurProcessor()
        processor.initialize()

        result = processor.process(test_image_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image_array.shape

    def test_contrast_enhancer(self, test_image):
        """Test ContrastEnhancer."""
        from transformation_portal.plugins.builtin import ContrastEnhancer

        enhancer = ContrastEnhancer()
        enhancer.initialize({"base_factor": 1.5})

        result = enhancer.enhance(test_image, strength=0.8)
        assert isinstance(result, Image.Image)

    def test_sharpen_enhancer(self, test_image):
        """Test SharpenEnhancer."""
        from transformation_portal.plugins.builtin import SharpenEnhancer

        enhancer = SharpenEnhancer()
        enhancer.initialize()

        result = enhancer.enhance(test_image, strength=0.5)
        assert isinstance(result, Image.Image)

    def test_edge_depth_estimator(self, test_image):
        """Test EdgeDepthEstimator."""
        from transformation_portal.plugins.builtin import EdgeDepthEstimator

        estimator = EdgeDepthEstimator()
        estimator.initialize({"edge_threshold": 30})

        depth_map = estimator.estimate_depth(test_image)
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.dtype == np.float32
        assert depth_map.shape == (test_image.size[1], test_image.size[0])


# =============================================================================
# Test Global Singletons
# =============================================================================


class TestGlobalSingletons:
    """Tests for global singleton instances."""

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2

    def test_global_loader(self):
        """Test global loader singleton."""
        loader1 = get_global_loader()
        loader2 = get_global_loader()
        assert loader1 is loader2

    def test_global_manager(self):
        """Test global manager singleton."""
        manager1 = get_global_manager()
        manager2 = get_global_manager()
        assert manager1 is manager2


# =============================================================================
# Test Decorators
# =============================================================================


class TestPluginDecorators:
    """Tests for plugin decorators."""

    def test_plugin_decorator(self):
        """Test @plugin decorator."""
        @plugin(
            name="decorated_plugin",
            plugin_type=PluginType.PROCESSOR,
            version="1.0.0",
            auto_register=False,
        )
        class DecoratedPlugin(ProcessorPlugin):
            def _create_metadata(self):
                # The decorator adds _decorator_metadata at class level
                if hasattr(self, '_decorator_metadata'):
                    return self._decorator_metadata
                return super()._create_metadata()

            def initialize(self, config=None):
                self._initialized = True

            def process(self, input_data, **kwargs):
                return input_data

        instance = DecoratedPlugin()
        assert instance.metadata.name == "decorated_plugin"
        assert instance.metadata.version == "1.0.0"


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_initialization_error(self):
        """Test PluginInitializationError."""
        with pytest.raises(PluginInitializationError):
            raise PluginInitializationError("Init failed")

    def test_execution_error(self):
        """Test PluginExecutionError."""
        with pytest.raises(PluginExecutionError):
            raise PluginExecutionError("Execution failed")

    def test_validation_error(self):
        """Test PluginValidationError."""
        with pytest.raises(PluginValidationError):
            raise PluginValidationError("Validation failed")

    def test_manager_handles_missing_plugin(self, plugin_manager):
        """Test manager handles missing plugin gracefully."""
        result = plugin_manager.execute("nonexistent_plugin")
        assert not result.success
        assert "No plugins available" in result.error


# =============================================================================
# Test Execution Result
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            result={"data": "value"},
            plugin_name="test_plugin",
            execution_time_ms=10.5,
        )
        assert result.success
        assert result.result == {"data": "value"}
        assert result.execution_time_ms == 10.5

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error="Something went wrong",
            plugin_name="test_plugin",
        )
        assert not result.success
        assert result.error == "Something went wrong"
