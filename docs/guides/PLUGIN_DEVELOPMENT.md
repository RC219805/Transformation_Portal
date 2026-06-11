# Plugin Development Guide

**Transformation Portal - Building Custom Plugins**

> Historical guide.
>
> This November 2025 plugin guide is not currently listed as maintained
> operator guidance. Check [Documentation Map](../governance/DOCUMENTATION_MAP.md)
> before using it as an implementation contract.

Version: 1.0.0
Last Updated: 2025-11-08

---

## Overview

The Transformation Portal plugin architecture enables extensibility through hot-swappable components. Build custom depth models, processors, enhancers, and more without modifying core code.

## Quick Start

### Basic Plugin Example

```python
from transformation_portal.plugins import (
    DepthModelPlugin,
    PluginMetadata,
    PluginType,
    plugin
)

@plugin(
    name="my_depth_model",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="Custom depth estimation model"
)
class MyDepthModel(DepthModelPlugin):
    """Custom depth model plugin."""

    def _create_metadata(self):
        return self._decorator_metadata

    def initialize(self, config=None):
        """Initialize model."""
        self._config = config or {}

        # Load your model
        self.model = self._load_model()

        self._initialized = True

    def estimate_depth(self, image, **kwargs):
        """Estimate depth from image."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        # Your depth estimation logic
        depth_map = self.model.predict(image)

        return depth_map

    def _load_model(self):
        """Load depth model."""
        # Your model loading logic
        pass
```

### Using Your Plugin

```python
from transformation_portal.plugins import get_global_registry

# Get plugin registry
registry = get_global_registry()

# Get your plugin
depth_model = registry.get_plugin(
    'depth_model',
    'my_depth_model',
    initialize=True,
    config={'device': 'cuda'}
)

# Use plugin
depth_map = depth_model.estimate_depth(image)
```

## Plugin Types

### 1. Depth Model Plugins

Implement custom depth estimation models:

```python
from transformation_portal.plugins import DepthModelPlugin

class CustomDepthModel(DepthModelPlugin):
    def _create_metadata(self):
        return PluginMetadata(
            name="custom_depth",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="My custom depth model"
        )

    def initialize(self, config=None):
        self.model = load_custom_model(config)
        self._initialized = True

    def estimate_depth(self, image, **kwargs):
        return self.model.predict(image)
```

### 2. Processor Plugins

Implement custom image/video processors:

```python
from transformation_portal.plugins import ProcessorPlugin

class CustomProcessor(ProcessorPlugin):
    def _create_metadata(self):
        return PluginMetadata(
            name="custom_processor",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
        )

    def initialize(self, config=None):
        self._config = config or {}
        self._initialized = True

    def process(self, input_data, **kwargs):
        """Process image/video."""
        # Your processing logic
        return processed_data
```

### 3. Enhancer Plugins

Implement custom enhancement algorithms:

```python
from transformation_portal.plugins import EnhancerPlugin

class CustomEnhancer(EnhancerPlugin):
    def _create_metadata(self):
        return PluginMetadata(
            name="custom_enhancer",
            version="1.0.0",
            plugin_type=PluginType.ENHANCER,
        )

    def initialize(self, config=None):
        self._initialized = True

    def enhance(self, image, strength=1.0, **kwargs):
        """Enhance image."""
        # Your enhancement logic
        return enhanced_image
```

### 4. Custom Plugins

Implement any custom functionality:

```python
from transformation_portal.plugins import PluginInterface

class CustomPlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name="custom_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
        )

    def initialize(self, config=None):
        self._initialized = True

    def execute(self, *args, **kwargs):
        """Execute custom logic."""
        # Your custom logic
        return result
```

## Plugin Discovery

### Automatic Discovery

Place plugins in standard locations:

```bash
~/.transformation_portal/plugins/
  my_depth_model/
    __init__.py          # Plugin implementation
    model_weights.pth    # Model files
  my_processor/
    __init__.py
```

Discover plugins:

```python
from transformation_portal.plugins import get_global_registry

registry = get_global_registry()
discovered = registry.discover_plugins()
print(f"Discovered {discovered} plugins")
```

### Manual Registration

Register plugins programmatically:

```python
from transformation_portal.plugins import get_global_registry

plugin_instance = MyDepthModel()
registry = get_global_registry()
registry.register(plugin_instance)
```

### Environment Variable

Set custom plugin path:

```bash
export TRANSFORMATION_PORTAL_PLUGINS=/path/to/plugins
```

## Plugin Decorators

### @plugin - Auto-registration

```python
@plugin(
    name="my_plugin",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="My plugin",
    author="Your Name",
    license="MIT"
)
class MyPlugin(DepthModelPlugin):
    pass
```

### @requires_version - Version enforcement

```python
from transformation_portal.plugins import requires_version

@requires_version(min_version="0.1.0", max_version="1.0.0")
class MyPlugin(PluginInterface):
    pass
```

### @deprecated_plugin - Mark as deprecated

```python
from transformation_portal.plugins import deprecated_plugin

@deprecated_plugin(
    replacement="new_plugin",
    removal_version="2.0.0"
)
class OldPlugin(PluginInterface):
    pass
```

### @cached_execution - Cache results

```python
from transformation_portal.plugins import cached_execution

class MyPlugin(PluginInterface):
    @cached_execution(maxsize=256)
    def execute(self, image_hash):
        # Expensive computation cached by image_hash
        return result
```

### @measure_performance - Track performance

```python
from transformation_portal.plugins import measure_performance

class MyPlugin(PluginInterface):
    @measure_performance
    def execute(self, image):
        # Automatically logs execution time
        return process(image)
```

## Advanced Features

### Hot-Reloading

Reload plugins without restarting:

```python
registry = get_global_registry()

# Reload plugin with new config
success = registry.reload_plugin(
    'depth_model',
    'my_depth_model',
    config={'device': 'cpu'}
)
```

### Plugin Metadata

Access plugin information:

```python
metadata = registry.get_metadata('depth_model', 'my_depth_model')

print(f"Name: {metadata.name}")
print(f"Version: {metadata.version}")
print(f"Author: {metadata.author}")
print(f"Dependencies: {metadata.dependencies}")
```

### List Available Plugins

```python
# List all plugins
all_plugins = registry.list_plugins()

# List by type
depth_models = registry.list_plugins(plugin_type='depth_model')

# Include deprecated
all_including_deprecated = registry.list_plugins(include_deprecated=True)
```

## Plugin Packaging

### Directory Structure

```
my_plugin/
  __init__.py           # Plugin implementation
  README.md             # Documentation
  requirements.txt      # Dependencies
  tests/                # Plugin tests
    test_plugin.py
  examples/             # Usage examples
    example.py
```

### Plugin Metadata File (Optional)

`plugin.json`:

```json
{
  "name": "my_depth_model",
  "version": "1.0.0",
  "description": "Custom depth estimation model",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/yourusername/my_plugin",
  "dependencies": [
    "torch>=2.12.0",
    "torchvision>=0.27.0"
  ],
  "min_portal_version": "0.1.0"
}
```

### Distribution

**PyPI Package**:

```bash
# Install from PyPI
pip install transformation-portal-my-plugin

# Plugin auto-registers on import
```

**Git Repository**:

```bash
# Install from git
pip install git+https://github.com/yourusername/my_plugin.git

# Or clone and install
git clone https://github.com/yourusername/my_plugin.git
cd my_plugin
pip install -e .
```

## Testing Plugins

### Unit Tests

```python
import pytest
from transformation_portal.plugins import get_global_registry

def test_plugin_registration():
    registry = get_global_registry()
    plugin = registry.get_plugin('depth_model', 'my_depth_model')
    assert plugin is not None

def test_plugin_execution():
    registry = get_global_registry()
    plugin = registry.get_plugin(
        'depth_model',
        'my_depth_model',
        initialize=True
    )

    result = plugin.estimate_depth(test_image)
    assert result.shape == expected_shape
```

### Integration Tests

```python
def test_plugin_in_pipeline():
    from transformation_portal.depth import ArchitecturalDepthPipeline

    # Use custom plugin in pipeline
    pipeline = ArchitecturalDepthPipeline(
        depth_model='my_depth_model'
    )

    result = pipeline.process(image)
    assert result is not None
```

## Best Practices

1. **Initialize properly**: Always implement `initialize()` and set `self._initialized = True`
2. **Handle errors gracefully**: Catch exceptions and provide helpful messages
3. **Document thoroughly**: Add docstrings to all public methods
4. **Test extensively**: Write unit tests and integration tests
5. **Version correctly**: Follow semantic versioning
6. **Declare dependencies**: List all required packages
7. **Optimize performance**: Use caching and lazy loading
8. **Clean up resources**: Implement `cleanup()` method

## Examples

See `examples/plugins/` for complete plugin examples:

- `examples/plugins/custom_depth_model/` - Custom depth model
- `examples/plugins/custom_processor/` - Custom processor
- `examples/plugins/custom_enhancer/` - Custom enhancer

## Troubleshooting

### Plugin Not Found

```python
# Check if plugin is registered
registry = get_global_registry()
all_plugins = registry.list_plugins()
print(all_plugins)
```

### Plugin Initialization Error

```python
# Initialize with error handling
try:
    plugin = registry.get_plugin(
        'depth_model',
        'my_depth_model',
        initialize=True,
        config={'device': 'cuda'}
    )
except Exception as e:
    print(f"Initialization failed: {e}")
```

### Version Compatibility

```python
# Check version compatibility
metadata = registry.get_metadata('depth_model', 'my_depth_model')
is_compatible = metadata.is_compatible(portal_version="0.1.0")
```

## Resources

- **Plugin API Reference**: `docs/api/plugins.md`
- **Example Plugins**: `examples/plugins/`
- **Community Plugins**: [Transformation Portal Plugin Registry](https://github.com/RC219805/Transformation_Portal/wiki/Plugins)

---

**Questions?** Open an issue on [GitHub](https://github.com/RC219805/Transformation_Portal/issues)
