# Custom Depth Model Plugin Example

Example plugin demonstrating how to create a custom depth estimation model for the Transformation Portal.

## Installation

This plugin is automatically discovered when the `examples/plugins/` directory is in your plugin search path.

```python
from transformation_portal.plugins import get_global_registry

registry = get_global_registry()

# Discover plugins in examples directory
registry.discover_plugins(['examples/plugins/'])
```

## Usage

```python
from transformation_portal.plugins import get_global_registry
from PIL import Image

# Get plugin registry
registry = get_global_registry()

# Get the simple depth model plugin
depth_model = registry.get_plugin(
    'depth_model',
    'simple_depth_model',
    initialize=True,
    config={'normalize': True, 'invert': False}
)

# Load image
image = Image.open('test_image.jpg')

# Estimate depth
depth_map = depth_model.estimate_depth(image)

# Save depth map
depth_image = Image.fromarray((depth_map * 255).astype('uint8'))
depth_image.save('depth_map.png')
```

## Configuration Options

- `normalize` (bool): Normalize depth values to 0-1 range (default: True)
- `invert` (bool): Invert depth values (far=1, near=0) (default: False)

## How It Works

This is a simplified example that uses gradient magnitude as a proxy for depth:

1. Convert image to grayscale
2. Compute gradients using Sobel filter
3. Calculate gradient magnitude
4. Normalize to 0-1 range
5. Optionally invert values

**Note**: Real depth models (like Depth Anything V2) use trained neural networks. This example demonstrates the plugin structure without requiring heavy ML dependencies.

## Creating Your Own Plugin

Use this as a template for creating your own depth model plugins:

1. Inherit from `DepthModelPlugin`
2. Implement `_create_metadata()` to define plugin info
3. Implement `initialize()` to set up your model
4. Implement `estimate_depth()` for depth estimation
5. Use `@plugin` decorator for auto-registration

See [docs/PLUGIN_DEVELOPMENT.md](../../../docs/PLUGIN_DEVELOPMENT.md) for complete guide.
