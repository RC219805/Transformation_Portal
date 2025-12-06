# Lux Depth V2 Examples

This directory contains practical examples demonstrating various use cases for the lux_depth_v2 pipeline.

## Examples

### Basic Usage

- **[01_basic_processing.py](01_basic_processing.py)** - Process a single image with default settings
- **[02_batch_processing.py](02_batch_processing.py)** - Batch process an entire directory
- **[03_with_depth_maps.py](03_with_depth_maps.py)** - Process images with pre-computed depth maps

### Advanced Features

- **[04_material_segmentation.py](04_material_segmentation.py)** - Enable automatic material detection
- **[05_custom_preset.py](05_custom_preset.py)** - Create and apply custom processing presets
- **[06_performance_tuning.py](06_performance_tuning.py)** - Optimize for speed vs quality

### Production Workflows

- **[07_production_pipeline.py](07_production_pipeline.py)** - Complete production workflow with error handling
- **[08_cli_wrapper.py](08_cli_wrapper.py)** - Command-line wrapper for batch operations
- **[09_monitoring.py](09_monitoring.py)** - Performance monitoring and telemetry

### Integration

- **[10_rest_api_server.py](10_rest_api_server.py)** - REST API server for on-demand processing
- **[11_automated_workflow.py](11_automated_workflow.py)** - Automated watch folder processing

## Running Examples

All examples can be run directly:

```bash
cd examples
python 01_basic_processing.py
```

Some examples require additional setup (input images, depth maps, etc.). See individual file headers for requirements.

## Requirements

Core requirements are installed with lux_depth_v2. Some examples may need additional dependencies:

```bash
# For REST API example
pip install fastapi uvicorn

# For material segmentation examples
pip install transformers

# For ONNX examples
pip install onnxruntime
```
