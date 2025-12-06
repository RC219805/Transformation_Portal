Lux Depth V2 Documentation
==========================

**Gold Standard Lux Depth Pipeline V2** - Modular, GPU-accelerated, production-oriented image processing pipeline with automatic material segmentation and depth-aware enhancement.

Features
--------

* **Depth-Aware Processing**: Zone-based grading using foreground/midground/background weights
* **Material Response**: Automatic material segmentation with surface-specific enhancements
* **AI Upscaling**: Real-ESRGAN and ONNX backend support for intelligent 4x upscaling
* **GPU Acceleration**: PyTorch-based with CUDA/MPS support and autocast precision
* **Production Ready**: Atomic file writes, comprehensive error handling, batch processing
* **Flexible Configuration**: YAML presets for common use cases (interiors, exteriors, archival)

Quick Start
-----------

.. code-block:: python

   from lux_depth_v2.pipeline import LuxPipelineV2
   from lux_depth_v2.config import PipelineConfig, Preset

   # Initialize pipeline with preset
   config = PipelineConfig(
       input_dir="input/",
       output_dir="output/",
       depth_dir="depth/",
       preset=Preset.INTERIOR_LUXURY,
       device="auto",
       upscale=4,
   )
   
   pipeline = LuxPipelineV2(config)
   
   # Process all images in directory
   results = pipeline.process_directory()
   
   # Or process single image
   result = pipeline.process_one("input/image.jpg")

Installation
------------

Core dependencies:

.. code-block:: bash

   pip install torch torchvision numpy opencv-python tifffile
   
   # For Real-ESRGAN upscaling
   pip install realesrgan basicsr
   
   # For ONNX backends
   pip install onnxruntime
   
   # For material segmentation
   pip install transformers

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   guides/installation
   guides/quickstart
   guides/presets
   guides/material_segmentation
   guides/depth_processing
   guides/batch_processing
   guides/performance

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/config
   api/pipeline
   api/torch_ops
   api/material_profiles
   api/material_segmentation
   api/upscaling
   api/io_utils
   api/weights

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   development/testing
   development/contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
