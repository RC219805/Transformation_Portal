Lux Depth V3
============

.. automodule:: transformation_portal.lux_depth_v3
   :members:
   :undoc-members:
   :show-inheritance:

Depth Estimation Pipeline
--------------------------

.. automodule:: transformation_portal.lux_depth_v3.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Depth Processors
----------------

.. automodule:: transformation_portal.lux_depth_v3.processors
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Lux Depth V3 module provides depth-aware image processing using Depth Anything V2
with Apple Neural Engine optimization. Features include:

- High-quality monocular depth estimation
- CoreML acceleration for M-series chips
- Zone-based tone mapping and atmospheric effects
- Material-aware depth processing

Usage Example
-------------

.. code-block:: python

    from transformation_portal.lux_depth_v3 import DepthPipeline

    # Initialize pipeline with CoreML optimization
    pipeline = DepthPipeline(
        model="depth_anything_v2",
        use_coreml=True,
        batch_size=4
    )

    # Process with depth awareness
    result = pipeline.process(
        input_path="input.jpg",
        output_path="output.jpg",
        depth_strength=0.8
    )

Performance Notes
-----------------

- **CoreML (M1/M2/M3)**: 3-5x faster than CPU
- **MPS Backend**: Recommended for batch processing
- **Memory**: ~2GB for 4K images with depth estimation
- **Throughput**: 400-600 images/hour (batch processing)
