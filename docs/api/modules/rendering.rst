Rendering
=========

.. automodule:: transformation_portal.rendering
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

High-performance rendering engines for batch processing and production workflows.

Rendering Engines
-----------------

.. autoclass:: transformation_portal.rendering.BatchRenderer
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.rendering.VideoRenderer
   :members:
   :special-members: __init__

Features
--------

- **Batch Processing**: Multi-threaded rendering with progress tracking
- **Video Rendering**: FFmpeg integration with HDR support
- **Metadata Preservation**: IPTC, XMP, and GPS data retention
- **Error Recovery**: Automatic retry and checkpoint management

Usage Example
-------------

.. code-block:: python

    from transformation_portal.rendering import BatchRenderer
    from pathlib import Path

    # Initialize batch renderer
    renderer = BatchRenderer(
        num_workers=4,
        checkpoint_interval=50
    )

    # Process directory
    input_dir = Path("input_images/")
    output_dir = Path("output/")

    renderer.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        preset="luxury_estate"
    )

Performance
-----------

- **Throughput**: 400-600 images/hour (4K images, full pipeline)
- **Memory**: Adaptive batch sizing to prevent OOM
- **Parallelism**: Multi-process for CPU, batched for GPU
- **Checkpoints**: Automatic resume on failure
