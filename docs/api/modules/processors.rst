Processors
==========

.. automodule:: transformation_portal.processors
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Core image processing modules for transformation pipelines.

Available Processors
--------------------

.. autoclass:: transformation_portal.processors.DepthProcessor
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.processors.MaterialDetector
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.processors.ToneMappingProcessor
   :members:
   :special-members: __init__

Processing Pipeline Order
-------------------------

Processors should be applied in the following order for optimal results:

1. **Depth Estimation**: Generate depth maps
2. **Material Detection**: Identify surface types
3. **Color Grading**: Apply LUTs and color transforms
4. **Tone Mapping**: HDR to SDR or creative tone mapping
5. **Sharpening**: Final detail enhancement

Usage Example
-------------

.. code-block:: python

    from transformation_portal.processors import (
        DepthProcessor,
        MaterialDetector,
        ToneMappingProcessor
    )

    # Initialize processors
    depth_proc = DepthProcessor(model="depth_anything_v2")
    material_proc = MaterialDetector(confidence_threshold=0.7)
    tone_proc = ToneMappingProcessor(operator="aces")

    # Process image through pipeline
    depth_map = depth_proc.process("input.jpg")
    materials = material_proc.detect("input.jpg", depth_map)
    result = tone_proc.apply("input.jpg", depth_map, materials)
