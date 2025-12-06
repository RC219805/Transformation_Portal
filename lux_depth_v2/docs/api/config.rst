Configuration (config)
======================

.. automodule:: lux_depth_v2.config
   :members:
   :undoc-members:
   :show-inheritance:

Enumerations
------------

.. autoclass:: Preset
   :members:
   :undoc-members:

Configuration Classes
---------------------

.. autoclass:: PipelineConfig
   :members:
   :undoc-members:
   
   .. automethod:: apply_preset

.. autoclass:: SegmentationConfig
   :members:
   :undoc-members:

.. autoclass:: ServiceConfig
   :members:
   :undoc-members:

Examples
--------

Basic Configuration
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lux_depth_v2.config import PipelineConfig, Preset
   
   config = PipelineConfig(
       preset=Preset.PHOTO_REALISTIC,
       upscale=4,
       device="cuda",
       enable_material=True,
   )
   
   # Apply preset values
   config.apply_preset()

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config = PipelineConfig(
       preset=Preset.INTERIOR_LUXURY,
       # Override specific values
       material_strength=0.85,
       detail_strength=0.75,
       clarity_fg=0.20,
   )
   config.apply_preset()

Material Segmentation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lux_depth_v2.config import SegmentationConfig
   
   seg_config = SegmentationConfig(
       backend="segformer",
       allow_downloads=True,
       input_long_side=768,
       soften_sigma_px=2.0,
       min_confidence=0.25,
   )
   
   config = PipelineConfig(segmentation=seg_config)
