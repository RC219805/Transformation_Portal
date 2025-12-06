Pipeline (pipeline)
===================

.. automodule:: lux_depth_v2.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Main Pipeline Class
-------------------

.. autoclass:: LuxPipelineV2
   :members:
   :undoc-members:
   :special-members: __init__

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from lux_depth_v2.pipeline import LuxPipelineV2
   from lux_depth_v2.config import PipelineConfig
   
   config = PipelineConfig(
       input_dir="renders/",
       output_dir="output/",
       depth_dir="depth_maps/",
   )
   
   pipeline = LuxPipelineV2(config)
   results = pipeline.process_directory()

Process Single Image
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   
   # Process with automatic depth discovery
   result = pipeline.process_one(Path("renders/image.jpg"))
   
   # Process with explicit depth path
   result = pipeline.process_one(
       Path("renders/image.jpg"),
       depth_path=Path("depth_maps/image.tif")
   )
   
   print(f"Status: {result['status']}")
   print(f"Processing time: {result['timing_s']}s")
   print(f"Zone weights: {result['zone_weights']}")

Custom Logger
^^^^^^^^^^^^^

.. code-block:: python

   import logging
   
   logger = logging.getLogger("my_pipeline")
   logger.setLevel(logging.DEBUG)
   
   pipeline = LuxPipelineV2(config, logger=logger)

Error Handling
^^^^^^^^^^^^^^

.. code-block:: python

   results = pipeline.process_directory()
   
   for result in results:
       if result['status'] == 'error':
           print(f"Failed: {result['image']}")
           print(f"Error: {result.get('error', 'Unknown')}")
       elif result['status'] == 'ok':
           print(f"Success: {result['image']} ({result['timing_s']:.2f}s)")

Output Files
------------

The pipeline generates the following outputs for each input image:

* **{stem}_master16.tif**: Graded master at original resolution (16-bit TIFF)
* **{stem}_upscaled16.tif**: Final upscaled result (16-bit TIFF)
* **{stem}_marketing.png**: Marketing deliverable (8-bit PNG)
* **{stem}_preview.jpg**: Quick preview (8-bit JPG, optional)
* **{stem}_report.json**: Processing report with metadata

Report JSON Structure
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "status": "ok",
     "image": "/path/to/input.jpg",
     "depth": "/path/to/depth.tif",
     "zone_weights": "depth_percentiles",
     "material_mods": "material_segmentation",
     "upscaler": "realesrgan",
     "ai_color_diff": 0.0234,
     "ai_luma_diff": 0.0156,
     "timing_s": 12.345,
     "config": {...}
   }
