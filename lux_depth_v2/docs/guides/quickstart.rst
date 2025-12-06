Quick Start Guide
=================

This guide will help you get started with lux_depth_v2 quickly.

Installation
------------

Install core dependencies:

.. code-block:: bash

   pip install torch torchvision numpy opencv-python tifffile

For AI upscaling with Real-ESRGAN:

.. code-block:: bash

   pip install realesrgan basicsr

Basic Usage
-----------

Process a Single Image
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from lux_depth_v2.pipeline import LuxPipelineV2
   from lux_depth_v2.config import PipelineConfig, Preset

   # Configure pipeline
   config = PipelineConfig(
       preset=Preset.PHOTO_REALISTIC,
       device="auto",
       output_dir=Path("output"),
   )

   # Initialize and process
   pipeline = LuxPipelineV2(config)
   result = pipeline.process_one(Path("input_image.jpg"))

   print(f"Status: {result['status']}")
   print(f"Time: {result['timing_s']:.2f}s")

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   from lux_depth_v2.pipeline import LuxPipelineV2
   from lux_depth_v2.config import PipelineConfig

   config = PipelineConfig(
       input_dir=Path("input_folder"),
       output_dir=Path("output_folder"),
       preset=Preset.INTERIOR_LUXURY,
   )

   pipeline = LuxPipelineV2(config)
   results = pipeline.process_directory()

   # Print summary
   success = sum(1 for r in results if r['status'] == 'ok')
   print(f"Processed {success}/{len(results)} images")

With Depth Maps
^^^^^^^^^^^^^^^

.. code-block:: python

   config = PipelineConfig(
       input_dir=Path("renders"),
       depth_dir=Path("depth_maps"),  # Depth maps with same stem
       output_dir=Path("output"),
   )

   pipeline = LuxPipelineV2(config)
   results = pipeline.process_directory()

Choosing a Preset
-----------------

lux_depth_v2 includes curated presets for different use cases:

**PHOTO_REALISTIC** (Default)
  Balanced processing for photorealistic architectural renders

**INTERIOR_LUXURY**
  Enhanced material response for interior luxury photography

**EXTERIOR_SHOWCASE**
  Vibrant processing for exterior architectural showcases

**ARCHITECTURAL**
  Clean, precise processing for architectural documentation

**ARCHIVAL_QUALITY**
  Conservative processing for archival/museum quality work

Example:

.. code-block:: python

   config = PipelineConfig(
       preset=Preset.EXTERIOR_SHOWCASE,
       # Preset applies: enhanced saturation, clarity, material strength
   )

Device Selection
----------------

Automatic (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config = PipelineConfig(device="auto")
   # Uses CUDA if available, otherwise CPU

Explicit Device
^^^^^^^^^^^^^^^

.. code-block:: python

   # Force CPU
   config = PipelineConfig(device="cpu")

   # Force CUDA
   config = PipelineConfig(device="cuda")

Precision Control
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # FP16 for faster CUDA processing (default)
   config = PipelineConfig(device="cuda", precision="fp16")

   # FP32 for maximum precision
   config = PipelineConfig(precision="fp32")

Output Files
------------

For each input image, the pipeline generates:

* **{stem}_master16.tif** - Graded master (16-bit)
* **{stem}_upscaled16.tif** - Upscaled final (16-bit)
* **{stem}_marketing.png** - Marketing deliverable (8-bit)
* **{stem}_preview.jpg** - Quick preview (8-bit, optional)
* **{stem}_report.json** - Processing report with metadata

Example report.json:

.. code-block:: json

   {
     "status": "ok",
     "image": "/path/to/input.jpg",
     "zone_weights": "depth_percentiles",
     "upscaler": "realesrgan",
     "timing_s": 8.234
   }

Common Options
--------------

Skip Existing Files
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   config = PipelineConfig(
       skip_existing=True,  # Resume interrupted batches
       overwrite=False,
   )

Control Outputs
^^^^^^^^^^^^^^^

.. code-block:: python

   config = PipelineConfig(
       save_master=True,
       save_upscaled=True,
       save_marketing_png=True,
       save_preview_jpg=True,
       preview_scale=0.25,  # 25% preview size
   )

Upscaling Backend
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use Real-ESRGAN (requires model file)
   config = PipelineConfig(
       upscaler_backend="realesrgan",
       model_path=Path("RealESRGAN_x4plus.pth"),
       upscale=4,
   )

   # Use ONNX model
   config = PipelineConfig(
       upscaler_backend="onnx",
       model_path=Path("upscaler_x4.onnx"),
   )

   # Use bicubic (fast, no AI)
   config = PipelineConfig(
       upscaler_backend="none",
   )

Material Segmentation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lux_depth_v2.config import SegmentationConfig

   config = PipelineConfig(
       enable_material=True,
       material_strength=0.75,
       segmentation=SegmentationConfig(
           backend="segformer",
           allow_downloads=True,
       ),
   )

Next Steps
----------

* Learn about :doc:`presets` for different use cases
* Explore :doc:`material_segmentation` for automatic surface detection
* Read about :doc:`depth_processing` for advanced depth-aware grading
* Check :doc:`batch_processing` for production workflows
* See :doc:`performance` for optimization tips
