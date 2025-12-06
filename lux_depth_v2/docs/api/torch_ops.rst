Torch Operations (torch_ops)
=============================

.. automodule:: lux_depth_v2.torch_ops
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

Device Management
^^^^^^^^^^^^^^^^^

.. autofunction:: require_torch
.. autofunction:: pick_device
.. autofunction:: configure_torch
.. autofunction:: maybe_autocast

Tensor Conversion
^^^^^^^^^^^^^^^^^

.. autofunction:: to_torch_rgb
.. autofunction:: from_torch_rgb

Image Operations
^^^^^^^^^^^^^^^^

.. autofunction:: luma
.. autofunction:: midtone_map
.. autofunction:: smoothstep
.. autofunction:: gaussian_blur
.. autofunction:: resize
.. autofunction:: edge_map

Color Adjustments
^^^^^^^^^^^^^^^^^

.. autofunction:: soft_clip01
.. autofunction:: apply_temperature
.. autofunction:: apply_saturation
.. autofunction:: apply_exp_con
.. autofunction:: apply_luma_ratio

Processing Pipeline
^^^^^^^^^^^^^^^^^^^

.. autofunction:: grade_core
.. autofunction:: detail_transfer
.. autofunction:: apply_clarity
.. autofunction:: apply_sharpen
.. autofunction:: material_highlight_compress

Utilities
^^^^^^^^^

.. autofunction:: param_map
.. autofunction:: mean_abs_rgb
.. autofunction:: mean_abs_luma

Classes
-------

.. autoclass:: GradeMaps
   :members:
   :undoc-members:

.. autoclass:: Tiler
   :members:
   :undoc-members:
   :special-members: __init__

Examples
--------

Basic Image Processing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import numpy as np
   from lux_depth_v2 import torch_ops
   
   # Load image
   rgb_np = np.random.rand(1080, 1920, 3).astype(np.float32)
   device = torch_ops.pick_device("auto")
   
   # Convert to torch
   rgb_t = torch_ops.to_torch_rgb(rgb_np, device)
   
   # Apply operations
   luma_t = torch_ops.luma(rgb_t)
   blurred = torch_ops.gaussian_blur(rgb_t, sigma=2.0)
   
   # Convert back
   result = torch_ops.from_torch_rgb(blurred)

Depth-Aware Grading
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create zone weights (foreground, midground, background)
   wfg = torch.ones((1, 1, H, W), device=device) * 0.3
   wmid = torch.ones((1, 1, H, W), device=device) * 0.4
   wbg = torch.ones((1, 1, H, W), device=device) * 0.3
   
   # Grade image
   graded = torch_ops.grade_core(rgb_t, wfg, wmid, wbg, config)
   
   # Apply clarity enhancement
   enhanced = torch_ops.apply_clarity(graded, wfg, wmid, wbg, config)

Tiled Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create tiler for large images
   tiler = torch_ops.Tiler(tile=512, overlap=32)
   
   def process_tile(tile, ya0, xa0, ya1, xa1, y0, x0, y1, x1):
       # Apply processing to tile
       return torch_ops.apply_clarity(tile, wfg, wmid, wbg, config)
   
   # Process with automatic tiling
   result = tiler.run(large_image, process_tile)
