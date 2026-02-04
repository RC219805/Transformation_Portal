Metrics
=======

.. automodule:: transformation_portal.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Quality metrics for image and video processing validation.

Available Metrics
-----------------

.. autoclass:: transformation_portal.metrics.LPIPS
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.metrics.TraditionalMetrics
   :members:
   :special-members: __init__

Supported Metrics
-----------------

- **LPIPS**: Learned Perceptual Image Patch Similarity
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error

Usage Example
-------------

.. code-block:: python

    from transformation_portal.metrics import LPIPS, TraditionalMetrics

    # Perceptual similarity
    lpips = LPIPS()
    score = lpips.compute("original.jpg", "processed.jpg")
    print(f"LPIPS: {score:.4f}")

    # Traditional metrics
    metrics = TraditionalMetrics()
    results = metrics.compute_all("original.jpg", "processed.jpg")
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"SSIM: {results['ssim']:.4f}")
