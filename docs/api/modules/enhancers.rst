Enhancers
=========

.. automodule:: transformation_portal.enhancers
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Image enhancement modules for luxury real estate processing:

- **AI Enhancement**: Stable Diffusion XL with ControlNet
- **Material Response**: Physics-based surface enhancement
- **Color Grading**: Professional LUT application and color science
- **Sharpening**: Adaptive sharpening with edge protection

Available Enhancers
-------------------

.. autoclass:: transformation_portal.enhancers.AIEnhancer
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.enhancers.MaterialResponseEnhancer
   :members:
   :special-members: __init__

.. autoclass:: transformation_portal.enhancers.ColorGrader
   :members:
   :special-members: __init__

Usage Example
-------------

.. code-block:: python

    from transformation_portal.enhancers import AIEnhancer, ColorGrader

    # AI-powered enhancement
    ai_enhancer = AIEnhancer(
        model="sdxl",
        controlnet="depth",
        strength=0.7
    )
    enhanced = ai_enhancer.enhance("input.jpg")

    # Professional color grading
    grader = ColorGrader(lut_path="assets/luts/film_emulation/Kodak_2383.cube")
    graded = grader.apply("enhanced.jpg")
