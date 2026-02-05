Scene Types
===========

.. automodule:: transformation_portal.scene_types
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

Scene type detection and classification for architectural rendering workflows.
Automatically identifies scene characteristics to optimize processing parameters.

Supported Scene Types
---------------------

- **Interior**: Indoor spaces (living rooms, kitchens, bedrooms)
- **Exterior**: Outdoor facades and landscaping
- **Aerial**: Drone and elevated photography
- **Detail**: Close-up architectural details and materials

Usage Example
-------------

.. code-block:: python

    from transformation_portal.scene_types import detect_scene_type

    # Detect scene type from image
    scene_type = detect_scene_type("estate_exterior.jpg")
    print(scene_type)  # SceneType.EXTERIOR

    # Get recommended processing parameters
    params = scene_type.get_recommended_params()
