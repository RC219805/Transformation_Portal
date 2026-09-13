Scene Types
===========

.. automodule:: transformation_portal.scene_types
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

This module defines a canonical taxonomy and normalizes supplied labels. It does
not inspect image pixels or automatically recommend processing parameters.

Usage Example
-------------

.. code-block:: python

    from transformation_portal.scene_types import (
        get_scene_type_description,
        normalize_scene_type,
        validate_scene_type,
    )

    scene_type = normalize_scene_type("pool")
    assert scene_type == "exterior_pool"
    assert validate_scene_type(scene_type)
    print(get_scene_type_description(scene_type))

Unknown labels raise ``ValueError`` during normalization. Enumerate the current
taxonomy with ``list_scene_types()`` rather than assuming a fixed four-item list.
