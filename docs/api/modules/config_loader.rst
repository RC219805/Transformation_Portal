Configuration Loader
====================

.. automodule:: transformation_portal.config_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

The configuration loader module provides utilities for loading and validating
YAML-based pipeline configurations. It supports:

- Preset configurations for common workflows
- Schema validation with clear error messages
- Environment variable interpolation
- Cascading configuration inheritance

Usage Example
-------------

.. code-block:: python

    from transformation_portal.config_loader import load_recipe

    # Load a preset recipe
    recipe = load_recipe("luxury_estate")

    # Access recipe metadata
    print(recipe.name)
    print(recipe.description)
