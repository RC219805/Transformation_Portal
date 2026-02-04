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

    from transformation_portal.config_loader import load_config

    # Load a preset configuration
    config = load_config("config/presets/luxury_estate.yaml")

    # Access configuration values
    print(config.depth_model)  # "depth_anything_v2"
    print(config.enhancement_strength)  # 0.75
