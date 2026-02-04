Transformation Portal API Documentation
========================================

Welcome to the Transformation Portal API reference. This documentation covers all public modules and APIs for luxury real estate rendering and architectural visualization.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   modules/config_loader
   modules/scene_types
   modules/lux_depth_v3

.. toctree::
   :maxdepth: 2
   :caption: Processing Pipelines

   modules/enhancers
   modules/rendering
   modules/processors

.. toctree::
   :maxdepth: 2
   :caption: Utilities & Tools

   modules/utils
   modules/cli
   modules/metrics

Quick Start
-----------

Basic usage example:

.. code-block:: python

    from transformation_portal.config_loader import load_config
    from transformation_portal.lux_depth_v3 import DA3InferenceEngine, DA3Config

    # Load configuration
    config = load_config("config/presets/luxury_estate.yaml")

    # Initialize depth inference engine
    da3_config = DA3Config()
    engine = DA3InferenceEngine(da3_config)

    # Process image with depth awareness
    result = engine.infer_depth("input.jpg")

Module Index
------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
