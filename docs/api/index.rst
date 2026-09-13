Transformation Portal API Documentation
========================================

Welcome to the Transformation Portal API reference. This documentation covers selected Python modules. The maintained HTTP contract
is implemented by ``app.py`` and its request/response models; see the portal
quickstarts under ``docs/guides/`` for startup and authentication.

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

Validate a recipe definition without loading a model:

.. code-block:: python

    from transformation_portal.config_loader import validate_recipe

    valid, errors = validate_recipe({"name": "Example", "stages": ["color_grading"]})
    assert valid, errors

For Lux model/license/input resolution without inference, use the maintained
module CLI with ``--plan``. Actual inference and produced artifacts require a
separate successful run with the selected runtime and model installed. Importing
the API or building this reference does not establish either result.

Module Index
------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
