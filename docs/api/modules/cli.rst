Command Line Interface
======================

.. automodule:: transformation_portal.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Command-line tools for image and video processing pipelines.

Available Console Scripts
-------------------------

The maintained entrypoints are declared in ``pyproject.toml``:

- ``lux-depth-v3``: Lux execution planning and image/depth processing.
- ``depth-aware-dof``: depth-aware focus rendering.
- ``presence-security``: Presence Compiler parameter, anchor, and watermark tools.
- ``luxury-tiff-batch``: TIFF batch finishing.
- ``transform-render``, ``transform-process``, ``transform-analyze``: compatibility
  command groups.

The recipe CLI is invoked with ``python -m transformation_portal``; there is no
installed ``transformation-portal`` console script or ``serve`` command in
``transformation_portal.cli``.

Usage Examples
--------------

.. code-block:: bash

    .venv/bin/python -m transformation_portal --help
    .venv/bin/python -m transformation_portal version
    .venv/bin/python -m transformation_portal list-recipes
    .venv/bin/lux-depth-v3 --help

To launch the backend, set ``TP_API_KEY`` in the same shell first, then run
``make run-backend-local`` or ``make run-backend-local-noreload``. The direct
no-reload equivalent is:

.. code-block:: bash

    .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
