Utilities
=========

.. automodule:: transformation_portal.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The package exposes the ``performance`` and ``error_handling`` utility modules.
It does not export ``load_image_with_metadata`` or ``setup_logging`` functions.

Usage Example
-------------

.. code-block:: python

    from transformation_portal.utils.performance import cache_result

    @cache_result(maxsize=16)
    def square(value: int) -> int:
        return value * value

    assert square(4) == 16

The cache is process-local. It is distinct from Lux's governed depth cache and
does not establish model/runtime identity or persistent artifact authority.
