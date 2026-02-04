Utilities
=========

.. automodule:: transformation_portal.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Utility modules for common operations:

- **File I/O**: Image loading with metadata preservation
- **Logging**: Structured logging with performance metrics
- **Validation**: Input validation and schema checking
- **Caching**: LRU caching for expensive operations

Usage Example
-------------

.. code-block:: python

    from transformation_portal.utils import load_image_with_metadata, setup_logging

    # Configure logging
    logger = setup_logging("processing.log", level="INFO")

    # Load image with metadata
    image, metadata = load_image_with_metadata("input.jpg")
    logger.info(f"Loaded {metadata['width']}x{metadata['height']} image")
