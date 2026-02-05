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

The package installs the following command-line tools:

- **transformation-portal**: Main CLI entry point

Usage Examples
--------------

.. code-block:: bash

    # Basic image processing
    transformation-portal process input.jpg output.jpg

    # Batch processing
    transformation-portal batch input_dir/ output_dir/

    # Check version and help
    transformation-portal --version
    transformation-portal --help
