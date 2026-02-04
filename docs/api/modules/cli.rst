Command Line Interface
======================

.. automodule:: transformation_portal.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Command-line interface built with Typer for user-friendly operation.

Available Commands
------------------

- **process**: Process single images or directories
- **batch**: High-performance batch processing
- **video**: Video processing with HDR support
- **preview**: Quick preview with draft settings
- **config**: Configuration management

Usage Examples
--------------

.. code-block:: bash

    # Process single image
    transformation-portal process input.jpg --preset luxury_estate

    # Batch process directory
    transformation-portal batch input_dir/ output_dir/ --workers 4

    # Video processing with HDR
    transformation-portal video input.mp4 output.mp4 --hdr --tone-map aces

    # Generate preview
    transformation-portal preview input.jpg --fast

CLI Reference
-------------

.. autofunction:: transformation_portal.cli.process
.. autofunction:: transformation_portal.cli.batch
.. autofunction:: transformation_portal.cli.video
.. autofunction:: transformation_portal.cli.preview
.. autofunction:: transformation_portal.cli.config
