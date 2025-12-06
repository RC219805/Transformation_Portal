Testing Guide
=============

The lux_depth_v2 package includes a comprehensive test suite with 80%+ code coverage.

Running Tests
-------------

Run All Tests
^^^^^^^^^^^^^

.. code-block:: bash

   # Using pytest directly
   pytest tests/

   # Using make
   make test

Run Fast Tests (Recommended for Development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Skip slow and GPU tests
   pytest tests/ -m "not slow and not gpu"

   # Using make
   make test-fast

Test Markers
------------

Tests are marked with pytest markers for selective execution:

**slow**
  Tests that take significant time (>5 seconds)

**gpu**
  Tests requiring CUDA GPU

**integration**
  End-to-end integration tests

Coverage
--------

Generate coverage reports:

.. code-block:: bash

   # HTML report
   pytest --cov=lux_depth_v2 --cov-report=html
   
   # Terminal report
   pytest --cov=lux_depth_v2 --cov-report=term-missing

Target: 80%+ coverage for all modules.
