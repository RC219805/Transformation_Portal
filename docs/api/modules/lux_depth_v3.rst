Lux Depth V3
============

.. automodule:: transformation_portal.lux_depth_v3
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Overview
--------

The Lux Depth V3 package exposes configuration, inference, PBR, and orchestration
APIs lazily so importing the package does not initialize optional ML stacks.
The maintained CLI prepares one canonical ``tp.execution.plan.v1`` before backend
initialization and output creation. ``--plan`` emits its canonical bytes without
running inference.

Usage Example
-------------

Resolve a plan against an existing image directory from the repository root:

.. code-block:: bash

    .venv/bin/lux-depth-v3 --input-dir input_images --output-dir output/planned \
      --model-key da3-metric --enable-v2 off --plan

The input directory must contain supported images. The plan resolves inputs,
model/license selection, and execution policy. A successful plan does not prove
that the isolated inference runtime can execute or that output artifacts exist.

Python callers that need governed cache access must prepare execution through
``transformation_portal.lux_depth_v3.execution_lifecycle.prepare_lux_execution``
and use ``EnhanceOrchestrator.from_prepared(...)``. Legacy structural invocation
projections cannot authorize execution or cache access. See
``docs/reference/EXECUTION_PLAN_V1.md`` for the core-owned contract.

Runtime Evidence
----------------

DA3 and Depth Pro use their selected runtime contracts. Device availability,
model weights, license acknowledgements, and complete runtime identity must be
verified for the actual run. Optional PBR, Materials V3, segmentation, captioning,
float-depth persistence, and output encoding are explicit configuration choices.
No throughput, memory ceiling, or acceleration factor is guaranteed here.
