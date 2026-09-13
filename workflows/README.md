# ComfyUI Workflows

This directory contains declarative graph examples for the Transformation
Portal ComfyUI integration. Building, exporting, or loading a graph does not
establish that its template can execute.

## Build and save templates

Graph construction uses the pure-Python builder and does not load models:

```python
from transformation_portal.comfyui import WorkflowTemplates

WorkflowTemplates.save_all_templates("/tmp/tp-workflow-templates")
```

The generated templates include full-estate, iterative, material-specific,
location-specific, coastal, and multi-variant graph examples. Their names
describe intended compositions, not measured quality or throughput.

```python
from transformation_portal.comfyui import Workflow, WorkflowBuilder

workflow = (
    WorkflowBuilder(name="Enhancement graph example")
    .add_input("input.jpg")
    .add_flux_enhancement(strength=0.45, variant="dev")
    .add_output("output.jpg")
    .build()
)
workflow.save("/tmp/tp-workflow.json")
loaded = Workflow.load("/tmp/tp-workflow.json")
assert loaded.connections == workflow.connections
```

This example writes graph JSON only. It does not read `input.jpg` or produce
`output.jpg`.

## Serialization and editing

`Workflow.save()` writes the node dictionary, workflow metadata, and an
explicit `connections` list. Nodes retain their `_meta` fields. Explicit
connections preserve named or numeric source outputs and distinguish edges
from ordinary list-valued parameters, including literals such as `["node", 1]`.
The encoded connection is removed from target parameters during loading so
it cannot override the executor's resolved value.

Edit graph connections through `Workflow.connections`, then call `save()`
to regenerate encoded node inputs. If editing JSON directly, keep the explicit
connection records and encoded node inputs consistent; the explicit records
are authoritative when present. JSON does not support `//` comments.

Legacy files without `connections` reconstruct links after all supported
nodes are loaded: `[source_node_id, non_negative_integer_slot]` is an edge
only when the source is a loaded node. Legacy literals with exactly that
shape are inherently ambiguous; new saves with an explicit connections list
remove that ambiguity. Unknown node types are logged and skipped. Export and
load continue to tolerate missing targets instead of turning them into new
validation failures.

Numeric slots use source-specific output contracts. For example, SkyGAN
slots 0/1/2 mean image/mask/report; SceneAnalysis slot 0 is a string report.
Unknown output contracts retain their numeric slot rather than assuming
slot zero is an image. The local executor still has the implementation gaps
below.

## Execution limits

[`executor.py`](../src/transformation_portal/comfyui/executor.py) currently has
handlers for SkyGAN and the atmospheric model. INPUT, OUTPUT, FLUX, and other
builder node types lack local executor handlers. An unsupported node returns
`success=False` with an error; template generation must not be presented as a
successful enhancement run.

The custom-node registry is a separate runtime surface. Its implemented
FLUX, SkyGAN, and SceneAnalysis nodes do not supply the local executor's
missing handlers. SkyGAN's custom node returns image, mask, and report, while
the local executor currently returns image and report only. Resolving a mask
connection cannot create that missing output. Handler completion and output
contract alignment require separate feature work and runtime validation.

`WorkflowExecutor(cache_models=False)` avoids eager blender construction,
but executing implemented nodes can still initialize runtime dependencies.
There is no `clear_cache()` public method. Do not install models merely to
check graph serialization.

## Validation and references

```bash
./.venv/bin/pytest tests/test_comfyui_workflow_builder.py \
  tests/comfyui/test_workflow_templates_contract.py -q
```

These deterministic checks prove graph behavior and fail-closed unsupported
execution. They do not prove model inference, ComfyUI GUI interoperability,
or delivered images.

- [Builder](../src/transformation_portal/comfyui/workflow_builder.py)
- [Templates](../src/transformation_portal/comfyui/workflow_templates.py)
- [Custom nodes](../src/transformation_portal/comfyui/custom_nodes.py)
- [Maintained Lux CLI](../docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
