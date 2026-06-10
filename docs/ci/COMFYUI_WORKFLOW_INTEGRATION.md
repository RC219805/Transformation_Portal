# ComfyUI Workflow Integration

The ComfyUI integration currently provides a declarative workflow construction
layer for Transformation Portal enhancement graphs. `WorkflowBuilder` and
`WorkflowTemplates` are safe to import in the core environment and can serialize
workflow JSON without optional ML dependencies.

Full programmatic execution is intentionally limited. `WorkflowExecutor` fails
fast for unsupported node types instead of silently passing inputs through an
unimplemented runtime path.

## Current Contract

- Package-level `transformation_portal.comfyui` imports expose pure workflow
  builder and template primitives eagerly.
- Runtime custom nodes and `WorkflowExecutor` are lazy imports because they may
  require optional ML or image-processing dependencies.
- `WorkflowTemplates` are build and serialization contracts, not proof that
  every emitted node has an executable runtime implementation.
- Unsupported executor nodes raise a clear `NotImplementedError`.

## Components

```text
comfyui/
├── __init__.py                    # Pure exports plus lazy runtime access
├── workflow_builder.py            # Declarative workflow construction
├── workflow_templates.py          # Pre-built declarative workflows
├── custom_nodes.py                # Implemented runtime custom nodes
└── executor.py                    # Programmatic execution dispatcher
```

## Implemented Runtime Nodes

These custom nodes are registered in `custom_nodes.py`:

- `FluxEnhancementNode`
- `SkyGANNode`
- `SceneAnalysisNode`

They are runtime-oriented and may require optional dependencies when imported
or executed. Keep imports lazy from package entry points.

## Declarative Workflow Node Types

The builder can emit JSON for the following planned node types, but runtime
implementations are not complete yet:

- `MaterialSegmentationNode`
- `NeuroaestheticsNode`
- `AtmosphericModelNode`
- `QualityValidationNode`

These nodes are safe for workflow serialization tests but must not be treated
as executable until their custom node and executor implementations are added.

## Building Workflows

```python
from transformation_portal.comfyui import WorkflowBuilder

workflow = (
    WorkflowBuilder(name="Luxury Estate Enhancement")
    .add_input("estate.jpg")
    .add_scene_analysis(detailed=True)
    .add_flux_enhancement(strength=0.45, num_steps=4)
    .add_skygan_sky(location="montecito", time_of_day="golden_hour")
    .add_quality_validation(pass_threshold=7.0)
    .add_output("enhanced.jpg")
    .build()
)

workflow.save("luxury_estate_enhancement.json")
```

`add_scene_analysis()` is a sidecar builder operation: it reads the current
image, but the image chain remains attached to the previous image-producing
node so downstream image operations do not consume the analysis report.
`add_quality_validation()` follows the same sidecar rule so `SaveImage` nodes
continue to receive images rather than validation reports.

## Template Coverage

The current template factories are expected to build and serialize:

- `full_luxury_estate_pipeline`
- `quick_iterative_enhancement`
- `material_specific_enhancement`
- `location_specific_atmospheric`
- `coastal_property_golden_hour`
- `multi_variant_generation`

`multi_variant_generation()` rejects `num_variants < 1` and rejects empty
`emotional_targets` or `flux_strengths` lists when those lists are provided.

## Execution Posture

Use `WorkflowExecutor` only for node types with explicit executor support. It
does not provide a pass-through fallback for unknown or planned node types:

```python
from transformation_portal.comfyui import WorkflowExecutor, WorkflowTemplates

workflow = WorkflowTemplates.quick_iterative_enhancement("input.jpg", "output.jpg")
result = WorkflowExecutor(cache_models=False).execute(workflow)

assert result["success"] is False
assert "No executor implementation" in result["errors"][0]
```

This fail-fast behavior is intentional. It prevents declarative workflow JSON
from being mistaken for completed FLUX, SkyGAN, segmentation,
neuroaesthetics, or quality-validation execution.

## Validation

Core declarative checks:

```bash
python -S -c 'import sys; sys.path.insert(0, "src"); import transformation_portal.comfyui as c; print(c.WorkflowBuilder)'
python -m pytest -q tests/comfyui/test_workflow_templates_contract.py
```

Runtime custom-node checks remain ML/runtime scoped:

```bash
python -m pytest -q tests/test_comfyui.py
```

Broader closeout for changes in this area:

```bash
make test-fast
make coverage-package
```
