# ComfyUI Workflows

This directory contains example and template workflows for the Transformation Portal ComfyUI integration.

## Directory Structure

```
workflows/
├── README.md                  # This file
├── examples/                  # Example workflows demonstrating capabilities
│   ├── simple_enhancement.json
│   └── coastal_golden_hour.json
└── templates/                 # Pre-built template workflows (generated)
```

## Quick Start

### Generate All Templates

```python
from transformation_portal.comfyui import WorkflowTemplates

# Save all pre-built templates to templates/
WorkflowTemplates.save_all_templates("workflows/templates")
```

This generates:
- `full_luxury_estate_pipeline.json` - Complete enhancement pipeline
- `quick_iterative_enhancement.json` - Fast enhancement for previews
- `material_specific_enhancement.json` - Material-aware processing
- `location_specific_atmospheric.json` - Atmospheric sky replacement
- `coastal_property_golden_hour.json` - Coastal estate specialization
- `multi_variant_1.json`, `multi_variant_2.json`, `multi_variant_3.json` - A/B testing variants

### Load and Execute a Workflow

```python
from transformation_portal.comfyui import Workflow, WorkflowExecutor

# Load workflow
workflow = Workflow.load("workflows/examples/simple_enhancement.json")

# Execute
executor = WorkflowExecutor(verbose=True)
results = executor.execute(workflow)

if results["success"]:
    print(f"Complete in {results['execution_time']:.1f}s")
```

### Modify a Workflow

Workflows are JSON files that can be edited:

```json
{
  "nodes": {
    "fluxenhancement_1": {
      "inputs": {
        "strength": 0.45,     // Adjust enhancement strength
        "num_steps": 4,       // Change diffusion steps
        "variant": "dev"      // Switch between "dev" and "schnell"
      }
    }
  }
}
```

## Example Workflows

### simple_enhancement.json

**Purpose:** Basic FLUX enhancement without additional processing

**Pipeline:**
1. Load image
2. FLUX enhancement (strength=0.45, 4 steps)
3. Save output

**Use for:**
- Quick tests
- Learning the workflow system
- Baseline comparisons

**Execution time:** ~30-45 seconds

---

### coastal_golden_hour.json

**Purpose:** Complete enhancement for coastal properties at golden hour

**Pipeline:**
1. Load image
2. Scene analysis (VLM)
3. Material segmentation (SAM+CLIP)
4. FLUX enhancement with ControlNet
5. SkyGAN golden hour sky
6. Atmospheric modeling
7. Neuroaesthetics optimization (aspiration)
8. Quality validation
9. Save high-quality output (quality=98)

**Use for:**
- Montecito/Santa Barbara coastal estates
- Final client deliverables
- Portfolio-quality images

**Execution time:** ~2.5-3 minutes

## Creating Custom Workflows

### Option 1: Programmatically with WorkflowBuilder

```python
from transformation_portal.comfyui import WorkflowBuilder

builder = WorkflowBuilder(name="My Custom Workflow")

workflow = (builder
    .add_input("my_image.jpg")
    .add_flux_enhancement(strength=0.50, variant="dev")
    .add_skygan_sky(location="montecito", time_of_day="sunset")
    .add_output("my_output.jpg")
    .build()
)

# Save for reuse
workflow.save("workflows/custom/my_workflow.json")
```

### Option 2: Modify Existing JSON

1. Copy an example workflow:
   ```bash
   cp workflows/examples/simple_enhancement.json workflows/custom/my_workflow.json
   ```

2. Edit JSON to change parameters:
   ```json
   {
     "nodes": {
       "fluxenhancement_2": {
         "inputs": {
           "strength": 0.50,    // Increased from 0.45
           "variant": "schnell" // Changed from "dev"
         }
       }
     }
   }
   ```

3. Update metadata:
   ```json
   {
     "metadata": {
       "name": "My Custom Enhancement",
       "description": "Custom workflow for my specific use case"
     }
   }
   ```

## Workflow Parameters

### Common Adjustments

**Enhancement Strength:**
- `0.35` - Subtle, conservative enhancement
- `0.45` - Balanced (recommended default)
- `0.55` - Aggressive enhancement
- `0.65+` - Maximum transformation (use carefully)

**FLUX Variant:**
- `"dev"` - Better quality, slower (4-50 steps)
- `"schnell"` - Faster, good quality (1-4 steps)

**Quality Threshold:**
- `6.0` - Acceptable for previews
- `7.0` - Good for most work
- `7.5+` - High bar for final deliverables

**Time of Day:**
- `"sunrise"` - Warm, low-angle morning light
- `"morning"` - Soft, diffuse light
- `"midday"` - Bright, overhead light
- `"golden_hour"` - Warm, magical quality
- `"sunset"` - Dramatic, colorful skies
- `"twilight"` - Blue hour, ethereal

**Emotional Target:**
- `"luxury"` - Opulent, refined, exclusive
- `"aspiration"` - Inspiring, desirable, elevated
- `"comfort"` - Warm, inviting, cozy
- `"serenity"` - Calm, peaceful, tranquil
- `"energy"` - Vibrant, dynamic, exciting

## Batch Processing

Process multiple images with the same workflow:

```python
from transformation_portal.comfyui import Workflow, WorkflowExecutor
from pathlib import Path
import json

# Load workflow template
with open("workflows/examples/simple_enhancement.json", 'r') as f:
    workflow_template = json.load(f)

executor = WorkflowExecutor(cache_models=True)

# Process all images
for image_path in Path("input").glob("*.jpg"):
    # Update input/output paths
    workflow_template["nodes"]["loadimage_1"]["inputs"]["image"] = str(image_path)
    workflow_template["nodes"]["saveimage_3"]["inputs"]["filename"] = f"output/{image_path.name}"

    # Create workflow instance
    workflow = Workflow()
    # ... load from modified template ...

    # Execute
    results = executor.execute(workflow)
    print(f"{image_path.name}: {'✓' if results['success'] else '✗'}")
```

## Best Practices

1. **Start with templates** - Modify existing workflows rather than building from scratch
2. **Version control workflows** - Track workflow changes in git
3. **Use descriptive names** - Name workflows by purpose (e.g., `montecito_estates_summer.json`)
4. **Test incrementally** - Build complex workflows step-by-step
5. **Cache models** - Use `WorkflowExecutor(cache_models=True)` for batch processing
6. **Validate quality** - Always include quality validation for client work

## Troubleshooting

### Workflow fails to load

```
ValueError: Unknown node type: ...
```

**Solution:** Check node class names match available node types in `custom_nodes.py`

### Execution errors

```
Error executing node_id: ...
```

**Solution:** Check execution results for detailed error:
```python
results = executor.execute(workflow)
print(results["errors"])  # See specific errors
```

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Use `variant="schnell"` instead of `"dev"`
- Enable CPU offload in node initialization
- Clear executor cache: `executor.clear_cache()`

## Additional Resources

- **Full Documentation:** `docs/COMFYUI_WORKFLOW_INTEGRATION.md`
- **API Reference:** See WorkflowBuilder, WorkflowTemplates, WorkflowExecutor classes
- **Custom Nodes:** `src/transformation_portal/comfyui/custom_nodes.py`

## Contributing

To add new workflow templates:

1. Create workflow with WorkflowBuilder
2. Test thoroughly
3. Save to `workflows/templates/`
4. Document in this README
5. Add to `WorkflowTemplates.save_all_templates()`

## License

Part of Transformation Portal - Luxury Real Estate AI Enhancement System
