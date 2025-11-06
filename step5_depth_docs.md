# Depth Estimation Documentation

Query: depth estimation CoreML

## Citations

### [1] docs/guides/README_VFX_EXTENSION.md:90-211
**Confidence**: 76%
**Relevance**: Type: readme | Text match

```
- `--save-depth`: Save depth map alongside output
- `--out-bitdepth`: Output bit depth (8, 16, or 32)

#### Batch Processing

Process multiple images:

```bash
python realize_v8_unified_cli_extension.py batch-vfx \
    --input renders/ \
...
```

### [2] docs/depth_pipeline/DEPTH_PIPELINE_README.md:301-427
**Confidence**: 62%
**Relevance**: Type: readme | Text match

```
```

**Effect**: Atmospheric haze, aerial desaturation, depth-based color shift

### Example 3: Depth-of-Field Simulation

```python
from depth_pipeline.utils import create_depth_of_field_map
import cv2

...
```

### [3] docs/guides/DEPTH_PIPELINE_README.md:301-427
**Confidence**: 53%
**Relevance**: Type: readme | Text match

```
```

**Effect**: Atmospheric haze, aerial desaturation, depth-based color shift

### Example 3: Depth-of-Field Simulation

```python
from depth_pipeline.utils import create_depth_of_field_map
import cv2

...
```
