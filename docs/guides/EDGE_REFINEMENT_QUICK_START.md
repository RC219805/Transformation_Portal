# Edge Refinement Quick Start

## 🚀 Quick Usage

### Python API
```python
from lux_depth_v3.edge_refinement import DepthRefiner, create_refinement_preset
import numpy as np

# Load your depth and RGB
depth = np.load("depth.npy")  # (H, W) float32 [0, 1]
rgb = np.array(Image.open("rgb.jpg"))  # (H, W, 3) uint8

# Create refiner with preset
config = create_refinement_preset("balanced")
refiner = DepthRefiner(config)

# Refine
refined = refiner.refine(depth, rgb)
```

### CLI
```bash
# Enable refinement with default preset
lux-depth-v3 process -i images/ -o output/ --enable-refinement

# Use specific preset
lux-depth-v3 process -i images/ -o output/ \
  --enable-refinement \
  --refinement-preset aggressive

# Custom stages
lux-depth-v3 process -i images/ -o output/ \
  --enable-refinement \
  --refinement-stages "guided,edge"
```

## 📋 Presets

| Preset       | Best For                      | Speed  |
|--------------|-------------------------------|--------|
| balanced     | General purpose (recommended) | Medium |
| aggressive   | Maximum edge fidelity         | Slow   |
| conservative | Fast processing, smooth scenes| Fast   |
| edge_focused | Strong RGB edges             | Medium |

## 🔧 Parameter Tuning

**Too smooth?** → Decrease `guided_eps` or `bilateral_sigma_color`  
**Too noisy?** → Increase `bilateral_sigma_space`  
**Edges blurred?** → Decrease `bilateral_d`  
**Missing edges?** → Lower Canny thresholds

## 📊 Expected Results

- **Edge F1**: +25-50% improvement
- **Processing**: +40-90ms per image
- **No regressions**: Chamfer distance ±5%

## 📚 Documentation

- Full guide: `docs/EDGE_REFINEMENT.md`
- Tests: `tests/test_edge_refinement.py`
- Implementation: `lux_depth_v3/edge_refinement.py`
