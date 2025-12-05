# Phase 2 Enhancement Tools

This directory contains the high-priority tools implemented in Phase 2 Strategic Enhancements.

## Tools Overview

### 1. Material Detector (`material_detector.py`)

Advanced material detection with confidence scores and heatmaps.

**Quick Start:**
```bash
# Basic detection
python tools/material_detector.py input.jpg --output-dir output_material/

# With heatmaps
python tools/material_detector.py input.jpg --generate-heatmaps

# Specific materials
python tools/material_detector.py input.jpg --materials wood metal glass
```

**Features:**
- 8 material types (wood, metal, glass, stone, fabric, water, concrete, ceramic)
- Per-pixel confidence maps
- Statistical analysis and JSON reports
- Confidence-weighted enhancement support

### 2. Depth-Aware LUT (`depth_aware_lut.py`)

Apply LUTs with depth-dependent strength for realistic atmospheric perspective.

**Quick Start:**
```bash
# Basic usage
python tools/depth_aware_lut.py input.jpg \
    --output result.png \
    --fg-lut foreground.cube \
    --bg-lut background.cube \
    --atmospheric 0.3

# Advanced: zone-specific adjustments
python tools/depth_aware_lut.py input.jpg \
    --output result.png \
    --fg-lut warm.cube --fg-strength 0.8 --fg-temp 0 \
    --mg-lut neutral.cube --mg-strength 0.7 --mg-temp 100 \
    --bg-lut cool.cube --bg-strength 0.6 --bg-temp 200 \
    --atmospheric 0.4
```

**Features:**
- Zone-based LUT application (foreground/midground/background)
- Atmospheric perspective simulation
- Per-zone color temperature and saturation control
- Automatic depth map creation (Depth Anything V2)

### 3. Comparison Tool (`comparison_tool.py`) [Phase 1]

Batch comparison for 16-bit vs 32-bit HDR outputs.

**Quick Start:**
```bash
python tools/comparison_tool.py \
    --dir1 output_16bit/ \
    --dir2 output_32bit/ \
    --output-dir comparison_results/
```

### 4. HDR Visualizer (`hdr_visualizer.py`) [Phase 1]

Visualize HDR images with histograms and tone mapping.

**Quick Start:**
```bash
python tools/hdr_visualizer.py input_hdr.tif --output output_viz/
```

### 5. QA Validator (`qa_validator.py`) [Phase 1]

Quality assurance validation for processed images.

**Quick Start:**
```bash
python tools/qa_validator.py \
    --input output_images/ \
    --report qa_report.json
```

### 6. Time Predictor (`time_predictor.py`) [Phase 1]

Predict processing time for batch operations.

**Quick Start:**
```bash
python tools/time_predictor.py \
    --num-images 100 \
    --resolution 4000x6000 \
    --operations depth,material,lut
```

## Utility Integration

These tools integrate with utilities in `utils/`:

- `utils/adaptive_tone_mapping.py` - Adaptive tone mapping [Phase 1]
- `utils/alpha_compositor.py` - Alpha compositing [Phase 1]
- `utils/enhanced_reporter.py` - Enhanced reporting [Phase 1]
- `utils/performance_profiler.py` - Performance profiling [Phase 2]
- `utils/exposure_fusion.py` - Multi-exposure fusion [Phase 2]

## Documentation

See `docs/PHASE2_ENHANCEMENTS.md` for complete documentation.

## Testing

```bash
# Run all Phase 2 tests
pytest tests/test_phase2_enhancements.py -v

# Fast tests only
pytest tests/test_phase2_enhancements.py -m "not slow" -v
```

## Performance Benchmarks

| Tool | Speed (2000x3000px) | Memory Peak |
|------|---------------------|-------------|
| Material Detector | 0.5-2s | 200-500MB |
| Depth-Aware LUT | 0.3-1s | 400-800MB |
| Performance Profiler | <1% overhead | ~50MB |
| Exposure Fusion | 2-3s (fusion) | 800MB-1.5GB |

## Examples

### Material Detection + Enhancement

```python
from tools.material_detector import MaterialDetector

detector = MaterialDetector(min_confidence=0.3)
result = detector.detect(image_path)

# Apply confidence-weighted enhancement
enhanced = detector.enhance_with_confidence(
    image,
    result,
    enhancement_func=my_enhancement,
    base_strength=0.8
)
```

### Depth-Aware Grading

```python
from tools.depth_aware_lut import DepthAwareLUT, DepthAwareLUTConfig

config = DepthAwareLUTConfig(
    zone_configs={...},
    atmospheric_strength=0.3
)

processor = DepthAwareLUT(config)
result = processor.apply(image, depth_map)
```

### Performance Profiling

```python
from utils.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.stage('processing', items=10):
    for img in images:
        process(img)

report = profiler.generate_report()
profiler.print_report(report)
```

## Contributing

When adding new tools:
1. Follow existing naming conventions
2. Add comprehensive CLI with `--help`
3. Include usage examples in docstring
4. Add tests in `tests/test_*.py`
5. Update this README
6. Document in `docs/PHASE2_ENHANCEMENTS.md`

## License

See repository LICENSE file.
