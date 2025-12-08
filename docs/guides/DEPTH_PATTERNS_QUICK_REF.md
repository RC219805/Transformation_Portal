# Depth Processing Patterns - Quick Reference 🚀

**Generated**: December 8, 2025 | **Source**: RAG Multi-Source Retrieval

## 📊 At a Glance

- **72 files** analyzed
- **1,143 patterns** identified  
- **2 implementations**: Lux Depth V2 (production) + Depth Anything V2 (legacy)
- **5 presets**: photo_realistic, interior_luxury, exterior_showcase, architectural, archival_quality
- **Performance**: 300-500ms/image, 7,200 images/hour

## 🎯 Primary Implementation

### Lux Depth V2 (Production) ⭐

```bash
# CLI batch processing
lux-depth-v2 --input-dir renders/ --output-dir out/ --preset interior_luxury

# Service mode
lux-depth-v2-service --port 8088 --workers 4
```

```python
# Python API
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset

config = PipelineConfig(preset=Preset.INTERIOR_LUXURY)
pipeline = LuxPipelineV2(config)
result = pipeline.process_image("render.jpg")
```

**Location**: `lux_depth_v2/` (29 files)  
**Features**: GPU-accelerated, 5 presets, security-hardened, FastAPI service

## ⚡ Performance

| Stage | Time (ms) | % Total |
|-------|-----------|---------|
| Depth Estimation | 42 | 8.4% |
| Material Segmentation | 25 | 5.0% |
| Zone Processing | 63 | 12.6% |
| **Upscaling** | **287** | **57.4%** |
| Export | 75 | 15.0% |
| **TOTAL** | **500** | **100%** |

**Throughput**: 7,200 images/hour (2 images/second)

## 🎨 Presets

| Preset | Use Case | Characteristics |
|--------|----------|-----------------|
| `photo_realistic` | General purpose | Balanced quality |
| `interior_luxury` | Interior spaces | Warm tones, high clarity |
| `exterior_showcase` | Exteriors | Dynamic range |
| `architectural` | Technical renders | Precision, neutrality |
| `archival_quality` | Preservation | Maximum fidelity |

## 🔧 Configuration

```python
from lux_depth_v2.config import PipelineConfig

config = PipelineConfig(
    preset=Preset.PHOTO_REALISTIC,
    upscale=4,                    # 2 or 4
    device="auto",                # auto|cuda|mps|cpu
    precision="fp16",             # fp16|fp32
    upscaler_backend="torch",     # torch|onnx|none
    save_master=True,             # 16-bit TIFF
    save_marketing_png=True       # 8-bit PNG
)
```

## 🎯 Depth-Aware Processing

### 1. Zone-Based Tone Mapping
```python
# Automatic quantile-based zones
fg_q = 0.35  # Foreground (closest 35%)
mg_q = 0.70  # Midground (35-70%)
bg_q = 1.0   # Background (70-100%)
```

### 2. Material-Aware Processing
- Wood (warmth, grain enhancement)
- Metal (specular, reflections)
- Glass (transparency, refraction)
- Fabric (texture, softness)
- Stone (micro-detail, roughness)
- Water (fluidity, reflections)

### 3. Atmospheric Effects
- Depth fog (aerial perspective)
- Clarity enhancement (edge-aware)
- Glow effects (bloom, halation)

## 🚀 Quick Commands

```bash
# Basic processing
lux-depth-v2 --input-dir renders/ --output-dir out/

# With preset
lux-depth-v2 --input-dir renders/ --preset interior_luxury

# Custom settings
lux-depth-v2 \
  --input-dir renders/ \
  --preset architectural \
  --upscale 4 \
  --device cuda \
  --precision fp16

# Service mode
lux-depth-v2-service --port 8088 --workers 4

# API request
curl -X POST http://localhost:8088/process \
  -F "file=@image.jpg" \
  -F "preset=interior_luxury"
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `lux_depth_v2/pipeline.py` | Main pipeline (LuxPipelineV2) |
| `lux_depth_v2/config.py` | Configuration + presets |
| `lux_depth_v2/cli.py` | Command-line interface |
| `lux_depth_v2/upscaling.py` | Safe upscaling (torch) |
| `lux_depth_v2/material_segmentation.py` | Material detection |
| `lux_depth_v2/service.py` | FastAPI REST API |

## 🔬 Depth Estimation

### Depth Anything V2

**Location**: `scripts/utilities/depth_anything_v2.py`

```python
from depth_anything_v2 import DepthAnythingV2Predictor

predictor = DepthAnythingV2Predictor(
    variant="small",           # small|base|large
    backend="pytorch_mps",     # pytorch_mps|coreml|onnx
    cache_size=100
)
depth_map = predictor.predict("image.jpg")
```

**Performance**:
- Small: 24-40ms (M4 Max), 400-600 images/hour
- Base: 40-55ms (M4 Max), 300-450 images/hour
- Large: 55-65ms (M4 Max), 200-350 images/hour

### CoreML Optimization

**Location**: `scripts/utilities/depth_predict_coreml.py`

- **3-5x speedup** on Apple Silicon (M1/M2/M3/M4)
- Uses Apple Neural Engine (dedicated hardware)
- Lower power consumption

## 🧪 Testing

**66 depth-related tests** covering:
- Pipeline functionality
- Preset validation
- Material segmentation
- Upscaling backends
- Error handling
- Format support

```bash
# Run depth tests
pytest tests/ -k depth -v

# Lux V2 specific
pytest tests/test_lux_depth_v2*.py -v
```

## 🎓 Integration Patterns

### 1. Standalone CLI
```bash
lux-depth-v2 --input-dir renders/ --preset interior_luxury
```

### 2. Python API
```python
pipeline = LuxPipelineV2(config)
result = pipeline.process_image(path)
```

### 3. REST API Service
```bash
lux-depth-v2-service --port 8088
curl -X POST http://localhost:8088/process -F "file=@image.jpg"
```

### 4. Custom Pipeline Integration
```python
# Your preprocessing
preprocessed = your_preprocess(image)

# Lux Depth V2
lux_result = pipeline.process_image(preprocessed)

# Your postprocessing
final = your_postprocess(lux_result['master'])
```

## 🔍 Key Recommendations

### High Priority
1. ✅ Create architectural diagram (visual flowchart)
2. ✅ Write migration guide (legacy → Lux V2)
3. ✅ Expand test coverage to 80%+ (edge cases)

### Medium Priority
4. 📦 Production deployment guide (Docker/Kubernetes)
5. 📈 Performance regression testing (CI/CD)
6. 📖 API reference documentation (Sphinx/mkdocs)

### Low Priority
7. 🎬 Video processing support (temporal consistency)
8. 🚀 Multi-GPU support (distributed processing)

## 📖 Full Documentation

- **Comprehensive Report**: `DEPTH_PROCESSING_PATTERNS_RAG_REPORT.md` (1,470 lines)
- **Structured Data**: `depth_pattern_analysis.json`
- **RAG System Status**: `RAG_SYSTEM_STATUS.md`
- **Quick Start**: `.github/agents/rag_system/QUICK_START.md`

## 🆘 Troubleshooting

**GPU not detected?**
```python
import torch
print(torch.cuda.is_available())  # CUDA
print(torch.backends.mps.is_available())  # MPS (Apple)
```

**Out of memory?**
- Reduce `upscale` (4 → 2)
- Use `precision="fp32"` → `"fp16"`
- Decrease `tile` size (512 → 256)

**Slow performance?**
- Use `device="cuda"` or `"mps"` (not CPU)
- Enable `cudnn_benchmark=True`
- Use smaller depth model variant ("small" vs "large")
- Check upscaler backend (`torch` recommended)

## 🎯 Next Steps

1. **Review comprehensive report**: `cat DEPTH_PROCESSING_PATTERNS_RAG_REPORT.md`
2. **Test pipeline**: `lux-depth-v2 --input-dir test_renders/`
3. **Explore presets**: Try all 5 presets on sample images
4. **Integrate**: Add to your processing workflow

---

**Need more detail?** See full report: `DEPTH_PROCESSING_PATTERNS_RAG_REPORT.md`
