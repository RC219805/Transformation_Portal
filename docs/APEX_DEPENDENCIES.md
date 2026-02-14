# APEX Command Dependencies - Installation Guide

## ✅ Current Installation Status

All dependencies for **APEX production and research commands** are installed and verified.

---

## 📦 Installed Components

### Core Runtime
- ✅ **NumPy** 1.26.4 - Array operations
- ✅ **Pillow** 11.3.0 - Image I/O
- ✅ **OpenCV** 4.13.0 - Computer vision utilities
- ✅ **PyTorch** 2.10.0 - ML framework with **MPS support** (Apple Silicon)
- ✅ **Transformers** 4.57.6 - HuggingFace model integration

### Depth Backends
- ✅ **Depth Anything V3 (DA3)** - Production depth estimation
  - License: Apache 2.0 (commercial-safe)
  - Installation: Editable from `depth-anything-3/`
  - Model: ~400 MB download on first run

- ✅ **Depth Pro** 0.1 - Research-grade metric depth
  - License: Apple ML Research (non-commercial only)
  - Model: Already cached
  - Features: Metric depth (meters), focal length estimation

### Material Segmentation Backends
- ✅ **SAM2 Materials Adapter** - Superior segmentation quality
  - License: Apache 2.0 (commercial-safe)
  - Model: facebook/sam2-hiera-base (~1.2 GB download on first run)
  - Quality: ⭐⭐⭐⭐⭐ Excellent
  - Speed: ~3-5s per image (CPU), ~1-2s (GPU)

- ✅ **EfficientSAM** - Faster alternative
  - License: Apache 2.0 (commercial-safe)
  - Model: ~50 MB
  - Quality: ⭐⭐⭐⭐ Very Good
  - Speed: ~500ms per image

### Pipeline Components
- ✅ **Materials V3 Pipeline** - Material-aware enhancement
- ✅ **V2 Enhancement** - Advanced color grading
- ✅ **PBR Texture Generation** - Normal, roughness, AO maps
- ✅ **Lux Depth V3 Orchestrator** - Production pipeline

---

## 🚀 Ready to Execute

### Production Command (Commercial-Safe)
```bash
./scripts/pipelines/run_750_picacho_apex_full.sh
```

**Features:**
- Depth: DA3 (Apache 2.0)
- Segmentation: SAM2 (Apache 2.0)
- Device: MPS (Apple Silicon)
- Performance: ~5-7s per image
- Quality: Production-grade

### Research Command (Non-Commercial)
```bash
./scripts/pipelines/run_source_tiffs_depth_pro_research.sh
```

**Features:**
- Depth: Depth Pro (AMLR research-only)
- Segmentation: SAM2 (Apache 2.0)
- Device: MPS (Apple Silicon)
- Performance: ~8-12s per image
- Quality: Research-grade (metric depth + focal length)

---

## 💻 Compute Devices

### Verified Available
- ✅ **MPS (Apple Silicon)** - Primary acceleration backend
  - All backends support MPS
  - Recommended for production use
  - 3-5x faster than CPU

### Fallback
- ✅ **CPU** - Available on all systems
  - Slower but reliable
  - Use if GPU unavailable

---

## 📥 First Run Downloads

On first execution, these models will be automatically downloaded:

| Model | Size | Backend | License |
|-------|------|---------|---------|
| SAM2-base | ~1.2 GB | facebook/sam2-hiera-base | Apache 2.0 |
| DA3 weights | ~400 MB | depth-anything/da3nested-giant-large | Apache 2.0 |
| Depth Pro | Already cached | Apple ML Research checkpoint | AMLR |

**Note:** Downloads happen once and are cached in:
- HuggingFace cache: `~/.cache/huggingface/`
- Depth Pro checkpoint: Repository checkpoints directory

---

## 🧪 Verification

### Quick Dependency Check
```bash
python -c "
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from pathlib import Path
import tempfile

config = EnhanceConfig(
    depth_backend='da3',
    depth_device='cpu',
    enable_material_segmentation=True,
    material_segmentation_backend='sam2',
    enable_v2=True,
    generate_pbr=True,
)

with tempfile.TemporaryDirectory() as tmpdir:
    orchestrator = EnhanceOrchestrator(config, Path(tmpdir))
    print(f'✅ All dependencies verified')
    print(f'   Depth: {orchestrator.depth_backend.__class__.__name__}')
"
```

Expected output:
```
✅ All dependencies verified
   Depth: DA3Backend
```

### Test SAM2 Backend
```bash
python -c "
from transformation_portal.lux_depth_v3.segmentation_backend import _get_backend_instance
backend = _get_backend_instance('sam2', device='cpu')
print(f'✅ SAM2 backend: {backend.__class__.__name__}')
"
```

Expected output:
```
✅ SAM2 backend: SAM2MaterialsAdapter
```

---

## 🔧 Troubleshooting

### If SAM2 model download fails
```bash
# Pre-download SAM2 model manually
python -c "
from transformers import pipeline
pipe = pipeline('mask-generation', model='facebook/sam2-hiera-base')
print('✅ SAM2 model downloaded')
"
```

### If DA3 installation is missing
```bash
# Reinstall DA3 in editable mode
cd depth-anything-3
pip install -e .
cd ..
```

### If MPS is not detected
```bash
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

If MPS is not available, commands will automatically fall back to CPU.

---

## 📊 Performance Expectations

### Production APEX (DA3 + SAM2)
| Stage | CPU Time | MPS Time | Notes |
|-------|----------|----------|-------|
| Depth (DA3) | ~1-2s | ~0.5-1s | Per image |
| Segmentation (SAM2) | ~3-5s | ~1-2s | Per image |
| Materials V3 | ~0.5s | ~0.2s | Pixel operations |
| V2 Enhancement | ~0.5-1s | ~0.3-0.5s | Color grading |
| PBR Generation | ~0.5s | ~0.2s | Texture maps |
| **Total** | **~5-7s** | **~2-3s** | Per image |

### Research APEX (Depth Pro + SAM2)
| Stage | CPU Time | MPS Time | Notes |
|-------|----------|----------|-------|
| Depth (Depth Pro) | ~3-5s | ~1.5-2s | Metric depth |
| Segmentation (SAM2) | ~3-5s | ~1-2s | Per image |
| Materials V3 | ~0.5s | ~0.2s | Pixel operations |
| V2 Enhancement | ~0.5-1s | ~0.3-0.5s | Color grading |
| PBR Generation | ~0.5s | ~0.2s | Texture maps |
| **Total** | **~8-12s** | **~3-5s** | Per image |

**Batch Processing (6 images):**
- Production: ~30-40s (MPS) / ~60-90s (CPU)
- Research: ~40-60s (MPS) / ~90-120s (CPU)

---

## 💡 Quick Test Command

Minimal APEX test on a single image:

```bash
python -m transformation_portal.lux_depth_v3 \
  input_images/source_tiffs/ \
  --output-root output_test_apex/ \
  --depth-backend "da3" \
  --enable-segmentation "on" \
  --segmentation-backend "sam2" \
  --depth-device "mps" \
  --limit 1
```

This will:
1. Process only the first image (`--limit 1`)
2. Use DA3 for depth estimation
3. Use SAM2 for material segmentation
4. Enable MPS acceleration
5. Output to `output_test_apex/`

---

## ✅ Installation Complete

All dependencies are installed and verified. The system is ready for:

- ✅ Production APEX processing (commercial-safe)
- ✅ Research APEX processing (non-commercial)
- ✅ Materials V3 with SAM2 segmentation
- ✅ MPS acceleration on Apple Silicon
- ✅ Full PBR texture generation
- ✅ V2 enhancement pipeline

**Last Verified:** 2026-02-13

**System:**
- Python: 3.11.14
- Platform: macOS (Apple Silicon)
- PyTorch: 2.10.0 with MPS support
- HuggingFace: 4.57.6
