# Professional Pipeline - Quick Reference Card

## 🚀 Quick Start

```bash
# Single image
python pro_pipeline.py process image.jpg --preset architectural-hero --out ./enhanced

# Batch processing
python pro_pipeline.py batch ./renders --preset interior-dramatic --out ./final

# List presets
python pro_pipeline.py list-presets

# Version info
python pro_pipeline.py version
```

## 📋 Available Presets

| Preset | Best For | Processing Time | Key Features |
|--------|----------|----------------|--------------|
| `architectural-hero` | Hero shots | 2-5 min | All stages, maximum quality |
| `interior-dramatic` | Dark interiors | 30-60 sec | High contrast, no AI |
| `exterior-golden-hour` | Outdoor scenes | 2-4 min | Warm tone, atmospheric |
| `aerial-estate` | Aerial photos | 30-60 sec | Depth perspective, no AI |
| `pool-luxury` | Water features | 1-2 min | Water enhancement |
| `kitchen-bright` | Bright interiors | 20-40 sec | Conservative, clean |
| `bedroom-cozy` | Bedrooms | 30-60 sec | Warm, inviting |
| `bathroom-spa` | Bathrooms | 30-60 sec | Material clarity |
| `courtyard-natural` | Outdoor spaces | 1-2 min | Natural enhancement |
| `custom` | Any | Variable | Manual configuration |

## 🎛️ Command Options

### Basic Options
```bash
--preset, -p       # Preset to use (default: architectural-hero)
--out, -o          # Output directory (default: ./output)
--format, -f       # Output format: jpg, png, tiff (default: tiff)
--bits             # Bit depth for TIFF: 8, 16, 32 (default: 16)
--device           # Device: auto, cpu, cuda, mps (default: auto)
--quality, -q      # Quality: draft, standard, high, ultra (default: high)
```

### Stage Toggles
```bash
--depth-aware / --no-depth        # Enable/disable depth processing
--ai-enhance / --no-ai            # Enable/disable AI enhancement
--material-response / --no-material  # Enable/disable Material Response
--color-grading / --no-grading    # Enable/disable color grading
--finishing / --no-finishing      # Enable/disable finishing
```

### Other Options
```bash
--workers, -w      # Parallel workers for batch (default: 4)
--keep-intermediates  # Keep intermediate outputs
--dry-run          # Preview without processing
```

## 📊 Processing Stages

1. **Depth-Aware** - Depth Anything V2 with CoreML (24-65ms)
2. **AI Enhancement** - SDXL + ControlNet (60-180s)
3. **Material Response** - Physics-based surfaces (10-20ms)
4. **Color Grading** - LUT + tone mapping (15-30ms)
5. **Finishing** - Sharpening, clarity (10-20ms)

## ⚡ Performance Tips

**For Speed:**
```bash
# Disable AI for 5-10x faster processing
python pro_pipeline.py process image.jpg --preset architectural-hero --no-ai

# Use standard quality
python pro_pipeline.py process image.jpg --quality standard

# Increase workers for batch
python pro_pipeline.py batch ./renders --workers 8
```

**For Quality:**
```bash
# Ultra quality (slow)
python pro_pipeline.py process image.jpg --quality ultra

# 16-bit TIFF output
python pro_pipeline.py process image.jpg --format tiff --bits 16

# Keep all intermediate outputs
python pro_pipeline.py process image.jpg --keep-intermediates
```

## 🎯 Use Case Examples

### Architectural Photography
```bash
python pro_pipeline.py process building.jpg \\
  --preset architectural-hero \\
  --format tiff --bits 16 \\
  --out ./client_deliverable
```

### Interior Design Portfolio
```bash
python pro_pipeline.py batch ./interiors \\
  --preset interior-dramatic \\
  --no-ai \\
  --workers 6 \\
  --out ./portfolio
```

### Real Estate Marketing
```bash
python pro_pipeline.py batch ./property_photos \\
  --preset exterior-golden-hour \\
  --format jpg \\
  --workers 8 \\
  --out ./marketing
```

### Aerial Drone Photography
```bash
python pro_pipeline.py process aerial.jpg \\
  --preset aerial-estate \\
  --depth-aware --material-response \\
  --no-ai \\
  --out ./enhanced
```

### Pool/Water Features
```bash
python pro_pipeline.py process pool.jpg \\
  --preset pool-luxury \\
  --material-response --color-grading \\
  --out ./enhanced
```

## 🔧 Troubleshooting

### Slow Processing
```bash
# Check device being used (should see CUDA or MPS)
python pro_pipeline.py process image.jpg --dry-run

# Use CPU explicitly
python pro_pipeline.py process image.jpg --device cpu

# Disable AI enhancement
python pro_pipeline.py process image.jpg --no-ai
```

### Out of Memory
```bash
# Reduce workers
python pro_pipeline.py batch ./renders --workers 2

# Use lower quality
python pro_pipeline.py process image.jpg --quality draft

# Process in smaller batches
python pro_pipeline.py batch ./subset --workers 1
```

### Dependencies Missing
```bash
# Install core package
pip install -e .

# Install ML dependencies
pip install -e ".[ml]"

# Install TIFF support
pip install -e ".[tiff]"
```

## 📖 Documentation

- **Full Guide**: `docs/PRO_PIPELINE_GUIDE.md`
- **Configuration**: `config/pro_pipeline_config.yaml`
- **Examples**: `examples/pro_pipeline_example.py`
- **Tests**: `tests/test_pro_pipeline.py`

## 💡 Pro Tips

1. **Use presets** - They're professionally tuned for specific use cases
2. **Disable AI for speed** - AI adds 2-4 minutes per image
3. **Batch process at night** - Let large batches run overnight
4. **Export as TIFF** - Keep master files in 16-bit TIFF
5. **Also export JPEG** - For client previews and sharing
6. **Use dry-run** - Preview settings before processing
7. **Profile first image** - Time one image to estimate batch duration
8. **Organize by preset** - Save outputs to preset-named folders
9. **Keep intermediates for iteration** - Useful for fine-tuning
10. **Use custom preset** - For unique workflows

## 🎨 Preset Selection Guide

**Choose:**
- `architectural-hero` → Maximum quality, all features
- `interior-dramatic` → Dark interiors needing contrast
- `exterior-golden-hour` → Outdoor shots with warm light
- `aerial-estate` → Drone/aerial photography
- `pool-luxury` → Any scene with water
- `kitchen-bright` → Bright, modern kitchens
- `bedroom-cozy` → Intimate bedroom spaces
- `bathroom-spa` → Luxury bathrooms
- `courtyard-natural` → Outdoor living spaces
- `custom` → Special requirements

## ⏱️ Expected Times (M4 Max)

| Configuration | Time per 4K Image |
|--------------|-------------------|
| All stages (with AI) | 2-5 minutes |
| No AI | 30-60 seconds |
| Depth + Material only | 15-30 seconds |
| Finishing only | 5-10 seconds |

**Batch Throughput:**
- With AI: 50-100 images/hour
- Without AI: 400-600 images/hour
- Selective AI: 200-300 images/hour

## 🆘 Getting Help

- **GitHub Issues**: Report bugs
- **GitHub Discussions**: Ask questions
- **Documentation**: `docs/PRO_PIPELINE_GUIDE.md`
- **Examples**: `examples/pro_pipeline_example.py`

---

**Version**: 1.0.0  
**Last Updated**: November 2025
