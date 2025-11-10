# Elite Architectural Pipeline - Quick Reference

## 🏆 Cutting-Edge Luxury Real Estate Processing

### One-Line Processing

```bash
# Full processing with all features
./process_750_picacho_elite.sh

# Fast mode (depth + tone mapping only)
./process_750_picacho_elite.sh --fast

# Custom output directory
./process_750_picacho_elite.sh --output my_output_dir/
```

### Feature Matrix

| Stage | Technology | Processing Time | Quality Impact |
|-------|-----------|-----------------|----------------|
| **Depth Estimation** | Depth Anything V2 + CoreML | 24-65ms | ★★★★★ |
| **Tone Mapping** | AgX/Filmic HDR | 1-2s | ★★★★★ |
| **Material Response** | Surface-aware enhancement | 2-3s | ★★★★☆ |
| **Color Grading** | LUT stacks (Location + Film) | 1-2s | ★★★★★ |
| **AI Enhancement** | ControlNet + SDXL | 60-90s | ★★★★★ |
| **4x Upscaling** | Real-ESRGAN | 15-25s | ★★★★☆ |

### Quick Commands

```bash
# Single image
python elite_architectural_pipeline.py -i input.tif -o output/

# Batch process
python elite_architectural_pipeline.py -d input_dir/ -o output/

# Preview configuration
python elite_architectural_pipeline.py -i input.tif --dry-run

# Fast processing (no AI/upscale)
python elite_architectural_pipeline.py -i input.tif --no-ai --no-upscale
```

### Output Files

For each input image, you get:
- ✅ **DELIVERY.jpg** - Final delivery file (98% quality JPEG)
- ✅ **MASTER.tiff** - 16-bit master file for archival
- 📊 **processing_report.json** - Detailed processing metadata
- 🎨 **Intermediate stages** - depth, material, graded, ai_enhanced, 4x_upscaled

### Performance

**Full Pipeline**: 90-120s per image on M4 Max
**Throughput**: 30-40 images/hour with all features
**Fast Mode**: 400-600 images/hour (depth + tone mapping only)

### 750 Picacho Optimized Presets

The pipeline auto-detects room types from filenames:
- `Aerial` → Aerial preset (atmospheric haze, wider depth zones)
- `Pool` → Pool preset (cool water tones, crisp clarity)
- `Bathroom`/`Bedroom`/`Great_Room`/`Kitchen` → Interior preset (warm tones, detailed enhancement)

### Documentation

📖 **Full Documentation**: [docs/ELITE_PIPELINE_GUIDE.md](docs/ELITE_PIPELINE_GUIDE.md)
⚙️ **Configuration Reference**: [config/750_picacho_elite_preset.yaml](config/750_picacho_elite_preset.yaml)
🎯 **Pipeline Script**: [elite_architectural_pipeline.py](elite_architectural_pipeline.py)

### Requirements

- Python 3.10+
- 16GB RAM minimum (32GB recommended for 4x upscaling)
- Apple Silicon (M1/M2/M3/M4) or NVIDIA GPU recommended
- Required packages: `tifffile`, `numpy`, `opencv-python`, `pillow`, `torch`
- Optional: `diffusers`, `realesrgan`, `opencolorio` for full features

### Installation

```bash
pip install -r requirements.txt
pip install tifffile imagecodecs
```

### Troubleshooting

**Can't read TIFF**: Install `tifffile` → `pip install tifffile imagecodecs`
**Out of memory**: Use `--no-upscale` or `--fast` mode
**Slow on Mac**: Ensure PyTorch MPS is installed for Apple Silicon

---

**Ready to Process?** Run `./process_750_picacho_elite.sh` to get started!
