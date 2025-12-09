# Processing Recommendations Quick Reference

**Last Updated**: 2025-12-08  
**Based On**: 750 Picacho empirical processing analysis

## TL;DR

✅ **Safe Zone**: 12-24MP images (80-163MB 16-bit TIFF)  
⚠️ **Warning**: 25-35MP (may OOM on MPS)  
❌ **Unsafe**: >35MP (use CPU fallback)  
💾 **Disk Space**: 2.5GB minimum per 20MP image  
⏱️ **Expected Time**: ~6 seconds per megapixel

---

## Decision Matrix

| Resolution | File Size | MPS Safe? | Processing Time | Disk Space | Recommendation |
|-----------|-----------|-----------|----------------|------------|----------------|
| 12MP | 80MB | ✅ Yes | 1-2 min | 1.6GB | Optimal batch processing |
| 20MP | 140MB | ✅ Yes | 2-3 min | 2.5GB | Recommended for production |
| 24MP | 163MB | ✅ Yes | 2-4 min | 3.0GB | Maximum safe MPS size |
| 35MP | 240MB | ⚠️ Risky | 3-5 min | 4.0GB | Use CPU fallback |
| 48MP | 341MB | ❌ OOM | 5-8 min* | 6.0GB | Requires CPU + tiling |

*With CPU and tile_size=256

---

## Command Templates

### Standard Batch (Recommended)
```bash
# For 12-24MP images
lux-depth-v2 \
  --input-dir input_images/property/ \
  --output-dir output_property/ \
  --preset photo_realistic \
  --upscale 4 \
  --device auto
```

### Large Format (>24MP)
```bash
# For 25-48MP images - CPU mode
lux-depth-v2 \
  --input-dir input_images/large/ \
  --output-dir output_large/ \
  --preset photo_realistic \
  --upscale 2 \
  --device cpu \
  --tile 256
```

### Low Disk Space (<20GB available)
```bash
# Process without 4x upscale to save space
lux-depth-v2 \
  --input-dir input_images/ \
  --output-dir output/ \
  --preset photo_realistic \
  --save-upscaled false \
  --save-marketing-png false
```

### High Quality (Optimal Conditions)
```bash
# Maximum quality with ample resources
lux-depth-v2 \
  --input-dir input_images/ \
  --output-dir output/ \
  --preset photo_realistic \
  --upscale 4 \
  --upscaler-backend torch \
  --tile 512 \
  --precision fp16 \
  --material-strength 0.7
```

---

## Pre-Flight Checklist

Before starting a batch:

1. **Check Input Files**
   ```bash
   identify -format "%f: %wx%h (%[fx:w*h/1000000]MP) %z-bit\n" input_images/*.tif
   ```

2. **Check Available Disk Space**
   ```bash
   df -h . | tail -1
   # Need: (num_images × 2.5GB) for 20MP images
   ```

3. **Estimate Processing Time**
   ```
   Time = (total_megapixels × 6 seconds) / 60 minutes
   Example: 4 images × 20MP = 80MP × 6s = 480s = 8 minutes
   ```

4. **Verify System Resources**
   ```bash
   # Check MPS availability
   python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
   
   # Check memory pressure
   memory_pressure  # macOS only
   ```

---

## Troubleshooting Quick Fixes

### Issue: MPS Out of Memory
```bash
# Solution: Use CPU mode
lux-depth-v2 --input-dir ... --device cpu --upscale 2
```

### Issue: Disk Full During Processing
```bash
# Solution 1: Free space
rm -rf output_*/750Picacho_*_upscaled16.tif.tmp

# Solution 2: Skip upscaling
lux-depth-v2 --input-dir ... --save-upscaled false
```

### Issue: Processing Too Slow
```bash
# Check disk usage
df -h .

# If >85% full, free space or reduce batch size
# Process 1-2 images at a time instead of 4-6
```

### Issue: Poor Quality Output
```bash
# Check input bit depth
identify -verbose input.tif | grep "Depth:"

# Should be 16-bit. If 8-bit, re-export from source
```

---

## Batch Size Guidelines

| Available Disk | Image Size | Max Batch Size |
|---------------|-----------|----------------|
| 10-20GB | 12MP | 6 images |
| 10-20GB | 20MP | 4 images |
| 20-50GB | 12MP | 12 images |
| 20-50GB | 20MP | 8 images |
| >50GB | Any ≤24MP | No limit |

**Rule of Thumb**: `available_GB / 2.5GB = max_20MP_images`

---

## Expected Output Sizes

| Input | Master TIFF | Upscaled 4x TIFF | Marketing PNG | Total |
|-------|------------|-----------------|---------------|-------|
| 12MP (80MB) | 59MB | 921MB | 159MB | 1.1GB |
| 20MP (140MB) | 103MB | 1.6GB | 322MB | 2.1GB |
| 24MP (163MB) | 123MB | 1.9GB | 414MB | 2.4GB |

**Multiplier**: Expect ~15x input file size for complete output set

---

## Quality Validation

After processing, check the report files:
```bash
cat output_dir/*_report.json | jq '{file: .image, quality: {color: .ai_color_diff, luma: .ai_luma_diff}}'
```

**Expected Values** (750 Picacho baseline):
- Color diff: <0.004 (excellent), <0.06 (acceptable)
- Luma diff: <0.002 (excellent), <0.06 (acceptable)

---

## System-Specific Notes

### Apple M4 Max (64GB)
- **MPS Limit**: 24MP safe, 35MP risky, 48MP+ fails
- **Optimal**: 4-6 images at 20MP in parallel
- **Expected**: 6s/MP processing rate

### Apple M1/M2/M3 (32GB)
- **MPS Limit**: 20MP safe, 24MP risky
- **Reduce tile size**: `--tile 256` for large images
- **Expected**: 8-10s/MP processing rate

### Intel/AMD CPU (No MPS)
- **Use**: `--device cpu` always
- **Tile**: `--tile 256` for >24MP images
- **Expected**: 60-120s/MP (10-20x slower than MPS)

---

## Production Workflow

1. **Intake**: Verify all inputs are 16-bit TIFF, sRGB
2. **Pre-check**: Validate disk space (images × 2.5GB)
3. **Batch**: Group by size (all 12MP, all 20MP, etc.)
4. **Process**: Run lux-depth-v2 with appropriate preset
5. **Validate**: Check AI quality scores in reports
6. **Deliver**: Master TIFF + marketing PNG for client

---

For detailed analysis, see [INPUT_SPECIFICATIONS.md](INPUT_SPECIFICATIONS.md)
