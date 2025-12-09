# Transformation Portal: Quick Reference Card

## 🚦 Safe Processing Limits (Apple M4 Max 64GB)

| Resolution | File Size | MPS Safe? | Time | Disk | Command |
|-----------|-----------|-----------|------|------|---------|
| ≤12MP | ≤80MB | ✅ Safe | 1-2min | 1.6GB | Default |
| 13-24MP | 81-163MB | ✅ Safe | 2-3min | 2.5GB | Default |
| 25-35MP | 164-240MB | ⚠️ Risk | 3-5min | 4GB | `--device auto` monitor |
| 36-48MP | >240MB | ❌ OOM | 5-8min | 6GB | `--device cpu` |
| >48MP | >341MB | ❌ Fail | N/A | N/A | `--device cpu --tile 256` |

## 📊 Critical Numbers

- **MPS Memory Formula**: `MP × 1.25GB` (for 4x upscale)
- **Disk Space Formula**: `Source_MB × 15x` (total output)
- **Processing Rate**: `6 seconds per megapixel` (optimal)
- **Quality Standard**: `<0.004 color diff` (excellent)

## 🎯 Pre-Flight Checklist

```bash
# 1. Check input specs
identify -format "%f: %wx%h (%[fx:w*h/1000000]MP) %z-bit\n" *.tif

# 2. Check disk space (need 2.5GB per 20MP image)
df -h . | tail -1

# 3. Calculate requirements
python tools/calculate_processing_requirements.py --megapixels 20 -n 4

# 4. Run processing
lux-depth-v2 --input-dir input/ --output-dir output/ --preset photo_realistic
```

## ⚡ Common Commands

**Standard Batch** (12-24MP):
```bash
lux-depth-v2 --input-dir renders/ --output-dir out/ --preset photo_realistic
```

**Large Format** (>24MP):
```bash
lux-depth-v2 --input-dir renders/ --output-dir out/ --device cpu --upscale 2
```

**Low Disk Space**:
```bash
lux-depth-v2 --input-dir renders/ --output-dir out/ --save-upscaled false
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| MPS OOM | `--device cpu` |
| Disk full | Free space or `--save-upscaled false` |
| Slow (>5min) | Check disk usage: `df -h .` |
| Poor quality | Verify 16-bit input: `identify -verbose input.tif` |

## 📚 Documentation

- **Full Guide**: `docs/INPUT_SPECIFICATIONS.md`
- **Workflows**: `docs/PROCESSING_RECOMMENDATIONS.md`
- **Calculator**: `tools/calculate_processing_requirements.py --help`
- **Analysis**: `750_PICACHO_ANALYSIS_SUMMARY.md`

---

**Updated**: 2025-12-08 | **Source**: 750 Picacho empirical analysis
