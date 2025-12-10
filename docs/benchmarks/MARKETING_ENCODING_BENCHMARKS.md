# Marketing Export Encoding Benchmarks

## PNG Compression Level Benchmarks (M1.1)

**Date**: 2025-12-10  
**Dataset**: Pool, Aerial, GreatRoom (750 Picacho)  
**Method**: Median-based comparison (15 runs total, 3 images × 5 levels)  
**Tool**: `scripts/analyze_marketing_export.py`

---

## Results Summary

### Median Performance (across all 3 images)

| Level | Median Time | Δ vs Level 6 | Median Size | Δ vs Level 6 | Verdict |
|-------|-------------|--------------|-------------|--------------|---------|
| **0** | 5.5s | **-92.7%** ⚡ | 928 MB | +184.7% ❌ | Too large |
| **1** | 12.1s | **-84.0%** ⚡ | 375 MB | +15.0% ✅ | **RECOMMENDED** |
| **3** | 23.8s | **-68.5%** ⚡ | 352 MB | +7.8% ✅ | **RECOMMENDED** |
| **6** | 75.7s | baseline | 326 MB | baseline | Old default |
| **9** | 417.0s | +450.7% 🐌 | 318 MB | -2.6% | **AVOID** |

---

## Key Findings

### ✅ Level 1 (NEW DEFAULT)
- **Time**: 12.1s (84% faster than level 6)
- **Size**: 375 MB (+15% vs baseline)
- **Savings**: ~63.6 seconds per image
- **Verdict**: Best balance of speed and size

### ✅ Level 3 (CONSERVATIVE OPTION)
- **Time**: 23.8s (68% faster than level 6)
- **Size**: 352 MB (+7.8% vs baseline)
- **Savings**: ~51.9 seconds per image

### ❌ Level 9 (NEVER USE)
- **Time**: 417.0s (~7 minutes!)
- **Size**: 318 MB (-2.6% vs baseline)
- **Verdict**: 7× slower to save only 2.6% size

---

## Recommendation

**Deploy level 1 as default immediately.**

The original hypothesis of 20-30s savings has been exceeded by 2-3×, delivering 60-70s of real-world performance improvement.
