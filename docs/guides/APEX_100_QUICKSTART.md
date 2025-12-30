# 🚀 100% APEX Quick Start

**Status:** Production-Ready ✅  
**Commit:** 082e493

---

## ⚡ **TL;DR**

Add one flag to achieve 100% APEX quality:

```bash
--materials-v2-backend segformer
```

**That's it.** MaterialsV2 will now use AI-based SegFormer instead of heuristic rules.

---

## 📋 **Full APEX Command**

```bash
lux-depth-v2 \
  --input INPUT.tif \
  --output-dir OUTPUT/ \
  --preset interior_luxury_max_quality \
  --quality-tier apex \
  --device auto \
  --precision fp32 \
  --tile 1024 --tile-pad 32 \
  --seg-backend segformer \
  --seg-long-side 2048 \
  --materials-v2 \
  --materials-v2-backend segformer \
  --max-segmentation-side 2048 \
  --edge-refinement --refinement-preset aggressive \
  --cache-masks --model-cache --depth-cache
```

---

## 🎯 **What You Get**

- **Depth:** FP32 precision, 1024px tiles, guided filter
- **Scene Seg:** SegFormer-B5 @ 2048px
- **Materials:** SegFormer AI backend (not heuristic)
- **Refinement:** CLAHE + guided filter + edge snap
- **Speed:** 47s cold, 6s cached

---

## 📊 **Quality Impact**

| Scene Type | Improvement |
|------------|-------------|
| Kitchen/interior | +5-8% |
| Glass-heavy | +15-20% |
| Multi-material | +10-15% |
| Metal/stone | +8-12% |

---

## ✅ **Validation**

**Check your logs for these lines:**

```
MaterialsV2Engine initialized | backend=segformer
Loading segmentation model: segformer
```

If you see `backend=heuristic`, you're **not** at 100% APEX.

---

## 🔄 **Cache Behavior**

**First run (cold):**
- MaterialsV2 loads SegFormer model
- Generates AI-based material masks
- Saves to `.mask_cache/`
- Time: ~47 seconds

**Second run (warm):**
- Loads cached SegFormer masks
- Skips model loading
- Time: ~6 seconds

**Cache key includes backend** → No collision between heuristic/segformer.

---

## 📚 **Full Documentation**

- **Validation:** `docs/guides/APEX_100_ACHIEVED.md`
- **Production:** `docs/deployment/APEX_100_PRODUCTION_READY.md`
- **Backend Analysis:** `APEX_BACKEND_ANALYSIS.md`

---

## 🐛 **Troubleshooting**

### **Problem:** Still shows `backend=heuristic`

**Solution:** You didn't add `--materials-v2-backend segformer` flag.

### **Problem:** Slow (47s every time)

**Solution:** Add `--cache-masks --model-cache --depth-cache` flags.

### **Problem:** Out of memory

**Solution:** Reduce `--max-segmentation-side 2048` to `1024`.

---

*Updated: 2025-12-30*  
*Commit: 082e493*
