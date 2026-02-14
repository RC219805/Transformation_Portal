# APEX Feature Gaps - Quick Reference Card

## Three Gaps, Three Fixes

| # | Gap | Fix | LOC | Risk |
|---|-----|-----|-----|------|
| 1 | 16-bit output broken | Add 16-bit TIFF handoff | ~30 | Medium |
| 2 | MPS acceleration unavailable | Add --v2-device flag | ~5 | Low |
| 3 | ML upscaling stubbed | Create backend registry | ~500 | High |

---

## Implementation Order

```
Phase 1 (Week 1):  Gap 2 - V2 MPS Acceleration
                   ├─ Add --v2-device CLI flag
                   ├─ Update docs
                   └─ Test on macOS + Linux

Phase 2 (Week 2-3): Gap 1 - 16-Bit Output Path
                    ├─ Add 16-bit TIFF handoff (orchestrator.py)
                    ├─ Update manifest schema to v1.7
                    └─ End-to-end 16-bit tests

Phase 3 (Week 4-6): Gap 3 - ML Super-Resolution
                    ├─ Create upscaler registry
                    ├─ Implement bicubic backend
                    ├─ Implement Real-ESRGAN backend
                    ├─ Add --v2-upscaler flag
                    └─ Add ML dependencies + tests
```

---

## Root Causes

### Gap 1: 16-Bit Output
**Location:** `orchestrator.py:846`
```python
# Current (8-bit):
enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)
enhanced_pil.save(enhanced_image_path)  # PNG, 8-bit

# Needed (16-bit when flags enabled):
if emit_master16 or emit_upscaled16:
    enhanced_uint16 = (np.clip(working_image, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(enhanced_image_path, enhanced_uint16, ...)  # TIFF, 16-bit
```

### Gap 2: V2 MPS
**Location:** `__main__.py` (missing flag)
```python
# Add this:
v2_device: str = typer.Option("cpu", "--v2-device", help="...")

# Wire to config:
config = EnhanceConfig(v2_device=v2_device)
```

### Gap 3: ML Upscaling
**Location:** `upscaling.py:141`
```python
# Current (hardcoded):
def _load_upscaler(self, device: str):
    self._upscaler = "bicubic"  # Always bicubic

# Needed (registry-based):
def _load_upscaler(self, device: str):
    self._upscaler = UpscalerRegistry.get(self.backend, device)
```

---

## Testing Commands

### Gap 1: 16-Bit Output
```bash
# Test 16-bit path
python -m transformation_portal.lux_depth_v3 \
  --input-dir input/ --output-dir output_16bit/ \
  --quality-tier apex \
  --emit-master16 on --emit-upscaled16 on

# Verify:
file output_16bit/v2/*_master16.tif  # Should be "16-bit"
```

### Gap 2: V2 MPS
```bash
# Test MPS acceleration (macOS)
python -m transformation_portal.lux_depth_v3 \
  --input-dir input/ --output-dir output_mps/ \
  --v2-device mps

# Verify:
grep "device: mps" output_mps/logs/v2_*.log
```

### Gap 3: ML Upscaling
```bash
# Test Real-ESRGAN
pip install -e .[ml]
python scripts/setup/download_upscaler_weights.py

python -m transformation_portal.lux_depth_v3 \
  --input-dir input/ --output-dir output_realesrgan/ \
  --v2-upscaler realesrgan

# Verify:
# Compare output_realesrgan vs bicubic for sharpness
```

---

## Files to Modify

### Gap 1 (16-Bit)
- `src/transformation_portal/lux_depth_v3/orchestrator.py` (handoff logic)
- `src/transformation_portal/lux_depth_v3/manifest.py` (schema v1.7)

### Gap 2 (MPS)
- `src/transformation_portal/lux_depth_v3/__main__.py` (CLI flag)
- `src/transformation_portal/lux_depth_v3/README.md` (docs)

### Gap 3 (Upscaling)
- `src/transformation_portal/upscaling/__init__.py` (new)
- `src/transformation_portal/upscaling/registry.py` (new)
- `src/transformation_portal/upscaling/backends/bicubic.py` (new)
- `src/transformation_portal/upscaling/backends/realesrgan.py` (new)
- `src/transformation_portal/stage_graph/stages/upscaling.py` (modify)
- `src/transformation_portal/lux_depth_v3/__main__.py` (CLI flag)
- `requirements/ml.in` (add basicsr, realesrgan)

---

## Performance Impact

### Gap 1: 16-Bit
- **Storage:** 2x larger files (uint16 vs uint8)
- **Quality:** Preserves tonal gradations for print/archival
- **Speed:** Minimal impact (<5% slower I/O)

### Gap 2: MPS
- **Speed:** 2-3x faster V2 enhancement on Apple Silicon
- **Quality:** Identical (device-agnostic algorithms)
- **Compatibility:** Auto-fallback to CPU if unavailable

### Gap 3: Real-ESRGAN
- **Quality:** Sharper upscaling, better detail preservation
- **Speed:** ~500ms/image (vs ~50ms bicubic)
- **Requirements:** +basicsr +realesrgan (~100MB total)

---

## Success Criteria

| Gap | Metric | Target |
|-----|--------|--------|
| Gap 1 | 16-bit files generated | `master16.tif` + `upscaled16.tif` exist |
| Gap 1 | Bit depth preserved | TIFF tags show 16 bits/sample |
| Gap 1 | Golden Path unchanged | 8-bit PNG handoff when flags off |
| Gap 2 | MPS speedup | 2-3x faster on M-series chips |
| Gap 2 | Fallback works | No crash on non-MPS platforms |
| Gap 2 | CLI exposure | `--v2-device` flag accepted |
| Gap 3 | Registry functional | `UpscalerRegistry.get()` returns backends |
| Gap 3 | Real-ESRGAN quality | Sharper than bicubic (visual test) |
| Gap 3 | Fallback works | Bicubic if ML deps missing |

---

## Common Issues & Solutions

### Gap 1: "Still getting 8-bit output"
- Check: `--emit-master16 on` flag set?
- Check: Materials V3 enabled? (if disabled, uses original input)
- Check: V2 log shows 16-bit input detected?

### Gap 2: "MPS not working"
- Check: macOS with Apple Silicon?
- Check: V2 subprocess log shows device?
- Check: Fallback to CPU happens gracefully?

### Gap 3: "Real-ESRGAN import error"
- Solution: `pip install -e .[ml]`
- Check: `python -c "import realesrgan"`
- Fallback: Should use bicubic automatically

---

## Documentation Links

- **Full Plan:** `docs/architecture/APEX_FEATURE_GAPS_IMPLEMENTATION_PLAN.md`
- **Executive Summary:** `docs/architecture/APEX_FEATURE_GAPS_SUMMARY.md`
- **Verification Script:** `scripts/verify_apex_gaps.py`
- **Governance:** `docs/architecture/agent_governance.md`

---

## Contact

- **Architect Approval:** ✅ Approved 2026-02-14
- **Implementation:** Delegated to @transformation-portal-specialist
- **Escalation:** Required if contract changes needed beyond plan

---

**Last Updated:** 2026-02-14
**Status:** Ready for Implementation
