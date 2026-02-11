# Phase 2 Quick Reference

**Status:** Implementation Guide
**Date:** 2026-02-11
**Related:** PHASE2_IMPLEMENTATION_PLAN.md, ADR-023, ADR-026

---

## One-Page Overview

### What Phase 2 Adds

```
Phase 1 (COMPLETE):     Phase 2 (THIS PLAN):
┌─────────────────┐     ┌──────────────────────┐
│ Linear Ingest   │ ──▶ │ SAM2 Segmentation    │
│ Depth Ensemble  │     │ MaterialGAN PBR      │
│ (DA3 + DepthPro)│     │ 3D Gaussian Splatting│
└─────────────────┘     └──────────────────────┘
```

### Critical Constraints (Non-Negotiable)

| Constraint | Enforcement | Violation = |
|------------|-------------|-------------|
| **No `lux_depth_v3` imports** | AST checker (CI) | ❌ Merge blocked |
| **HF revisions pinned** (stable) | Validation script (CI) | ❌ Merge blocked |
| **gamma=1.0 always** | Contract validation | ❌ RuntimeError |
| **EXR fail-loud** | Preflight check | ❌ Clear error message |

---

## Namespace Map

```
src/transformation_portal/
├── spatial_ai/              # ✅ Phase 2 code goes here
│   ├── ingest/              # Phase 1 (done)
│   ├── segmentation/        # Phase 2.1 (SAM2)
│   ├── materials/           # Phase 2.2 (MaterialGAN)
│   ├── reconstruction/      # Phase 2.3 (3DGS)
│   └── orchestration/       # Phase 2.4 (pipeline)
│
├── core/                    # ✅ Shared utilities OK
│   └── geometry/            # NEW: camera, transforms, depth_utils
│
└── lux_depth_v3/            # ❌ DO NOT IMPORT from here
```

---

## Import Cheat Sheet

```python
# ✅ ALLOWED: Phase 2 → Phase 1
from transformation_portal.spatial_ai.ingest import LinearDecoder

# ✅ ALLOWED: Phase 2 → core
from transformation_portal.core.geometry.camera import CameraIntrinsics

# ✅ ALLOWED: Phase 2 internal
from transformation_portal.spatial_ai.segmentation import SAM2Backend

# ❌ FORBIDDEN: Phase 2 → lux_depth_v3
from transformation_portal.lux_depth_v3.utils import anything  # CI FAIL
```

---

## Preset Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CREATE in experimental/                                   │
│    - Placeholders OK: revision: "NEEDS_VERIFICATION_..."     │
│    - CI skips HF validation                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. DEVELOP & TEST                                            │
│    - Iterate quickly with placeholder                        │
│    - Run local tests                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. VERIFY HF commit hash                                     │
│    - Visit HuggingFace /commits/main                         │
│    - Copy latest stable commit SHA (40 chars)                │
│    - Update preset: revision: "abc123..."                    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. PROMOTE to stable                                         │
│    - Move from experimental/ to presets/                     │
│    - Run: python scripts/validation/validate_hf_revisions.py │
│    - CI validates on merge                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## OpenEXR Preflight Pattern

```python
# Add to ALL scripts that use emit_exr=True
def check_exr_support():
    """Verify OpenEXR/Imath before using emit_exr=True."""
    try:
        import OpenEXR
        import Imath
        return True
    except ImportError:
        return False

# Usage A: Default to safe option
emit_exr = check_exr_support()

# Usage B: Fail with friendly message
if args.emit_exr and not check_exr_support():
    raise RuntimeError(
        "EXR requested but OpenEXR not installed.\n"
        "Install: pip install OpenEXR Imath"
    )
```

---

## Lane-Based Strictness

```yaml
# Development lane (fast iteration)
# config/presets/experimental/spatial_ai_dev.yaml
ingest:
  strict_ingest: false  # Allow 8-bit
  emit_exr: false       # Skip EXR

# Research lane (scientific rigor)
# config/presets/experimental/spatial_ai_research.yaml
ingest:
  strict_ingest: true   # Reject 8-bit
  emit_exr: true        # Require EXR
```

---

## Testing Checklist

Before submitting PR:

- [ ] All unit tests pass (≥85% coverage)
- [ ] Integration tests pass
- [ ] Isolation checker passes: `python scripts/security/verify_pipeline_isolation.py`
- [ ] HF revision validator passes: `python scripts/validation/validate_hf_revisions.py`
- [ ] OpenEXR preflight checks added
- [ ] Documentation updated
- [ ] No imports from `lux_depth_v3` (grep check)
- [ ] Experimental preset created (if new model)

---

## Common Mistakes to Avoid

### ❌ Don't: Import from lux_depth_v3

```python
# WRONG
from transformation_portal.lux_depth_v3.utils import normalize_depth
```

**Fix:** Copy small helpers or use `core/geometry/depth_utils.py`

---

### ❌ Don't: Use HF placeholders in stable presets

```yaml
# WRONG (in config/presets/sam2.yaml)
revision: "NEEDS_VERIFICATION_0000..."  # CI FAIL
```

**Fix:** Verify SHA, or keep in `experimental/`

---

### ❌ Don't: Skip OpenEXR preflight

```python
# WRONG
result = decoder.decode(path, emit_exr=True)  # Crashes if missing
```

**Fix:** Add preflight check (see pattern above)

---

### ❌ Don't: Weaken strict_ingest globally

```python
# WRONG
LinearDecoder(strict_ingest=True)  # Now dev fixtures fail
```

**Fix:** Use lane-based presets (dev vs research)

---

### ❌ Don't: Work around AST isolation checker

```python
# WRONG
import importlib
lux = importlib.import_module('transformation_portal.lux_depth_v3.utils')
```

**Fix:** Respect isolation. Governance exists for reasons.

---

## Implementation Sequence

| Week | Phase | Deliverable | Owner |
|------|-------|-------------|-------|
| 1 | 2.1 SAM2 | Segmentation working | Specialist |
| 2 | 2.2 MaterialGAN | PBR textures working | Specialist |
| 3 | 2.3 3DGS | Reconstruction working | Specialist |
| 4 | 2.4 Orchestration | E2E pipeline working | Specialist |
| 5 | 2.5 Hardening | Architect approval, merge ready | Architect + Specialist |

---

## Key Metrics

| Metric | Target | Check |
|--------|--------|-------|
| Test coverage | ≥85% | `pytest --cov` |
| CI green | 100% | All checks passing |
| Isolation | 100% | AST checker |
| HF compliance | 100% stable | Validation script |
| Performance | <5% regression | Benchmark suite |

---

## Quick Commands

```bash
# Check isolation compliance
python scripts/security/verify_pipeline_isolation.py

# Validate HF revisions
python scripts/validation/validate_hf_revisions.py

# Run Phase 2 tests only
pytest tests/spatial_ai/ -v

# Full test suite
pytest tests/ -v --cov=src/transformation_portal/spatial_ai

# Run spatial AI pipeline (dev mode)
transformation_portal spatial-ai process \
    --preset experimental/spatial_ai_dev \
    --input scene.jpg \
    --output output/

# Run spatial AI pipeline (research mode)
transformation_portal spatial-ai process \
    --preset experimental/spatial_ai_research \
    --input scene.tiff \
    --output output/ \
    --strict-mode
```

---

## Escalation Path

| Issue | Action |
|-------|--------|
| **ADR-023 violation** | Stop. Fix imports. Re-run AST checker. |
| **HF revision failure** | Verify SHA manually. Update preset. Re-validate. |
| **OpenEXR missing** | Add preflight check. Document dependency. |
| **Performance regression** | Profile. Optimize. Or document trade-off. |
| **Architectural uncertainty** | Escalate to Architect. Don't guess. |

---

## Resources

- **Full Plan:** `docs/architecture/PHASE2_IMPLEMENTATION_PLAN.md`
- **ADR-023:** Pipeline isolation policy
- **ADR-026:** APEX Research Ultra architecture
- **Phase 1.1 Summary:** Session file `phase1.1_completion_summary.md`
- **Compatibility Checklist:** Session file `phase2_compatibility_checklist.md`
- **Governance Policy:** `docs/architecture/agent_governance.md`

---

**Last Updated:** 2026-02-11
**Status:** Implementation guide (active)
**Maintainer:** Transformation Portal Architect
