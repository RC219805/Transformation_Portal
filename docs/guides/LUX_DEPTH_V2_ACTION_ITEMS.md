# Lux Depth V2 - Action Items Checklist

**Date**: December 25, 2025  
**Priority**: Immediate → Sprint PR-1 → Post-Deployment

---

## ✅ Immediate Actions (30 Minutes)

### Action 1: Fix Default Upscaler Backend (5 minutes)

**File**: `lux_depth_v2/config.py`

**Current**:
```python
upscaler_backend: str = "realesrgan"  # Line ~80
```

**Change To**:
```python
upscaler_backend: str = "torch"  # Safe default, CVE-free
```

**Verification**:
```bash
cd "$(git rev-parse --show-toplevel)/lux_depth_v2"
grep -n "upscaler_backend.*=" config.py
# Should show: upscaler_backend: str = "torch"
```

**Commit Message**:
```
fix(lux_depth_v2): Change default upscaler from realesrgan to torch

- Removes vulnerable basicsr dependency from default code path
- Aligns with SECURITY.md recommendations
- Legacy --upscaler-backend realesrgan still works (maps to torch with warning)

Refs: CVE-2024-27763, SECURITY.md line 14-43
```

---

### Action 2: Clean Vulnerable Packages (1 minute)

**Command**:
```bash
cd "$(git rev-parse --show-toplevel)"
pip uninstall basicsr realesrgan gfpgan -y
```

**Verification**:
```bash
pip show basicsr
# Should output: WARNING: Package(s) not found: basicsr

pip show realesrgan
# Should output: WARNING: Package(s) not found: realesrgan
```

**Why**: Removes CVE-2024-27763 vulnerable packages from environment

---

### Action 3: Document Immediate Fixes (5 minutes)

**File**: Create `lux_depth_v2/IMMEDIATE_FIXES_APPLIED.md`

**Content**:
```markdown
# Immediate Fixes Applied - December 25, 2025

## 1. Default Upscaler Backend ✅
- Changed `config.py` default from `realesrgan` to `torch`
- Removed vulnerable code path
- Verified with grep/tests

## 2. Environment Cleanup ✅
- Uninstalled basicsr, realesrgan, gfpgan
- Verified removal with pip show
- Clean security posture

## Next: Sprint PR-1 (Depth Contract + Cache Fix)
```

---

### Action 4: Update Feature Freeze Exception (10 minutes)

**File**: `lux_depth_v2/FEATURE_FREEZE.md`

**Add Section** (after line 80):
```markdown
## Approved Exceptions

### Exception #1: Sprint PR-1 (Depth Contract + Cache Fix)
**Approved**: December 25, 2025  
**Justification**: Critical production quality risk (silent degradation)  
**Scope**:
- DepthMode enum (REQUIRED/AUTO/OPTIONAL)
- Auto-depth generation with caching
- Materials V2 cache type safety fix
- Depth provenance tracking

**Timeline**: 1 week (Dec 25 - Jan 1)  
**Risk**: Low (additive changes, backward-compatible)  
**Approver**: Repo Maintainer (RC219805)
```

---

### Action 5: Commit Immediate Fixes (10 minutes)

**Commands**:
```bash
cd "$(git rev-parse --show-toplevel)"

# Stage changes
git add lux_depth_v2/config.py
git add lux_depth_v2/IMMEDIATE_FIXES_APPLIED.md
git add lux_depth_v2/FEATURE_FREEZE.md

# Commit
git commit -m "fix(lux_depth_v2): Apply immediate security and config fixes

- Change default upscaler from realesrgan to torch (CVE-free)
- Document environment cleanup (basicsr, realesrgan, gfpgan removed)
- Approve Sprint PR-1 as feature freeze exception

This prepares lux_depth_v2 for Sprint PR-1 implementation next week.

Refs: LUX_DEPTH_V2_EVALUATION.md, CVE-2024-27763"

# Push (if on feature branch)
git push origin <branch-name>
```

---

## 🔧 Sprint PR-1: Depth Contract + Cache Fix (1 Week)

### Day 1-2: DepthMode Implementation

**Files to Modify**:
1. `lux_depth_v2/config.py` - Add DepthMode enum
2. `lux_depth_v2/depth_cache_manager.py` - New file (like MaskCacheManager)
3. `lux_depth_v2/pipeline.py` - Integrate auto-depth generation

**Code Skeleton**:
```python
# config.py
from enum import Enum

class DepthMode(Enum):
    REQUIRED = "required"  # Fail if depth missing
    AUTO = "auto"          # Auto-generate if missing
    OPTIONAL = "optional"  # Allow uniform fallback

@dataclass
class DepthConfig:
    mode: DepthMode = DepthMode.AUTO
    model_name: str = "depth_anything_v2_vitl"
    cache_dir: Optional[Path] = None
    cache_enabled: bool = True
    confidence_threshold: float = 0.70
    tile_size: int = 512
    tile_overlap: int = 64

@dataclass  
class PipelineConfig:
    # ... existing fields ...
    depth: DepthConfig = field(default_factory=DepthConfig)
```

**Tests**:
```python
# tests/test_depth_contract.py
def test_depth_required_fails_without_depth():
    cfg = PipelineConfig()
    cfg.depth.mode = DepthMode.REQUIRED
    # Should raise FileNotFoundError

def test_depth_auto_generates_and_caches():
    cfg = PipelineConfig()
    cfg.depth.mode = DepthMode.AUTO
    # First run: generates depth
    # Second run: cache hit

def test_depth_optional_allows_fallback():
    cfg = PipelineConfig()
    cfg.depth.mode = DepthMode.OPTIONAL
    # Should succeed with uniform weights
```

---

### Day 3-4: Auto-Depth Integration

**Files to Modify**:
1. `lux_depth_v2/pipeline.py` - Add `_ensure_depth()` method
2. `lux_depth_v2/depth_inference.py` - Already exists, wire in
3. `lux_depth_v2/depth_cache_manager.py` - Implement caching

**Code Skeleton**:
```python
# pipeline.py
def _ensure_depth(self, rgb: np.ndarray, stem: str) -> np.ndarray:
    """Ensure depth map exists (load, generate, or fail)."""
    # 1. Try loading from depth_dir
    if self.depth_dir:
        depth_path = _find_depth(self.depth_dir, stem)
        if depth_path:
            return io_utils.read_depth_u16(depth_path)
    
    # 2. Check cache
    if self.cfg.depth.cache_enabled:
        cached = self.depth_cache.get(stem, rgb)
        if cached is not None:
            return cached
    
    # 3. Auto-generate or fail
    if self.cfg.depth.mode == DepthMode.REQUIRED:
        raise FileNotFoundError(f"Depth required but missing: {stem}")
    elif self.cfg.depth.mode == DepthMode.AUTO:
        depth = self._generate_depth(rgb)
        self.depth_cache.set(stem, rgb, depth)
        return depth
    else:  # OPTIONAL
        return None  # Caller will use uniform fallback
```

---

### Day 5: Materials V2 Cache Fix

**Files to Modify**:
1. `lux_depth_v2/cache_manager.py` - Add `CachedSegmentationResult` adapter
2. `lux_depth_v2/pipeline.py` - Fix cache hit/miss type mismatch

**Code Skeleton**:
```python
# cache_manager.py
@dataclass
class CachedSegmentationResult:
    """Type-safe wrapper for cached segmentation results."""
    masks: Dict[str, np.ndarray]
    confidence: float
    quality_flags: Dict[str, bool]
    cache_hit: bool = False
    
    @classmethod
    def from_dict(cls, data: dict) -> "CachedSegmentationResult":
        """Convert cached dict to SegmentationResult-compatible object."""
        return cls(
            masks=data["masks"],
            confidence=data["confidence"],
            quality_flags=data["quality_flags"],
            cache_hit=True
        )
```

**Tests**:
```python
# tests/test_materials_v2_cache.py
def test_cache_roundtrip():
    # Save result
    result = SegmentationResult(masks={...}, confidence=0.85)
    cache.set(key, result)
    
    # Load result
    cached = cache.get(key)
    assert isinstance(cached, CachedSegmentationResult)
    assert cached.confidence == 0.85
    assert cached.cache_hit == True
```

---

### Day 6-7: Validation & Testing

**Dataset**: 750 Picacho kitchen (or equivalent)

**Test Scenarios**:
1. **Depth AUTO** - First run generates, second run caches
2. **Depth REQUIRED** - Fails fast without depth
3. **Depth OPTIONAL** - Allows uniform fallback
4. **Materials V2 Cache** - Hit/miss produce identical output

**Validation Script**:
```bash
#!/bin/bash
# validate_sprint_pr1.sh

# Test 1: AUTO mode (first run)
lux-depth-v2 --input test.tif --preset interior_luxury --output-dir out1
jq '.depth.source' out1/*_report.json
# Expected: "generated"

# Test 2: AUTO mode (cache hit)
lux-depth-v2 --input test.tif --preset interior_luxury --output-dir out2
jq '.depth.source' out2/*_report.json
# Expected: "cache"

# Test 3: REQUIRED mode (should fail)
lux-depth-v2 --input test.tif --preset apex_quality --output-dir out3
# Expected: FileNotFoundError

# Test 4: OPTIONAL mode (fallback)
lux-depth-v2 --input test.tif --preset ci_baseline --output-dir out4
jq '.depth.source' out4/*_report.json
# Expected: "uniform_fallback"
```

**Acceptance Criteria**:
- [ ] Zero silent depth fallbacks in production presets
- [ ] Cache hit rate >90% on repeated runs
- [ ] Auto-depth confidence >0.70 for 95% of images
- [ ] Materials V2 cache hit/miss type-safe
- [ ] Performance regression <1.5s overhead

---

## 📊 Post-Deployment (Week 3+)

### Monitoring & Metrics

**Setup Prometheus/Grafana** (optional but recommended):
```yaml
# prometheus.yml
- job_name: 'lux-depth-v2-service'
  static_configs:
    - targets: ['localhost:8088']
  metrics_path: '/metrics'
```

**Key Metrics**:
- Depth cache hit rate (target: >90%)
- Auto-depth generation time (target: <1.5s)
- Materials V2 cache hit rate (target: >90%)
- Service mode request rate
- Error rate (target: <1%)

---

### Artist Training

**Topics**:
1. New presets (`ci_baseline`, `production_standard`, `production_ultra`)
2. Depth auto-generation (when it happens, cache benefits)
3. Report JSON (depth provenance, cache_hit flags)
4. Troubleshooting (missing depth errors, cache misses)

**Materials**:
- User guide updates
- Video walkthrough
- FAQ document

---

### Documentation Updates

**Files to Update**:
1. `lux_depth_v2/README.md` - Add DepthMode section
2. `docs/LUX_DEPTH_V2_QUICK_START.md` - Update quickstart examples
3. `.github/copilot-instructions.md` - Add depth contract info
4. `lux_depth_v2/PHASE4_COMPLETE.md` - New phase report

---

## ✅ Completion Checklist

### Immediate Actions
- [ ] Fix default upscaler backend (`config.py`)
- [ ] Clean vulnerable packages (`pip uninstall`)
- [ ] Document fixes (`IMMEDIATE_FIXES_APPLIED.md`)
- [ ] Update feature freeze (`FEATURE_FREEZE.md`)
- [ ] Commit and push changes

### Sprint PR-1
- [ ] Implement DepthMode enum + DepthConfig
- [ ] Create DepthCacheManager
- [ ] Integrate auto-depth generation
- [ ] Fix Materials V2 cache type safety
- [ ] Add depth/cache provenance to reports
- [ ] Write tests (depth contract, cache roundtrip)
- [ ] Validate on production dataset
- [ ] Update documentation
- [ ] Create PR with phase report

### Post-Deployment
- [ ] Configure service mode (HTTPS, monitoring)
- [ ] Train artists on new presets
- [ ] Monitor metrics (cache hit rate, performance)
- [ ] Collect user feedback
- [ ] Plan Phase 2 (materials pipeline hardening)

---

**Status Tracking**: Update this file as tasks complete

**Next Review**: After Sprint PR-1 merge (estimated Jan 1-5, 2026)
