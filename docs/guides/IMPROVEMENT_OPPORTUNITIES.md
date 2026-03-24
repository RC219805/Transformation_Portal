# Transformation Portal: Improvement Opportunities

**Document Version:** 1.0.0
**Date:** February 2026
**Status:** Active Recommendations

This document identifies improvement opportunities across all sectors of the Transformation Portal repository, with risk/reward analysis for each recommendation.

---

## Executive Summary

The Transformation Portal is a mature, production-ready toolkit at v2.0.0 with comprehensive quality gates and extensive capabilities. This analysis identifies **21 actionable improvement opportunities** across 6 sectors:

| Sector | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| **Security** | 1 | 1 | 1 | 0 | 3 |
| **Testing** | 1 | 2 | 2 | 1 | 6 |
| **Performance** | 0 | 2 | 2 | 0 | 4 |
| **Documentation** | 0 | 1 | 2 | 1 | 4 |
| **CI/CD** | 0 | 2 | 1 | 0 | 3 |
| **Code Quality** | 0 | 0 | 1 | 0 | 1 |
| **Total** | **2** | **8** | **9** | **2** | **21** |

---

## 1. Security Improvements

### SEC-001: Fix shell=True in subprocess call [CRITICAL]

**Status:** ✅ **IMPLEMENTED**

**Location:** `scripts/utilities/fix_quality_issues.py:13-25`

**Current Code (FIXED):**
```python
def run_command(cmd, description):
    """Run a command and report results.

    Security Note: Uses shlex.split() instead of shell=True to prevent
    command injection vulnerabilities (SEC-001). This safely parses the
    command string into a list of arguments without invoking a shell.
    """
    # Convert string to list for safe subprocess execution (no shell injection risk)
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
```

~~**Risk:** Command injection vulnerability if `cmd` variable contains user-controlled input.~~ **MITIGATED**

~~**Recommendation:**~~ **IMPLEMENTED:** Uses `shlex.split()` to safely parse command strings without shell invocation.

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | ~~Critical~~ → N/A |
| **Implementation Risk** | N/A |
| **Effort** | ~~Low (15 minutes)~~ → DONE |
| **Reward** | ✅ Command injection vulnerability eliminated |
| **Breaking Change** | No |

---

### SEC-002: Implement Input Validation for Config Loader [HIGH]

**Status:** ✅ **IMPLEMENTED**

**Location:** `src/transformation_portal/config_loader.py:_expand_env_vars` (lines 35-93)

**Current Code (IMPLEMENTED):**
```python
def _expand_env_vars(value: str) -> str:
    """Expand environment variables in a string with path validation.

    Validates expanded values that appear to be paths for basic safety checks.
    """
    # ... expansion logic ...

    # Validate expanded values that look like paths
    if expanded != value and (os.path.sep in expanded or "/" in expanded or expanded.startswith((".", "/"))):
        try:
            test_path = Path(expanded)
            # Check for excessive parent directory traversal (>5 levels)
            parent_count = sum(1 for part in test_path.parts if part == "..")
            if parent_count > 5:
                logger.warning(f"Config path contains excessive parent traversal (..): {expanded}")

            # Check for sensitive system directories
            if test_path.exists():
                resolved = test_path.resolve()
                sensitive_dirs = {"/etc", "/sys", "/proc", "/dev", "/root"}
                if any(str(resolved).startswith(d) for d in sensitive_dirs):
                    logger.warning(f"Config path resolves to sensitive system directory: {resolved}")
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug(f"Config path validation warning for '{expanded}': {e}")

    return expanded
```

~~**Issue:** Environment variable expansion without output sanitization.~~ **RESOLVED**

~~**Recommendation:**~~ **IMPLEMENTED:** Defense-in-depth path validation with parent traversal checks (>5 levels) and sensitive directory detection.

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | ~~High~~ → N/A |
| **Implementation Risk** | N/A |
| **Effort** | ~~Medium (1-2 hours)~~ → DONE |
| **Reward** | ✅ Defense-in-depth for config loading |
| **Breaking Change** | No |

---

### SEC-003: Add Runtime Dependency Integrity Check [MEDIUM]

**Issue:** While CI scans dependencies, runtime verification is not performed.

**Recommendation:** Add optional runtime integrity check for production deployments:
```python
# src/transformation_portal/__init__.py
def verify_dependency_integrity():
    """Verify known-good hashes of critical dependencies."""
    # Check torch, transformers, etc. have expected hashes
    pass
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Medium |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours) |
| **Reward** | Supply chain attack protection |
| **Breaking Change** | No |

---

## 2. Testing Improvements

### TEST-001: Create Shared Test Fixtures in conftest.py [CRITICAL]

**Location:** `tests/conftest.py` (currently 21 lines, no fixtures)

**Issue:** Tests create fixtures locally, leading to duplication and inconsistency. Critical fixtures like temp directories, mock images, and model mocks are recreated in many test files.

**Recommendation:** Add comprehensive shared fixtures:
```python
# tests/conftest.py
import pytest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_image(temp_dir):
    """Sample test image (256x256 RGB)."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = temp_dir / "sample.png"
    img.save(path)
    return path

@pytest.fixture
def mock_depth_map():
    """Mock depth map for unit tests."""
    return np.random.rand(256, 256).astype(np.float32)

@pytest.fixture
def mock_config():
    """Standard test configuration."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    return EnhanceConfig(preset="draft")
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | ~~Low~~ → N/A |
| **Implementation Risk** | N/A |
| **Effort** | ~~Medium (4-6 hours)~~ → DONE |
| **Reward** | ✅ Reduced test duplication, consistent test behavior |
| **Breaking Change** | No |

**Status:** ✅ **IMPLEMENTED**

`tests/conftest.py` now has comprehensive shared fixtures with a tiered architecture:

- **Tier 1 (Pure fixtures):** `deterministic_rng`, `sample_config_dict`, `sample_pbr_config_dict`
- **Tier 2 (IO fixtures):** `temp_workspace`, `sample_rgb_image`, `sample_rgb_pil`, `sample_depth_map`, `sample_depth_pil`, `sample_image_file`, `sample_depth_file`, `sample_yaml_config`
- **Tier 3 (ML fixtures):** `transformers_offline`, `mock_depth_model`
- **Availability helpers:** `has_depth_anything_v3()`, `has_torch()`, `has_transformers()`, `is_offline_mode()`, `can_run_da3_compute()`
- **Benchmark guard:** `pytest_collection_modifyitems()` skips benchmark tests unless explicitly requested

---

### TEST-002: Add Tests for Untested Modules [HIGH]

**Gap Analysis (Updated 2026-03-24):**

| Module | Path | Lines of Code | Test Status |
|--------|------|---------------|-------------|
| enhancers/ | `src/transformation_portal/enhancers/` | ~500 | ⚠️ Partial (aerial/board enhancers tested) |
| rendering/ | `src/transformation_portal/rendering/` | ~800 | ⚠️ Partial |
| comfyui/ | `src/transformation_portal/comfyui/` | ~300 | ❌ No tests |
| scene_types.py | `src/transformation_portal/` | 185 | ❌ No tests |
| config_loader.py | `src/transformation_portal/` | 385 | ✅ Tests exist (tests/utils/test_config_loader.py + tests/test_config_loader_security.py) |
| pipeline_unified.py | `src/transformation_portal/` | 1,077 | ❌ No tests |

*Note: Line counts measured via `wc -l` for consistency and reproducibility.*

**Recommendation:** Prioritize tests by usage frequency:
1. `config_loader.py` - Core functionality, high usage
2. `scene_types.py` - Data structures used throughout
3. `enhancers/` - Production workflows rely on these
4. `comfyui/` - Integration point with external system

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Medium |
| **Implementation Risk** | Low |
| **Effort** | High (8-16 hours) |
| **Reward** | 15-20% coverage increase, regression protection |
| **Breaking Change** | No |

---

### TEST-003: Enable Disabled Integration Tests [HIGH]

**Location:** `tests/test_da3_inference_integration.py`

**Issue:** 29 tests are disabled by default, requiring `TP_RUN_HF_MODEL_TESTS=1`. While appropriate for CI, local development misses these tests.

**Recommendation:**
1. Add `make test-integration` target to Makefile
2. Document integration test requirements clearly
3. Add CI job for weekly integration tests with real models

```makefile
# Makefile addition
test-integration:
	@echo "Running integration tests (requires HF_TOKEN)..."
	TP_RUN_HF_MODEL_TESTS=1 "$(PY)" -m pytest -v tests/test_da3_inference_integration.py
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Low (1 hour) |
| **Reward** | Better integration test visibility |
| **Breaking Change** | No |

---

### TEST-004: Add Property-Based Tests for Core Algorithms [MEDIUM]

**Location:** Various processing modules

**Issue:** Current tests are example-based. Property-based tests with hypothesis would catch edge cases.

**Recommendation:** Add hypothesis tests for:
- Color space conversions (roundtrip invariants)
- Depth map normalization (range preservation)
- Filter graph composition (associativity)

```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.floats(min_value=0, max_value=1))
def test_normalize_roundtrip(value):
    """Normalization should be reversible."""
    result = denormalize(normalize(value))
    assert abs(result - value) < 1e-6
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (4-8 hours) |
| **Reward** | Edge case detection, mathematical correctness |
| **Breaking Change** | No |

---

### TEST-005: Implement Golden Fixture Regression Tests [MEDIUM]

**Location:** `tests/fixtures/golden/` (exists but underutilized)

**Issue:** Golden fixtures exist but aren't consistently used for regression testing.

**Recommendation:** Create systematic golden tests:
```python
@pytest.mark.golden
def test_depth_estimation_golden(golden_fixtures):
    """Verify depth estimation matches golden reference."""
    result = estimate_depth(golden_fixtures["input_image"])
    expected = golden_fixtures["expected_depth"]
    assert np.allclose(result, expected, rtol=0.01)
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (4-6 hours) |
| **Reward** | Detect visual/numerical regressions |
| **Breaking Change** | No |

---

### TEST-006: Fix Draft Preset Performance Regression [LOW]

**Location:** `tests/stress/test_stress_large_batch.py:282`

**Issue:** Draft preset is slower than standard, opposite of expected behavior.
```python
# TODO: Investigate preset performance ordering regression
# Expected: draft < standard < premium (in time)
# Actual: premium < standard < draft (in time)
```

**Recommendation:** Profile draft preset to identify bottleneck:
```python
import cProfile
cProfile.run('process_with_preset("draft")', 'draft_profile.stats')
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours to diagnose) |
| **Reward** | Improved UX for draft previews |
| **Breaking Change** | No |

---

## 3. Performance Improvements

### PERF-001: Complete YAML Loading Implementation [HIGH]

**Location:** `src/transformation_portal/depth_canonical/config.py:143`

```python
# TODO: Implement YAML loading in Phase 2
```

**Issue:** YAML loading for canonical config is incomplete, forcing programmatic configuration.

**Recommendation:** Implement Phase 2 YAML loading to enable declarative configuration:
```python
@classmethod
def from_yaml(cls, path: Path) -> "CanonicalConfig":
    """Load configuration from YAML file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return cls(**data)
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours) |
| **Reward** | Better developer experience, pipeline portability |
| **Breaking Change** | No (additive) |

---

### PERF-002: Implement Batch Processing for Depth Estimation [HIGH]

**Location:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Issue:** Depth estimation processes images sequentially even in batch mode.

**Recommendation:** Implement true batch inference:
```python
def estimate_depth_batch(images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
    """Batch depth estimation for 2-3x throughput improvement."""
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack([preprocess(img) for img in batch])
        with torch.no_grad():
            depths = model(batch_tensor)
        results.extend([postprocess(d) for d in depths])
    return results
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Medium |
| **Implementation Risk** | Medium (numerical precision) |
| **Effort** | High (8-12 hours) |
| **Reward** | 2-3x throughput for batch processing |
| **Breaking Change** | No (new API) |

---

### PERF-003: Add xxHash as Default Hasher [MEDIUM]

**Location:** `src/transformation_portal/lux_depth_v3/orchestrator.py:137-140`

**Issue:** xxHash is supported but not default. SHA256 is 5x slower.

**Recommendation:** Make xxHash default when available:
```python
try:
    import xxhash
    DEFAULT_HASHER = xxhash.xxh64
except ImportError:
    import hashlib
    DEFAULT_HASHER = hashlib.sha256
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Low (1 hour) |
| **Reward** | 5x faster hashing for skip detection |
| **Breaking Change** | Optional (add xxhash to base dependencies) |

---

### PERF-004: Implement Lazy Model Loading [MEDIUM]

**Location:** Various ML model imports

**Issue:** Some models load at import time, slowing startup.

**Recommendation:** Use `__getattr__` pattern for lazy loading:
```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_heavy_model()
    return _model
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours) |
| **Reward** | Faster CLI startup, reduced memory for unused features |
| **Breaking Change** | No |

---

## 4. Documentation Improvements

### DOC-001: Consolidate Documentation Structure [HIGH]

**Issue:** 200+ markdown files across 30+ directories with significant duplication.

**Duplicated Content Examples:**
- `LUXURY_ESTATE_PIPELINE_README.md` / `UNIFIED_LUXURY_PIPELINE.md` / `LUXURY_ESTATE_PIPELINE.md`
- `ELITE_PIPELINE_README.md` / `ELITE_PIPELINE_GUIDE.md`
- Multiple `QUALITY_CONTROL_*.md` variants
- Historical session logs mixed with active docs

**Recommendation:**
1. Archive session logs to `docs/archive/sessions/`
2. Merge duplicate pipeline docs into single canonical versions
3. Create clear documentation hierarchy:
   ```
   docs/
   ├── getting-started/     # Quick starts, installation
   ├── user-guide/          # How-to guides
   ├── reference/           # API, CLI, presets
   ├── architecture/        # ADRs, design docs
   └── archive/             # Historical, deprecated
   ```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Medium (link breakage) |
| **Effort** | High (8-16 hours) |
| **Reward** | Better developer experience, easier maintenance |
| **Breaking Change** | Documentation links may break |

---

### DOC-002: Generate API Documentation [MEDIUM]

**Location:** `docs/api/` (currently empty)

**Issue:** No Python API documentation exists. Only CLI documentation available.

**Recommendation:** Set up Sphinx autodoc:
```bash
pip install sphinx sphinx-autodoc-typehints
sphinx-quickstart docs/api
sphinx-apidoc -o docs/api/reference src/transformation_portal
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (4-6 hours) |
| **Reward** | Programmatic usage becomes accessible |
| **Breaking Change** | No |

---

### DOC-003: Add Missing Docstrings to Orchestrator [MEDIUM]

**Location:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Missing Docstrings:**
- `_generate_pbr_stage()` (critical PBR generation)
- `_verify_pbr_outputs()` (validation logic)
- `_load_cached_depth()` (caching behavior)
- `enhance_batch()` (main batch API)

**Recommendation:** Add comprehensive docstrings with Args/Returns/Raises:
```python
def enhance_batch(
    self,
    inputs: List[Path],
    output_dir: Path,
    *,
    parallel: bool = True,
) -> BatchResult:
    """Enhance a batch of images with depth-aware processing.

    Args:
        inputs: List of input image paths.
        output_dir: Directory for output files.
        parallel: Enable parallel processing (default: True).

    Returns:
        BatchResult with success/failure counts and timing metrics.

    Raises:
        ValidationError: If inputs are invalid or inaccessible.
        ResourceError: If insufficient memory or disk space.
    """
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours) |
| **Reward** | IDE support, API clarity |
| **Breaking Change** | No |

---

### DOC-004: Complete Sample Data URLs [LOW]

**Location:** `data/sample_images/README.md`

**Issue:** Sample URLs marked as TODO, not yet uploaded to GitHub Release.

```markdown
# TODO: Upload samples to GitHub Release
```

**Recommendation:** Upload curated sample images to GitHub Release and update URLs.

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Low (1-2 hours) |
| **Reward** | Better onboarding experience |
| **Breaking Change** | No |

---

## 5. CI/CD Improvements

### CI-001: Consolidate Overlapping Workflows [HIGH]

**Issue:** Multiple workflows perform the same tasks:
- **Lint**: build.yml, ci.yml, quality-gate.yml, python-app.yml (4 separate)
- **Test**: build.yml, ci.yml, python-app.yml, enforcement.yml (4 separate)
- **Security**: ci.yml, security-unified.yml, codeql.yml (3 separate)

**Current State:** 14 workflow files, significant overlap.

**Recommendation:** Consolidate to 6 focused workflows:
```
.github/workflows/
├── ci.yml                 # Unified: lint + test + coverage (PR/push)
├── security.yml           # Unified: CodeQL + dependency scan (PR/push/schedule)
├── nightly.yml            # Deep checks: stress, performance, integration
├── release.yml            # Build + PyPI publish (release)
├── maintenance.yml        # Dependency updates (scheduled)
└── automation.yml         # Issue management, AI review (events)
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Medium |
| **Implementation Risk** | Medium (workflow migration) |
| **Effort** | High (8-16 hours) |
| **Reward** | Faster CI, easier maintenance, cost reduction |
| **Breaking Change** | Branch protection rules may need updating |

---

### CI-002: Add Caching to build.yml [HIGH]

**Status:** ✅ **IMPLEMENTED**

**Location:** `.github/workflows/build.yml:356-360, 407-411, 506-510`

**Issue:** ~~build.yml reinstalls dependencies on every run (no pip cache).~~ **RESOLVED**

**Implementation:**
- `actions/cache@v5` now caches `~/.cache/pip` (lines 356-360, 407-411, 506-510)
- Cache key based on `requirements*.txt` hash files
- Comments note 3-5 min savings per job (line 25)

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | ~~Low~~ → N/A |
| **Implementation Risk** | N/A |
| **Effort** | ~~Low (30 minutes)~~ → DONE |
| **Reward** | 2-5 minute reduction per run |
| **Breaking Change** | No |

---

### CI-003: Add Concurrency Control [MEDIUM]

**Status:** ✅ **IMPLEMENTED**

**Location:** `.github/workflows/ci.yml:11`, `.github/workflows/build.yml:14`

**Issue:** ~~Missing concurrency control, allowing redundant parallel runs.~~ **RESOLVED**

**Implementation:**
```yaml
# ci.yml:11
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

# build.yml:14
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | ~~Low~~ → N/A |
| **Implementation Risk** | N/A |
| **Effort** | ~~Low (15 minutes)~~ → DONE |
| **Reward** | Reduced CI minutes, faster feedback |
| **Breaking Change** | No |

---

## 6. Code Quality Improvements

### CODE-001: Complete Phase 2 Configuration [MEDIUM]

**Status:** 📦 **OBSOLETE** - Module archived to `archive/depth_canonical/`

**Location:** ~~`src/transformation_portal/depth_canonical/config.py:143`~~

**Issue:** ~~YAML configuration loading marked as Phase 2 TODO remains incomplete.~~

**Resolution:** The `depth_canonical` module was superseded by `src/transformation_portal/depth/backends/` (ADR-019) and archived. No action required.

**Recommendation:** ~~Implement full YAML support:~~
```python
@classmethod
def from_yaml(cls, path: Path) -> "CanonicalConfig":
    """Load canonical configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Populated CanonicalConfig instance.

    Raises:
        ConfigurationError: If YAML is invalid or missing required fields.
    """
    import yaml
    from transformation_portal.utils.security import validate_filepath

    validate_filepath(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    # Validate required fields
    required = ["model", "preset"]
    for field in required:
        if field not in data:
            raise ConfigurationError(f"Missing required field: {field}")

    return cls(**data)
```

| Aspect | Assessment |
|--------|------------|
| **Issue Severity** | Low |
| **Implementation Risk** | Low |
| **Effort** | Medium (2-4 hours) |
| **Reward** | Declarative configuration, better tooling support |
| **Breaking Change** | No (additive) |

---

## Implementation Priority Matrix

### Quick Wins (< 2 hours, High Impact)

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| SEC-001 | Fix shell=True subprocess | 15 min | Critical security fix |
| CI-002 | Add caching to build.yml | 30 min | 2-5 min CI speedup |
| CI-003 | Add concurrency control | 15 min | Reduced CI waste |
| TEST-003 | Add test-integration target | 1 hour | Better test visibility |

### High-Value Medium Effort (2-8 hours)

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| TEST-001 | Create shared fixtures | 4-6 hours | Test quality improvement |
| PERF-001 | YAML loading for canonical | 2-4 hours | Developer experience |
| DOC-003 | Add orchestrator docstrings | 2-4 hours | API clarity |
| PERF-003 | Default xxHash hasher | 1 hour | 5x faster hashing |

### Strategic Investments (8+ hours)

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| TEST-002 | Add tests for untested modules | 8-16 hours | 15-20% coverage increase |
| CI-001 | Consolidate workflows | 8-16 hours | Simplified CI/CD |
| DOC-001 | Consolidate documentation | 8-16 hours | Better DX |
| PERF-002 | Batch depth estimation | 8-12 hours | 2-3x throughput |

---

## Risk/Reward Summary

**Note:** "Issue Severity" reflects the risk of *not* addressing the recommendation. "Implementation Risk" reflects the risk of *making* the change.

```
Implementation Risk Distribution:
┌─────────────────────────────────────────────────────────────┐
│ Low        ███████████████████████ 19 recommendations (90%) │
│ Medium     ██ 2 recommendations (10%)                       │
│ High       0 recommendations (0%)                           │
└─────────────────────────────────────────────────────────────┘

Effort Distribution:
┌─────────────────────────────────────────────────────────────┐
│ Low (< 2h)     ██████████ 6 recommendations (29%)           │
│ Medium (2-8h)  ███████████████ 11 recommendations (52%)     │
│ High (8h+)     ████ 4 recommendations (19%)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix: Current State Metrics

### Test Coverage
- **Test Files:** ~100
- **Test Cases:** ~1,600
- **Untested Modules:** 6 identified
- **Disabled Tests:** 29 (integration tier)

### CI/CD
- **Workflow Files:** 14
- **Average Run Time:** 40-60 minutes
- **Caching:** Partial (missing in build.yml)

### Documentation
- **Total Files:** 200+
- **Duplicate Sets:** 10+ identified
- **API Docs:** None (empty docs/api/)

### Security
- **Critical Issues:** 1 (shell=True)
- **CVEs Blocked:** 2 (basicsr, protobuf)
- **Scanning Coverage:** CodeQL + pip-audit + Safety

---

*This document should be reviewed quarterly and updated as improvements are implemented.*
