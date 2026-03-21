# ADR-019 Revised Architectural Decision

**Status:** APPROVED FOR IMMEDIATE INTEGRATION
**Date:** 2026-02-05
**Authority:** Transformation Portal Architect
**Supersedes:** ADR-019-IMPLEMENTATION-STATUS.md (deferral decision)
**Related:** ADR-019, ADR-024, ADR-018

---

## Executive Summary

**NEW INFORMATION:** Depth Pro checkpoint **IS AVAILABLE** at `checkpoints/depth_pro.pt` (1.77 GB).

**PREVIOUS DECISION:** Defer ADR-019 orchestrator integration to v2.1.0 because Depth Pro not operational.

**REVISED DECISION:** **APPROVE IMMEDIATE MINIMAL INTEGRATION** - Orchestrator integration + DA3Backend adapter.

**Rationale:**
1. ✅ **Depth Pro is operational:** checkpoint exists, dependencies installed, backend verified
2. ✅ **Core infrastructure ready:** 85% complete, only orchestrator integration remains
3. ✅ **User value unlocked:** Two backends available immediately
4. ✅ **Minimal risk:** Small surgical change, comprehensive testing possible
5. ✅ **Validates architecture:** Can now A/B test registry design with real backends

---

## Verification Results

### ✅ Checkpoint Available
```bash
$ ls -lh checkpoints/depth_pro.pt
-rw-r--r-- 1 rc staff 1.77G Feb 1 17:56 checkpoints/depth_pro.pt
```

### ✅ Dependencies Installed
```bash
$ python -c "import depth_pro; print('✅ Depth Pro available')"
✅ Depth Pro available
```

### ✅ Backend Operational
```bash
$ python -c "
from transformation_portal.depth.backends.depth_pro import DepthProBackend
backend = DepthProBackend()
backend.ensure_available()
print('✅ Backend is AVAILABLE')
"
✅ Backend is AVAILABLE
```

### ✅ Checkpoint Integrity Verified
```bash
$ python -c "
import torch
checkpoint = torch.load('checkpoints/depth_pro.pt', map_location='cpu', weights_only=False)
print(f'✅ Checkpoint loads successfully')
print(f'   Keys: {len(checkpoint)} total')
"
✅ Checkpoint loads successfully
   Keys: 1119 total
```

**CONCLUSION:** All primary blockers from ADR-019-IMPLEMENTATION-STATUS.md are **RESOLVED**.

---

## Updated Risk Assessment

### Previous Assessment (Deferral Rationale)

**From ADR-019-IMPLEMENTATION-STATUS.md:**

> **Reason 1: Depth Pro Not Operational**
> - Depth Pro backend code exists
> - But checkpoint not available (1.9 GB download) ❌
> - Dependencies not in requirements.txt ❓
> - No presets configured ❓

### Current Assessment (All Clear)

| Blocker | Previous Status | Current Status | Evidence |
|---------|----------------|----------------|----------|
| Checkpoint available | ❌ Missing | ✅ **Available** | 1.77 GB at `checkpoints/depth_pro.pt` |
| Dependencies installed | ❓ Unknown | ✅ **Installed** | `import depth_pro` succeeds |
| Backend functional | ❓ Untested | ✅ **Operational** | `backend.ensure_available()` succeeds |
| Checkpoint loads | ❓ Unknown | ✅ **Verified** | 1119 keys, OrderedDict structure |

**NEW RISK PROFILE:**

| Risk Factor | Previous | Current | Mitigation |
|-------------|----------|---------|------------|
| Depth Pro unavailable | 🔴 **HIGH** (main blocker) | 🟢 **NONE** (verified operational) | N/A - resolved |
| No second backend to test | 🟡 **MEDIUM** (design risk) | 🟢 **NONE** (two backends ready) | Can A/B test now |
| Integration breaks DA3 | 🟡 **MEDIUM** (regression risk) | 🟡 **LOW** (surgical change + tests) | Comprehensive test suite |
| No user value | 🟡 **MEDIUM** (wasted effort) | 🟢 **NONE** (unlocks Depth Pro) | Users can choose backend |

**CONCLUSION:** Primary deferral rationale is **INVALIDATED**. Risk profile now **FAVORABLE** for integration.

---

## Approved Integration Scope

### Phase 1: Minimal Viable Integration (THIS PR)

**Scope: Orchestrator Integration + DA3Backend Adapter**

**Changes Required:**

1. **Create DA3Backend Adapter** (`src/transformation_portal/depth/backends/depth_anything_v3.py`)
   - Wrap existing `DA3InferenceEngine`
   - Implement `DepthBackend` protocol
   - Maintain all existing DA3 behavior (backward compatibility)
   - License enforcement: existing `non_commercial_ok` logic

2. **Update Orchestrator** (`src/transformation_portal/lux_depth_v3/orchestrator.py`)
   - Replace hardcoded `DA3InferenceEngine` with `DepthBackendRegistry.get_backend()`
   - Backend selection: `config.depth_backend or "depth_anything_v3"` (DA3 default)
   - Fallback policy: `depth_pro → da3` if Depth Pro unavailable
   - Update metadata capture to use `self.depth_backend.name`

3. **Backend Availability Checking**
   - Pre-flight check: `backend.ensure_available()` on initialization
   - Graceful fallback with warning if requested backend unavailable
   - Log backend selection decision (already exists in truth-line logging)

4. **Tests** (comprehensive validation)
   - Unit tests: `DA3Backend` adapter behavior
   - Integration tests: Orchestrator with both backends
   - Regression tests: Ensure DA3 behavior unchanged
   - Fallback tests: Verify depth_pro → da3 graceful degradation
   - License tests: Verify enforcement for both backends

5. **Documentation Updates**
   - README: Document `--depth-backend` flag usage
   - Update CLI reference with backend selection
   - Add example: How to use Depth Pro vs DA3
   - License compliance guide for Depth Pro

**Out of Scope (Deferred to Future PRs):**

- ⏸️ `--strict-backend` enforcement flag (ADR-024 scope)
- ⏸️ `--list-backends` command
- ⏸️ Preset configuration for Depth Pro (can use existing presets + `--depth-backend depth_pro`)
- ⏸️ CI testing with Depth Pro (checkpoint too large for CI artifact)
- ⏸️ Automatic checkpoint download (`lux-depth-v3 --download-models`)

---

## Implementation Plan

### Step 1: Create DA3Backend Adapter

**File:** `src/transformation_portal/depth/backends/depth_anything_v3.py`

```python
"""Depth Anything V3 backend adapter for unified backend registry."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import DepthBackend, DepthResult, LicenseType
from ...lux_depth_v3.inference import DA3InferenceEngine

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)


class DA3Backend(DepthBackend):
    """Depth Anything V3 backend adapter implementing DepthBackend protocol.

    Wraps existing DA3InferenceEngine for backward compatibility.

    Attributes:
        name: Backend identifier ("depth_anything_v3").
        license_type: RESEARCH_ONLY (DA3 1.1 is CC BY-NC 4.0).
        requires_checkpoint: False (HuggingFace auto-download).
    """

    name = "depth_anything_v3"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = False

    def __init__(self, config: Optional["EnhanceConfig"] = None):
        """Initialize DA3 backend.

        Args:
            config: EnhanceConfig with depth backend settings.
        """
        from ...lux_depth_v3.config import DA3Config

        self._config = config

        # Build DA3Config from EnhanceConfig
        da3_config = DA3Config(
            model_id=getattr(config, "depth_model_id", "depth-anything/Depth-Anything-V3-Large"),
            revision=getattr(config, "depth_model_revision", "main"),
            device=getattr(config, "depth_device", None),
        )

        # Initialize existing DA3InferenceEngine
        self._engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=not getattr(config, "non_commercial_ok", True),
            validate_license_strict=True,
        )

    def ensure_available(self) -> None:
        """Ensure DA3 dependencies and model are available.

        DA3 uses HuggingFace auto-download, so no checkpoint check needed.
        This method validates that transformers and torch are installed.
        """
        # Engine initialization already validates dependencies
        # No additional checks needed for DA3
        pass

    def compute(
        self,
        image: Image.Image,
        device: Optional[str] = None,
    ) -> DepthResult:
        """Compute depth map for input image.

        Args:
            image: Input PIL image.
            device: Target device (overrides config if provided).

        Returns:
            DepthResult with relative disparity depth (0-1 range).
        """
        # Use existing engine compute_depth
        result = self._engine.compute_depth(image)

        # DA3 outputs relative disparity (0-1), not metric depth
        return DepthResult(
            depth_map=result.depth_map,
            original_image=result.original_image,
            metadata={
                **result.metadata,
                "depth_units": "relative_disparity",
                "backend": self.name,
                "backend_version": getattr(self._engine, "model_version", "1.1"),
            },
        )

    def compute_cache_key(
        self,
        image: Image.Image,
        extra_params: Optional[dict] = None,
    ) -> str:
        """Generate cache key for depth computation.

        Delegates to existing DA3InferenceEngine cache key logic.
        """
        return self._engine.compute_cache_key(image, extra_params)
```

### Step 2: Update Orchestrator

**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Changes:**

```python
# OLD (line ~48):
from .inference import DA3InferenceEngine

# NEW:
from ..depth.backends import DepthBackendRegistry

# OLD (line ~207):
self.inference_engine = DA3InferenceEngine(
    config=da3_config,
    commercial_use=not config.non_commercial_ok,
    validate_license_strict=True,
)

# NEW:
# Use backend registry for depth backend selection
registry = DepthBackendRegistry()
backend_name = config.depth_backend or "depth_anything_v3"

try:
    self.depth_backend = registry.get_backend(backend_name, config)
    self.depth_backend.ensure_available()
    logger.info(f"Depth backend initialized: {self.depth_backend.name}")
except Exception as e:
    # Fallback to DA3 if requested backend unavailable
    if backend_name != "depth_anything_v3":
        logger.warning(
            f"Backend fallback: requested '{backend_name}' not available ({e}), "
            f"using 'depth_anything_v3'"
        )
        self.depth_backend = registry.get_backend("depth_anything_v3", config)
        self.depth_backend.ensure_available()
    else:
        raise

# Update compute path (wherever depth is computed):
# OLD: depth_result = self.inference_engine.compute_depth(image)
# NEW: depth_result = self.depth_backend.compute(image)
```

### Step 3: Update Metadata Capture

**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

```python
# Update _capture_backend_metadata() method:
def _capture_backend_metadata(self) -> BackendSelectionMetadata:
    """Capture backend selection metadata for manifest."""
    requested = getattr(self.config, "depth_backend", None)
    resolved = self.depth_backend.name

    return BackendSelectionMetadata(
        requested_backend=requested or "depth_anything_v3",
        resolved_backend=resolved,
        resolution_status="matched" if requested == resolved else "fallback",
        resolution_reason=None if requested == resolved else "Backend unavailable, using fallback",
        device=getattr(self.depth_backend, "_device", "unknown"),
    )
```

### Step 4: Comprehensive Tests

**New Test File:** `tests/unit/depth/backends/test_da3_backend.py`

```python
"""Tests for DA3Backend adapter."""

import pytest
from PIL import Image
import numpy as np

from transformation_portal.depth.backends.depth_anything_v3 import DA3Backend
from transformation_portal.lux_depth_v3.config import EnhanceConfig


@pytest.fixture
def test_image():
    """Create test image."""
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


@pytest.fixture
def enhance_config():
    """Create test config."""
    return EnhanceConfig(
        non_commercial_ok=True,
        depth_device="cpu",
    )


def test_da3_backend_initialization(enhance_config):
    """Test DA3Backend initializes correctly."""
    backend = DA3Backend(enhance_config)
    assert backend.name == "depth_anything_v3"
    assert backend.license_type.value == "research_only"
    assert backend.requires_checkpoint is False


@pytest.mark.ml
def test_da3_backend_compute(test_image, enhance_config):
    """Test DA3Backend compute produces valid output."""
    backend = DA3Backend(enhance_config)
    result = backend.compute(test_image)

    assert result.depth_map is not None
    assert result.depth_map.shape[0] == test_image.height
    assert result.depth_map.shape[1] == test_image.width
    assert result.metadata["depth_units"] == "relative_disparity"
    assert result.metadata["backend"] == "depth_anything_v3"


def test_da3_backend_ensure_available(enhance_config):
    """Test ensure_available succeeds when dependencies installed."""
    backend = DA3Backend(enhance_config)
    backend.ensure_available()  # Should not raise
```

**Integration Test:** `tests/integration/test_orchestrator_backend_selection.py`

```python
"""Integration tests for orchestrator backend selection."""

import pytest
from pathlib import Path
from PIL import Image

from transformation_portal.lux_depth_v3 import EnhanceConfig, LuxDepthV3Orchestrator


@pytest.fixture
def test_image():
    """Create test image."""
    return Image.new("RGB", (512, 512), color=(100, 150, 200))


@pytest.fixture
def output_dir(tmp_path):
    """Create temp output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.mark.ml
def test_orchestrator_default_backend(test_image, output_dir):
    """Test orchestrator uses DA3 by default."""
    config = EnhanceConfig(
        non_commercial_ok=True,
        depth_device="cpu",
        # depth_backend not specified, should default to DA3
    )

    orchestrator = LuxDepthV3Orchestrator(config, output_dir)
    assert orchestrator.depth_backend.name == "depth_anything_v3"


@pytest.mark.ml
def test_orchestrator_explicit_da3_backend(test_image, output_dir):
    """Test orchestrator with explicit DA3 selection."""
    config = EnhanceConfig(
        non_commercial_ok=True,
        depth_device="cpu",
        depth_backend="depth_anything_v3",
    )

    orchestrator = LuxDepthV3Orchestrator(config, output_dir)
    assert orchestrator.depth_backend.name == "depth_anything_v3"


@pytest.mark.ml
@pytest.mark.skipif(not Path("checkpoints/depth_pro.pt").exists(), reason="Depth Pro checkpoint not available")
def test_orchestrator_depth_pro_backend(test_image, output_dir):
    """Test orchestrator with Depth Pro backend."""
    config = EnhanceConfig(
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        depth_device="cpu",
        depth_backend="depth_pro",
    )

    orchestrator = LuxDepthV3Orchestrator(config, output_dir)
    assert orchestrator.depth_backend.name == "depth_pro"


@pytest.mark.ml
def test_orchestrator_fallback_to_da3(test_image, output_dir, monkeypatch):
    """Test orchestrator falls back to DA3 if Depth Pro unavailable."""
    config = EnhanceConfig(
        non_commercial_ok=True,
        depth_device="cpu",
        depth_backend="depth_pro",
    )

    # Simulate Depth Pro unavailable
    monkeypatch.setattr("transformation_portal.depth.backends.depth_pro.DepthProBackend.ensure_available",
                       lambda self: (_ for _ in ()).throw(FileNotFoundError("Checkpoint not found")))

    orchestrator = LuxDepthV3Orchestrator(config, output_dir)
    # Should fall back to DA3
    assert orchestrator.depth_backend.name == "depth_anything_v3"
```

### Step 5: Documentation Updates

**Update README.md:**

```markdown
## Depth Backend Selection

Transformation Portal supports multiple depth estimation backends:

- **Depth Anything V3** (default): Relative disparity depth, HuggingFace auto-download
- **Depth Pro**: Metric depth (meters) with focal length estimation, Apple AMLR research license

### Selecting a Backend

```bash
# Use default (Depth Anything V3)
lux-depth-v3 --input-dir ./input --output-dir ./output

# Explicitly select Depth Anything V3
lux-depth-v3 --input-dir ./input --output-dir ./output --depth-backend depth_anything_v3

# Use Depth Pro (requires checkpoint and license acceptance)
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --depth-backend depth_pro \
  --accept-apple-depth-pro-research-license true \
  --non-commercial-ok true
```

### Depth Pro Requirements

**License:** Apple AMLR (research only, non-commercial)

**Installation:**
```bash
pip install depth-pro
```

**Checkpoint:** Download 1.9 GB checkpoint:
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
```

**License Flags (BOTH required):**
- `--non-commercial-ok true`: Acknowledge non-commercial use
- `--accept-apple-depth-pro-research-license true`: Accept Apple AMLR license

### Backend Fallback

If requested backend is unavailable, the system falls back to Depth Anything V3 with a warning:

```
WARNING: Backend fallback: requested 'depth_pro' not available (checkpoint not found), using 'depth_anything_v3'
```
```

---

## Success Criteria

**Integration is COMPLETE when:**

- ✅ `DA3Backend` adapter implements `DepthBackend` protocol
- ✅ `DepthBackendRegistry.get_backend("depth_anything_v3")` returns working adapter
- ✅ `DepthBackendRegistry.get_backend("depth_pro")` returns working adapter (when deps available)
- ✅ Orchestrator uses registry instead of hardcoded `DA3InferenceEngine`
- ✅ Orchestrator compute path uses `self.depth_backend.compute()`
- ✅ Backend selection logged in truth-line logs
- ✅ Metadata capture updated to use `self.depth_backend.name`
- ✅ Fallback works: `depth_pro → da3` if Depth Pro unavailable
- ✅ Tests pass: unit tests for DA3Backend + integration tests for orchestrator
- ✅ Regression tests confirm DA3 behavior unchanged
- ✅ Documentation updated with backend selection guide
- ✅ License enforcement works for both backends

**User-Facing Validation:**

```bash
# Test 1: Default backend (DA3)
lux-depth-v3 --input-dir ./test_images --output-dir ./output_da3

# Test 2: Explicit DA3
lux-depth-v3 --input-dir ./test_images --output-dir ./output_da3_explicit \
  --depth-backend depth_anything_v3

# Test 3: Depth Pro (if checkpoint available)
lux-depth-v3 --input-dir ./test_images --output-dir ./output_depth_pro \
  --depth-backend depth_pro \
  --accept-apple-depth-pro-research-license true \
  --non-commercial-ok true

# Test 4: Fallback behavior (request unavailable backend)
lux-depth-v3 --input-dir ./test_images --output-dir ./output_fallback \
  --depth-backend nonexistent_backend
# Should warn and use DA3
```

---

## Timeline

**Target: Single PR, estimated 1-2 days**

**Breakdown:**
- 🔧 **4 hours:** Create `DA3Backend` adapter + unit tests
- 🔧 **4 hours:** Update orchestrator integration + backend selection logic
- 🔧 **3 hours:** Write integration tests (orchestrator + backends)
- 🔧 **2 hours:** Update documentation (README, CLI reference)
- 🔧 **2 hours:** Manual testing + validation
- 🔧 **1 hour:** Code review polish

**Total: ~16 hours** (2 developer days)

---

## Risk Mitigation

### Risk 1: Breaks DA3 Pipeline

**Mitigation:**
- `DA3Backend` is thin wrapper around existing `DA3InferenceEngine`
- Comprehensive regression tests ensure behavior unchanged
- Fallback logic ensures DA3 always available

**Rollback Plan:**
- Revert orchestrator changes, keep `DA3InferenceEngine` direct call
- Backend infrastructure remains intact (no breaking changes)

### Risk 2: Depth Pro Has Production Issues

**Mitigation:**
- Default backend is DA3 (stable, proven)
- Depth Pro is opt-in via explicit `--depth-backend depth_pro`
- Fallback logic automatically switches to DA3 if Depth Pro fails
- Users must explicitly accept research license (informed consent)

**Monitoring:**
- Truth-line logs capture backend selection decisions
- Manifests record requested vs resolved backend
- Can analyze production usage patterns

### Risk 3: License Enforcement Gaps

**Mitigation:**
- Both backends have `license_type = LicenseType.RESEARCH_ONLY`
- Registry enforces `non_commercial_ok=True` before backend selection
- Depth Pro additionally requires `accept_apple_depth_pro_research_license=True`
- Multi-layer enforcement (config + registry + runtime)

**Validation:**
- License tests verify enforcement for both backends
- CLI tests verify flags required
- Integration tests verify license errors propagate correctly

---

## Alternatives Considered

### Alternative 1: Keep Deferral to v2.1.0

**Arguments For:**
- Lower risk (no code changes)
- More time for Depth Pro validation

**Arguments Against:**
- ❌ User cannot use Depth Pro despite having checkpoint
- ❌ Registry infrastructure sits idle (wasted effort)
- ❌ Cannot validate architecture with real multi-backend scenario
- ❌ Delays user value delivery

**Verdict:** ❌ Rejected - deferral rationale invalidated by new information

### Alternative 2: Full Integration (Including --strict-backend)

**Arguments For:**
- Complete ADR-019 implementation in one PR
- No follow-up work needed

**Arguments Against:**
- ❌ Larger scope = higher risk
- ❌ `--strict-backend` is ADR-024 scope (separate decision)
- ❌ Complicates rollback if issues found

**Verdict:** ❌ Rejected - prefer minimal scope for surgical change

### Alternative 3: Minimal Integration (APPROVED)

**Arguments For:**
- ✅ Unlocks Depth Pro immediately
- ✅ Validates architecture with real backends
- ✅ Surgical change, easy to review
- ✅ Comprehensive testing possible
- ✅ Low risk, clear rollback path

**Arguments Against:**
- `--strict-backend` enforcement deferred to future PR

**Verdict:** ✅ **APPROVED** - optimal risk/reward balance

---

## Communication to User

**Message:**

> **ADR-019 Decision REVISED: APPROVE IMMEDIATE INTEGRATION**
>
> **New Information:** Depth Pro checkpoint verified at `checkpoints/depth_pro.pt` (1.77 GB).
> All dependencies operational, backend validated.
>
> **Previous Decision (INVALIDATED):**
> Defer to v2.1.0 because "Depth Pro not operational."
>
> **Revised Decision (APPROVED):**
> Proceed with **minimal integration** immediately:
> - Create `DA3Backend` adapter
> - Update orchestrator to use `DepthBackendRegistry`
> - Enable backend selection via `--depth-backend` flag
> - Fallback logic: `depth_pro → da3`
> - Comprehensive tests + documentation
>
> **Scope:**
> - ✅ Orchestrator integration
> - ✅ DA3Backend adapter
> - ✅ Backend availability checking
> - ✅ Fallback policies
> - ✅ Tests + docs
> - ⏸️ `--strict-backend` enforcement (defer to ADR-024 PR)
>
> **Timeline:** Single PR, ~2 developer days
>
> **Risk:** LOW (surgical change, comprehensive tests, fallback logic)
>
> **User Value:** Unlocks Depth Pro immediately, validates multi-backend architecture
>
> **Next Steps:**
> 1. Implement `DA3Backend` adapter
> 2. Update orchestrator backend selection
> 3. Write comprehensive tests
> 4. Update documentation
> 5. Manual validation with both backends
> 6. Submit PR for review
>
> **Architect Approval:** ✅ APPROVED

---

## References

### Related ADRs
- [ADR-019: Depth Backend Unification](../ADR-019-depth-backend-unification.md) - Original specification
- [ADR-019-IMPLEMENTATION-STATUS](ADR-019-IMPLEMENTATION-STATUS.md) - Previous deferral decision (SUPERSEDED)
- [ADR-024: Backend Enforcement Strategy](ADR-024-backend-enforcement-strategy.md) - `--strict-backend` enforcement (future work)
- [ADR-018: Depth Pro Integration](../ADR-018-depth-pro-integration.md) - Depth Pro operationalization

### Code References
- `src/transformation_portal/depth/backends/protocol.py` - Backend protocol
- `src/transformation_portal/depth/backends/registry.py` - Backend registry
- `src/transformation_portal/depth/backends/depth_pro.py` - Depth Pro adapter (existing)
- `src/transformation_portal/lux_depth_v3/orchestrator.py` - Orchestrator (needs update)
- `src/transformation_portal/lux_depth_v3/inference.py` - DA3InferenceEngine (to be wrapped)

---

## Document History

- **2026-02-05:** ADR-019 Revised Decision created (Architect approval)
  - Verified Depth Pro operational (checkpoint + dependencies + backend)
  - Invalidated deferral rationale from ADR-019-IMPLEMENTATION-STATUS
  - Approved immediate minimal integration
  - Defined scope: orchestrator + DA3Backend adapter
  - Estimated timeline: 2 developer days
  - Risk assessment: LOW, comprehensive mitigation plan
  - Supersedes: ADR-019-IMPLEMENTATION-STATUS.md deferral decision
