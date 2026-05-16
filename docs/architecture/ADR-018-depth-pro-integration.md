# ADR-018: Depth Pro Integration Decision

**Status:** Adopted
**Date:** 2026-02-02
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** PR #780, ADR-001 (PBR Integration), ADR-015 (DA3 1.1 Research Tier)

---

## Context

Apple's [Depth Pro](https://github.com/apple/ml-depth-pro) offers metric (absolute scale) depth estimation with strong Apple Silicon optimization. This ADR documents the architectural decision to integrate Depth Pro as an **experimental, optional** depth backend alongside the existing Depth Anything V3 (DA3) production pipeline.

### Motivation

1. **Metric Depth:** Depth Pro outputs absolute depth values (meters), unlike relative depth from DA3. This enables physically-based workflows requiring accurate scale.
2. **Apple Silicon Optimization:** Native MPS acceleration on M1/M2/M3/M4 hardware.
3. **Research Exploration:** Evaluate metric depth for architectural visualization use cases (scale-aware rendering, BIM integration).

### Current State

- **Production Default:** Depth Anything V3 (DA3) remains the sole production depth backend.
- **Experimental:** Depth Pro is available via explicit `depth_backend: depth_pro` preset selection.
- **PR #780:** Implements the `DepthProStage` class (429 LOC) with full mocking test coverage (22 tests).

---

## Decision

**Integrate Depth Pro as an isolated, feature-flagged experimental depth backend with zero behavior change to existing pipelines.**

### Implementation Roadmap

| Phase | PR | Scope | Status |
|-------|-----|-------|--------|
| **PR1: Stage** | #780 | Add `DepthProStage` class as isolated leaf stage with strict caching and provenance. | ✅ Merged |
| **PR2: Wiring** | #859, ADR-019 | Wire backend via unified backend registry with `depth_backend: depth_pro` configuration. | ✅ Merged |
| **PR3: Validation** | TBD | Integration tests with real checkpoint, benchmark against DA3. | Planned |

### Tier Classification

Depth Pro is classified as **Experimental Tier** per repository governance:

```yaml
tier: experimental
```

**Experimental Tier Constraints:**
- Not used in production workflows by default
- Requires explicit opt-in via preset configuration
- May change or be removed without deprecation notice
- No stability guarantees for output format or API
- Not covered by v2.0.0 Golden Path contracts

---

## Technical Constraints

### Checkpoint Requirements

| Attribute | Value |
|-----------|-------|
| **File** | `checkpoints/depth_pro.pt` |
| **Size** | ~1.9 GB |
| **Source** | Apple CDN (not bundled) |
| **Integrity** | SHA-256 verified at runtime |

**Download Command:**
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
```

### Device Support

| Device | Support | Notes |
|--------|---------|-------|
| **MPS (Apple Silicon)** | ✅ Primary | Auto-detected when available |
| **CUDA** | ✅ Supported | GPU acceleration for NVIDIA |
| **CPU** | ✅ Fallback | Significantly slower |

### Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `depth-pro` | Apple's inference package | `./.venv-depth-pro/bin/python -m pip install depth-pro` |
| `torch` | ML backend | Already in ML extras |

**Note:** `depth-pro` is NOT added to core dependencies and should live in a
dedicated environment to avoid downgrading the main repo to `numpy<2`:
```bash
python3 -m venv .venv-depth-pro
./.venv-depth-pro/bin/python -m pip install --upgrade pip
./.venv-depth-pro/bin/python -m pip install depth-pro

# Wire the isolated env into Transformation Portal
export TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON=./.venv-depth-pro/bin/python
```

### Output Contract

The `DepthProStage` produces:

| Artifact | Type | Description |
|----------|------|-------------|
| `depth_map` | `np.ndarray[float32]` | Metric depth array (H, W) |
| `depth_float_path` | `Path` | `.npy` file (source of truth) |
| `depth_preview_path` | `Path` | 16-bit PNG visualization |
| `depth_provenance` | `dict` | Audit-quality provenance JSON |

### Provenance Schema

```json
{
  "status": "ok",
  "engine": "apple_depth_pro",
  "device": "mps",
  "checkpoint": {
    "path": "checkpoints/depth_pro.pt",
    "sha256": "...",
    "bytes": 1900000000
  },
  "outputs": {
    "depth_shape": [1080, 1920],
    "depth_dtype": "float32",
    "depth_stats": {
      "finite_pct": 100.0,
      "min": 0.5,
      "median": 3.2,
      "p95": 8.7
    }
  },
  "timing": {
    "inference_sec": 0.42
  },
  "run": {
    "timestamp_epoch": 1738473600,
    "timestamp_iso_utc": "2026-02-02T04:00:00+00:00"
  },
  "env": {
    "python": "3.11.0",
    "platform": "macOS-14.0-arm64",
    "torch": "2.1.0",
    "depth_pro_pkg": "0.1.2"
  }
}
```

---

## Cache Key Design

Deterministic cache keys include:

1. **Checkpoint SHA-256** (first 16 chars)
2. **depth_pro package version**
3. **Transform hash** (model preprocessing)
4. **Input image hash** (SHA-256, first 16 chars)
5. **Device class** (mps/cuda/cpu)

Format: `depthpro_{ckpt}_{ver}_{transform}_{img}_{device}`

Cache invalidation triggers:
- Checkpoint file change
- Package version update
- Input image change
- Device change

---

## Alternatives Considered

### Alternative 1: Replace DA3 with Depth Pro
**Rejected:** Depth Pro lacks the validation history and stability of DA3 for production use.

**Trade-offs:**
- ✅ Simpler architecture (single depth backend)
- ❌ Breaks v2.0.0 Golden Path contracts
- ❌ No fallback for non-Apple hardware
- ❌ Metric depth incompatible with existing zone mapping

### Alternative 2: Deep Integration (Unified Interface)
**Rejected:** Premature abstraction before evaluating Depth Pro in production.

**Trade-offs:**
- ✅ Cleaner API surface
- ❌ Higher coupling (harder to remove if Depth Pro fails evaluation)
- ❌ Forces contract changes before validation
- ❌ Adds complexity before proving value

### Alternative 3: Separate Repository
**Rejected:** Fragments ecosystem and CI coverage.

**Trade-offs:**
- ✅ Complete isolation
- ❌ Duplicates infrastructure
- ❌ Harder to maintain parity with main pipeline
- ❌ Users must manage multiple installations

**Chosen:** Isolated Stage with Feature Flag (minimal coupling, reversible)

---

## Consequences

### Positive

1. **Zero Breaking Changes:** Existing workflows unaffected.
2. **Reversibility:** Can remove Depth Pro without migration.
3. **Evaluation Path:** Real-world testing before promotion.
4. **Provenance:** Full audit trail for metric depth outputs.
5. **Apple Silicon:** Native MPS optimization available.

### Negative

1. **Manual Install:** Users must install `depth-pro` separately.
2. **Large Checkpoint:** 1.9 GB download not bundled.
3. **Experimental Status:** No stability guarantees.

### Neutral

1. **Parallel Backends:** DA3 and Depth Pro coexist without interference.
2. **CI Isolation:** Depth Pro tests are mocked (no model downloads in CI).

---

## Success Criteria for Tier Promotion

Depth Pro may be promoted from **experimental** to **canary** tier when:

### Quality Gates

- [ ] **Benchmark Parity:** Inference time within 2x of DA3 for equivalent resolution
- [ ] **Output Quality:** Visual quality validation on 10+ architectural scenes
- [ ] **Metric Accuracy:** Ground truth validation against known-scale scenes
- [ ] **Integration Tests:** End-to-end tests with real checkpoint (CI excluded)

### Stability Gates

- [ ] **6-Week Soak:** No critical bugs during experimental period
- [ ] **API Stability:** Output contract unchanged for 4+ weeks
- [ ] **Documentation:** User guide and troubleshooting complete
- [ ] **Preset Validation:** At least 2 validated preset configurations

### Governance Gates

- [ ] **ADR Update:** This ADR updated with promotion decision
- [ ] **Architect Approval:** Explicit approval for tier promotion
- [ ] **CI Integration:** Checkpoint download script for integration tests

---

## Migration Path

### For Existing Users
**No changes required.** DA3 remains default. Depth Pro is opt-in only.

### For Experimental Users (PR2+)
1. Create a dedicated Depth Pro environment and install `depth-pro` there
2. Download checkpoint: `curl -L <url> -o checkpoints/depth_pro.pt`
3. Set `TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON` or pass `--depth-pro-python`
4. Use preset: `--preset depth-pro-example` or `depth_backend: depth_pro`

### For Future Promotion
If promoted to canary/stable:
1. Add `depth-pro` to optional extras (`pip install .[depth-pro]`)
2. Add checkpoint download to setup script
3. Update presets with stability marker

---

## Enforcement

### Required Tests (PR1 - Complete)

- [x] Stage instantiation and configuration
- [x] Cache key determinism and invalidation
- [x] Error handling (missing checkpoint, missing image)
- [x] Provenance structure validation
- [x] Depth statistics computation
- [x] Device auto-detection (MPS/CUDA/CPU)

### CI Gates

- [x] Unit tests pass with 100% mocking (no model downloads)
- [x] Import tests verify conditional imports
- [ ] Integration tests (PR3, excluded from CI)

### Preset Validation

```yaml
# config/presets/depth_pro_example.yaml
name: depth-pro-example
description: "Apple Depth Pro for metric depth estimation (experimental preset)"
tier: experimental
depth_backend: depth_pro
```

---

## Implementation Details

### File Locations

| File | Purpose |
|------|---------|
| `src/transformation_portal/stage_graph/stages/depth_pro.py` | DepthProStage class |
| `src/transformation_portal/stage_graph/stages/__init__.py` | Stage exports |
| `config/presets/depth_pro_example.yaml` | Example preset |
| `tests/unit/depth/test_depth_pro_stage.py` | Unit tests (22 tests) |

### Key Implementation Decisions

1. **Lazy Loading:** Model loads on first `compute()`, not at init.
2. **Graceful Degradation:** Returns actionable errors when `depth-pro` not installed.
3. **Checkpoint Validation:** SHA-256 verification with cached hash.
4. **Atomic Writes:** Output files use temp-then-rename pattern.
5. **Provenance First:** Every output includes full audit metadata.

---

## References

### External

- [Apple Depth Pro Repository](https://github.com/apple/ml-depth-pro)
- [Depth Pro Paper (arXiv)](https://arxiv.org/abs/2312.04527)
- [Apple ML Models CDN](https://ml-site.cdn-apple.com/models/)

### Internal

- [PR #780: Add Depth Pro as optional leaf stage](https://github.com/RC219805/Transformation_Portal/pull/780)
- [ADR-001: PBR Integration Architecture](ADR-001-PBR-Integration-Architecture.md)
- [ADR-015: DA3 1.1 Non-Commercial Research Tier](ADR-015-da3-1-1-non-commercial-research-tier.md)
- [Agent Governance Policy](agent_governance.md)

---

**Document History**
- **2026-02-02:** Initial ADR-018 created following PR #780 merge
