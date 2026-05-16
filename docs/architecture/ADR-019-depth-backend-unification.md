# ADR-019: Depth Backend Unification Architecture

**Status:** Implemented
**Date:** 2026-02-02
**Implementation Date:** 2026-02-09 (PR #906)
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-018 (Depth Pro Integration), ADR-015 (DA3 1.1 Research Tier), PR #780, PR #906

---

## Executive Summary

This ADR defines the architecture for unified depth backend abstraction, enabling first-class integration of Depth Pro alongside existing Depth Anything V2/V3 implementations while maintaining strict license governance and backward compatibility.

**Key Decision:** Introduce `DepthBackend` Protocol and `DepthBackendRegistry` following existing `ModelRegistry` patterns, with dual-gated license enforcement for research-only models (Depth Pro AMLR, DA3 1.1 CC BY-NC 4.0).

---

## Context

### Current State

The codebase has **three separate depth implementations** with overlapping but inconsistent patterns:

| Implementation | Location | Contract | License | Status |
|----------------|----------|----------|---------|--------|
| **DA3 (V2)** | `lux_depth_v3/inference.py` (DA3InferenceEngine) | `DepthResult(depth_map, original_image, metadata)` | Commercial | Production |
| **DA2** | `depth/models/depth_anything_v2.py` (DepthAnythingV2Model) | Custom | Commercial | Stable |
| **Depth Pro** | `stage_graph/stages/depth_pro.py` (DepthProStage) | Stage-based | Research (AMLR) | Experimental |

**Integration Points:**
- `lux_depth_v3/orchestrator.py` → DA3InferenceEngine
- `depth_canonical/models/registry.py` → ModelRegistry (DA2/DA3 only)
- `stage_graph/graph.py` → DepthProStage (isolated)

**License Governance:**
- `EnhanceConfig.non_commercial_ok: bool` exists but **not enforced** for Depth Pro
- DA3 1.1 has `@require_non_commercial` decorator (ADR-015)
- Depth Pro (Apple AMLR license) requires research-only use

### Problem Statement

1. **No Unified Contract:** Each depth backend has different input/output contracts
2. **License Gaps:** Depth Pro bypasses existing `non_commercial_ok` enforcement
3. **Metadata Loss:** Depth Pro outputs metric depth + focal length, but contract doesn't expose it
4. **Caching Inconsistency:** `.npy` (legacy) vs `.npz` + `.json` sidecar
5. **No Registry:** Cannot select backend via configuration (`depth_backend: depth_pro`)

### Requirements (from User)

1. **Backend Contract & Registry**: Unified DepthBackend Protocol + DepthBackendRegistry
2. **DepthResult Enhancement**: Add `depth_units`, `focal_length_px`, `field_of_view_deg`
3. **License Gating**: BOTH `non_commercial_ok=True` AND `accept_apple_depth_pro_research_license=True`
4. **Enhanced Caching**: `.npz` + `.json` metadata sidecar, backward-compatible
5. **Presets**: `depth_pro_metric_mps`, `depth_pro_metric_cpu` experimental presets

---

## Decision

### 1. Backend Contract Location

**DECISION: `src/transformation_portal/depth/backends/`** (new module)

**Rationale:**
- Depth backends are cross-cutting: used by `lux_depth_v3`, `stage_graph`, and `depth_canonical`
- Placing in `src/transformation_portal/depth/` makes it a **shared utility**, not pipeline-specific
- Parallel to existing `depth/models/` (model wrappers) and `depth/processors/` (postprocessing)

**Structure:**
```
src/transformation_portal/depth/backends/
├── __init__.py              # Exports: DepthBackend, DepthResult, DepthBackendRegistry
├── protocol.py              # DepthBackend Protocol, DepthResult dataclass
├── registry.py              # DepthBackendRegistry (factory + device auto-detection)
├── depth_anything_v2.py     # DA2 backend adapter
├── depth_anything_v3.py     # DA3 backend adapter
└── depth_pro.py             # Depth Pro backend adapter
```

**Rejected Alternatives:**

| Location | Pros | Cons | Verdict |
|----------|------|------|---------|
| `stage_graph/stages/` | Co-located with DepthProStage | Tight coupling to stage execution model, harder to use from `lux_depth_v3` | ❌ Rejected |
| `lux_depth_v3/` | Near orchestrator | DA3-centric, excludes DA2 and stage-based usage | ❌ Rejected |
| `depth_canonical/` | Canonical phase module | Phase 3 experimental, not yet stable baseline | ❌ Rejected |

---

### 2. License Gating Strategy

**DECISION: Multi-Layer Enforcement (All Three Layers)**

1. **EnhanceConfig Validation** (Entry Point)
   - Validate on config construction
   - Fail fast before model loading

2. **DepthBackendRegistry Selection** (Factory)
   - Check licenses before instantiating backend
   - Provide actionable error messages

3. **Backend.compute() Runtime** (Defense-in-Depth)
   - Final gate before inference
   - Log license acceptance for audit trail

**Enforcement Code:**

```python
# src/transformation_portal/depth/backends/protocol.py
from enum import Enum

class LicenseType(Enum):
    COMMERCIAL = "commercial"
    RESEARCH_ONLY = "research_only"

@dataclass
class DepthResult:
    """Unified depth estimation result.

    Enhanced to support both relative depth (0-1 normalized) and
    metric depth (absolute scale in meters).
    """
    depth_map: np.ndarray
    original_image: np.ndarray
    metadata: Dict[str, Any]

    # New fields for metric depth support
    depth_units: Literal["relative", "meters"] = "relative"
    focal_length_px: Optional[float] = None
    field_of_view_deg: Optional[float] = None

class DepthBackend(Protocol):
    """Unified depth estimation backend contract."""

    name: str
    license_type: LicenseType
    requires_checkpoint: bool

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None
    ) -> DepthResult:
        """Estimate depth from image."""
        ...

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key."""
        ...
```

```python
# src/transformation_portal/lux_depth_v3/config.py (MODIFICATION)
@dataclass
class EnhanceConfig:
    # Existing fields...
    non_commercial_ok: bool = False

    # NEW: Explicit Depth Pro license acceptance
    accept_apple_depth_pro_research_license: bool = False

    def validate(self) -> None:
        """Validate configuration before execution."""
        # Check depth backend license requirements
        if self.depth_backend == "depth_pro":
            if not self.non_commercial_ok:
                raise LicenseRestrictionError(
                    "Depth Pro requires non_commercial_ok=True (Apple AMLR research license)"
                )
            if not self.accept_apple_depth_pro_research_license:
                raise LicenseRestrictionError(
                    "Depth Pro requires explicit license acceptance.\n"
                    "Set accept_apple_depth_pro_research_license=True to acknowledge:\n"
                    "  - Apple Machine Learning Research License (AMLR)\n"
                    "  - Research and non-commercial use only\n"
                    "  - See: https://github.com/apple/ml-depth-pro/blob/main/LICENSE"
                )
```

```python
# src/transformation_portal/depth/backends/registry.py
class DepthBackendRegistry:
    """Factory for depth backends with license governance."""

    def get_backend(
        self,
        backend_name: str,
        config: EnhanceConfig
    ) -> DepthBackend:
        """Get depth backend with license validation."""

        backend_cls = self._backends.get(backend_name)
        if backend_cls is None:
            raise ValueError(f"Unknown backend: {backend_name}")

        # License gate (Layer 2)
        if backend_cls.license_type == LicenseType.RESEARCH_ONLY:
            if not config.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"Backend '{backend_name}' requires non_commercial_ok=True"
                )

            # Depth Pro specific gate
            if backend_name == "depth_pro":
                if not config.accept_apple_depth_pro_research_license:
                    raise LicenseRestrictionError(
                        f"Backend 'depth_pro' requires accept_apple_depth_pro_research_license=True"
                    )

        return backend_cls(config)
```

**Rationale:**
- **Defense-in-Depth:** No single point of failure
- **Fail-Fast:** Config validation catches errors before model download/loading
- **Audit Trail:** License acceptance logged in runtime provenance
- **Clear UX:** Explicit config flags document legal requirements

---

### 3. DepthResult Dataclass Design

**DECISION: Extend Existing `lux_depth_v3/inference.py:DepthResult`**

**Rationale:**
- `DepthResult` is already the established contract for DA3InferenceEngine
- Adding fields is **backward compatible** (default values preserve existing behavior)
- Avoids proliferation of `DepthResult` variants across modules

**Migration Path:**
1. Update `lux_depth_v3/inference.py:DepthResult` with new fields (defaults preserve compatibility)
2. Move to `depth/backends/protocol.py` (centralized location)
3. Add deprecation alias in `lux_depth_v3/inference.py` for 6 months
4. Update all imports to new location

**Enhanced DepthResult:**

```python
@dataclass
class DepthResult:
    """Unified depth estimation result contract.

    Supports both relative depth (0-1 normalized) and metric depth (meters).
    """
    # Existing fields (v2.0.0 contract)
    depth_map: np.ndarray        # Shape: (H, W), values depend on depth_units
    original_image: np.ndarray   # Shape: (H, W, 3), RGB [0-255]
    metadata: Dict[str, Any]     # Backend-specific metadata

    # New fields for metric depth (backward compatible)
    depth_units: Literal["relative", "meters"] = "relative"
    focal_length_px: Optional[float] = None  # Focal length in pixels (metric depth)
    field_of_view_deg: Optional[float] = None  # Horizontal FOV in degrees

    @property
    def depth(self) -> np.ndarray:
        """Alias for depth_map (backward compatibility)."""
        return self.depth_map

    @property
    def is_metric(self) -> bool:
        """Check if depth is metric (absolute scale)."""
        return self.depth_units == "meters"
```

**Rejected Alternatives:**

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| New type in backend contract | Clean separation | Breaks existing code, forces migration | ❌ Rejected |
| Extend `stage_graph/stage.py:StageResult` | Stage-based only | Excludes non-stage usage (lux_depth_v3) | ❌ Rejected |
| Multiple DepthResult variants | Type safety per backend | Proliferation, hard to share utilities | ❌ Rejected |

---

### 4. Refactor Existing Backends?

**DECISION: Phased Migration (Do NOT Refactor Immediately)**

**Phase 1 (Current PR):** Add Depth Pro backend adapter ONLY
- Wrap `DepthProStage` logic in `DepthProBackend` class
- Wire into `DepthBackendRegistry`
- Keep DA2/DA3 unchanged

**Phase 2 (Future PR, Post-Validation):** Migrate DA3 if Depth Pro proves stable
- Create `DA3Backend` adapter wrapping `DA3InferenceEngine`
- Preserve existing `lux_depth_v3/inference.py` for 6 months (deprecation period)
- Update orchestrator to use registry

**Phase 3 (Optional):** Migrate DA2 if unified interface proves valuable
- Create `DA2Backend` adapter

**Rationale:**
- **Risk Minimization:** Don't disrupt stable DA2/DA3 code paths for experimental feature
- **Reversibility:** If Depth Pro fails evaluation, registry removal is clean
- **Validation First:** Prove backend abstraction with one adapter before refactoring existing code
- **v2.0.0 Stability:** DA3 is production Golden Path, changes require ADR justification

**Adapter Pattern (Phase 1):**

```python
# src/transformation_portal/depth/backends/depth_pro.py
class DepthProBackend:
    """Adapter wrapping DepthProStage for backend registry."""

    name = "depth_pro"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = True

    def __init__(self, config: EnhanceConfig):
        self._stage = DepthProStage(
            checkpoint_path=config.depth_pro_checkpoint_path,
            device=config.depth_device,
        )

    def compute(self, image, device=None) -> DepthResult:
        """Run Depth Pro inference."""
        context = StageContext(artifacts={"image": image})
        result = self._stage.compute(context)

        if result.status != StageStatus.COMPLETED:
            raise RuntimeError(f"Depth Pro failed: {result.error}")

        # Extract DepthResult from stage artifacts
        return DepthResult(
            depth_map=result.artifacts["depth_map"],
            original_image=image,
            metadata=result.artifacts.get("depth_provenance", {}),
            depth_units="meters",  # Depth Pro is metric
            focal_length_px=result.metadata.get("focal_length_px"),
            field_of_view_deg=result.metadata.get("fov_deg"),
        )
```

---

### 5. Enhanced Caching Strategy

**DECISION: `.npz` + `.json` Sidecar (Backward Compatible)**

**Cache File Patterns:**

```python
# Legacy (existing DA2/DA3 caches)
depth_cache/abc123.npy          # Depth map only, no metadata

# Enhanced (Depth Pro + future)
depth_cache/abc123.npz          # Depth map + focal_length + fov (compressed)
depth_cache/abc123.json         # Provenance metadata (human-readable)
```

**Cache Writer (with backward compatibility):**

```python
class DepthCacheWriter:
    """Write depth results to cache with metadata."""

    def write(self, cache_key: str, result: DepthResult) -> Path:
        """Write depth result to cache."""

        if result.is_metric:
            # Enhanced format for metric depth
            npz_path = self.cache_dir / f"{cache_key}.npz"
            json_path = self.cache_dir / f"{cache_key}.json"

            # Write compressed depth + metadata
            np.savez_compressed(
                npz_path,
                depth=result.depth_map,
                focal_length_px=result.focal_length_px,
                fov_deg=result.field_of_view_deg,
            )

            # Write provenance sidecar
            with open(json_path, 'w') as f:
                json.dump(result.metadata, f, indent=2)

            return npz_path
        else:
            # Legacy format for relative depth (backward compatible)
            npy_path = self.cache_dir / f"{cache_key}.npy"
            np.save(npy_path, result.depth_map)
            return npy_path

    def read(self, cache_key: str) -> Optional[DepthResult]:
        """Read depth result from cache (backward compatible)."""

        # Try enhanced format first
        npz_path = self.cache_dir / f"{cache_key}.npz"
        json_path = self.cache_dir / f"{cache_key}.json"

        if npz_path.exists():
            data = np.load(npz_path)
            metadata = {}
            if json_path.exists():
                with open(json_path) as f:
                    metadata = json.load(f)

            return DepthResult(
                depth_map=data["depth"],
                original_image=None,  # Not cached
                metadata=metadata,
                depth_units="meters",
                focal_length_px=data.get("focal_length_px"),
                field_of_view_deg=data.get("fov_deg"),
            )

        # Fallback to legacy format
        npy_path = self.cache_dir / f"{cache_key}.npy"
        if npy_path.exists():
            return DepthResult(
                depth_map=np.load(npy_path),
                original_image=None,
                metadata={},
                depth_units="relative",  # Legacy is always relative
            )

        return None
```

**Rationale:**
- **`.npz`:** Compressed, supports multiple arrays (depth + focal_length + fov)
- **`.json` Sidecar:** Human-readable provenance, easy debugging, no Python unpickling risk
- **Backward Compatible:** Existing `.npy` caches work unchanged
- **Migration Path:** Old caches remain valid, new writes use enhanced format

---

### 6. Experimental Presets

**DECISION: Add Two Depth Pro Presets**

**Preset 1: `depth_pro_metric_mps.yaml`**
```yaml
name: depth-pro-metric-mps
description: "Depth Pro metric depth for Apple Silicon (M1/M2/M3/M4)"
tier: experimental
license_restriction: research_only

depth_backend: depth_pro

model:
  variant: depth-pro
  device: mps
  checkpoint_path: checkpoints/depth_pro.pt
  expected_sha256: 3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce

processing:
  apply_bilateral: false
  enable_zone_mapping: false  # Metric depth incompatible with relative zone mapping

  pbr:
    enabled: true
    normal_strength: 1.2  # Metric depth has different gradient characteristics

io:
  cache_enabled: true
  output_format: npz  # Enhanced format
  depth_bit_depth: 32  # Float32 for metric values

compliance:
  non_commercial_ok: true
  accept_apple_depth_pro_research_license: true
```

**Preset 2: `depth_pro_metric_cpu.yaml`**
```yaml
name: depth-pro-metric-cpu
description: "Depth Pro metric depth CPU fallback (slow, for compatibility testing)"
tier: experimental
license_restriction: research_only

depth_backend: depth_pro

model:
  variant: depth-pro
  device: cpu  # Explicit CPU (no MPS/CUDA)

# Inherit rest from depth_pro_metric_mps.yaml
extends: depth_pro_metric_mps
```

---

## Consequences

### Positive

1. **Unified Contract:** All depth backends share `DepthBackend` Protocol
2. **License Safety:** Multi-layer enforcement prevents accidental research-only usage
3. **Metric Depth Support:** First-class support for absolute scale depth
4. **Backward Compatible:** Existing DA2/DA3 code unchanged, legacy caches work
5. **Reversible:** Depth Pro can be removed without affecting stable pipelines
6. **Clear Governance:** License requirements explicit in presets and config

### Negative

1. **Migration Debt:** Eventually need to migrate DA2/DA3 to backend contract (Phase 2/3)
2. **Caching Complexity:** Need to support both `.npy` (legacy) and `.npz` + `.json` (enhanced)
3. **License Friction:** Two boolean flags (`non_commercial_ok` + `accept_apple_depth_pro_research_license`) is verbose
4. **Adapter Overhead:** Wrapping `DepthProStage` in `DepthProBackend` adds indirection

### Risks

| Risk | Mitigation |
|------|------------|
| License enforcement bypassed | Multi-layer gates (config + registry + runtime) |
| Breaking changes to DA3 during migration | Phased approach, 6-month deprecation period |
| Cache invalidation on format change | Backward-compatible reader, cache_key includes format version |
| Depth Pro fails validation, wasted effort | Minimal refactoring, easy removal, adapter pattern isolates impact |

---

## Migration Plan

### Phase 1: Depth Pro Integration (Current PR)

1. Create `src/transformation_portal/depth/backends/` module
2. Implement `DepthBackend` Protocol and enhanced `DepthResult`
3. Implement `DepthBackendRegistry` with license gates
4. Create `DepthProBackend` adapter (wraps existing `DepthProStage`)
5. Add `accept_apple_depth_pro_research_license` to `EnhanceConfig`
6. Implement enhanced caching (`.npz` + `.json` sidecar)
7. Add two experimental presets
8. Update orchestrator to use registry when `depth_backend` is set

### Phase 2: DA3 Migration (Post-Validation, 6-8 weeks)

1. Create `DA3Backend` adapter
2. Add deprecation warnings to `lux_depth_v3/inference.py`
3. Update orchestrator to use `DA3Backend` by default
4. Maintain `DA3InferenceEngine` for 6 months

### Phase 3: DA2 Migration (Optional, 12+ weeks)

1. Evaluate if DA2 migration adds value
2. Create `DA2Backend` adapter if needed
3. Update `depth_canonical/models/registry.py` to use backend registry

---

## Required Enforcement

### Tests (PR1)

- [ ] `DepthBackend` Protocol compliance tests
- [ ] License gating unit tests (all three layers)
- [ ] Enhanced `DepthResult` construction and property tests
- [ ] Cache read/write tests (both `.npy` and `.npz` + `.json`)
- [ ] Preset validation tests (license flags required)
- [ ] `DepthBackendRegistry` selection tests

### CI Gates

- [ ] Unit tests pass with license enforcement
- [ ] Preset schema validation (requires license flags for `depth_pro`)
- [ ] Import tests verify new backend module
- [ ] Backward compatibility tests (legacy `.npy` caches)

### Documentation

- [ ] Update `README.md` with Depth Pro setup instructions
- [ ] Add license acceptance documentation
- [ ] Update preset documentation with tier markers
- [ ] Add `depth/backends/README.md` with backend contract specification

---

## Alternatives Considered

### Alternative 1: Single License Flag

Use only `non_commercial_ok=True` for all research-only models.

**Rejected:** Apple AMLR license is distinct from CC BY-NC 4.0 (DA3 1.1). Explicit acknowledgment of each license reduces legal risk.

### Alternative 2: Immediate DA2/DA3 Refactoring

Refactor all depth backends to use unified contract before adding Depth Pro.

**Rejected:** High risk, disrupts stable v2.0.0 code paths. Violates "validate before refactoring" principle.

### Alternative 3: Separate Depth Pro Repository

Fork Depth Pro integration into separate repository.

**Rejected:** Fragments ecosystem, duplicates CI, harder to maintain integration with LuxRender pipeline.

### Alternative 4: JSON-Only Caching

Use JSON for both depth data and metadata.

**Rejected:** JSON arrays are inefficient for large float32 arrays (5-10x larger than `.npz`).

---

## References

### Internal ADRs

- [ADR-018: Depth Pro Integration Decision](ADR-018-depth-pro-integration.md)
- [ADR-015: DA3 1.1 Non-Commercial Research Tier](ADR-015-da3-1-1-non-commercial-research-tier.md)
- [ADR-001: PBR Integration Architecture](ADR-001-PBR-Integration-Architecture.md)
- [Agent Governance Policy](agent_governance.md)

### External Resources

- [Apple Depth Pro Repository](https://github.com/apple/ml-depth-pro)
- [Apple Machine Learning Research License (AMLR)](https://github.com/apple/ml-depth-pro/blob/main/LICENSE)
- [Depth Anything V3 License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)

### Code References

- `src/transformation_portal/stage_graph/stages/depth_pro.py` (DepthProStage implementation)
- `src/transformation_portal/lux_depth_v3/inference.py` (DA3InferenceEngine + DepthResult)
- `src/transformation_portal/depth_canonical/models/registry.py` (ModelRegistry pattern)
- `config/presets/depth_pro_example.yaml` (Existing experimental preset)

---

**Document History**
- **2026-02-02:** Initial ADR-019 created (Architect architectural guidance response)
