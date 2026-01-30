# PBR Integration Architecture - Visual Overview

## Current State (Fragmented)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAGMENTED DEPTH MODULES                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  src/transformation_portal/depth/          (19 files, 872KB)    │
│  ├── models/depth_anything_v2.py           [DeviceType #1]      │
│  ├── pipeline.py (ArchitecturalDepthPipeline) [DepthConfig #1]  │
│  ├── processors/                                                 │
│  │   ├── zone_tone_mapping.py                                   │
│  │   ├── depth_aware_denoise.py                                 │
│  │   └── atmospheric_effects.py                                 │
│  └── utils/cache.py                                              │
│                                                                   │
│  src/transformation_portal/depth_intelligence/ (2 files, 36KB)  │
│  ├── depth_estimator.py                    [DepthEstimator #1]  │
│  └── __init__.py                           [DepthConfig #2]      │
│                                                                   │
│  src/transformation_portal/lux_depth_v3/   (20 files, 340KB)    │
│  ├── orchestrator.py (EnhanceOrchestrator)                      │
│  ├── da3_model_backend.py                  [DeviceType #2]      │
│  ├── inference.py (DA3InferenceEngine)     [DepthEstimator #2]  │
│  ├── pbr.py                            ⭐ NEW PBR MODULE         │
│  ├── pbr_writer.py                                               │
│  ├── depth_writer.py                                             │
│  └── security.py                                                 │
│                                                                   │
│  OTHER DUPLICATES:                                               │
│  ├── core/config/schemas.py               [DeviceType #3]      │
│  ├── core/device/detector.py              [DeviceType #4]      │
│  ├── foundation/device_manager.py         [DeviceType #5]      │
│  └── pipelines/rendering_4k_pipeline.py   (DepthConfig, DeviceType) │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
              ⬇ CONSOLIDATION (Phase 1-2)
```

## Target State (Unified)

```
┌─────────────────────────────────────────────────────────────────┐
│             UNIFIED DEPTH CANONICAL MODULE (~25 files)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  src/transformation_portal/depth_canonical/                     │
│  │                                                                │
│  ├── __init__.py              ⭐ PUBLIC API SURFACE              │
│  │   ├── DepthConfig (unified)                                   │
│  │   ├── PBRConfig                                               │
│  │   ├── DeviceType (canonical)                                  │
│  │   ├── ModelVariant (DA2 + DA3)                                │
│  │   ├── DepthPipeline (orchestrator)                            │
│  │   ├── generate_pbr_maps                                       │
│  │   └── write_pbr_maps                                          │
│  │                                                                │
│  ├── config.py                 [SINGLE SOURCE OF TRUTH]         │
│  │   ├── DeviceType (Enum)                                       │
│  │   ├── ModelVariant (Enum)                                     │
│  │   ├── PBRConfig (frozen dataclass)                            │
│  │   ├── ProcessingConfig                                        │
│  │   └── DepthConfig                                             │
│  │                                                                │
│  ├── device.py                 [CANONICAL DEVICE DETECTION]     │
│  │                                                                │
│  ├── models/                                                     │
│  │   ├── base.py              (DepthEstimator interface)         │
│  │   ├── depth_anything_v3.py (DA3 - from lux_depth_v3)         │
│  │   └── depth_anything_v2.py (DA2 - from depth/)               │
│  │                                                                │
│  ├── processing/                                                 │
│  │   ├── inference.py         (from lux_depth_v3)                │
│  │   ├── postprocessing.py    (from lux_depth_v3)                │
│  │   ├── pbr.py               ⭐ INTEGRATED PBR                  │
│  │   ├── zone_mapping.py      (from depth/)                      │
│  │   ├── denoise.py           (from depth/)                      │
│  │   └── atmospheric.py       (from depth/)                      │
│  │                                                                │
│  ├── io/                                                         │
│  │   ├── depth_writer.py      (atomic writes)                    │
│  │   ├── pbr_writer.py        (atomic writes)                    │
│  │   └── cache.py             (LRU caching)                      │
│  │                                                                │
│  ├── security/                                                   │
│  │   └── validation.py        (path sanitization)                │
│  │                                                                │
│  └── pipeline.py              [UNIFIED ORCHESTRATOR]            │
│      └── DepthPipeline                                           │
│          ├── process_image()                                     │
│          ├── batch_process()                                     │
│          └── Optional PBR generation                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    USER ENTRY POINTS                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  CLI                    Python API              Pipeline          │
│  ├─ depth_process       ├─ DepthPipeline        ├─ Lux Render    │
│  │  --pbr               │  .process_image()     │                 │
│  │  --preset interior   │  .batch_process()     ├─ Unified Luxury │
│  └─ batch_tiff          └─ generate_pbr_maps()  └─ Video Master  │
│                                                                    │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│            DEPTH CANONICAL PIPELINE ORCHESTRATOR                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  DepthPipeline.__init__(config: DepthConfig)                     │
│  │                                                                 │
│  ├─ Load config from preset (YAML)                                │
│  ├─ Initialize InferenceEngine (DA2/DA3)                          │
│  ├─ Initialize Postprocessor                                      │
│  ├─ Initialize Cache (LRU, optional)                              │
│  └─ Initialize PBR config (if enabled)                            │
│                                                                    │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Step 1: DEPTH ESTIMATION                                         │
│  ┌──────────────────────────────────────────┐                    │
│  │ InferenceEngine.estimate(image)          │                    │
│  │  ├─ Check cache (SHA256 key)             │                    │
│  │  ├─ Load image                            │                    │
│  │  ├─ Preprocess (resize, normalize)       │                    │
│  │  ├─ Model inference (DA2/DA3)            │                    │
│  │  └─ Return depth map (H, W) float32      │                    │
│  └──────────────────────────────────────────┘                    │
│                 │                                                  │
│                 ▼                                                  │
│  Step 2: DEPTH POSTPROCESSING                                     │
│  ┌──────────────────────────────────────────┐                    │
│  │ Postprocessor.refine(depth)              │                    │
│  │  ├─ Bilateral filter (optional)          │                    │
│  │  ├─ Edge preservation                    │                    │
│  │  ├─ Metric scaling                       │                    │
│  │  └─ Return refined depth                 │                    │
│  └──────────────────────────────────────────┘                    │
│                 │                                                  │
│                 ▼                                                  │
│  Step 3: PBR GENERATION (if enabled)                              │
│  ┌──────────────────────────────────────────┐                    │
│  │ generate_pbr_maps(depth, pbr_config)     │                    │
│  │  ├─ Normal Map:                          │                    │
│  │  │   ├─ Blur depth (optional)            │                    │
│  │  │   ├─ Sobel gradients (dx, dy)         │                    │
│  │  │   ├─ Cross product → normals          │                    │
│  │  │   └─ Encode RGB: [-1,1]→[0,255]       │                    │
│  │  │                                        │                    │
│  │  ├─ Roughness Map:                       │                    │
│  │  │   ├─ Laplacian (2nd derivative)       │                    │
│  │  │   ├─ Box blur smoothing               │                    │
│  │  │   └─ Normalize to [0, 255]            │                    │
│  │  │                                        │                    │
│  │  └─ Ambient Occlusion Map:               │                    │
│  │      ├─ Laplacian for concavity          │                    │
│  │      ├─ Wide box blur (occlusion spread) │                    │
│  │      ├─ Apply AO bias                    │                    │
│  │      └─ Normalize to [0, 255]            │                    │
│  │                                           │                    │
│  │  Performance: ~420ms @ 4K (CPU)          │                    │
│  └──────────────────────────────────────────┘                    │
│                 │                                                  │
│                 ▼                                                  │
│  Step 4: ATOMIC WRITES                                            │
│  ┌──────────────────────────────────────────┐                    │
│  │ Write outputs atomically                 │                    │
│  │  ├─ write_depth_u16_png(depth)           │                    │
│  │  │   └─ {basename}_depth.png             │                    │
│  │  │                                        │                    │
│  │  └─ write_pbr_maps(normal, rough, ao)    │                    │
│  │      ├─ {basename}_normal.png            │                    │
│  │      ├─ {basename}_roughness.png         │                    │
│  │      └─ {basename}_ao.png                │                    │
│  │                                           │                    │
│  │  Security: Temp file + atomic rename     │                    │
│  └──────────────────────────────────────────┘                    │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONFIGURATION SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  config/architectural_interior_pbr.yaml                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ depth_model:                             │                   │
│  │   variant: "da3-metric-large"            │                   │
│  │   device: "cpu"                          │                   │
│  │   cache_enabled: true                    │                   │
│  │                                           │                   │
│  │ processing:                               │                   │
│  │   apply_bilateral: true                   │                   │
│  │   enable_zone_mapping: true               │                   │
│  │   num_zones: 4                            │                   │
│  │                                           │                   │
│  │   pbr:                         ⭐ PBR    │                   │
│  │     enabled: true                         │                   │
│  │     normal_strength: 1.2                  │                   │
│  │     roughness_strength: 1.5               │                   │
│  │     ao_strength: 1.0                      │                   │
│  │     ao_bias: 0.3                          │                   │
│  └──────────────────────────────────────────┘                   │
│                 │                                                 │
│                 ▼                                                 │
│  DepthConfig.from_preset("architectural_interior_pbr")           │
│  ┌──────────────────────────────────────────┐                   │
│  │ DepthConfig(                             │                   │
│  │   model = DA3_METRIC_LARGE,              │                   │
│  │   device = CPU,                          │                   │
│  │   processing = ProcessingConfig(         │                   │
│  │     apply_bilateral = True,              │                   │
│  │     enable_zone_mapping = True,          │                   │
│  │     pbr = PBRConfig(                     │                   │
│  │       enabled = True,                    │                   │
│  │       normal_strength = 1.2,             │                   │
│  │       ...                                 │                   │
│  │     )                                     │                   │
│  │   )                                       │                   │
│  │ )                                         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Migration Path

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIGRATION TIMELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 1: Consolidation (Weeks 1-2)                             │
│  ┌──────────────────────────────────────────┐                   │
│  │ ✓ Create depth_canonical/                │                   │
│  │ ✓ Migrate PBR module (unchanged)         │                   │
│  │ ✓ Unified DepthConfig                    │                   │
│  │ ✓ Canonical DeviceType                   │                   │
│  │ ✓ Unit tests                             │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  PHASE 2: Integration (Weeks 3-4)                               │
│  ┌──────────────────────────────────────────┐                   │
│  │ ✓ DepthPipeline orchestrator             │                   │
│  │ ✓ Migrate processors                     │                   │
│  │ ✓ CLI tools                              │                   │
│  │ ✓ Update pipelines                       │                   │
│  │ ✓ Integration tests                      │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  PHASE 3: Deprecation (Weeks 5-6)                               │
│  ┌──────────────────────────────────────────┐                   │
│  │ ✓ Deprecation warnings                   │                   │
│  │ ✓ Compatibility shims                    │                   │
│  │ ✓ Migration guide                        │                   │
│  │ ✓ CI enforcement                         │                   │
│  │ ✓ Performance validation                 │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  PHASE 4: Removal (v2.0.0, 3-6 months)                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ ✓ Remove depth/                          │                   │
│  │ ✓ Remove lux_depth_v3/                   │                   │
│  │ ✓ Remove depth_intelligence/             │                   │
│  │ ✓ Update all documentation               │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  Backward Compatibility Window: 6 months                         │
│  Support Policy: Security fixes + critical bugs only             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Targets

```
┌─────────────────────────────────────────────────────────────────┐
│                  PERFORMANCE BENCHMARKS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Component                 │ 4K Image  │ Batch Throughput       │
│  ──────────────────────────┼───────────┼────────────────────    │
│  Depth Estimation (DA3)    │  24-65ms  │ 150-200 img/hr        │
│  Depth Postprocessing      │  10-20ms  │ -                      │
│  PBR Generation (Total)    │  ~420ms   │ 150 img/hr            │
│    ├─ Normal Map          │  ~140ms   │ -                      │
│    ├─ Roughness Map       │  ~140ms   │ -                      │
│    └─ Ambient Occlusion   │  ~140ms   │ -                      │
│  Atomic Writes (3 PBR)     │  30-50ms  │ -                      │
│  ──────────────────────────┼───────────┼────────────────────    │
│  TOTAL (Depth + PBR)       │  ~500ms   │ 100-120 img/hr        │
│                                                                   │
│  Optimization Opportunities:                                     │
│  ✓ Parallel PBR generation (ThreadPoolExecutor)                 │
│  ✓ Model batching (process multiple images in single pass)      │
│  ✓ I/O overlap (write while processing next image)              │
│  ✓ LRU caching (10-20x speedup for repeated images)             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   SECURITY LAYERS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: INPUT VALIDATION                                       │
│  ┌──────────────────────────────────────────┐                   │
│  │ validate_input_path(path)                │                   │
│  │  ├─ Prevent path traversal (..)          │                   │
│  │  ├─ Reject unsafe filenames              │                   │
│  │  └─ Validate file extensions             │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  Layer 2: ATOMIC WRITES                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ write_pbr_maps() / write_depth_u16_png() │                   │
│  │  ├─ Write to .tmp file                   │                   │
│  │  ├─ Verify write success                 │                   │
│  │  ├─ Atomic rename (all-or-nothing)       │                   │
│  │  └─ Cleanup on failure                   │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  Layer 3: CONFIGURATION IMMUTABILITY                             │
│  ┌──────────────────────────────────────────┐                   │
│  │ PBRConfig (frozen=True)                  │                   │
│  │  ├─ Prevent runtime modification         │                   │
│  │  ├─ Safe for caching (hash-able)         │                   │
│  │  └─ Reproducible results                 │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
│  Layer 4: CI ENFORCEMENT                                         │
│  ┌──────────────────────────────────────────┐                   │
│  │ ✓ Security tests in CI                   │                   │
│  │ ✓ Banned import detection                │                   │
│  │ ✓ Pre-commit hooks                       │                   │
│  │ ✓ CodeQL scanning                        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies Graph

```
┌─────────────────────────────────────────────────────────────────┐
│              MODULE DEPENDENCY ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Code / CLI / Pipelines                                     │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────────────────────────┐                        │
│  │  depth_canonical/__init__.py        │  ⭐ PUBLIC API         │
│  │  (Stable public interface)          │                        │
│  └────────┬────────────────────────────┘                        │
│           │                                                       │
│           ├─────────────┬──────────────┬──────────────┐         │
│           ▼             ▼              ▼              ▼         │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  config.py │  │pipeline.py│  │processing│  │   io/    │     │
│  └────────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                         │              │              │         │
│                         ▼              ▼              ▼         │
│                  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│                  │  models/ │  │   pbr.py │  │  cache   │     │
│                  └──────────┘  └──────────┘  └──────────┘     │
│                         │                                       │
│                         ▼                                       │
│                  ┌──────────┐                                   │
│                  │ security/│                                   │
│                  └──────────┘                                   │
│                                                                   │
│  RULE: No circular dependencies                                 │
│  RULE: No imports from deprecated modules                       │
│  RULE: Private modules hidden from public API                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

**Legend:**
- ⭐ = New/Enhanced Feature
- ✓ = Deliverable/Checkpoint
- [CAPS] = Architectural Invariant
