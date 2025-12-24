# LuxPipelineV2 Flow Diagram

**Last Updated:** 2025-12-23  
**Author:** Pipeline Architecture Team  
**Purpose:** Visual reference for LuxPipelineV2 architecture, stage dependencies, and error handling

---

## Quick Reference

**For Onboarding:** Jump to [Flow Diagram](#flow-diagram) and [Visual Legend](#visual-legend)  
**For Performance Analysis:** See [Timing Checkpoint Mapping](#timing-checkpoint-mapping-to-python-implementation) and [Performance Optimization Guidelines](#performance-optimization-guidelines)  
**For Code Reviews:** Check [Code Traceability Matrix](#code-traceability-matrix) and [Stage Dependencies](#stage-dependencies--execution-order)  
**For Debugging:** Review [Error Handling Patterns](#error-handling-patterns) and [Configuration Flags](#configuration-flags)

---

## Pipeline Overview

This diagram illustrates the complete LuxPipelineV2 workflow from input to export, including:
- **Main pipeline stages** (Input → Material Processing → Grading → Upscaling → Export)
- **Optional paths** (Materials v2, Materials v3, AI upscaler, tiling)
- **Error fallbacks** (dashed lines show graceful degradation)
- **Timing checkpoints** (instrumentation points for performance profiling)
- **Decision nodes** (configuration-driven branching)

---

## Flow Diagram

```mermaid
flowchart TD
    %% -------------------------
    %% Input Stage
    %% -------------------------
    subgraph Input["Input Stage"]
        A[Image File] -->|Exists?| B{File Found?}
        B -->|Yes| C[Read Image (io_utils.read_rgb_any)]
        B -->|No| Z[Skip / Error Log]
        C --> TC1((t_io_read_input))
        C --> D{Depth Provided?}
        D -->|Yes| E[Load Depth Map (_find_depth)]
        D -->|No| F{strict_depth?}
        F -->|True| Z
        F -->|False| G[Use Uniform Weights]
        E --> TC2((t_io_read_depth))
    end

    %% -------------------------
    %% Material Processing Stage
    %% -------------------------
    subgraph Material["Material Processing"]
        G --> H{Segmentation Enabled?}
        H -->|Yes| I[Legacy Material Segmentation]
        H -->|No| J[Skip Segmentation]
        I --> TC3((t_material_segmentation))
        
        %% Materials V2 path
        I --> K{Materials V2 Enabled?}
        K -->|Yes| L[Materials V2 Engine]
        K -->|No| M[Skip Materials V2]
        L --> TC4((t_materials_v2))
        
        %% Materials V3 path
        L --> N{Materials V3 Enabled?}
        M --> N
        I --> N
        N -->|Yes| O[Materials V3 Engine (Plan + Pixel Ops)]
        N -->|No| P[Skip Materials V3]
        O --> TC5((t_materials_v3))

        %% Error fallbacks
        I -.->|Segmentation Fail| J
        L -.->|V2 Fail| M
        O -.->|V3 Fail| P
    end

    %% -------------------------
    %% Grading Stage
    %% -------------------------
    subgraph Grading["Grading & Master Image"]
        P --> Q[Grade Image (torch_ops.grade_core)]
        Q --> TC6((t_grade_master))
        Q --> R[Soft Clip / Highlight Compress]
    end

    %% -------------------------
    %% Upscaling Stage
    %% -------------------------
    subgraph Upscaling["Upscaling & AI Enhancement"]
        R --> S{Use AI Upscaler?}
        S -->|Yes| T[GPU Upscaler (torch_ops.upscaler.upscale)]
        S -->|No| U[Bicubic Upscale]
        T --> TC7((t_ai_upscale))
        T --> V{AI Drift Check (RGB/Luma)}
        V -->|Pass| W[Use AI Detail Transfer]
        V -->|Fail| U
        W --> X[Optional Tiling Processing (_stage / Tiler)]
        U --> X
        X --> TC8((t_tile_postprocessing))
    end

    %% -------------------------
    %% Export Stage
    %% -------------------------
    subgraph Export["Export & Reporting"]
        X --> Y{Write Outputs?}
        Y -->|Yes| Z1[Export Master TIFF (ExportManager / atomic_write)]
        Y -->|Yes| Z2[Export Upscaled TIFF]
        Y -->|Yes| Z3[Export Marketing PNG]
        Y -->|Yes| Z4[Export Preview JPG]
        Y -->|Yes| Z5[Export JSON Report]
        Y -->|No| ZZ[Skip Writing Outputs]
        Z1 --> TC9((t_export_master))
        Z5 --> TC10((t_export_report))
    end

    %% -------------------------
    %% Connect Main Flow
    %% -------------------------
    C --> H
    G --> H
    H --> K
    K --> N
    P --> Q
    Q --> S
    X --> Y

    %% -------------------------
    %% Decision Nodes Styling
    %% -------------------------
    classDef decision fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    class B,D,F,H,K,N,S,Y decision

    %% -------------------------
    %% Optional / Fallback Paths Styling
    %% -------------------------
    classDef optional fill:#cff,stroke-dasharray: 5 5,stroke:#00f,color:#000
    class J,M,P,U,ZZ optional

    %% -------------------------
    %% Timing / Logging Nodes Styling
    %% -------------------------
    classDef timing fill:#ffd,stroke:#333,stroke-width:1px,color:#000
    class TC1,TC2,TC3,TC4,TC5,TC6,TC7,TC8,TC9,TC10 timing

    %% -------------------------
    %% Critical Paths Highlight
    %% -------------------------
    classDef critical fill:#ffa,stroke:#f60,stroke-width:2px,color:#000
    class C,E,G,I,L,O,Q,R,T,X,Z1,Z2,Z3,Z4,Z5 critical
```

---

## Visual Legend

**Color Coding:**
- 🟠 **Orange (Critical Paths):** Core processing stages that always execute when conditions are met
- 🔵 **Blue Dashed (Optional/Fallback):** Graceful degradation paths when features are disabled or fail
- 🟡 **Yellow (Timing Checkpoints):** Performance instrumentation points for profiling
- 🩷 **Pink (Decision Nodes):** Configuration-driven branching logic

**Line Styles:**
- **Solid Lines:** Normal execution flow
- **Dashed Lines:** Error fallback or optional skip paths

---

## Stage Descriptions

### Input Stage
- **Purpose:** Validate input files and load image + depth map
- **Key Decision:** `strict_depth` - fail if depth map missing vs. use uniform weights
- **Timing Checkpoints:** `t_io_read_input`, `t_io_read_depth`
- **Error Handling:** Skip processing if file not found; warn and continue if depth missing (non-strict mode)

### Material Processing Stage
- **Purpose:** Segment materials and apply surface-aware enhancements
- **Components:**
  - **Legacy Segmentation:** Optional backward-compatible material detection
  - **Materials V2:** Confidence-scored segmentation with caching
  - **Materials V3:** Plan-based processing with glass/stone pixel operations
- **Timing Checkpoints:** `t_material_segmentation`, `t_materials_v2`, `t_materials_v3`
- **Error Handling:** Graceful fallback for each material version (dashed lines)

### Grading Stage
- **Purpose:** Apply core color grading with material-aware highlight compression
- **Function:** `torch_ops.grade_core()` with soft clipping
- **Timing Checkpoint:** `t_grade_master`
- **Error Handling:** None (critical path)

### Upscaling Stage
- **Purpose:** Enhance resolution with GPU-accelerated upscaling
- **Key Decision:** `enable_ai_upscaler` - use AI detail transfer vs. bicubic fallback
- **AI Validation:** RGB/Luma drift check before applying AI enhancement
- **Tiling:** Optional tile-based processing for large images (>50MP)
- **Timing Checkpoints:** `t_ai_upscale`, `t_tile_postprocessing`
- **Error Handling:** Fallback to bicubic if AI drift exceeds threshold

### Export Stage
- **Purpose:** Write multiple output formats and metadata
- **Outputs:**
  - Master TIFF (16-bit, full precision)
  - Upscaled TIFF (high-resolution master)
  - Marketing PNG (web-optimized)
  - Preview JPG (thumbnail)
  - JSON Report (reproducibility metadata)
- **Timing Checkpoints:** `t_export_master`, `t_export_report`
- **Error Handling:** Skip all outputs if `write_outputs=False`

---

## Configuration Flags

| Flag | Stage | Effect |
|------|-------|--------|
| `strict_depth` | Input | Fail if depth map missing |
| `enable_legacy_material_mods` | Material | Enable backward-compatible segmentation |
| `enable_materials_v2` | Material | Enable confidence-scored segmentation |
| `enable_materials_v3` | Material | Enable plan-based pixel operations |
| `enable_ai_upscaler` | Upscaling | Use AI detail transfer (with drift validation) |
| `use_tiles` | Upscaling | Enable tile-based processing for large images |
| `write_outputs` | Export | Enable/disable file writes |
| `validate_ai` | Upscaling | Enable AI drift validation (RGB/Luma checks) |

---

## Timing Checkpoint Mapping to Python Implementation

This table maps each timing checkpoint (TC1–TC10) in the flow diagram to the corresponding `_stage()` context manager in `pipeline.py::LuxPipelineV2.process_one()`:

| Checkpoint | Stage Name | Python Location | Description | Dependencies |
|------------|------------|-----------------|-------------|--------------|
| **TC1** | `io/read_input` | `pipeline.py:422` | Reads input RGB image using `io_utils.read_rgb_any()`. Validates file existence and format. | File I/O |
| **TC2** | `io/read_depth` | `pipeline.py:493` | Loads depth map via `_find_depth()` and `io_utils.read_depth_u16()`. Critical for weighted grading; warns if missing and `strict_depth=False`. | File I/O, Depth Map |
| **TC3** | `material/segmentation` | `pipeline.py:512` | Runs base material segmentation via `create_material_segmenter()` and `segmenter.predict()`. Produces masks for Materials V2/V3 or legacy grading. | Segmentation Model |
| **TC4** | `material/materials_v2` | `pipeline.py:535` | Materials V2 engine integration. Checks cache via `MaskCacheManager`. Performs `segment_with_confidence()` and stores metrics (confidence, coverage, quality). Optional fallback if Materials V2 fails. | Materials V2 Engine, Cache |
| **TC5** | `material/materials_v3` | `pipeline.py:609` | Materials V3 integration. Processes `seg_result_for_v3` including plan mode, pixel operations (glass/stone), and response plan. Fallback logging if fails. Environment killswitch: `DISABLE_MATERIALS_V3`. | Materials V3 Engine |
| **TC6** | `grade/master` | `pipeline.py:695` | Master image grading via `torch_ops.grade_core()`. Applies soft clip and highlight compress with optional legacy material mods. | GPU/MPS, Materials |
| **TC7** | `upscale/{backend}` | `pipeline.py:749` | Upscaling stage: GPU upscaler (`self.upscaler.upscale()`) or bicubic fallback. Includes AI drift check for RGB/Luma differences. Backend-specific timing (e.g., `upscale/torch`, `upscale/onnx`). | GPU/MPS, AI Upscaler |
| **TC7 (alt)** | `upscale/base` | `pipeline.py:741` | Bicubic fallback upscaling when AI upscaler is disabled or drift validation fails. | GPU/MPS |
| **TC8** | *(implicit)* | N/A (inferred) | Optional tiling for large images via `torch_ops.Tiler`. Applies detail transfer, clarity, sharpen, and material highlight compress per tile. Not explicitly instrumented as separate stage. | GPU/MPS, Tiling |
| **TC9** | `export_master` | `pipeline.py:706` | Writes master TIFF image via `ExportManager.write_master()` or direct `atomic_write()`. Critical for archival output. 16-bit precision maintained. | File I/O, ExportManager |
| **TC10** | `export_report` | `pipeline.py:879` | Writes JSON report with full metadata: reproducibility info (git commit, config hash, device), stage timings, Materials V2/V3 stats, AI validation results, and export paths. | File I/O |

### Additional Export Stages (Not in Diagram)

These stages are also instrumented but not shown in the main flow diagram:

| Stage Name | Python Location | Description |
|------------|-----------------|-------------|
| `export/autotune` | `pipeline.py:433` | JIT autotuning for ExportManager configuration (runs once per session). |
| `export_preview` | `pipeline.py:714` | Writes preview JPG thumbnail for quick inspection. |
| `export_upscaled` | `pipeline.py:814` | Writes upscaled TIFF (high-resolution master). |
| `export_marketing` | `pipeline.py:821` | Writes marketing PNG (web-optimized, sRGB color space). |
| `material/cleanup` | `pipeline.py:733` | Cleanup of temporary material segmentation artifacts. |

---

## Code Traceability Matrix

For **QA validation** and **performance analysis**, use this matrix to trace diagram nodes to implementation:

```python
# Example: Finding TC4 (Materials V2) timing in JSON report
{
  "timings": {
    "io/read_input": 0.052,          # TC1
    "io/read_depth": 0.031,          # TC2
    "material/segmentation": 0.287,  # TC3
    "material/materials_v2": 0.145,  # TC4
    "material/materials_v3": 0.213,  # TC5
    "grade/master": 0.089,           # TC6
    "upscale/torch": 1.234,          # TC7
    "export_master": 0.321,          # TC9
    "export_report": 0.012           # TC10
  }
}
```

### Timing Checkpoint Verification

To verify all timing checkpoints are being captured:

```bash
# Run pipeline and extract timing keys from JSON report
lux-depth-v2 --input-dir test/ --output-dir out/ --preset interior_luxury
cat out/test_image_report.json | jq '.timings | keys[]'

# Expected output should include:
# "io/read_input"              (TC1)
# "io/read_depth"              (TC2)
# "material/segmentation"      (TC3)
# "material/materials_v2"      (TC4, if enabled)
# "material/materials_v3"      (TC5, if enabled)
# "grade/master"               (TC6)
# "upscale/torch"              (TC7, if AI upscaler enabled)
# "export_master"              (TC9)
# "export_report"              (TC10)
```

---

## Timing Instrumentation

All timing checkpoints are captured via the `_stage()` context manager and logged to the JSON report:

```python
with self._stage("stage_name"):
    # Stage processing logic
    pass
# Timing automatically recorded in self.timings
```

**Checkpoint Reference:**
- `t_io_read_input` - Input image read time
- `t_io_read_depth` - Depth map read time
- `t_material_segmentation` - Legacy segmentation time
- `t_materials_v2` - Materials V2 processing time
- `t_materials_v3` - Materials V3 processing time
- `t_grade_master` - Core grading time
- `t_ai_upscale` - AI upscaling time
- `t_tile_postprocessing` - Tiling overhead
- `t_export_master` - Master TIFF export time
- `t_export_report` - JSON report generation time

---

## Error Handling Patterns

### Graceful Fallback (Dashed Lines)
- **Segmentation Fail** → Skip to grading
- **Materials V2 Fail** → Skip to Materials V3 check
- **Materials V3 Fail** → Continue to grading
- **AI Drift Fail** → Fallback to bicubic upscaling

### Critical Failures (Stop Processing)
- **File Not Found** → Skip image, log error
- **Strict Depth Missing** → Skip image, log error

---

## Performance Characteristics

| Stage | Typical Time (4K) | Memory Usage | GPU Utilized |
|-------|-------------------|--------------|--------------|
| Input | 50-100ms | ~50MB | No |
| Material Segmentation | 200-400ms | ~200MB | Optional |
| Materials V2 | 100-200ms | ~150MB | Yes |
| Materials V3 | 150-300ms | ~200MB | Yes |
| Grading | 50-100ms | ~100MB | Yes |
| AI Upscaling | 500-1500ms | ~1-2GB | Yes |
| Bicubic Upscaling | 100-200ms | ~200MB | Yes |
| Export | 200-500ms | ~300MB | No |

**Total (Typical 4K, AI Upscaler):** ~2-3 seconds  
**Total (Typical 4K, Bicubic):** ~1-1.5 seconds

---

## Future Improvements

Based on recent code review feedback:

1. **Memory Management**
   - Add streaming chunked reads for extreme resolutions (>100MP)
   - Document memory requirements per resolution tier

2. **Retry Mechanism**
   - Implement configurable retry logic for Materials V3 and AI upscaler
   - Exponential backoff with graceful failure after max retries

3. **Perceptual Validation**
   - Integrate SSIM or LPIPS for AI drift detection
   - Configurable thresholds for perceptual vs. fast validation

4. **Code Readability**
   - Move `_stage()` context manager to dedicated `timing.py` module
   - Enable reuse across other pipeline modules

---

## Stage Dependencies & Execution Order

Understanding the critical path through the pipeline is essential for optimization:

### Critical Path (Always Executes)
1. **TC1** (`io/read_input`) → **TC6** (`grade/master`) → **TC7/alt** (`upscale/base`) → **TC9** (`export_master`) → **TC10** (`export_report`)

**Minimum Execution Time (4K image, bicubic upscale):** ~0.5-0.7 seconds

### Extended Path (Materials V2 Enabled)
1. **TC1** → **TC2** → **TC3** (`material/segmentation`) → **TC4** (`material/materials_v2`) → **TC6** → **TC7** → **TC9** → **TC10**

**Typical Execution Time (4K image, AI upscale):** ~2-3 seconds

### Full Path (All Features Enabled)
1. **TC1** → **TC2** → **TC3** → **TC4** → **TC5** (`material/materials_v3`) → **TC6** → **TC7** (`upscale/torch` + AI drift check) → **TC8** (tiling if >50MP) → **TC9** → **TC10**

**Maximum Execution Time (4K image, all features):** ~3-4 seconds

### Parallel Export Operations
The following exports can run in parallel after grading:
- `export_preview` (JPG thumbnail)
- `export_upscaled` (high-res TIFF)
- `export_marketing` (web PNG)

### Optional Paths (Conditional Execution)
- **TC2** skipped if depth map missing and `strict_depth=False` → uses uniform weights
- **TC3/TC4/TC5** skipped if segmentation disabled or fails → continues to grading
- **TC7** uses bicubic fallback if AI drift exceeds threshold (RGB/Luma validation)
- **TC8** only runs for images >50MP when `use_tiles=True`

---

## Performance Optimization Guidelines

Based on timing checkpoint data:

### Bottleneck Identification
| Stage | Typical % of Total | Optimization Strategy |
|-------|-------------------|----------------------|
| **TC3** (Segmentation) | 10-15% | Use cached masks via `MaskCacheManager` |
| **TC4** (Materials V2) | 5-8% | Pre-compute segmentation offline |
| **TC5** (Materials V3) | 8-12% | Disable if not needed for material type |
| **TC7** (AI Upscale) | 40-60% | Use bicubic for batch processing, AI for final deliverables |
| **TC9** (Export Master) | 10-15% | Use SSD storage, enable compression |

### Recommended Presets by Use Case

**Speed Priority (Real-time Preview):**
```bash
lux-depth-v2 --preset quick_preview \
  --upscaler-backend bicubic \
  --enable-materials-v3 false
```
**Expected:** <1 second per 4K image

**Quality Priority (Final Deliverable):**
```bash
lux-depth-v2 --preset ultra_quality \
  --upscaler-backend torch \
  --enable-materials-v3 true \
  --validate-ai true
```
**Expected:** 3-4 seconds per 4K image

**Balanced (Production Batch):**
```bash
lux-depth-v2 --preset interior_luxury \
  --upscaler-backend torch \
  --enable-materials-v2 true \
  --enable-materials-v3 false
```
**Expected:** 2-3 seconds per 4K image

---

## Related Documentation

- **Pipeline Implementation:** `lux_depth_v2/pipeline.py`
- **Material Segmentation:** `lux_depth_v2/materials_v2.py`, `lux_depth_v2/materials_v3.py`
- **Upscaling Backends:** `lux_depth_v2/upscaling.py`
- **Export Manager:** `lux_depth_v2/io_utils.py` (ExportManager class)
- **Configuration Guide:** `lux_depth_v2/config.py`
- **Security Guidelines:** `lux_depth_v2/SECURITY.md`
- **Phase Completion Reports:** `lux_depth_v2/PHASE2_COMPLETE.md`, `lux_depth_v2/PHASE3_COMPLETE.md`

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-23 | Initial diagram creation with full stage breakdown | Pipeline Team |
| 2025-12-23 | Added timing checkpoint mapping to Python implementation | Pipeline Team |
| 2025-12-23 | Added code traceability matrix and performance guidelines | Pipeline Team |
| 2025-12-23 | Added stage dependencies and execution order analysis | Pipeline Team |

---

## Validation Checklist

Use this checklist to verify pipeline integrity after code changes:

### Diagram Consistency
- [ ] All decision nodes (pink diamonds) correspond to actual configuration flags
- [ ] All timing checkpoints (yellow circles) have matching `_stage()` calls in `pipeline.py`
- [ ] All error fallback paths (blue dashed) have corresponding exception handling
- [ ] Critical path stages (orange) are correctly identified and instrumented

### Code Traceability
- [ ] Each TC1-TC10 checkpoint maps to a valid line number in `pipeline.py`
- [ ] JSON report contains all expected timing keys (run verification command)
- [ ] Stage names in diagram match `_stage()` context manager names exactly
- [ ] Additional export stages are documented (autotune, preview, marketing)

### Performance Validation
- [ ] Typical execution times align with documented characteristics table
- [ ] Bottleneck identification percentages are still accurate (re-profile if needed)
- [ ] Recommended presets produce expected timing results
- [ ] AI drift validation thresholds are correctly configured

### Documentation Accuracy
- [ ] All Python line numbers are current (update if code has changed)
- [ ] Configuration flags table reflects actual `PipelineConfig` options
- [ ] Error handling patterns match actual exception handling code
- [ ] Future improvements section is up-to-date with roadmap

### Integration Testing
```bash
# Run full pipeline with all features enabled
lux-depth-v2 --input-dir test/ --output-dir out/ \
  --preset interior_luxury \
  --enable-materials-v2 true \
  --enable-materials-v3 true \
  --validate-ai true

# Verify JSON report structure
test -f out/*_report.json && \
  jq -e '.timings | has("io/read_input")' out/*_report.json && \
  jq -e '.reproducibility | has("git_commit")' out/*_report.json && \
  echo "✅ Pipeline integrity validated"
```

---

**End of Document**
