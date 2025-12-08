# Depth Pipeline Architecture

**Version**: 2.0  
**Date**: December 2025  
**Status**: Production-Ready

---

## Overview

This document provides a comprehensive visual and textual overview of the Lux Depth V2 architecture, including data flow, component interactions, and processing stages.

---

## 1. High-Level Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        IMG[Input Image<br/>TIFF/PNG/JPEG]
        DEPTH[Pre-computed Depth<br/>Optional]
        CONFIG[Configuration<br/>Preset/YAML]
    end
    
    subgraph Core["Core Pipeline - LuxPipelineV2"]
        INIT[Pipeline Init<br/>Device/Model Setup]
        DEPTHGEN[Depth Estimation<br/>Depth Anything V2]
        ZONE[Zone Synthesis<br/>Quantile-based]
        MATSEG[Material Segmentation<br/>ONNX/SegFormer/Heuristic]
        MATPROF[Material Profile<br/>Application]
        PROC[Depth-Aware<br/>Processing]
        UPSCALE[Upscaling<br/>torch/onnx]
    end
    
    subgraph Output["Output Layer"]
        MASTER[16-bit TIFF Master]
        UPSCALED[Upscaled TIFF]
        MARKETING[8-bit PNG Marketing]
        PREVIEW[Preview JPEG]
        REPORT[Processing Report JSON]
    end
    
    IMG --> INIT
    DEPTH -.Optional.-> DEPTHGEN
    CONFIG --> INIT
    INIT --> DEPTHGEN
    DEPTHGEN --> ZONE
    ZONE --> MATSEG
    MATSEG --> MATPROF
    MATPROF --> PROC
    PROC --> UPSCALE
    UPSCALE --> MASTER
    UPSCALE --> UPSCALED
    UPSCALE --> MARKETING
    UPSCALE --> PREVIEW
    PROC --> REPORT
    
    style Input fill:#e1f5ff
    style Core fill:#fff3cd
    style Output fill:#d4edda
```

---

## 2. Detailed Processing Flow

```mermaid
flowchart TD
    START([Start]) --> LOAD[Load Image & Config]
    LOAD --> PRESET{Apply<br/>Preset?}
    PRESET -->|Yes| APPLY[Apply Preset Defaults]
    PRESET -->|No| VALIDATE
    APPLY --> VALIDATE[Validate Config]
    
    VALIDATE --> DEVICE[Select Device<br/>CUDA/MPS/CPU]
    DEVICE --> HASDDEPTH{Depth Map<br/>Provided?}
    
    HASDDEPTH -->|Yes| LOADDEPTH[Load Depth Map]
    HASDDEPTH -->|No| GENDEPTH[Generate Depth<br/>Depth Anything V2]
    
    LOADDEPTH --> NORMALIZE[Normalize Depth<br/>0-1 range]
    GENDEPTH --> NORMALIZE
    
    NORMALIZE --> ZONESYN[Zone Synthesis<br/>Foreground/Mid/Background]
    ZONESYN --> MANMASKS{Manual<br/>Masks?}
    
    MANMASKS -->|Yes| LOADMASKS[Load Zone Masks]
    MANMASKS -->|No| AUTOSYN[Auto Quantile Zones]
    
    LOADMASKS --> MATSEG
    AUTOSYN --> MATSEG[Material Segmentation]
    
    MATSEG --> BACKEND{Segmentation<br/>Backend}
    BACKEND -->|ONNX| ONNXSEG[ONNX Model]
    BACKEND -->|SegFormer| SEGFORMER[SegFormer Model]
    BACKEND -->|Heuristic| HEURISTIC[Color-based Rules]
    
    ONNXSEG --> MATPROF
    SEGFORMER --> MATPROF
    HEURISTIC --> MATPROF[Apply Material Profiles]
    
    MATPROF --> DENOISE[Depth-Aware Denoising<br/>Bilateral Filter]
    DENOISE --> TONEMAP[Zone Tone Mapping<br/>AgX/Reinhard/Filmic/ACES]
    TONEMAP --> ATMO{Atmospheric<br/>Effects?}
    
    ATMO -->|Yes| HAZE[Apply Haze/Fog]
    ATMO -->|No| CLARITY
    HAZE --> CLARITY[Clarity Enhancement<br/>Zone-weighted]
    
    CLARITY --> UPSCALEQ{Upscale<br/>Enabled?}
    UPSCALEQ -->|Yes| UPSCALER[Upscaling Backend<br/>torch/onnx]
    UPSCALEQ -->|No| EXPORT
    
    UPSCALER --> EXPORT[Export Outputs<br/>TIFF/PNG/JPEG]
    EXPORT --> TELEMETRY[Record Telemetry<br/>Timing/Memory]
    TELEMETRY --> END([End])
    
    style START fill:#90ee90
    style END fill:#90ee90
    style GENDEPTH fill:#ffd700
    style MATSEG fill:#ffd700
    style TONEMAP fill:#ffd700
    style UPSCALER fill:#ffd700
```

---

## 3. Component Architecture

```mermaid
graph LR
    subgraph API["API Layer"]
        CLI[CLI<br/>cli.py]
        SERVICE[FastAPI Service<br/>service.py]
        PYAPI[Python API<br/>pipeline.process_one]
    end
    
    subgraph Pipeline["Pipeline Core"]
        LUXPIPE[LuxPipelineV2<br/>pipeline.py]
        CONFIG[PipelineConfig<br/>config.py]
        PRESET[Preset Enum<br/>5 presets]
    end
    
    subgraph Processing["Processing Modules"]
        TORCHOPS[GPU Operations<br/>torch_ops.py]
        IOUTILS[I/O Utilities<br/>io_utils.py]
        WEIGHTS[Model Weights<br/>weights.py]
    end
    
    subgraph Materials["Material System"]
        MATSEGMENT[Material Segmentation<br/>material_segmentation.py]
        MATPROFILES[Material Profiles<br/>material_profiles.py]
    end
    
    subgraph Upscaling["Upscaling System"]
        TORCHUP[TorchUpscaler]
        ONNXUP[OnnxUpscaler]
        NOUP[NoOpUpscaler]
    end
    
    subgraph External["External Models"]
        DEPTHANYTHING[Depth Anything V2<br/>HuggingFace]
        SEGFORMER[SegFormer<br/>Optional]
        ONNXMODEL[ONNX Models<br/>Optional]
    end
    
    CLI --> LUXPIPE
    SERVICE --> LUXPIPE
    PYAPI --> LUXPIPE
    
    LUXPIPE --> CONFIG
    LUXPIPE --> TORCHOPS
    LUXPIPE --> IOUTILS
    LUXPIPE --> MATSEGMENT
    LUXPIPE --> TORCHUP
    LUXPIPE --> ONNXUP
    
    CONFIG --> PRESET
    MATSEGMENT --> MATPROFILES
    MATSEGMENT --> DEPTHANYTHING
    MATSEGMENT --> SEGFORMER
    MATSEGMENT --> ONNXMODEL
    
    LUXPIPE --> WEIGHTS
    
    style API fill:#e1f5ff
    style Pipeline fill:#fff3cd
    style Processing fill:#f8d7da
    style Materials fill:#d4edda
    style Upscaling fill:#cfe2ff
    style External fill:#f5c6cb
```

---

## 4. Depth Processing Stages

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant DepthModel
    participant ZoneSynth
    participant MaterialSeg
    participant Processor
    participant Upscaler
    
    User->>Pipeline: process_one(image_path)
    activate Pipeline
    
    Pipeline->>Pipeline: Load & Validate Config
    Pipeline->>Pipeline: Select Device (CUDA/MPS/CPU)
    
    alt Depth Map Provided
        Pipeline->>Pipeline: Load existing depth
    else Generate Depth
        Pipeline->>DepthModel: estimate_depth(image)
        activate DepthModel
        DepthModel-->>Pipeline: depth_map (normalized)
        deactivate DepthModel
    end
    
    Pipeline->>ZoneSynth: synthesize_zones(depth)
    activate ZoneSynth
    ZoneSynth-->>Pipeline: foreground, midground, background masks
    deactivate ZoneSynth
    
    opt Material Enhancement Enabled
        Pipeline->>MaterialSeg: segment_materials(image)
        activate MaterialSeg
        MaterialSeg-->>Pipeline: material_masks
        deactivate MaterialSeg
        
        Pipeline->>Pipeline: Apply material profiles per surface
    end
    
    Pipeline->>Processor: apply_depth_aware_processing
    activate Processor
    Processor->>Processor: Depth-aware denoising
    Processor->>Processor: Zone tone mapping
    Processor->>Processor: Atmospheric effects (if enabled)
    Processor->>Processor: Clarity enhancement
    Processor-->>Pipeline: processed_image
    deactivate Processor
    
    opt Upscaling Enabled
        Pipeline->>Upscaler: upscale(image, scale=4)
        activate Upscaler
        Upscaler-->>Pipeline: upscaled_image
        deactivate Upscaler
    end
    
    Pipeline->>Pipeline: Export outputs (TIFF/PNG/JPEG)
    Pipeline->>Pipeline: Generate processing report
    
    Pipeline-->>User: result dict (status, paths, timing)
    deactivate Pipeline
```

---

## 5. Preset Configuration Flow

```mermaid
stateDiagram-v2
    [*] --> ConfigCreation
    
    ConfigCreation --> PresetSelection: Set preset enum
    PresetSelection --> DefaultValues: Apply preset defaults
    
    state DefaultValues {
        [*] --> PhotoRealistic
        [*] --> InteriorLuxury
        [*] --> ExteriorShowcase
        [*] --> Architectural
        [*] --> ArchivalQuality
        
        PhotoRealistic: Balanced, conservative
        InteriorLuxury: High clarity, 4 zones
        ExteriorShowcase: Atmospheric effects
        Architectural: Technical accuracy
        ArchivalQuality: Maximum fidelity
    }
    
    DefaultValues --> ParameterOverride: User customizations
    ParameterOverride --> Validation: Validate config
    Validation --> DeviceSelection: Pick device
    DeviceSelection --> PipelineInit: Initialize pipeline
    PipelineInit --> [*]
```

---

## 6. Material Segmentation Backends

```mermaid
graph TD
    START([Material Segmentation Request]) --> SELECT{Backend<br/>Selection}
    
    SELECT -->|onnx| ONNX[ONNX Backend]
    SELECT -->|segformer| SEGFORMER[SegFormer Backend]
    SELECT -->|heuristic| HEURISTIC[Heuristic Backend]
    SELECT -->|auto| AUTO[Auto-select fastest available]
    
    ONNX --> ONNXLOAD[Load ONNX Model]
    ONNXLOAD --> ONNXINFER[Run Inference<br/>20-30ms]
    ONNXINFER --> MERGE
    
    SEGFORMER --> SEGLOAD[Load SegFormer Model]
    SEGLOAD --> SEGINFER[Run Inference<br/>50-80ms]
    SEGINFER --> MERGE
    
    HEURISTIC --> COLORSPACE[Convert to LAB]
    COLORSPACE --> RULES[Apply Color Rules<br/>5-10ms]
    RULES --> MERGE
    
    AUTO --> AVAIL{ONNX<br/>Available?}
    AVAIL -->|Yes| ONNX
    AVAIL -->|No| AVAIL2{SegFormer<br/>Available?}
    AVAIL2 -->|Yes| SEGFORMER
    AVAIL2 -->|No| HEURISTIC
    
    MERGE([Material Masks]) --> PROFILE[Apply Material Profiles]
    PROFILE --> OUTPUT([Enhanced Image])
    
    style START fill:#90ee90
    style MERGE fill:#90ee90
    style OUTPUT fill:#90ee90
    style ONNX fill:#ffd700
    style SEGFORMER fill:#ffd700
    style HEURISTIC fill:#87ceeb
```

---

## 7. Zone-Based Tone Mapping

```mermaid
graph TB
    INPUT[Image + Depth Map] --> QUANTILE[Compute Depth Quantiles]
    QUANTILE --> FG[Foreground Zone<br/>0-35th percentile]
    QUANTILE --> MG[Midground Zone<br/>35-70th percentile]
    QUANTILE --> BG[Background Zone<br/>70-100th percentile]
    
    FG --> FGPARAMS[FG Parameters<br/>High contrast: 1.3<br/>High saturation: 1.15<br/>Exposure: +0.1]
    MG --> MGPARAMS[MG Parameters<br/>Medium contrast: 1.1<br/>Medium saturation: 1.05<br/>Exposure: 0.0]
    BG --> BGPARAMS[BG Parameters<br/>Low contrast: 0.95<br/>Low saturation: 0.95<br/>Exposure: -0.05]
    
    FGPARAMS --> TONEMAP{Tone Map<br/>Operator}
    MGPARAMS --> TONEMAP
    BGPARAMS --> TONEMAP
    
    TONEMAP -->|AgX| AGX[Film-inspired<br/>Shoulder roll-off]
    TONEMAP -->|Reinhard| REIN[Classic HDR<br/>Compression]
    TONEMAP -->|Filmic| FILM[Hable Uncharted 2<br/>S-curve]
    TONEMAP -->|ACES| ACES[Academy ODT<br/>Industry standard]
    
    AGX --> BLEND
    REIN --> BLEND
    FILM --> BLEND
    ACES --> BLEND[Blend Zones<br/>Depth-weighted]
    
    BLEND --> OUTPUT[Final Tone-Mapped Image]
    
    style INPUT fill:#e1f5ff
    style OUTPUT fill:#d4edda
    style TONEMAP fill:#ffd700
```

---

## 8. Data Flow Summary

### Input → Processing → Output

| Stage | Input | Processing | Output | Time (ms) |
|-------|-------|------------|--------|-----------|
| **1. Depth Estimation** | RGB Image | Depth Anything V2 inference | Depth map (normalized) | 25-65 |
| **2. Zone Synthesis** | Depth map | Quantile computation | 3-4 zone masks | 5-10 |
| **3. Material Segmentation** | RGB Image | ONNX/SegFormer/Heuristic | Material masks (6+ types) | 5-80 |
| **4. Material Profiles** | Image + Material masks | Per-material enhancements | Enhanced image | 20-40 |
| **5. Denoising** | Image + Depth | Bilateral filtering | Denoised image | 15-25 |
| **6. Tone Mapping** | Image + Zones | Zone-based operators | Tone-mapped image | 30-50 |
| **7. Atmospheric** | Image + Depth | Haze/fog simulation | Enhanced image | 10-20 |
| **8. Clarity** | Image + Zones | Zone-weighted sharpening | Sharpened image | 15-25 |
| **9. Upscaling** | Processed image | torch/onnx upscaler | 2x or 4x upscaled | 150-300 |
| **10. Export** | Final image | TIFF/PNG/JPEG encoding | Output files | 50-100 |

**Total End-to-End**: 300-500ms per image (1024×1024, 4x upscale, CUDA)

---

## 9. Memory Architecture

```mermaid
graph TB
    subgraph GPU["GPU Memory (VRAM)"]
        MODEL[Depth Model<br/>~500MB]
        TENSORS[Image Tensors<br/>~1-2GB]
        SEGMODEL[Segmentation Model<br/>~500MB-2GB]
        UPMODEL[Upscaling Model<br/>~500MB]
    end
    
    subgraph CPU["System Memory (RAM)"]
        IMAGE[Raw Images<br/>~50-200MB each]
        DEPTH[Depth Maps<br/>~10-50MB each]
        BUFFER[Processing Buffer<br/>~500MB]
    end
    
    subgraph Disk["Storage"]
        INPUT[Input Images<br/>TIFF/PNG/JPEG]
        OUTPUT[Output Files<br/>TIFF/PNG/JPEG]
        CACHE[Depth Cache<br/>~/.cache/]
    end
    
    INPUT --> IMAGE
    IMAGE --> TENSORS
    TENSORS --> MODEL
    MODEL --> DEPTH
    DEPTH --> CACHE
    TENSORS --> SEGMODEL
    TENSORS --> UPMODEL
    UPMODEL --> BUFFER
    BUFFER --> OUTPUT
    
    style GPU fill:#ffd700
    style CPU fill:#87ceeb
    style Disk fill:#d4edda
```

**Peak Memory by Configuration**:
- Minimal (no upscale): 2-3 GB VRAM, 2-3 GB RAM
- Standard (4x upscale): 4-5 GB VRAM, 3-4 GB RAM  
- High throughput (batch 4): 8-10 GB VRAM, 4-6 GB RAM
- Maximum quality (2048px, 4x): 12-16 GB VRAM, 6-8 GB RAM

---

## 10. Security Architecture

```mermaid
graph TB
    subgraph Public["Public Interface"]
        CLI[CLI Entry Point]
        API[FastAPI Service]
    end
    
    subgraph Validation["Input Validation"]
        FILETYPE[File Type Check<br/>TIFF/PNG/JPEG only]
        FILESIZE[File Size Limit<br/>Max 50MB]
        PATHSEC[Path Sanitization<br/>No ../ traversal]
        RATELIMIT[Rate Limiting<br/>Requests/min]
    end
    
    subgraph Core["Core Pipeline"]
        PIPELINE[LuxPipelineV2<br/>Safe operations]
    end
    
    subgraph Dependencies["Dependencies"]
        SAFEDEPS[Safe Dependencies<br/>torch, onnx]
        NOVULN[No CVE-2024-27763<br/>No basicsr/realesrgan]
    end
    
    CLI --> VALIDATION
    API --> VALIDATION
    VALIDATION --> PIPELINE
    PIPELINE --> SAFEDEPS
    PIPELINE --> NOVULN
    
    style Validation fill:#ffd700
    style NOVULN fill:#d4edda
```

**Security Features**:
- ✅ Input validation (file type, size, path sanitization)
- ✅ Rate limiting (prevent abuse)
- ✅ CVE-2024-27763 mitigation (removed vulnerable dependencies)
- ✅ Safe upscaling backends (torch/onnx only)
- ✅ No arbitrary code execution risks

---

## 11. Deployment Architecture

```mermaid
graph TB
    subgraph Local["Local Development"]
        LOCALPYTHON[Python Script]
        LOCALCLI[CLI Tool]
    end
    
    subgraph Docker["Docker Container"]
        DOCKERFILE[Dockerfile]
        COMPOSE[docker-compose.yml]
    end
    
    subgraph Cloud["Cloud Deployment"]
        K8S[Kubernetes Pod]
        LB[Load Balancer]
        STORAGE[Persistent Storage]
    end
    
    LOCALPYTHON --> PIPELINE[Lux Depth V2 Pipeline]
    LOCALCLI --> PIPELINE
    DOCKERFILE --> PIPELINE
    COMPOSE --> PIPELINE
    K8S --> PIPELINE
    LB --> K8S
    PIPELINE --> STORAGE
    
    style Local fill:#e1f5ff
    style Docker fill:#fff3cd
    style Cloud fill:#d4edda
```

---

## 12. Performance Optimization Paths

```mermaid
graph LR
    START([Baseline<br/>500ms/image]) --> OPT1{Reduce<br/>Upscale?}
    
    OPT1 -->|2x instead of 4x| FASTER1[~200ms/image<br/>2.5x faster]
    OPT1 -->|No| OPT2
    
    OPT2{Use<br/>Heuristic<br/>Segmentation?}
    OPT2 -->|Yes| FASTER2[~450ms/image<br/>10% faster]
    OPT2 -->|No| OPT3
    
    OPT3{Batch<br/>Processing?}
    OPT3 -->|Batch 4-8| FASTER3[~400ms/image<br/>20% faster]
    OPT3 -->|No| OPT4
    
    OPT4{Use<br/>FP16?}
    OPT4 -->|Yes| FASTER4[~350ms/image<br/>30% faster]
    OPT4 -->|No| OPT5
    
    OPT5{Pre-compute<br/>Depth?}
    OPT5 -->|Yes| FASTER5[~460ms/image<br/>8% faster]
    
    FASTER1 --> END([Optimized])
    FASTER2 --> END
    FASTER3 --> END
    FASTER4 --> END
    FASTER5 --> END
    
    style START fill:#f8d7da
    style END fill:#d4edda
    style FASTER1 fill:#ffd700
    style FASTER2 fill:#ffd700
    style FASTER3 fill:#ffd700
    style FASTER4 fill:#ffd700
    style FASTER5 fill:#ffd700
```

---

## 13. Key Design Decisions

### 13.1 Modular Architecture
- **Why**: Enables independent testing, easy swapping of backends (e.g., segmentation, upscaling)
- **Trade-off**: Slightly more complex initialization vs monolithic design

### 13.2 GPU-First Design
- **Why**: Modern depth/segmentation models require GPU for acceptable performance
- **Trade-off**: CPU fallback is ~10x slower but available

### 13.3 Preset System
- **Why**: Simplified UX for common use cases, reduces parameter overwhelm
- **Trade-off**: Less granular control unless overriding presets

### 13.4 Optional Depth Input
- **Why**: Allows pre-computed depth maps for faster iteration
- **Trade-off**: Users must manage depth map lifecycle separately

### 13.5 Safe Upscaling Only
- **Why**: CVE-2024-27763 security vulnerability in RealESRGAN/basicsr
- **Trade-off**: Removed one upscaling backend, but torch/onnx alternatives are safe and high-quality

---

## 14. Extension Points

Future enhancements can plug into these extension points:

1. **Custom Tone Mapping Operators**: Add to `torch_ops.py`
2. **New Material Types**: Extend `material_profiles.py`
3. **Additional Segmentation Backends**: Implement interface in `material_segmentation.py`
4. **Custom Upscaling Models**: Implement `Upscaler` interface in `upscaling.py`
5. **New Presets**: Add to `Preset` enum and `apply_preset()` in `config.py`
6. **Video Processing**: Add temporal smoothing to `pipeline.py`

---

## 15. References

- **Pipeline Implementation**: `lux_depth_v2/pipeline.py`
- **Configuration**: `lux_depth_v2/config.py`
- **Test Suite**: `lux_depth_v2/tests/`
- **Documentation**: `lux_depth_v2/README.md`, `lux_depth_v2/SECURITY.md`
- **RAG Analysis**: `DEPTH_PROCESSING_PATTERNS_RAG_REPORT.md`

---

**Document Version**: 2.0  
**Last Updated**: December 8, 2025  
**Maintainer**: Transformation Portal Team
