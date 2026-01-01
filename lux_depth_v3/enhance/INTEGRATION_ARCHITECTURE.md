# V3 + V2 Integration Architecture Guide

## Overview

This document describes the architectural design of the V3 + V2 enhancement orchestrator, which integrates Depth Anything 3 (DA3) depth estimation with the production-ready V2 enhancement pipeline.

## Design Principles

### 1. Single Source of Truth
- **V2 remains canonical** for enhancement logic (weights, grading, upscaling, export, reporting)
- **V3 provides depth backend** using DA3 models
- **No code duplication** - V3 does not reimplement V2's enhancement stack

### 2. Clean Boundaries
- **Stage A (V3)**: Depth generation only
- **Stage B (V2)**: Enhancement only
- **Orchestrator**: Coordinates the two stages, writes combined manifest

### 3. Contract Compliance
- **Depth output format**: uint16 PNG, single-channel, `{stem}_depth.png`
- **Shape guarantee**: (H, W) exactly matching input image
- **V2 input contract**: Depth directory with `{stem}_depth.png` files

### 4. Provenance & Reproducibility
- **Combined manifests** link all processing stages
- **Git hashes** tracked for both V3 and V2
- **Model versions** and parameters recorded
- **Timing breakdowns** for performance analysis

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    V3 Orchestrator                          │
│                                                             │
│  ┌─────────────┐         ┌─────────────────────────────┐   │
│  │   Input     │         │    Output Structure         │   │
│  │   Manager   │─────▶   │                             │   │
│  └─────────────┘         │  depth/                     │   │
│                          │    <stem>_depth.png         │   │
│  ┌─────────────┐         │  v2/                        │   │
│  │    DA3      │         │    <stem>_master16.tif      │   │
│  │  Inference  │─────▶   │    <stem>_report.json       │   │
│  │   Engine    │         │  manifests/                 │   │
│  └─────────────┘         │    <stem>_combined.json     │   │
│         │                │  logs/                      │   │
│         │                │    v2_<stem>.log            │   │
│         ▼                └─────────────────────────────┘   │
│  ┌─────────────┐                                           │
│  │   Depth     │                                           │
│  │   Writer    │                                           │
│  └─────────────┘                                           │
│         │                                                   │
│         │ uint16 PNG                                       │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │           V2 Runner (Subprocess)            │           │
│  │                                             │           │
│  │  python -m lux_depth_v2.cli \              │           │
│  │    --input <image> \                       │           │
│  │    --depth-dir depth/ \                    │           │
│  │    --output-dir v2/ \                      │           │
│  │    --preset production_ultra               │           │
│  │                                             │           │
│  └─────────────────────────────────────────────┘           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │  Manifest   │                                           │
│  │   Writer    │                                           │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### `lux_depth_v3/enhance/`

```
enhance/
├── __init__.py           # Lazy imports, public API
├── depth_writer.py       # Depth I/O (uint16 PNG contract)
├── manifest.py          # Combined manifest schema
├── orchestrator.py      # Two-stage pipeline coordination
├── v2_runner.py         # V2 subprocess invocation
├── README.md            # Full documentation
└── QUICK_START.md       # Quick start guide
```

### Key Components

#### 1. Depth Writer (`depth_writer.py`)

**Responsibility**: Write depth maps in V2-compatible format

**Contract**:
- Input: float32 or uint16 depth array
- Output: uint16 PNG, single-channel, shape (H, W)
- Quantization: Percentile-based (p1p99, p0.5p99.5, minmax)
- Validation: Optional debug mode verifies read-back

**Usage**:
```python
from lux_depth_v3.enhance import write_depth_u16_png

depth = inference_engine.predict(image).depth
p1, p99 = write_depth_u16_png(
    Path("depth/image_depth.png"),
    depth,
    method="p1p99",
    debug_verify=True,
)
```

#### 2. Manifest Writer (`manifest.py`)

**Responsibility**: Combined manifest linking V3 + V2

**Schema Version**: `lux-depth-v3.enhance.v1`

**Components**:
- `InputMetadata`: Image path, SHA256 hash
- `DepthMetadata`: Model, license, quantization, runtime
- `V2Metadata`: Preset, status, report path
- `TimingMetadata`: Stage and total timings
- `ReproMetadata`: Git hashes, Python version, device

**Usage**:
```python
from lux_depth_v3.enhance import CombinedManifest

manifest = CombinedManifest(
    input=input_metadata,
    depth=depth_metadata,
    v2=v2_metadata,
    timing=timing_metadata,
    repro=repro_metadata,
)
manifest.write(Path("manifests/image_combined.json"))
```

#### 3. V2 Runner (`v2_runner.py`)

**Responsibility**: Subprocess invocation of V2

**Features**:
- Auto-detects V2 module location
- Captures stdout/stderr to log files
- Timeout handling
- Error detection and reporting
- Report finding utilities

**Usage**:
```python
from lux_depth_v3.enhance.v2_runner import V2Runner

runner = V2Runner()
result = runner.run(
    input_path=image_path,
    depth_dir=Path("depth"),
    output_dir=Path("v2"),
    preset="production_ultra",
    log_file=Path("logs/v2_image.log"),
    timeout=600,
)
```

#### 4. Orchestrator (`orchestrator.py`)

**Responsibility**: Coordinate V3 depth + V2 enhancement

**Execution Flow**:
1. Initialize DA3 inference engine (V3)
2. Initialize V2 runner
3. For each image:
   - Check resume conditions (skip if outputs exist)
   - Stage A: Generate depth with DA3
   - Write depth as uint16 PNG
   - Stage B: Run V2 with depth input
   - Write combined manifest
4. Report summary statistics

**Configuration**:
```python
from lux_depth_v3.enhance import EnhanceConfig

config = EnhanceConfig(
    model_variant=ModelVariant.METRIC_LARGE,
    v2_preset="production_ultra",
    depth_fallback="skip",
    force_depth=False,
    force_v2=False,
    non_commercial_ok=True,
)
```

## Data Flow

### Input → Depth → V2 → Manifest

```
1. Input Image (renders/luxury_estate.jpg)
   ↓
2. DA3 Inference (Stage A)
   - Load image
   - Run depth estimation
   - Quantize to uint16
   ↓
3. Depth Write
   - depth/luxury_estate_depth.png (uint16 PNG)
   - Shape: (H, W)
   - Scaling: p1p99
   ↓
4. V2 Enhancement (Stage B)
   - Load image + depth
   - Apply weights (depth-aware)
   - Color grading
   - Upscaling
   - Export (16-bit TIFF, 8-bit PNG/JPG)
   - Generate report
   ↓
5. V2 Outputs
   - v2/luxury_estate_master16.tif
   - v2/luxury_estate_report.json
   - v2/luxury_estate_marketing.png
   ↓
6. Combined Manifest
   - manifests/luxury_estate_combined.json
   - Links: depth metadata + V2 report + timings
```

## Resume Logic

### Skip Conditions

**Depth Stage**:
- Skip if `depth/<stem>_depth.png` exists AND `--force-depth` not set
- Load existing depth metadata if available

**V2 Stage**:
- Skip if V2 report found in `v2/` AND `--force-v2` not set
- Report search patterns: `{stem}_report.json`, `report_{stem}.json`

**Manifest Stage**:
- Always regenerate (cheap, ensures consistency)

### Force Flags

**`--force-depth`**: Regenerate depth even if exists
**`--force-v2`**: Re-run V2 even if outputs exist

## Failure Handling

### Depth Fallback Policies

#### `fail` (default)
- Stop immediately on depth failure
- Raise exception
- Best for: Production validation, critical workflows

#### `skip`
- Log warning, continue to next image
- Write manifest with `status: skipped`
- Best for: Batch processing with error tolerance

#### `v2-auto`
- If DA3 depth fails, clear depth_dir for this image
- V2 will auto-generate depth using DA2
- Manifest marks `strict_depth: false`
- Best for: Fallback to V2's depth when DA3 unavailable

## Performance Considerations

### Execution Modes

#### Sequential (Default)
- Process one image at a time
- Depth → V2 for image N, then image N+1
- Safe for single-GPU setups
- No OOM risk

#### Pipelined (Future)
- Worker A: Depth for image N
- Worker B: V2 for image N-1
- Requires: Multi-GPU or device pinning
- 1.5-2x throughput improvement

### Optimization Strategies

1. **Model caching**: Pre-cache DA3 models (see V3 model cache)
2. **Resume support**: Skip existing outputs
3. **Device pinning**: Separate GPUs for depth/V2
4. **Timeout tuning**: Adjust `--v2-timeout` for large images

## Security Considerations

### License Enforcement

**DA3 Models (CC-BY-NC)**:
- Require explicit `--non-commercial-ok` flag
- Write to every manifest: `non_commercial_ok: true`
- Prevents accidental commercial misuse

**Validation**:
```python
# V3 validates license before DA3 initialization
validate_license(
    config.model_variant,
    commercial_use=not config.non_commercial_ok,
    strict=True,
)
```

### Input Validation

**V2 Runner**:
- Validates file paths before subprocess
- Sanitizes command arguments
- Captures stderr for error analysis

**Depth Writer**:
- Validates shape (2D or 3D with single channel)
- Checks for NaN/Inf values
- Ensures uint16 output contract

## Testing Strategy

### Unit Tests

**Depth Writer**:
- `test_write_depth_u16_png_from_uint16`: Direct uint16 write
- `test_write_depth_u16_png_from_float`: Float quantization
- `test_write_depth_u16_png_3channel`: Multi-channel handling
- `test_write_depth_u16_png_invalid_shape`: Shape validation
- `test_write_depth_u16_png_with_nan`: NaN detection

**Manifest**:
- `test_manifest_creation`: Component assembly
- `test_manifest_to_dict`: Dictionary conversion
- `test_manifest_to_json`: JSON serialization
- `test_manifest_write_and_load`: Persistence roundtrip

**V2 Runner**:
- `test_v2_runner_initialization`: Auto-detection
- `test_find_v2_report`: Report location utilities

### Integration Tests

**Orchestrator**:
- Requires: Full environment (torch, V2, test images)
- Tests: End-to-end workflow, resume logic, failure modes

## Deployment Checklist

### Prerequisites
- [ ] V3 installed: `pip install -e lux_depth_v3/`
- [ ] V2 installed: `pip install -e lux_depth_v2/`
- [ ] Models cached: `lux-depth-v3 cache-download --set production`
- [ ] Test images prepared

### Validation
- [ ] Test depth writer: `pytest lux_depth_v3/tests/test_enhance.py -k depth_writer`
- [ ] Test manifest: `pytest lux_depth_v3/tests/test_enhance.py -k manifest`
- [ ] Test CLI help: `lux-depth-v3 enhance --help`

### Production Run
- [ ] Small batch first (1-5 images)
- [ ] Review manifests for correctness
- [ ] Check V2 reports for quality metrics
- [ ] Verify output directory structure
- [ ] Monitor for timeouts/failures
- [ ] Scale to full batch

## Troubleshooting Guide

### Common Issues

**"V2 not found"**
- Ensure `lux_depth_v2` in PYTHONPATH or installed
- Check `V2Runner` auto-detection logic

**"License error"**
- Add `--non-commercial-ok` flag
- For commercial: use `--model metric-large`

**"V2 timeout"**
- Increase `--v2-timeout` (default 600s)
- Check GPU availability
- Review V2 logs in `logs/v2_<stem>.log`

**"Resume not working"**
- Remove `--force-*` flags
- Check file paths in manifests
- Verify output directory structure

## Future Enhancements

### Planned Features
- [ ] Pipelined execution mode (multi-GPU)
- [ ] Batch depth processing (multiple images per DA3 call)
- [ ] Distributed processing (multiple machines)
- [ ] Real-time monitoring dashboard
- [ ] Quality gate automation (reject low-quality depth)

### Considered But Deferred
- V2 inline mode (import instead of subprocess) - increases coupling
- Depth caching across runs - filesystem already provides this
- Automatic preset selection - requires domain-specific heuristics

## References

- **V2 Documentation**: `lux_depth_v2/README.md`
- **DA3 Integration**: `lux_depth_v3/DA3_API_INTEGRATION_COMPLETE.md`
- **Model Caching**: `lux_depth_v3/docs/MODEL_CACHING_GUIDE.md`
- **Metric Depth**: `lux_depth_v3/docs/METRIC_DEPTH_GUIDE.md`
- **License Guide**: `lux_depth_v3/docs/LICENSE_GUIDE.md`
