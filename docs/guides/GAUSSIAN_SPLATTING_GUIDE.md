# 3D Gaussian Splatting Integration Guide

**Status:** Phase 2.3 (Experimental)
**Backend:** graphdeco-inria/gaussian-splatting
**License:** Inria Research License (non-commercial)
**Video Mode:** ⚠️ Not applicable (multi-view only)

---

## Overview

3D Gaussian Splatting (3DGS) is a state-of-the-art neural rendering technique for real-time radiance field reconstruction. In the Transformation Portal, it provides geometric verification for multi-view luxury real estate captures.

### Key Features

- ✅ **Depth-guided optimization** - Integrates with Depth Pro / DA3 depth maps
- ✅ **Multi-view reconstruction** - 3+ views for geometric consistency
- ✅ **PLY/OBJ export** - Standard 3D format output
- ✅ **RMSE quality validation** - Automatic convergence metrics
- ⚠️ **Research-only license** - Inria non-commercial license

### Use Cases

1. **Geometric Verification** - Validate depth maps against 3D reconstruction
2. **Novel View Synthesis** - Generate new camera angles
3. **Quality Assessment** - RMSE-based depth consistency checks
4. **Scene Export** - PLY point clouds for external rendering

---

## Installation

### 1. Prerequisites

```bash
# Ensure PyTorch is installed (required)
pip install torch torchvision

# Install gsplat (optional, for GPU acceleration)
pip install gsplat
```

### 2. Tier Restriction

3D Gaussian Splatting is restricted to research tiers due to the Inria license:

```python
# ✅ Valid tiers
tier = "apex_research"      # OK
tier = "apex_research_ultra"  # OK
tier = "experimental"       # OK

# ❌ Invalid tiers (raises LicenseRestrictionError)
tier = "commercial"         # NOT ALLOWED
tier = "elite"              # NOT ALLOWED
tier = "production"         # NOT ALLOWED
```

---

## Quick Start

### Basic 3-View Reconstruction

```python
from transformation_portal.spatial_ai.reconstruction import (
    GaussianBackend,
    ReconstructionInput,
    CameraParams,
)
import numpy as np

# 1. Initialize backend (research tier required)
backend = GaussianBackend(
    tier="apex_research",
    device="auto",  # auto-detect: cuda > mps > cpu
)

# 2. Prepare multi-view input
images = [load_image(f"view_{i}.tiff") for i in range(3)]
cameras = [CameraParams(intrinsics, extrinsics, width, height) for _ in range(3)]

# 3. Create reconstruction input
input_data = ReconstructionInput(
    images=images,
    gamma=1.0,  # Must be linear RGB
    cameras=cameras,
    tier="apex_research",
)

# 4. Run reconstruction
scene = backend.reconstruct(input_data, iterations=7000)

# 5. Check quality metrics
print(f"RMSE: {scene.rmse:.4f}")
print(f"Gaussians: {scene.splats.num_gaussians}")
print(f"Convergence: {scene.convergence}")
```

### Command-Line Usage

```bash
# Multi-view reconstruction with default preset
python -m transformation_portal.spatial_ai reconstruct \
  --input-dir projects/example/views/ \
  --output-dir output_3dgs/ \
  --preset experimental/gaussian_splat_3d.yaml \
  --iterations 7000
```

---

## Camera Setup

### CameraParams Contract

```python
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams
import numpy as np

# Intrinsic matrix (3x3)
intrinsics = np.array([
    [focal_x, 0, cx],
    [0, focal_y, cy],
    [0, 0, 1]
], dtype=np.float32)

# Extrinsic matrix (4x4 world-to-camera)
extrinsics = np.eye(4, dtype=np.float32)
extrinsics[:3, :3] = rotation_matrix
extrinsics[:3, 3] = translation

# Create camera parameters
camera = CameraParams(
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    width=image_width,
    height=image_height,
)
```

### Multi-View Requirements

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| Views | 3 | 5-10 | More views = better convergence |
| Baseline | 0.1m | 0.5-2m | Distance between camera positions |
| Overlap | 30% | 50%+ | Scene overlap between adjacent views |
| Resolution | 480p | 1080p+ | Higher = better detail, slower |

---

## Configuration

### Preset: `gaussian_splat_3d.yaml`

```yaml
# config/presets/experimental/gaussian_splat_3d.yaml
tier: experimental
license_restriction: research_only

backend:
  type: gaussian_splatting
  model:
    repo_id: "graphdeco-inria/gaussian-splatting"
    revision: "NEEDS_VERIFICATION_0000000000000000000000"

optimization:
  iterations: 30000
  position_lr: 0.00016
  scaling_lr: 0.005
  rotation_lr: 0.001
  opacity_lr: 0.05

quality:
  rmse_threshold: 0.02  # 2% maximum error
  min_views: 3
  max_gaussians: 1000000
```

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `iterations` | 30000 | 1000-50000 | Optimization steps |
| `position_lr` | 0.00016 | 0.0001-0.001 | Position learning rate |
| `rmse_threshold` | 0.02 | 0.01-0.10 | Convergence threshold |
| `max_gaussians` | 1000000 | 100000-5000000 | Memory limit |

---

## Quality Validation

### RMSE Thresholds

| RMSE | Quality | Action |
|------|---------|--------|
| < 1% | Excellent | Proceed |
| 1-2% | Good | Proceed with verification |
| 2-5% | Acceptable | Review depth maps |
| > 5% | Poor | Check camera calibration |

### Convergence States

```python
from transformation_portal.spatial_ai.reconstruction.contracts import ConvergenceState

# Possible states
ConvergenceState.CONVERGED      # RMSE < threshold
ConvergenceState.STALLED        # RMSE plateau detected
ConvergenceState.DIVERGING      # RMSE increasing
ConvergenceState.IN_PROGRESS    # Still optimizing
```

### Checking Depth Consistency

```python
# After reconstruction
scene = backend.reconstruct(input_data, iterations=7000)

# Validate depth consistency
from transformation_portal.spatial_ai.reconstruction import GeometricValidator

validator = GeometricValidator(rmse_threshold=0.05)
is_consistent, report = validator.validate_depth_consistency(
    scene=scene,
    depth_maps=original_depth_maps,  # From Depth Pro / DA3
)

if not is_consistent:
    print(f"Depth inconsistency detected: {report['rmse']:.2%}")
```

---

## Output Formats

### PLY Export (Default)

```python
from transformation_portal.spatial_ai.reconstruction import MeshExporter

exporter = MeshExporter()
exporter.export_ply(
    scene,
    output_path="output/scene.ply",
    binary=True,
    include_attributes=True,  # scales, rotations, opacities
)
```

### OBJ Export

```python
exporter.export_obj(
    scene,
    output_path="output/scene.obj",
    vertex_colors=True,
)
```

### Scene3D Contract

```python
@dataclass
class Scene3D:
    splats: GaussianSplat     # Point cloud data
    cameras: List[CameraParams]  # Input cameras
    rmse: float                # Reconstruction error
    convergence: str           # Convergence state
    iteration: int             # Final iteration
    metadata: Dict[str, Any]   # Additional info
```

---

## Performance Tuning

### Memory Optimization

```python
# Reduce VRAM usage
backend = GaussianBackend(
    tier="apex_research",
    optimization_max_gaussians=5000,  # Cap gaussian count
)
```

### Speed Optimization

```python
# Faster convergence (lower quality)
scene = backend.reconstruct(
    input_data,
    iterations=3000,  # Reduced from 7000
)
```

### Device Selection

| Device | Performance | Memory | Availability |
|--------|-------------|--------|--------------|
| CUDA | Fastest | High | NVIDIA GPUs |
| MPS | Fast | Medium | Apple Silicon |
| CPU | Slowest | Low | Always available |

```python
# Force specific device
backend = GaussianBackend(
    tier="apex_research",
    device="cpu",  # Force CPU (for testing)
)
```

---

## Integration with APEX Ultra

### Preset Integration

```yaml
# config/presets/experimental/apex_research_ultra.yaml
reconstruction:
  backend: gsplat
  enabled: auto          # Auto-enable if ≥3 views
  min_views_required: 3
  iterations: 7000
  depth_consistency_threshold: 0.05
  fail_on_high_rmse: false  # Log warning only
```

### Automatic Triggering

3DGS is automatically enabled in APEX Research Ultra when:

1. Input directory contains ≥3 images
2. Camera metadata is available (EXIF or sidecar JSON)
3. Tier is `apex_research_ultra` or higher

### Fallback Behavior

If reconstruction fails:

1. Log warning with RMSE details
2. Continue pipeline (fail_on_high_rmse: false)
3. Output depth consistency report
4. Skip PLY/OBJ export

---

## Troubleshooting

### Issue: "LicenseRestrictionError: tier must be apex_research or higher"

**Cause:** Attempting to use 3DGS with commercial tier.

**Solution:** Use research tier:

```python
backend = GaussianBackend(tier="apex_research")  # Required
```

### Issue: "RMSE > 10%, geometric consistency violated"

**Causes:**
1. Insufficient camera overlap
2. Incorrect intrinsics/extrinsics
3. Moving objects in scene

**Solutions:**
1. Increase view overlap (≥50%)
2. Verify camera calibration
3. Use static scene captures

### Issue: "Out of memory during optimization"

**Solutions:**
1. Reduce `max_gaussians` (e.g., 500000)
2. Reduce image resolution
3. Use CPU fallback (slower but more memory)

### Issue: "Convergence state: DIVERGING"

**Causes:**
1. Learning rates too high
2. Insufficient views
3. Scene too complex

**Solutions:**
1. Reduce learning rates by 50%
2. Add more views (target 5+)
3. Increase iterations

---

## API Reference

### GaussianBackend

```python
class GaussianBackend:
    """3D Gaussian Splatting backend."""
    
    def __init__(
        self,
        tier: str = "apex_research",
        device: Optional[str] = None,
        model_repo: str = "graphdeco-inria/gaussian-splatting",
        model_revision: str = "NEEDS_VERIFICATION_...",
        optimization_seed: Optional[int] = None,
        optimization_max_gaussians: int = 5000,
    ) -> None: ...
    
    def reconstruct(
        self,
        input_data: ReconstructionInput,
        iterations: int = 30000,
    ) -> Scene3D: ...
    
    def render_view(
        self,
        scene: Scene3D,
        camera: CameraParams,
    ) -> np.ndarray: ...
```

### GeometricValidator

```python
class GeometricValidator:
    """Validates reconstruction against depth maps."""
    
    def __init__(
        self,
        rmse_threshold: float = 0.05,
    ) -> None: ...
    
    def validate_depth_consistency(
        self,
        scene: Scene3D,
        depth_maps: List[np.ndarray],
    ) -> Tuple[bool, Dict[str, Any]]: ...
```

---

## References

- **ADR-026:** `docs/architecture/ADR-026-apex-research-ultra.md`
- **Preset:** `config/presets/experimental/gaussian_splat_3d.yaml`
- **Contracts:** `src/transformation_portal/spatial_ai/reconstruction/contracts.py`
- **Protocol:** `src/transformation_portal/spatial_ai/reconstruction/protocol.py`

### External Resources

- [Inria 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Original Paper (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## License Notice

This feature uses 3D Gaussian Splatting from Inria GraphDeco (research license).

**Non-commercial use only.** Commercial applications require separate license agreement with Inria GraphDeco team.

For commercial alternatives, consider:
- NeRF-based approaches (MIT licensed)
- Commercial 3D reconstruction SDKs
- Photogrammetry tools (RealityCapture, Metashape)
