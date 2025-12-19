# DA3 Feature Gap Analysis & Integration Roadmap

**Document Version:** 1.0  
**Date:** December 19, 2025  
**Author:** Transformation Portal Architect  
**Status:** Strategic Planning

---

## Executive Summary

### Top 3 Recommended Integrations

1. **🔴 CRITICAL: Model Versioning Support** (Priority 1)
   - **Effort:** 4-8 hours
   - **Value:** HIGH - Ensures users can access improved models with bug fixes
   - **Risk:** LOW - Clean configuration extension

2. **🟡 HIGH-VALUE: Metric Depth Conversion Utilities** (Priority 1)
   - **Effort:** 4-6 hours
   - **Value:** HIGH - Essential for architectural measurement workflows
   - **Risk:** LOW - Self-contained utility functions

3. **🟡 HIGH-VALUE: License Validation & Warnings** (Priority 2)
   - **Effort:** 6-10 hours
   - **Value:** MEDIUM-HIGH - Critical for commercial use compliance
   - **Risk:** LOW - Metadata validation, no performance impact

### Strategic Recommendation

**Phase 1 (Immediate - Week 1):** Implement model versioning and metric depth utilities to unblock production workflows requiring absolute measurements and latest model improvements.

**Phase 2 (Next Sprint - Week 2-3):** Add license validation to prevent compliance issues for commercial clients.

**Phase 3 (Future - Month 2+):** Evaluate DA3-Streaming after monitoring user feedback on long-video processing needs.

---

## Feature Analysis Matrix

| Feature | Priority | Effort | Value | Complexity | Maintenance | Integration Risk |
|---------|----------|--------|-------|------------|-------------|------------------|
| **Model Versioning** | P1 | 6h | HIGH | LOW | LOW | LOW |
| **Metric Depth Utilities** | P1 | 5h | HIGH | LOW | LOW | LOW |
| **License Validation** | P2 | 8h | MED-HIGH | LOW | LOW | LOW |
| **XFormers Fallback** | P2 | 10h | MEDIUM | MEDIUM | MEDIUM | MEDIUM |
| **DA3-Streaming** | P2 | 20h | MEDIUM | HIGH | MEDIUM | MEDIUM-HIGH |
| **Custom Model Configs** | P3 | 16h | LOW | HIGH | HIGH | HIGH |
| **Gradio/Gallery UI** | P3 | 12h | LOW-MED | MEDIUM | MEDIUM | LOW |
| **AUC3 Performance Tracking** | P3 | 8h | LOW | LOW | MEDIUM | LOW |
| **Community Tool Integration** | P4 | 40h+ | LOW | HIGH | HIGH | MEDIUM |

**Legend:**
- **Priority:** P1 (Critical), P2 (High-Value), P3 (Nice-to-Have), P4 (Out-of-Scope)
- **Effort:** Hours of development time
- **Value:** Business/user impact (LOW/MEDIUM/HIGH)
- **Complexity:** Technical difficulty (LOW/MEDIUM/HIGH)
- **Maintenance:** Ongoing update burden (LOW/MEDIUM/HIGH)
- **Risk:** Integration risk (LOW/MEDIUM/HIGH)

---

## Detailed Feature Analysis

### Priority 1: Critical Missing Features

#### 1.1 Model Versioning Support (`-1.1` Suffix Models) 🔴

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Models like `DA3NESTED-GIANT-LARGE-1.1` and `DA3-GIANT-1.1` have bug fixes and better street scene performance
- Our `ModelVariant` enum doesn't support version selection
- Users cannot specify model versions explicitly

**Business Impact:**
- Users stuck with older models containing known bugs
- Street scene performance degradation for luxury exterior properties
- Cannot leverage official bug fixes and improvements

**Technical Analysis:**

*Feasibility:* ✅ **HIGH** - Clean extension of existing enum pattern

*User Value:* ✅ **HIGH** - Luxury real estate heavily uses exterior/street scenes

*Maintenance:* ✅ **LOW** - Only update when new versions released (rare)

*Performance:* ✅ **NEUTRAL** - No impact, just metadata

**Recommended Approach:**

```python
# config.py
class ModelVariant(str, Enum):
    # Existing models
    GIANT = "depth-anything-3-giant"
    LARGE = "depth-anything-3-large"
    
    # Versioned models (bug fixes, better street scenes)
    GIANT_V1_1 = "depth-anything-3-giant-1.1"
    LARGE_V1_1 = "depth-anything-3-large-1.1"
    NESTED_GIANT_LARGE_V1_1 = "depth-anything-3-nested-giant-large-1.1"
    
@dataclass
class DA3Config:
    # Optional: Auto-upgrade to latest version
    auto_upgrade_to_latest: bool = False
    
    def _resolve_model_version(self, variant: ModelVariant) -> str:
        """Resolve model variant to specific version."""
        if self.auto_upgrade_to_latest:
            # Map base models to latest versions
            version_map = {
                ModelVariant.GIANT: ModelVariant.GIANT_V1_1,
                ModelVariant.LARGE: ModelVariant.LARGE_V1_1,
                # ...
            }
            return version_map.get(variant, variant).value
        return variant.value
```

**Implementation Plan:**

1. **Phase 1A** (2h): Extend `ModelVariant` enum with `-1.1` versions
2. **Phase 1B** (2h): Add `auto_upgrade_to_latest` configuration flag
3. **Phase 1C** (1h): Update documentation with version notes
4. **Phase 1D** (1h): Add tests for version resolution

**Risks & Mitigation:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| Version naming changes upstream | LOW | Follow official naming conventions |
| Model not found errors | MEDIUM | Graceful fallback to base version with warning |
| Breaking changes in new versions | LOW | Maintain separate enum entries, no auto-replace |

**Success Metrics:**
- Users can explicitly select `-1.1` models via CLI/API
- Auto-upgrade flag allows seamless migration
- Documentation clearly indicates version differences

---

#### 1.2 Metric Depth Conversion Utilities 🟡

**Status:** ⚠️ PARTIALLY IMPLEMENTED (DA3METRIC-LARGE supported, no conversion utilities)

**Current Gap:**
- DA3METRIC-LARGE outputs network units, not meters
- FAQ states: `metric_depth = focal * net_output / 300.0`
- No utility function to perform this conversion
- Users must manually implement conversion

**Business Impact:**
- Architectural measurement workflows blocked
- CAD integration requires manual conversion
- Real estate staging apps need metric dimensions (furniture placement, room measurements)

**Technical Analysis:**

*Feasibility:* ✅ **HIGH** - Simple mathematical conversion

*User Value:* ✅ **HIGH** - Essential for architectural visualization and measurements

*Maintenance:* ✅ **LOW** - Formula is stable, unlikely to change

*Performance:* ✅ **NEUTRAL** - Fast NumPy operation

**Recommended Approach:**

```python
# postprocessing.py
from typing import Optional, Tuple

def convert_to_metric_depth(
    depth_map: np.ndarray,
    focal_length: float,
    scale_factor: float = 300.0,
) -> np.ndarray:
    """Convert DA3METRIC model output to metric depth (meters).
    
    Based on official DA3 formula:
        metric_depth = focal * net_output / 300.0
    
    Args:
        depth_map: Raw depth output from DA3METRIC model (H, W)
        focal_length: Camera focal length in pixels (fx or fy)
        scale_factor: Model-specific scale factor (default: 300.0 for DA3METRIC-LARGE)
    
    Returns:
        Depth map in meters (H, W)
    
    Example:
        >>> from lux_depth_v3.postprocessing import convert_to_metric_depth
        >>> # Depth from DA3METRIC-LARGE
        >>> raw_depth = prediction.depth[0]
        >>> focal = 1000.0  # pixels
        >>> metric_depth = convert_to_metric_depth(raw_depth, focal)
        >>> print(f"Room depth: {metric_depth.mean():.2f}m")
    
    Notes:
        - Only valid for DA3METRIC-* models
        - Focal length should match image resolution
        - For images without EXIF: estimate as (image_width * 1.2)
    """
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_map.shape}")
    
    return focal_length * depth_map / scale_factor


def estimate_focal_length(
    image_width: int,
    sensor_width_mm: float = 36.0,  # Full-frame 35mm
    focal_length_mm: float = 50.0,  # Standard lens
) -> float:
    """Estimate focal length in pixels from image dimensions.
    
    Args:
        image_width: Image width in pixels
        sensor_width_mm: Camera sensor width (default: 35mm full-frame)
        focal_length_mm: Lens focal length in mm (default: 50mm standard lens)
    
    Returns:
        Focal length in pixels
    
    Example:
        >>> focal_px = estimate_focal_length(image_width=1920)
        >>> print(f"Estimated focal length: {focal_px:.1f} pixels")
    
    Notes:
        - Defaults assume full-frame DSLR with 50mm lens
        - For smartphones: sensor_width_mm ≈ 6-8mm, focal_length_mm ≈ 4-6mm
        - More accurate to extract from EXIF when available
    """
    return (focal_length_mm / sensor_width_mm) * image_width


class Postprocessor:
    """Enhanced postprocessor with metric depth support."""
    
    def convert_metric_depth(
        self,
        result: DepthResult,
        focal_length: Optional[float] = None,
        auto_estimate_focal: bool = True,
    ) -> DepthResult:
        """Convert depth result to metric units.
        
        Args:
            result: Depth result from DA3METRIC model
            focal_length: Focal length in pixels (optional)
            auto_estimate_focal: Estimate focal if not provided
        
        Returns:
            DepthResult with depth_map in meters
        
        Raises:
            ValueError: If depth result is not from DA3METRIC model
        """
        if not result.metadata.get("model", "").startswith("da3metric"):
            raise ValueError(
                "Metric depth conversion only valid for DA3METRIC models. "
                f"Got: {result.metadata.get('model')}"
            )
        
        if focal_length is None:
            if auto_estimate_focal:
                h, w = result.depth_map.shape
                focal_length = estimate_focal_length(w)
                logger.warning(
                    f"Focal length not provided. Estimated {focal_length:.1f} px "
                    "from image dimensions. For accurate measurements, provide "
                    "focal length from camera EXIF."
                )
            else:
                raise ValueError("focal_length required for metric conversion")
        
        metric_depth = convert_to_metric_depth(result.depth_map, focal_length)
        
        result.depth_map = metric_depth
        result.metadata["metric_conversion"] = {
            "focal_length_px": focal_length,
            "scale_factor": 300.0,
            "units": "meters",
        }
        
        return result
```

**Implementation Plan:**

1. **Phase 2A** (2h): Implement `convert_to_metric_depth()` utility
2. **Phase 2B** (1h): Implement `estimate_focal_length()` helper
3. **Phase 2C** (1h): Add `convert_metric_depth()` to Postprocessor
4. **Phase 2D** (1h): Add CLI flag `--metric-conversion`
5. **Phase 2E** (1h): Add examples to documentation

**Risks & Mitigation:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| Incorrect focal length estimates | MEDIUM | Warn users, document EXIF extraction methods |
| Scale factor changes in future models | LOW | Make scale_factor configurable |
| Non-metric models misuse | MEDIUM | Validate model type, raise clear error |

**Success Metrics:**
- Users can convert DA3METRIC output to meters in one function call
- CLI supports `--metric-conversion` flag
- Documentation includes EXIF extraction examples

---

### Priority 2: High-Value Features

#### 2.1 License Validation & Warnings 🟡

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Mixed licenses: Apache 2.0 (BASE/SMALL/METRIC/MONO) vs CC BY-NC 4.0 (GIANT/LARGE/NESTED)
- No runtime warnings for commercial-restricted models
- Users may unknowingly violate CC BY-NC 4.0 terms

**Business Impact:**
- Legal risk for commercial luxury real estate clients
- Potential license violations in production deployments
- Architectural firms require Apache 2.0 for commercial work

**Technical Analysis:**

*Feasibility:* ✅ **HIGH** - Metadata validation, no model changes

*User Value:* ✅ **MEDIUM-HIGH** - Critical for commercial compliance

*Maintenance:* ✅ **LOW** - License changes are rare

*Performance:* ✅ **NEUTRAL** - One-time check on init

**Recommended Approach:**

```python
# config.py
from enum import Enum

class ModelLicense(str, Enum):
    """License types for DA3 models."""
    APACHE_2_0 = "Apache-2.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"


MODEL_LICENSE_MAP = {
    # Commercial-friendly (Apache 2.0)
    ModelVariant.BASE: ModelLicense.APACHE_2_0,
    ModelVariant.SMALL: ModelLicense.APACHE_2_0,
    ModelVariant.METRIC_LARGE: ModelLicense.APACHE_2_0,
    ModelVariant.MONO_LARGE: ModelLicense.APACHE_2_0,
    
    # Non-commercial only (CC BY-NC 4.0)
    ModelVariant.GIANT: ModelLicense.CC_BY_NC_4_0,
    ModelVariant.LARGE: ModelLicense.CC_BY_NC_4_0,
    ModelVariant.NESTED_GIANT_LARGE: ModelLicense.CC_BY_NC_4_0,
    ModelVariant.GIANT_V1_1: ModelLicense.CC_BY_NC_4_0,
    ModelVariant.LARGE_V1_1: ModelLicense.CC_BY_NC_4_0,
}


def get_model_license(variant: ModelVariant) -> ModelLicense:
    """Get license for model variant."""
    return MODEL_LICENSE_MAP.get(variant, ModelLicense.CC_BY_NC_4_0)


def validate_commercial_use(
    variant: ModelVariant,
    commercial_use: bool = False,
    strict: bool = False,
) -> None:
    """Validate model license for commercial use.
    
    Args:
        variant: Model variant to check
        commercial_use: Whether this is commercial use
        strict: Raise error instead of warning
    
    Raises:
        ValueError: If strict=True and license incompatible with commercial use
    
    Warnings:
        UserWarning: If commercial use with CC-BY-NC model
    """
    license_type = get_model_license(variant)
    
    if commercial_use and license_type == ModelLicense.CC_BY_NC_4_0:
        message = (
            f"\n{'='*70}\n"
            f"⚠️  LICENSE WARNING: {variant.value}\n"
            f"{'='*70}\n"
            f"This model is licensed under CC BY-NC 4.0 (Non-Commercial).\n"
            f"\n"
            f"Commercial use is NOT permitted.\n"
            f"\n"
            f"For commercial applications (luxury real estate rendering, \n"
            f"architectural visualization, client deliverables), please use:\n"
            f"  - depth-anything-3-base (Apache 2.0)\n"
            f"  - depth-anything-3-small (Apache 2.0)\n"
            f"  - depth-anything-3-metric-large (Apache 2.0)\n"
            f"  - depth-anything-3-mono-large (Apache 2.0)\n"
            f"\n"
            f"License details: https://creativecommons.org/licenses/by-nc/4.0/\n"
            f"{'='*70}\n"
        )
        
        if strict:
            raise ValueError(message)
        else:
            import warnings
            warnings.warn(message, UserWarning, stacklevel=2)


@dataclass
class DA3Config:
    model: ModelVariant = ModelVariant.METRIC_LARGE
    
    # License validation
    commercial_use: bool = False
    strict_license_check: bool = False
    
    def __post_init__(self):
        """Validate license on initialization."""
        validate_commercial_use(
            self.model,
            commercial_use=self.commercial_use,
            strict=self.strict_license_check,
        )
```

**CLI Integration:**

```python
# cli.py
@app.command()
def process(
    # ... existing params
    
    commercial_use: bool = typer.Option(
        False,
        "--commercial-use",
        help="Flag for commercial use (validates model license)",
    ),
    strict_license: bool = typer.Option(
        False,
        "--strict-license",
        help="Raise error on license violations (default: warning only)",
    ),
):
    """Process images with DA3 depth estimation."""
    config = DA3Config(
        model=model,
        commercial_use=commercial_use,
        strict_license_check=strict_license,
    )
    # ... rest of processing
```

**Implementation Plan:**

1. **Phase 3A** (2h): Create license mapping and validation functions
2. **Phase 3B** (2h): Integrate validation into DA3Config
3. **Phase 3C** (2h): Add CLI flags for commercial use
4. **Phase 3D** (2h): Add license info to documentation
5. **Phase 3E** (2h): Add tests for license validation

**Risks & Mitigation:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| License changes upstream | LOW | Monitor official repo, update mapping |
| False positives blocking workflow | MEDIUM | Default to warning, not error |
| Users ignore warnings | HIGH | Document clearly, consider strict mode for CI |

**Success Metrics:**
- Commercial users receive clear warnings for CC-BY-NC models
- Documentation includes license comparison table
- CLI supports `--commercial-use` and `--strict-license` flags

---

#### 2.2 XFormers Support & Fallback 🟡

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Older GPUs without XFormers support may fail (Issue #11 in official repo)
- No detection of XFormers availability
- No graceful degradation path

**Business Impact:**
- Deployment failures on older GPU infrastructure
- Difficult to diagnose XFormers-related errors
- Lost productivity troubleshooting compatibility

**Technical Analysis:**

*Feasibility:* ✅ **MEDIUM** - Requires XFormers detection and optional patching

*User Value:* ✅ **MEDIUM** - Important for diverse GPU fleet compatibility

*Maintenance:* ⚠️ **MEDIUM** - May need updates as XFormers/PyTorch evolve

*Performance:* ⚠️ **MEDIUM** - Fallback may be slower but functional

**Recommended Approach:**

```python
# da3_wrapper.py
import logging

logger = logging.getLogger(__name__)


def check_xformers_available() -> bool:
    """Check if xformers is available."""
    try:
        import xformers
        return True
    except ImportError:
        return False


def check_gpu_xformers_compatible() -> Tuple[bool, str]:
    """Check if GPU supports xformers.
    
    Returns:
        (is_compatible, reason)
    """
    if not torch.cuda.is_available():
        return False, "No CUDA GPU available"
    
    # Check compute capability (xformers requires ≥6.0)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    compute_capability = float(f"{capability[0]}.{capability[1]}")
    
    if compute_capability < 6.0:
        return False, f"Compute capability {compute_capability} < 6.0 required"
    
    return True, "Compatible"


class DepthAnything3Wrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        enable_xformers: bool = True,
        fallback_on_xformers_error: bool = True,
    ):
        """Initialize DA3 wrapper with XFormers support detection.
        
        Args:
            model_name: DA3 model name
            device: Device to use
            enable_xformers: Try to use XFormers if available
            fallback_on_xformers_error: Fallback to standard attention on error
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.enable_xformers = enable_xformers
        self.fallback_on_xformers_error = fallback_on_xformers_error
        
        # XFormers compatibility check
        self._xformers_available = check_xformers_available()
        self._xformers_compatible, self._xformers_reason = check_gpu_xformers_compatible()
        
        self._log_xformers_status()
        self._init_model()
    
    def _log_xformers_status(self):
        """Log XFormers availability and compatibility."""
        if self.device == "cpu":
            logger.info("XFormers not needed for CPU inference")
            return
        
        if not self.enable_xformers:
            logger.info("XFormers disabled by user")
            return
        
        if not self._xformers_available:
            logger.warning(
                "XFormers not installed. Install with: pip install xformers\n"
                "Fallback to standard attention (may be slower)."
            )
            return
        
        if not self._xformers_compatible:
            logger.warning(
                f"XFormers installed but GPU incompatible: {self._xformers_reason}\n"
                f"Fallback to standard attention (may be slower).\n"
                f"See: https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11"
            )
            return
        
        logger.info("✅ XFormers enabled for optimized attention")
    
    def _init_model(self):
        """Initialize model with XFormers fallback."""
        try:
            # Try with XFormers if available
            if self.enable_xformers and self._xformers_available and self._xformers_compatible:
                self.model = DepthAnything3.from_pretrained(
                    self.model_name,
                    device=self.device,
                    enable_xformers=True,
                )
            else:
                # Standard initialization
                self.model = DepthAnything3.from_pretrained(
                    self.model_name,
                    device=self.device,
                    enable_xformers=False,
                )
        
        except Exception as e:
            if "xformers" in str(e).lower() and self.fallback_on_xformers_error:
                logger.warning(
                    f"XFormers initialization failed: {e}\n"
                    "Retrying with standard attention..."
                )
                self.model = DepthAnything3.from_pretrained(
                    self.model_name,
                    device=self.device,
                    enable_xformers=False,
                )
            else:
                raise
```

**Implementation Plan:**

1. **Phase 4A** (3h): Implement XFormers detection utilities
2. **Phase 4B** (3h): Add fallback logic to wrapper initialization
3. **Phase 4C** (2h): Add configuration flags for XFormers control
4. **Phase 4D** (2h): Document XFormers requirements and troubleshooting

**Risks & Mitigation:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| XFormers API changes | MEDIUM | Pin xformers version in requirements |
| Detection false positives | LOW | Conservative checks, allow manual override |
| Performance regression in fallback | MEDIUM | Document expected performance difference |

**Success Metrics:**
- Automatic XFormers detection on initialization
- Graceful fallback with clear warning messages
- Documentation for GPU compatibility requirements

---

#### 2.3 DA3-Streaming (Ultra-Long Video Support) 🚀

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- No support for ultra-long video sequences (>1000 frames)
- DA3-Streaming uses sliding-window approach with <12GB GPU memory
- Separate submodule `da3_streaming/` not integrated

**Business Impact:**
- Cannot process long walkthrough videos (5+ minutes)
- Property tour videos (10-30 minutes) require manual chunking
- Time-lapse sequences (thousands of frames) unsupported

**Technical Analysis:**

*Feasibility:* ⚠️ **MEDIUM-HIGH** - Requires integration of separate codebase

*User Value:* ⚠️ **MEDIUM** - Valuable for video workflows, but niche use case

*Maintenance:* ⚠️ **MEDIUM** - Separate module may evolve independently

*Performance:* ✅ **POSITIVE** - Enables processing that's currently impossible

**User Demand Assessment:**

**When is DA3-Streaming needed?**
- Property walkthrough videos: 5-30 minutes (7,500-45,000 frames @ 25fps)
- Time-lapse sequences: Construction progress (thousands of frames)
- Multi-property tour videos: Concatenated sequences

**Current alternatives:**
- Manual chunking (complex, requires stitching)
- Reduce resolution (quality loss)
- Skip frames (temporal discontinuities)

**Architectural Concerns:**

1. **Separate codebase:** `da3_streaming/` is a distinct module
2. **API surface:** Different from standard DA3 API
3. **Memory management:** Sliding window complexity
4. **Temporal coherence:** Needs careful stitching at window boundaries

**Recommended Approach:**

**Option A: Minimal Integration (8-12 hours)**
```python
# streaming.py (new module)
class DA3StreamingWrapper:
    """Wrapper for DA3-Streaming (ultra-long videos)."""
    
    def __init__(
        self,
        model_name: str,
        window_size: int = 100,
        overlap: int = 10,
        max_memory_gb: int = 12,
    ):
        """Initialize streaming wrapper.
        
        Args:
            model_name: DA3 model name
            window_size: Frames per window
            overlap: Overlap between windows (for smooth stitching)
            max_memory_gb: Maximum GPU memory to use
        """
        # Import da3_streaming if available
        try:
            from da3_streaming import StreamingDepthEstimator
            self.estimator = StreamingDepthEstimator(
                model_name=model_name,
                window_size=window_size,
                overlap=overlap,
            )
            self.available = True
        except ImportError:
            logger.warning(
                "DA3-Streaming not available. Install with:\n"
                "  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
                "  cd Depth-Anything-3/da3_streaming\n"
                "  pip install -e ."
            )
            self.available = False
    
    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        export_format: str = "mini_npz",
    ):
        """Process ultra-long video with streaming."""
        if not self.available:
            raise RuntimeError("DA3-Streaming not installed")
        
        # Use streaming API
        results = self.estimator.process_video(
            video_path=str(video_path),
            output_dir=str(output_dir),
            export_format=export_format,
        )
        
        return results
```

**Option B: Full Integration (16-24 hours)**
- Copy `da3_streaming/` into `lux_depth_v3/streaming/`
- Refactor to match our API patterns
- Integrate with existing CLI and configuration
- Add comprehensive tests

**Recommendation: DEFER to Priority 3**

**Rationale:**
1. **Niche use case:** Most luxury real estate content is <5 minutes
2. **Workaround exists:** Manual chunking is acceptable for now
3. **High complexity:** Separate codebase integration is risky
4. **Uncertain demand:** No user requests yet for ultra-long video support

**Decision Point:**
- **If** users request long video support (>1000 frames) in next 2 months
- **Then** implement Option A (minimal integration)
- **Otherwise** defer indefinitely

**Implementation Plan (if prioritized):**

1. **Phase 5A** (4h): Survey user needs for long video support
2. **Phase 5B** (6h): Implement minimal wrapper (Option A)
3. **Phase 5C** (4h): Add CLI command `lux-depth-v3 stream-video`
4. **Phase 5D** (4h): Document installation and usage
5. **Phase 5E** (2h): Add integration tests

**Risks & Mitigation:**

| Risk | Severity | Mitigation |
|------|----------|------------|
| da3_streaming API instability | HIGH | Vendor as optional dependency |
| Memory leaks in long processing | MEDIUM | Add checkpointing, restart capability |
| Temporal discontinuities at boundaries | MEDIUM | Use adequate overlap, test stitching |

**Success Metrics:**
- Can process 10+ minute videos without chunking
- GPU memory usage stays <12GB
- Temporal coherence maintained across windows

---

### Priority 3: Nice-to-Have Features

#### 3.1 Custom Model Architecture Configs 🔧

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Official DA3 supports YAML-based custom architectures
- `create_object(load_config(...))` pattern not exposed
- Users cannot customize backbone/head configurations

**Business Impact:**
- Research users cannot experiment with custom architectures
- Cannot optimize model for specific property types (interiors vs exteriors)
- Advanced users blocked from fine-tuning

**Technical Analysis:**

*Feasibility:* ⚠️ **MEDIUM-HIGH** - Complex YAML parsing and validation

*User Value:* ❌ **LOW** - Luxury real estate users want pre-trained models, not custom architectures

*Maintenance:* ⚠️ **HIGH** - Config schema may change, validation complexity

*Performance:* ⚠️ **VARIABLE** - Custom configs may be unstable

**Architectural Assessment:**

**Who needs custom configs?**
- ML researchers (not our target audience)
- Model fine-tuning experiments (advanced use case)
- Custom backbone integration (rare)

**What's the cost?**
- YAML config parser
- Model architecture validation
- Error handling for invalid configs
- Documentation for config schema
- Support burden for custom configs

**Recommendation: OUT OF SCOPE (Priority 4)**

**Rationale:**
1. **Not aligned with user base:** Luxury real estate rendering requires stability, not experimentation
2. **High maintenance:** Config schema changes upstream create breaking changes
3. **Support burden:** Debugging custom configs is complex
4. **Pre-trained models sufficient:** Official model zoo covers use cases

**Alternative:**
- Document how to use official DA3 API directly for research
- Focus our effort on production-ready preset configs
- Point advanced users to official DA3 documentation

---

#### 3.2 Gradio Web UI & Gallery Integration 🌐

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Official DA3 has `da3 gradio` and `da3 gallery` commands
- No wrapper or integration in our CLI
- Users must call DA3 CLI directly

**Business Impact:**
- No web UI for non-technical users (clients, designers)
- Gallery visualization requires manual command line use
- Potential demo/showcase tool for sales

**Technical Analysis:**

*Feasibility:* ✅ **MEDIUM** - Gradio integration is straightforward

*User Value:* ⚠️ **LOW-MEDIUM** - Nice for demos, but not core workflow

*Maintenance:* ✅ **LOW** - Gradio API is stable

*Performance:* ✅ **NEUTRAL** - Optional feature, no impact on batch processing

**Recommended Approach:**

**Option 1: CLI Passthrough (2-3 hours)**
```python
# cli.py
@app.command()
def gradio(
    model: str = typer.Option("da3-large", help="Model to use"),
    port: int = typer.Option(7860, help="Port for Gradio UI"),
):
    """Launch Gradio web UI (requires official DA3 CLI)."""
    if not check_da3_cli_available():
        raise RuntimeError(
            "DA3 CLI not found. Install with:\n"
            "  pip install depth-anything-3"
        )
    
    # Pass through to official CLI
    subprocess.run(["da3", "gradio", "-m", model, "-p", str(port)])


@app.command()
def gallery(
    output_dir: Path = typer.Argument(..., help="Directory with DA3 outputs"),
    port: int = typer.Option(8080, help="Port for gallery server"),
):
    """Launch gallery server for visualizing results."""
    if not check_da3_cli_available():
        raise RuntimeError("DA3 CLI not found")
    
    subprocess.run(["da3", "gallery", str(output_dir), "-p", str(port)])
```

**Option 2: Custom Gradio Interface (8-12 hours)**
- Build our own Gradio interface with luxury real estate theming
- Integrate with our presets and postprocessing
- Add Material Response and LUT controls
- Better for brand consistency and UX customization

**Recommendation: Priority 3 (Nice-to-Have)**

**Implementation Plan:**

1. **Phase 6A** (1h): Add CLI passthrough commands (Option 1)
2. **Phase 6B** (2h): Document Gradio/Gallery usage
3. **Future:** Custom Gradio UI if user demand emerges

**Success Metrics:**
- Users can launch Gradio UI via `lux-depth-v3 gradio`
- Gallery server accessible via CLI
- Documentation includes screenshots

---

#### 3.3 AUC3 Performance Tracking & Model Documentation 📊

**Status:** ❌ NOT IMPLEMENTED

**Current Gap:**
- Official README links to AUC3 results for models
- No performance documentation in our integration
- Users cannot compare model quality/speed tradeoffs

**Business Impact:**
- Users don't know which model to choose for their use case
- No benchmarks for speed vs quality tradeoffs
- Difficult to justify model selection decisions

**Technical Analysis:**

*Feasibility:* ✅ **LOW** - Documentation task, no code changes

*User Value:* ⚠️ **LOW-MEDIUM** - Helpful for model selection, but not blocking

*Maintenance:* ✅ **LOW** - Update when new models released (rare)

*Performance:* ✅ **NEUTRAL** - Documentation only

**Recommended Approach:**

Create `docs/MODEL_PERFORMANCE.md`:

```markdown
# DA3 Model Performance Guide

## Model Comparison

| Model | Parameters | AUC3↑ | Speed* | Memory | License | Best For |
|-------|-----------|-------|--------|--------|---------|----------|
| **da3nested-giant-large** | 1.40B | **0.XXX** | Slow | 16GB+ | CC-BY-NC | Highest quality, metric + multi-view |
| **da3-giant** | 1.15B | 0.XXX | Slow | 12GB+ | CC-BY-NC | Multi-view reconstruction |
| **da3-large** | 0.35B | 0.XXX | **Medium** | 6GB | CC-BY-NC | General purpose (recommended) |
| **da3-base** | 0.12B | 0.XXX | Fast | 4GB | Apache-2.0 | **Commercial use** |
| **da3-small** | 0.08B | 0.XXX | **Fastest** | 2GB | Apache-2.0 | Real-time / low-end GPUs |
| **da3metric-large** | 0.35B | N/A | Medium | 6GB | Apache-2.0 | Metric depth + measurements |
| **da3mono-large** | 0.35B | N/A | Medium | 6GB | Apache-2.0 | High-quality monocular |

*Speed benchmarks on RTX 4090, 1920×1080 input

## Use Case Recommendations

### Luxury Real Estate Rendering
**Recommended:** `da3-large` or `da3metric-large`
- High quality depth estimation
- Fast enough for batch processing (200-300 images/hour)
- Supports multi-view if needed

### Commercial Architectural Visualization
**Recommended:** `da3-base` or `da3-small`
- Apache 2.0 license (commercial-friendly)
- Good balance of quality and speed
- 400-600 images/hour throughput

### Metric Measurements (CAD, Staging Apps)
**Recommended:** `da3metric-large`
- Absolute depth in meters
- Sky segmentation included
- Apache 2.0 license

### 3D Reconstruction & Gaussian Splatting
**Recommended:** `da3nested-giant-large` or `da3-giant`
- Best multi-view accuracy
- Highest quality pose estimation
- Note: CC-BY-NC license (non-commercial)

## Benchmark Details

See official AUC3 results: https://github.com/ByteDance-Seed/Depth-Anything-3#...

## Model Versions

### Version 1.1 Models (-1.1 suffix)
Released: November 2025

**Improvements:**
- Bug fixes in pose estimation
- Better street scene performance
- More robust for exterior architectural shots

**Available:**
- `da3-giant-1.1`
- `da3-large-1.1`
- `da3nested-giant-large-1.1`

**Recommendation:** Use -1.1 versions for exterior/street photography
```

**Implementation Plan:**

1. **Phase 7A** (4h): Create MODEL_PERFORMANCE.md with benchmarks
2. **Phase 7B** (2h): Add model selection flowchart
3. **Phase 7C** (2h): Link from main README and CLI help

**Success Metrics:**
- Users can quickly identify best model for their use case
- Performance expectations documented (speed, memory)
- License compatibility clearly stated

---

### Priority 4: Out-of-Scope Features

#### 4.1 Community Tools Integration (Blender, ComfyUI, ROS2, WebXR)

**Status:** ❌ NOT IMPLEMENTED

**Recommendation:** **OUT OF SCOPE**

**Rationale:**
1. **Different target audience:** Plugin developers, not luxury real estate users
2. **High maintenance:** Each integration has its own API and lifecycle
3. **Not core competency:** We focus on Python/CLI batch processing
4. **Better handled externally:** Let community maintain integrations

**Alternative:**
- Document how to export compatible formats (PLY, GLB, NPZ)
- Link to community projects in README
- Provide examples of file format specifications

---

## Integration Timeline & Resource Allocation

### Sprint 1 (Week 1) - Critical Features
**Total Effort:** 11-14 hours (1.5-2 developer days)

| Feature | Effort | Owner | Priority |
|---------|--------|-------|----------|
| Model Versioning Support | 6h | Backend Dev | P1 |
| Metric Depth Utilities | 5h | ML Engineer | P1 |

**Deliverables:**
- ✅ Support for `-1.1` model versions
- ✅ `convert_to_metric_depth()` utility
- ✅ CLI flag `--metric-conversion`
- ✅ Documentation updates
- ✅ Test coverage

### Sprint 2 (Week 2-3) - High-Value Features
**Total Effort:** 18-20 hours (2.5-3 developer days)

| Feature | Effort | Owner | Priority |
|---------|--------|-------|----------|
| License Validation & Warnings | 8h | Architect | P2 |
| XFormers Fallback | 10h | Backend Dev | P2 |

**Deliverables:**
- ✅ License warnings for CC-BY-NC models
- ✅ `--commercial-use` CLI flag
- ✅ XFormers detection and graceful fallback
- ✅ Compatibility documentation

### Future Sprints (Month 2+) - Nice-to-Have
**Total Effort:** 22-28 hours (3-4 developer days)

| Feature | Effort | Owner | Priority |
|---------|--------|-------|----------|
| DA3-Streaming (conditional) | 20h | ML Engineer | P2-P3 |
| Gradio/Gallery CLI Passthrough | 2h | Backend Dev | P3 |
| Model Performance Documentation | 8h | Tech Writer | P3 |

**Deliverables:**
- ⏸️ Streaming support (if user demand emerges)
- ✅ Gradio UI launcher
- ✅ Model comparison documentation

---

## Risk Assessment & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|---------|---------------------|
| **Upstream API breaking changes** | MEDIUM | HIGH | Pin DA3 version, test before upgrades |
| **License mapping outdated** | LOW | HIGH | Monitor official repo, automated checks |
| **XFormers compatibility issues** | MEDIUM | MEDIUM | Conservative detection, allow override |
| **Metric depth formula changes** | LOW | MEDIUM | Make scale_factor configurable |
| **Model version naming changes** | LOW | MEDIUM | Follow official conventions, flexible parsing |

### Business Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|---------|---------------------|
| **License violations in production** | MEDIUM | CRITICAL | Implement strict validation, documentation |
| **User confusion on model selection** | HIGH | MEDIUM | Clear documentation, decision flowchart |
| **Support burden for custom features** | LOW | MEDIUM | Defer custom configs, focus on presets |

### Operational Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|---------|---------------------|
| **DA3-Streaming integration complexity** | HIGH | MEDIUM | Defer to P3, wait for user demand |
| **Documentation drift** | MEDIUM | LOW | Version docs with code releases |
| **Test coverage gaps** | LOW | MEDIUM | Require tests for all P1/P2 features |

---

## Success Criteria

### Phase 1 (Sprint 1) - Critical Features
- [ ] Users can select `-1.1` model versions via CLI
- [ ] Metric depth conversion documented with examples
- [ ] CLI supports `--metric-conversion` flag
- [ ] All tests passing (coverage ≥85%)

### Phase 2 (Sprint 2) - High-Value Features
- [ ] License warnings appear for CC-BY-NC models
- [ ] `--commercial-use` flag validates licenses
- [ ] XFormers detection automatic on init
- [ ] Graceful fallback on XFormers errors
- [ ] Documentation updated with license table

### Phase 3 (Future) - Nice-to-Have
- [ ] Model performance comparison documented
- [ ] Gradio UI accessible via CLI
- [ ] User feedback collected on long video needs

---

## Monitoring & Feedback Loop

### Key Metrics to Track

1. **Feature Adoption:**
   - % of users using `-1.1` models
   - Metric depth conversion usage
   - License warnings triggered

2. **Error Rates:**
   - XFormers fallback frequency
   - License validation errors
   - Model loading failures

3. **User Feedback:**
   - Feature requests for DA3-Streaming
   - Model selection confusion
   - Documentation clarity

### Feedback Collection

- Add telemetry (opt-in) for feature usage
- GitHub issue templates for feature requests
- User survey after Sprint 1 completion

---

## Conclusion

### Immediate Actions (Next Week)

1. **Implement model versioning** (6h) - Unblocks users needing bug-fixed models
2. **Add metric depth utilities** (5h) - Enables architectural measurement workflows
3. **Update documentation** (2h) - Clarify new features

### Short-Term Actions (Next Sprint)

4. **License validation** (8h) - Protect commercial clients from violations
5. **XFormers fallback** (10h) - Improve GPU compatibility

### Long-Term Strategy

- **Monitor user demand** for DA3-Streaming before investing
- **Defer custom configs** - Not aligned with user base
- **Document community integrations** - Let others maintain plugins

### Architectural Philosophy

**Prioritize:**
- ✅ Production stability over experimental features
- ✅ Commercial compliance over feature completeness
- ✅ Clear documentation over complex APIs
- ✅ User-driven development over speculative work

This analysis ensures that `lux_depth_v3/` remains focused on luxury real estate rendering workflows while avoiding technical debt from features that don't align with our core user base.

---

**Document Status:** Ready for Review  
**Next Review:** After Sprint 1 completion  
**Owner:** Transformation Portal Architect

