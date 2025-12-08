# Architecture Hardening Plan: Transformation Portal

**Status**: 🟢 **APPROVED FOR IMPLEMENTATION**  
**Version**: 1.0  
**Date**: 2025-12-08  
**Author**: Transformation Portal Architect  
**Review Date**: 2025-12-22 (2-week milestone)

---

## Executive Summary

This document outlines a comprehensive, PR-sequenced architecture optimization plan that transforms Transformation Portal from a feature-rich toolkit into a production-grade platform while maintaining backward compatibility and protecting ongoing validation system work.

**Key Goals**:
- ✅ Eliminate security risks (CVE-2024-27763, sensitive artifacts)
- ✅ Establish platform core for unified configuration and device management
- ✅ Enable graph-based pipelines with caching and intelligent routing
- ✅ Optimize GPU performance with systematic profiling
- ✅ Make reproducibility and validation default outputs
- ✅ Fill test coverage gaps with checkpoint/resume architecture

**Success Criteria**:
- Zero security vulnerabilities in production dependencies
- 66/66 Lux Depth V2 tests remain passing
- Backward compatible CLI interfaces
- Validation system integration without blocking
- <5% performance regression (ideally 10-20% improvement)
- 85%+ test coverage across all modules

---

## Current State Assessment

### ✅ Strengths
1. **Lux Depth V2**: Production-ready (66/66 tests, 80%+ coverage)
2. **Validation System**: Complete architecture (7 docs, 156KB)
3. **Security Awareness**: CVE-2024-27763 documented, mitigation in place
4. **Modular Design**: Lux Depth V2 demonstrates clean module boundaries
5. **Test Infrastructure**: pytest + CI/CD foundation operational

### ⚠️ Risks Identified

#### Priority 1: Security & Repo Hygiene
- **Sensitive artifacts in root**: `.bash_history`, `.local_backup/`, client folders
- **No CI enforcement** of banned dependencies (basicsr/realesrgan/gfpgan)
- **Documentation mismatch**: README references Real-ESRGAN but it's removed
- **Secrets exposure risk**: Git history may contain credentials

#### Priority 2: Management & Maintainability
- **Feature duplication**: Multiple pipelines reimplement config, device detection, logging
- **Inconsistent conventions**: 5+ different CLI patterns, 3 logging approaches
- **Coupling**: Pipelines import from each other creating circular dependencies

#### Priority 3: Efficiency & Intelligence
- **Monolithic processing**: No stage-level caching or measurement
- **Manual parameter tuning**: No context-aware preset selection
- **Limited observability**: Performance bottlenecks unidentified

#### Priority 4: GPU Performance
- **CPU↔GPU copy overhead**: Unnecessary round-trips between device and host
- **Missing profiling**: No systematic measurement of GPU utilization
- **Tiling gaps**: UHR support inconsistent across pipelines

#### Priority 5: Reproducibility
- **Inconsistent metadata**: Some pipelines emit reports, others don't
- **No validation integration**: Quality claims lack proof artifacts

#### Priority 6: Test Coverage
- **Fallback branches untested**: 82% coverage misses rare failure paths
- **No checkpoint/resume**: Large batch failures require full restart
- **Edge cases**: Multi-GPU, HDR, disk-full scenarios missing

---

## The 6-Priority Architecture Optimization Plan

### PR-1: Security + Repo Hygiene (IMMEDIATE)

**Target**: Eliminate critical security risks before any other changes.

#### Deliverables
1. **Artifact Purge & Gitignore Update**
   - Remove: `.bash_history`, `.local_backup/`, client folders, temp backups
   - Update `.gitignore` with comprehensive exclusions
   - Use BFG Repo-Cleaner or `git filter-repo` to purge history

2. **CI Security Gate**
   - Add `scripts/ci/enforce_safe_deps.py` (fail on banned packages)
   - Integrate into `security-scan.yml` workflow
   - Scan imports: `grep -r "import basicsr" src/ lux_depth_v2/`

3. **Secret Rotation & Scanning**
   - Run `gitleaks` or `trufflehog` on full history
   - Rotate any exposed API keys, credentials
   - Add `detect-secrets` pre-commit hook

4. **Documentation Sync**
   - Update README.md to reflect actual security posture
   - Document "unsafe extras" boundary if Real-ESRGAN ever re-enabled
   - Add security policy reference to all module READMEs

**Success Criteria**:
- ✅ CI fails if banned packages detected
- ✅ No sensitive files in `git status`
- ✅ README matches implementation
- ✅ Secret scanner passes

**Timeline**: 2-3 days  
**Risk**: 🟢 LOW (no code changes, only hygiene)

---

### PR-2: Platform Core Extraction (FOUNDATION)

**Target**: Establish unified infrastructure to eliminate duplication.

#### New Module Structure
```
transformation_portal/
├── core/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py          # Pydantic models for all config types
│   │   ├── validator.py       # Config validation rules
│   │   ├── loader.py          # Unified YAML/JSON loader
│   │   └── presets.py         # Preset registry
│   ├── device/
│   │   ├── __init__.py
│   │   ├── manager.py         # CPU/CUDA/MPS detection & allocation
│   │   ├── profiler.py        # Device metrics (optional, <5% overhead)
│   │   └── fallback.py        # Graceful degradation strategies
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── store.py           # Cache management (depth maps, masks)
│   │   ├── manifest.py        # Content-addressed storage
│   │   └── hashing.py         # SHA256 + config hashing
│   ├── security/
│   │   ├── __init__.py
│   │   ├── paths.py           # Safe path validation
│   │   ├── images.py          # Safe image loading (PIL verify)
│   │   └── validation.py      # Input sanitization
│   └── observability/
│       ├── __init__.py
│       ├── logging.py         # Structured logging (JSON lines)
│       ├── metrics.py         # Prometheus-compatible metrics
│       └── tracing.py         # OpenTelemetry hooks (optional)
```

#### API Contracts

**Config Schema** (`core/config/schema.py`):
```python
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Literal, Optional

class DeviceConfig(BaseModel):
    """Device configuration for compute resources."""
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    gpu_id: Optional[int] = None
    enable_amp: bool = True  # Automatic Mixed Precision
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    
    @validator("device")
    def validate_device(cls, v):
        if v == "auto":
            return detect_best_device()
        return v

class ProcessingConfig(BaseModel):
    """Base configuration for all processing pipelines."""
    input_path: Path
    output_dir: Path
    preset: str = "default"
    device: DeviceConfig = DeviceConfig()
    enable_caching: bool = True
    verbose: bool = False
    dry_run: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class ReportConfig(BaseModel):
    """Configuration for processing reports."""
    emit_report: bool = True
    include_metrics: bool = True
    include_git_info: bool = True
    include_device_info: bool = True
```

**Device Manager** (`core/device/manager.py`):
```python
from typing import Optional, Tuple
import torch
from enum import Enum

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class DeviceManager:
    """Unified device management for all pipelines."""
    
    def __init__(self, preferred: str = "auto"):
        self.device = self._detect_device(preferred)
        self.dtype = self._get_optimal_dtype()
        
    def _detect_device(self, preferred: str) -> torch.device:
        """Detect best available device with fallback chain."""
        if preferred == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(preferred)
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype for device."""
        if self.device.type == "cuda":
            return torch.float16  # AMP-safe
        elif self.device.type == "mps":
            return torch.float32  # MPS doesn't support all float16 ops
        return torch.float32
    
    def allocate_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Allocate tensor on managed device."""
        return torch.zeros(shape, device=self.device, dtype=self.dtype)
    
    def profile(self) -> dict:
        """Get device metrics (optional, for observability)."""
        if self.device.type == "cuda":
            return {
                "device": str(self.device),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "utilization": self._get_gpu_utilization()
            }
        return {"device": str(self.device)}
```

**Artifact Store** (`core/artifacts/store.py`):
```python
from pathlib import Path
import hashlib
import json
from typing import Optional, Dict, Any

class ArtifactStore:
    """Content-addressed cache for expensive computations."""
    
    def __init__(self, cache_dir: Path = Path(".cache/artifacts")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = cache_dir / "manifest.json"
        self._load_manifest()
    
    def _load_manifest(self):
        """Load cache manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
    
    def _compute_key(self, input_path: Path, config: Dict[str, Any]) -> str:
        """Compute cache key from input + config."""
        hasher = hashlib.sha256()
        hasher.update(input_path.read_bytes())
        hasher.update(json.dumps(config, sort_keys=True).encode())
        return hasher.hexdigest()
    
    def get(self, input_path: Path, config: Dict[str, Any]) -> Optional[Path]:
        """Retrieve cached artifact if exists."""
        key = self._compute_key(input_path, config)
        if key in self.manifest:
            cached_path = self.cache_dir / self.manifest[key]
            if cached_path.exists():
                return cached_path
        return None
    
    def put(self, input_path: Path, config: Dict[str, Any], 
            output_path: Path) -> None:
        """Store artifact in cache."""
        key = self._compute_key(input_path, config)
        cached_name = f"{key}.npy"
        cached_path = self.cache_dir / cached_name
        # Copy output to cache
        import shutil
        shutil.copy(output_path, cached_path)
        # Update manifest
        self.manifest[key] = cached_name
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
```

**Security Primitives** (`core/security/paths.py`):
```python
from pathlib import Path
from typing import Union

class PathValidator:
    """Secure path validation to prevent traversal attacks."""
    
    def __init__(self, allowed_base: Union[str, Path]):
        self.allowed_base = Path(allowed_base).resolve()
    
    def validate(self, user_input: Union[str, Path]) -> Path:
        """Validate path is within allowed base directory."""
        path = Path(user_input).resolve()
        
        # Prevent path traversal
        if not path.is_relative_to(self.allowed_base):
            raise ValueError(
                f"Path traversal attempt: {user_input} outside {self.allowed_base}"
            )
        
        # Prevent symlink attacks
        if path.is_symlink():
            raise ValueError(f"Symlinks not allowed: {user_input}")
        
        return path
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Remove dangerous characters from filename."""
        import re
        # Allow alphanumeric, underscore, hyphen, period
        safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
        # Prevent double extensions (.png.exe)
        parts = safe.split('.')
        if len(parts) > 2:
            safe = f"{'_'.join(parts[:-1])}.{parts[-1]}"
        return safe
```

#### Migration Strategy

**Phase 1: Extract without breaking**
1. Create `transformation_portal/core/` module
2. Implement all components with comprehensive tests
3. Add to CI but don't require usage yet

**Phase 2: Migrate Lux Depth V2 first**
1. Refactor `lux_depth_v2/config.py` to use `core.config`
2. Replace device detection with `core.device.DeviceManager`
3. Integrate `core.artifacts.ArtifactStore` for depth caching
4. Verify 66/66 tests still pass

**Phase 3: Migrate legacy pipelines**
1. Update `luxury_video_master_grader.py` to use core
2. Update `luxury_tiff_batch_processor.py` to use core
3. Deprecate old patterns with warnings

**Backward Compatibility**:
- All existing CLI interfaces remain unchanged
- Old config formats still work (converted internally)
- Deprecation warnings guide users to new patterns

**Success Criteria**:
- ✅ Lux Depth V2 tests pass with core integration
- ✅ Zero feature regressions
- ✅ 90%+ test coverage on core modules
- ✅ Performance neutral or improved

**Timeline**: 1-2 weeks  
**Risk**: 🟡 MEDIUM (refactoring requires careful testing)

---

### PR-3: Stage Graph Refactor (EFFICIENCY)

**Target**: Enable caching, measurement, and intelligent routing.

#### Stage Graph Architecture

**Core Concepts**:
1. **Stage**: Deterministic transformation `(input, config) -> output`
2. **Graph**: DAG of stages with dependencies
3. **Cache**: Stage outputs cached by `(input_hash, config_hash, stage_version)`
4. **Policy Engine**: Selects graph parameters based on context

#### Implementation

**Stage Base Class** (`core/pipeline/stage.py`):
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from dataclasses import dataclass
import hashlib
import json

@dataclass
class StageResult:
    """Result of stage execution."""
    output: Any
    metadata: Dict[str, Any]
    duration_ms: float
    cache_hit: bool

class Stage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
    
    @abstractmethod
    def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        """Execute stage transformation."""
        pass
    
    def compute_cache_key(self, input_data: Any, config: Dict[str, Any]) -> str:
        """Compute deterministic cache key."""
        hasher = hashlib.sha256()
        # Hash input
        if hasattr(input_data, 'tobytes'):
            hasher.update(input_data.tobytes())
        else:
            hasher.update(str(input_data).encode())
        # Hash config
        hasher.update(json.dumps(config, sort_keys=True).encode())
        # Hash stage version
        hasher.update(f"{self.name}:{self.version}".encode())
        return hasher.hexdigest()
    
    def run(self, input_data: Any, config: Dict[str, Any], 
            cache: 'ArtifactStore') -> StageResult:
        """Run stage with caching and timing."""
        import time
        
        # Check cache
        cache_key = self.compute_cache_key(input_data, config)
        cached = cache.get_by_key(cache_key)
        if cached is not None:
            return StageResult(
                output=cached,
                metadata={"cache_key": cache_key},
                duration_ms=0,
                cache_hit=True
            )
        
        # Execute
        start = time.perf_counter()
        output = self.execute(input_data, config)
        duration = (time.perf_counter() - start) * 1000
        
        # Store in cache
        cache.put_by_key(cache_key, output)
        
        return StageResult(
            output=output,
            metadata={"cache_key": cache_key},
            duration_ms=duration,
            cache_hit=False
        )
```

**Pipeline Graph** (`core/pipeline/graph.py`):
```python
from typing import List, Dict, Any
from .stage import Stage, StageResult

class PipelineGraph:
    """Directed acyclic graph of processing stages."""
    
    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.results: Dict[str, StageResult] = {}
    
    def execute(self, input_data: Any, config: Dict[str, Any], 
                cache: 'ArtifactStore') -> Any:
        """Execute pipeline graph."""
        data = input_data
        
        for stage in self.stages:
            result = stage.run(data, config, cache)
            self.results[stage.name] = result
            data = result.output
        
        return data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for all stages."""
        return {
            stage_name: {
                "duration_ms": result.duration_ms,
                "cache_hit": result.cache_hit
            }
            for stage_name, result in self.results.items()
        }
```

**Example: Lux Depth V2 Stage Graph**

```python
# lux_depth_v2/stages.py
from transformation_portal.core.pipeline.stage import Stage
import torch

class DepthEstimationStage(Stage):
    """Depth estimation stage."""
    
    def __init__(self, model):
        super().__init__(name="depth_estimation", version="1.0.0")
        self.model = model
    
    def execute(self, input_data, config):
        """Estimate depth map."""
        with torch.inference_mode():
            depth = self.model(input_data)
        return depth

class MaterialSegmentationStage(Stage):
    """Material segmentation stage."""
    
    def __init__(self, segmenter):
        super().__init__(name="material_segmentation", version="1.0.0")
        self.segmenter = segmenter
    
    def execute(self, input_data, config):
        """Segment materials."""
        return self.segmenter.segment(input_data)

class ToneMappingStage(Stage):
    """Tone mapping stage."""
    
    def __init__(self):
        super().__init__(name="tone_mapping", version="1.0.0")
    
    def execute(self, input_data, config):
        """Apply tone mapping."""
        operator = config.get("tone_map_operator", "agx")
        return apply_tone_mapping(input_data, operator)

# Build graph
def build_lux_depth_graph(config):
    """Build Lux Depth V2 processing graph."""
    stages = [
        DepthEstimationStage(load_depth_model()),
        MaterialSegmentationStage(load_segmenter()),
        ToneMappingStage()
    ]
    return PipelineGraph(stages)
```

**Policy Engine** (`core/pipeline/policy.py`):
```python
from typing import Dict, Any
from pathlib import Path
from PIL import Image

class ContextExtractor:
    """Extract context from input image."""
    
    def extract(self, image_path: Path) -> Dict[str, Any]:
        """Extract image context."""
        img = Image.open(image_path)
        width, height = img.size
        
        return {
            "resolution": (width, height),
            "aspect_ratio": width / height,
            "is_landscape": width > height,
            "is_portrait": height > width,
            "is_uhd": width * height > 3840 * 2160,
            "is_hdr": self._detect_hdr(img)
        }
    
    def _detect_hdr(self, img: Image.Image) -> bool:
        """Detect if image is HDR."""
        # Simple heuristic: check for 16-bit depth or extended gamut
        return img.mode in ("I;16", "I;16L", "I;16B")

class PolicyEngine:
    """Intelligent parameter selection based on context."""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load policy rules."""
        return {
            "uhd_tiling": {
                "condition": lambda ctx: ctx["is_uhd"],
                "params": {"enable_tiling": True, "tile_size": 512}
            },
            "hdr_tone_mapping": {
                "condition": lambda ctx: ctx["is_hdr"],
                "params": {"tone_map_operator": "aces"}
            },
            "portrait_clarity": {
                "condition": lambda ctx: ctx["is_portrait"],
                "params": {"clarity_strength": 0.3}
            }
        }
    
    def apply(self, context: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply policy rules to configuration."""
        config = base_config.copy()
        
        for rule_name, rule in self.rules.items():
            if rule["condition"](context):
                config.update(rule["params"])
        
        return config
```

**Migration Path**:
1. Implement stage graph infrastructure in `core/pipeline/`
2. Refactor Lux Depth V2 to use stage graph (backward compatible)
3. Measure performance impact (target: 10-20% improvement from caching)
4. Document stage graph pattern for future pipelines

**Success Criteria**:
- ✅ 10x speedup on re-processing with warm cache
- ✅ Stage-level timing metrics available
- ✅ Zero feature regressions
- ✅ Lux Depth V2 tests pass

**Timeline**: 1-2 weeks  
**Risk**: 🟡 MEDIUM (new architecture requires validation)

---

### PR-4: Performance + Profiling Hooks (OPTIMIZATION)

**Target**: Systematic GPU performance optimization with measurement.

#### GPU-First Fast Path Checklist

**1. Minimize CPU↔GPU Copies**
```python
# ❌ BAD: Unnecessary round-trips
depth_cpu = depth_gpu.cpu().numpy()
processed = apply_filter(depth_cpu)
depth_gpu = torch.from_numpy(processed).to(device)

# ✅ GOOD: Keep on device
depth_gpu = apply_filter_torch(depth_gpu)
```

**2. Inference-Only Discipline**
```python
# ✅ Always use inference mode
with torch.inference_mode():
    output = model(input_tensor)
```

**3. Automatic Mixed Precision (AMP)**
```python
# ✅ Enable AMP for compatible operations
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input_tensor)
```

**4. UHR Tiling Mechanism**
```python
# core/processing/tiling.py
class TiledProcessor:
    """Process ultra-high-resolution images with tiling."""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process(self, image: torch.Tensor, processor_fn) -> torch.Tensor:
        """Process image in tiles."""
        h, w = image.shape[-2:]
        
        if h <= self.tile_size and w <= self.tile_size:
            # Small enough, process directly
            return processor_fn(image)
        
        # Split into tiles
        tiles = self._split_tiles(image)
        processed_tiles = [processor_fn(tile) for tile in tiles]
        # Blend overlaps and reassemble
        return self._merge_tiles(processed_tiles, (h, w))
```

**5. Profiler Integration**
```python
# core/device/profiler.py
from contextlib import contextmanager
import time
import torch

class GPUProfiler:
    """Lightweight GPU profiler (<5% overhead)."""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events = []
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code section."""
        if not self.enabled:
            yield
            return
        
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        cpu_start = time.perf_counter()
        
        yield
        
        cpu_duration = (time.perf_counter() - cpu_start) * 1000
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            gpu_duration = start_event.elapsed_time(end_event)
            self.events.append({
                "name": name,
                "cpu_ms": cpu_duration,
                "gpu_ms": gpu_duration
            })
        else:
            self.events.append({
                "name": name,
                "cpu_ms": cpu_duration
            })
    
    def report(self) -> dict:
        """Generate profiling report."""
        total_time = sum(e.get("gpu_ms", e["cpu_ms"]) for e in self.events)
        return {
            "total_ms": total_time,
            "stages": self.events
        }
```

**6. Performance Regression Tests**
```python
# tests/performance/test_lux_depth_regression.py
import pytest
import time
from lux_depth_v2.pipeline import LuxDepthPipeline

@pytest.mark.performance
def test_lux_depth_throughput_regression():
    """Ensure throughput doesn't regress below baseline."""
    pipeline = LuxDepthPipeline.from_preset("interior_luxury")
    
    # Baseline: 24-65ms per image on M4 Max (from docs)
    # CI baseline: 200ms per image on CPU (GitHub Actions)
    baseline_ms = 200  # Conservative for CI
    
    test_image = load_test_image("sample_interior.jpg")
    
    start = time.perf_counter()
    result = pipeline.process(test_image)
    duration_ms = (time.perf_counter() - start) * 1000
    
    assert duration_ms < baseline_ms * 1.05, \
        f"Performance regression: {duration_ms}ms > {baseline_ms}ms"
```

**Implementation Plan**:
1. Add `core/device/profiler.py` with GPU profiler
2. Add `core/processing/tiling.py` with UHR support
3. Integrate profiler into Lux Depth V2
4. Add performance regression tests to CI
5. Document optimization patterns in `docs/PERFORMANCE_OPTIMIZATION.md`

**Success Criteria**:
- ✅ Profiler <5% overhead when enabled
- ✅ UHR images (324MP+) process without OOM
- ✅ Performance regression tests in CI
- ✅ 10-20% speedup on GPU workloads

**Timeline**: 1 week  
**Risk**: 🟢 LOW (additive, doesn't break existing code)

---

### PR-5: Validation-First Defaults (REPRODUCIBILITY)

**Target**: Make every processing run emit reproducible validation data.

#### Reproducibility Manifest

**Report Schema** (`core/validation/report.py`):
```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import platform
import torch

@dataclass
class GitInfo:
    """Git repository state."""
    commit: str
    branch: str
    is_dirty: bool
    
    @classmethod
    def capture(cls) -> 'GitInfo':
        """Capture current git state."""
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
        is_dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD"],
            check=False
        ).returncode != 0
        return cls(commit=commit, branch=branch, is_dirty=is_dirty)

@dataclass
class DeviceInfo:
    """Hardware and software environment."""
    device_type: str
    device_name: Optional[str]
    torch_version: str
    cuda_version: Optional[str]
    python_version: str
    platform: str
    
    @classmethod
    def capture(cls) -> 'DeviceInfo':
        """Capture device information."""
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon"
            cuda_version = None
        else:
            device_type = "cpu"
            device_name = platform.processor()
            cuda_version = None
        
        return cls(
            device_type=device_type,
            device_name=device_name,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            python_version=platform.python_version(),
            platform=platform.platform()
        )

@dataclass
class ModelInfo:
    """Model checksums and versions."""
    model_name: str
    checkpoint_sha256: str
    
    @classmethod
    def from_weights(cls, model_name: str, weights_path: Path) -> 'ModelInfo':
        """Compute model checksum."""
        import hashlib
        hasher = hashlib.sha256()
        with open(weights_path, "rb") as f:
            hasher.update(f.read())
        return cls(model_name=model_name, checkpoint_sha256=hasher.hexdigest())

@dataclass
class ProcessingReport:
    """Comprehensive processing report for reproducibility."""
    git_info: GitInfo
    device_info: DeviceInfo
    model_info: ModelInfo
    config_hash: str
    preset: str
    input_path: str
    output_path: str
    timestamp: str
    duration_ms: float
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save report to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def create(cls, config, input_path, output_path, duration_ms, metrics):
        """Create report from processing run."""
        import hashlib
        import json
        from datetime import datetime
        
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        
        return cls(
            git_info=GitInfo.capture(),
            device_info=DeviceInfo.capture(),
            model_info=ModelInfo.from_weights("depth_anything_v2", Path("weights/depth.pth")),
            config_hash=config_hash,
            preset=config.get("preset", "default"),
            input_path=str(input_path),
            output_path=str(output_path),
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration_ms,
            metrics=metrics
        )
```

**Metrics Integration** (`core/validation/metrics.py`):
```python
from typing import Dict
import torch
import numpy as np

class MetricsComputer:
    """Compute quality metrics with categorization."""
    
    def __init__(self):
        self.weights = {
            "ssim": 0.3,      # Structural similarity
            "psnr": 0.2,      # Peak signal-to-noise ratio
            "lpips": 0.3,     # Perceptual similarity
            "nima": 0.2       # Neural image assessment
        }
    
    def compute(self, reference: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            "ssim": self._compute_ssim(reference, processed),
            "psnr": self._compute_psnr(reference, processed),
            "lpips": self._compute_lpips(reference, processed),
            "nima": self._compute_nima(processed)
        }
    
    def compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted quality score."""
        return sum(metrics[k] * self.weights[k] for k in self.weights)
```

**Baseline Comparison** (`core/validation/comparison.py`):
```python
from pathlib import Path
import json

class BaselineComparator:
    """Compare against validation baseline."""
    
    def __init__(self, baseline_dir: Path):
        self.baseline_dir = baseline_dir
        self.baseline_metrics = self._load_baseline()
    
    def _load_baseline(self) -> dict:
        """Load baseline metrics."""
        baseline_path = self.baseline_dir / "baseline_metrics.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return {}
    
    def compare(self, preset: str, metrics: Dict[str, float]) -> dict:
        """Compare metrics against baseline."""
        if preset not in self.baseline_metrics:
            return {"status": "no_baseline", "delta": {}}
        
        baseline = self.baseline_metrics[preset]
        delta = {
            k: metrics[k] - baseline[k]
            for k in metrics
            if k in baseline
        }
        
        # Determine status
        if any(d < -0.05 for d in delta.values()):
            status = "regression"
        elif any(d > 0.05 for d in delta.values()):
            status = "improvement"
        else:
            status = "stable"
        
        return {"status": status, "delta": delta, "baseline": baseline}
```

**Implementation Plan**:
1. Add `core/validation/` module with report, metrics, comparison
2. Integrate into Lux Depth V2 pipeline (emit report by default)
3. Add `--no-report` flag to disable if needed
4. Update CI to collect and archive reports

**Success Criteria**:
- ✅ Every processing run emits reproducibility manifest
- ✅ Metrics computed and categorized
- ✅ Baseline comparison integrated
- ✅ <1% performance impact

**Timeline**: 1 week  
**Risk**: 🟢 LOW (additive feature)

---

### PR-6: Test Strategy - Fill Coverage Gaps (ROBUSTNESS)

**Target**: Turn test gaps into architecture improvements.

#### Checkpoint/Resume Architecture

**Problem**: Large batch failures require full restart.

**Solution**: Job model with checkpoint/resume.

**Implementation** (`core/batch/job.py`):
```python
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobItem:
    """Single item in batch job."""
    input_path: str
    output_path: str
    status: JobStatus
    error: Optional[str] = None
    duration_ms: Optional[float] = None

@dataclass
class BatchJob:
    """Resumable batch processing job."""
    job_id: str
    items: List[JobItem]
    checkpoint_path: Path
    
    def save_checkpoint(self):
        """Save job state to disk."""
        with open(self.checkpoint_path, "w") as f:
            json.dump({
                "job_id": self.job_id,
                "items": [asdict(item) for item in self.items]
            }, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> 'BatchJob':
        """Load job from checkpoint."""
        with open(checkpoint_path) as f:
            data = json.load(f)
        
        items = [
            JobItem(
                input_path=item["input_path"],
                output_path=item["output_path"],
                status=JobStatus(item["status"]),
                error=item.get("error"),
                duration_ms=item.get("duration_ms")
            )
            for item in data["items"]
        ]
        
        return cls(
            job_id=data["job_id"],
            items=items,
            checkpoint_path=checkpoint_path
        )
    
    def get_pending_items(self) -> List[JobItem]:
        """Get items that still need processing."""
        return [
            item for item in self.items
            if item.status == JobStatus.PENDING
        ]
    
    def mark_completed(self, item: JobItem, duration_ms: float):
        """Mark item as completed."""
        item.status = JobStatus.COMPLETED
        item.duration_ms = duration_ms
        self.save_checkpoint()
    
    def mark_failed(self, item: JobItem, error: str):
        """Mark item as failed."""
        item.status = JobStatus.FAILED
        item.error = error
        self.save_checkpoint()

class BatchProcessor:
    """Batch processor with checkpoint/resume."""
    
    def __init__(self, pipeline, checkpoint_dir: Path):
        self.pipeline = pipeline
        self.checkpoint_dir = checkpoint_dir
    
    def process_batch(self, input_paths: List[Path], output_dir: Path,
                      resume_from: Optional[Path] = None) -> BatchJob:
        """Process batch with checkpoint/resume."""
        import uuid
        
        if resume_from:
            job = BatchJob.load_checkpoint(resume_from)
            print(f"Resuming job {job.job_id}")
        else:
            job_id = str(uuid.uuid4())
            items = [
                JobItem(
                    input_path=str(p),
                    output_path=str(output_dir / p.name),
                    status=JobStatus.PENDING
                )
                for p in input_paths
            ]
            checkpoint_path = self.checkpoint_dir / f"{job_id}.json"
            job = BatchJob(job_id=job_id, items=items, checkpoint_path=checkpoint_path)
            job.save_checkpoint()
        
        # Process pending items
        for item in job.get_pending_items():
            try:
                import time
                start = time.perf_counter()
                
                result = self.pipeline.process(Path(item.input_path))
                result.save(Path(item.output_path))
                
                duration = (time.perf_counter() - start) * 1000
                job.mark_completed(item, duration)
                
            except Exception as e:
                job.mark_failed(item, str(e))
        
        return job
```

#### Rare Fallback Testing

**Segmentation Failure Fallback**:
```python
# tests/test_fallbacks.py
import pytest
from unittest.mock import patch

def test_segmentation_failure_falls_back_to_heuristic():
    """Test graceful fallback when ONNX segmentation fails."""
    from lux_depth_v2.material_segmentation import MaterialSegmenter
    
    segmenter = MaterialSegmenter(backend="onnx")
    
    # Mock ONNX failure
    with patch.object(segmenter.onnx_backend, "segment", side_effect=RuntimeError("ONNX error")):
        # Should fall back to heuristic
        result = segmenter.segment(test_image)
        
        # Verify fallback was used
        assert result.backend_used == "heuristic"
        assert result.success
```

**Disk-Full Recovery**:
```python
def test_batch_processor_handles_disk_full():
    """Test checkpoint/resume when disk fills up mid-batch."""
    processor = BatchProcessor(pipeline, checkpoint_dir=Path("/tmp"))
    
    # Simulate disk full on 5th image
    with patch("builtins.open", side_effect=OSError("No space left on device")):
        job = processor.process_batch(image_paths[:10], output_dir)
    
    # Verify checkpoint saved successfully for completed items
    assert len([i for i in job.items if i.status == JobStatus.COMPLETED]) == 4
    
    # Resume processing
    resumed_job = processor.process_batch([], output_dir, resume_from=job.checkpoint_path)
    assert len([i for i in resumed_job.items if i.status == JobStatus.COMPLETED]) == 10
```

**Multi-GPU Edge Case**:
```python
def test_multi_gpu_selection():
    """Test explicit GPU selection when multiple GPUs available."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU test requires 2+ GPUs")
    
    # Test GPU 0
    manager = DeviceManager(preferred="cuda:0")
    assert manager.device.index == 0
    
    # Test GPU 1
    manager = DeviceManager(preferred="cuda:1")
    assert manager.device.index == 1
```

**Implementation Plan**:
1. Add `core/batch/job.py` with checkpoint/resume
2. Add fallback tests to `tests/test_fallbacks.py`
3. Add edge case tests to `tests/test_edge_cases.py`
4. Integrate batch processor into Lux Depth V2 CLI
5. Update CI to run edge case tests

**Success Criteria**:
- ✅ 85%+ overall test coverage
- ✅ All rare fallback branches tested
- ✅ Batch processor handles interruptions gracefully
- ✅ Edge cases documented

**Timeline**: 1-2 weeks  
**Risk**: 🟢 LOW (test improvements, no production code changes)

---

## Integration with Validation System

**Key Principle**: Validation work continues unblocked.

### Non-Blocking Strategy

1. **PR-1 (Security)**: No impact on validation system
2. **PR-2 (Platform Core)**: Core modules available but not required
3. **PR-3 (Stage Graph)**: Optional feature, existing pipelines still work
4. **PR-4 (Performance)**: Additive profiling, no breaking changes
5. **PR-5 (Validation)**: **Direct integration** - validation reports use same schema
6. **PR-6 (Testing)**: Improves coverage, no validation system changes

### Integration Points

**Validation Report Schema**:
- Validation system uses `core/validation/report.py` schema
- Baseline comparison integrates with validation baselines
- Metrics use same categorization (SSIM, PSNR, LPIPS, NIMA)

**Artifact Storage**:
- Validation artifacts stored in `core/artifacts/store.py`
- Content-addressed for deduplication
- Manifests track validation runs

**Configuration**:
- Validation configs use `core/config/schema.py`
- Preset registry shared between validation and production

---

## Timeline & Risk Assessment

### Overall Schedule (6-8 weeks)

| PR | Timeline | Risk | Blocking? |
|----|----------|------|-----------|
| PR-1: Security | Week 1 (2-3 days) | 🟢 LOW | No |
| PR-2: Platform Core | Week 1-2 (1-2 weeks) | 🟡 MEDIUM | No |
| PR-3: Stage Graph | Week 3-4 (1-2 weeks) | 🟡 MEDIUM | No |
| PR-4: Performance | Week 5 (1 week) | 🟢 LOW | No |
| PR-5: Validation | Week 6 (1 week) | 🟢 LOW | No |
| PR-6: Testing | Week 7-8 (1-2 weeks) | 🟢 LOW | No |

### Parallel Execution

**Can run in parallel**:
- PR-1 (Security) - independent
- PR-4 (Performance) - after PR-2 core device manager available
- PR-5 (Validation) - after PR-2 core validation module available
- PR-6 (Testing) - continuous throughout

**Must be sequential**:
- PR-2 (Platform Core) must complete before PR-3 (Stage Graph)
- PR-3 (Stage Graph) benefits from PR-2 artifact store

### Risk Mitigation

**Medium-Risk PRs (PR-2, PR-3)**:
- ✅ Comprehensive test coverage before merge
- ✅ Feature flags to disable if issues found
- ✅ Rollback plan documented
- ✅ Performance benchmarks before/after

**Rollback Strategy**:
- Each PR is independently revertible
- Core modules have feature flags
- Backward compatibility maintained

---

## Success Metrics

### Technical Metrics

**Security**:
- ✅ Zero banned dependencies in `pip list`
- ✅ CI security gate passes
- ✅ No secrets in git history

**Performance**:
- ✅ 10-20% speedup on GPU workloads (from caching + optimization)
- ✅ <5% overhead when profiling enabled
- ✅ UHR images (324MP+) process without OOM

**Quality**:
- ✅ 66/66 Lux Depth V2 tests passing
- ✅ 85%+ overall test coverage
- ✅ Zero feature regressions

**Maintainability**:
- ✅ 50% reduction in duplicated code
- ✅ Single source of truth for config/device/logging
- ✅ All pipelines use platform core

### Operational Metrics

**Validation Integration**:
- ✅ Validation reports use standardized schema
- ✅ Baseline comparison automated
- ✅ No blocking of validation work

**Developer Experience**:
- ✅ Clear migration guides for each PR
- ✅ Examples updated with new patterns
- ✅ Deprecation warnings guide to new APIs

---

## Next Steps

### Immediate Actions (Week 1)

1. **Review & Approve Plan**: Stakeholder sign-off on this document
2. **PR-1 Branch Creation**: `feature/architecture-hardening-pr1-security`
3. **Security Audit**: Run `gitleaks` and `trufflehog` on full history
4. **CI Gate Implementation**: `scripts/ci/enforce_safe_deps.py`

### Week 1-2: PR-1 & PR-2

1. **Complete Security Hardening** (PR-1)
   - Purge artifacts
   - Enforce CI gate
   - Update docs
   - Merge to `main`

2. **Begin Platform Core** (PR-2)
   - Create `transformation_portal/core/` module
   - Implement config, device, artifacts, security, observability
   - 90%+ test coverage on core modules
   - Integration with Lux Depth V2

### Week 3-4: PR-3

1. **Stage Graph Refactor**
   - Implement stage graph infrastructure
   - Migrate Lux Depth V2 to stage graph
   - Performance validation
   - Documentation

### Week 5-8: PR-4, PR-5, PR-6

1. **Parallel Execution**
   - PR-4: Performance profiling
   - PR-5: Validation integration
   - PR-6: Test coverage improvements

---

## Appendices

### Appendix A: File Tree After Completion

```
transformation_portal/
├── core/                           # Platform core (PR-2)
│   ├── __init__.py
│   ├── config/                     # Unified configuration
│   │   ├── schema.py
│   │   ├── validator.py
│   │   ├── loader.py
│   │   └── presets.py
│   ├── device/                     # Device management
│   │   ├── manager.py
│   │   ├── profiler.py
│   │   └── fallback.py
│   ├── artifacts/                  # Caching & storage
│   │   ├── store.py
│   │   ├── manifest.py
│   │   └── hashing.py
│   ├── security/                   # Security primitives
│   │   ├── paths.py
│   │   ├── images.py
│   │   └── validation.py
│   ├── observability/              # Logging & metrics
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   ├── pipeline/                   # Stage graph (PR-3)
│   │   ├── stage.py
│   │   ├── graph.py
│   │   └── policy.py
│   ├── validation/                 # Validation (PR-5)
│   │   ├── report.py
│   │   ├── metrics.py
│   │   └── comparison.py
│   ├── batch/                      # Batch processing (PR-6)
│   │   ├── job.py
│   │   └── processor.py
│   └── processing/                 # Processing utilities (PR-4)
│       └── tiling.py
├── lux_depth_v2/                   # Lux Depth V2 (refactored)
│   ├── pipeline.py                 # Uses core modules
│   ├── stages.py                   # Stage graph implementation
│   └── ...
└── scripts/
    └── ci/
        └── enforce_safe_deps.py    # Security gate (PR-1)
```

### Appendix B: Migration Checklist

**For Pipeline Developers**:
- [ ] Replace custom config with `core.config`
- [ ] Replace device detection with `core.device.DeviceManager`
- [ ] Use `core.artifacts.ArtifactStore` for caching
- [ ] Integrate `core.validation.ProcessingReport` for reproducibility
- [ ] Use `core.security.PathValidator` for user inputs
- [ ] Adopt `core.pipeline.Stage` pattern for new stages

**For CI/CD Maintainers**:
- [ ] Add security gate to `security-scan.yml`
- [ ] Add performance regression tests
- [ ] Enable artifact collection for reports
- [ ] Update baseline metrics after optimization

**For Documentation Writers**:
- [ ] Update README with new patterns
- [ ] Create migration guides for each PR
- [ ] Update examples to use core modules
- [ ] Document deprecation timeline

---

## Conclusion

This Architecture Hardening Plan provides a clear, risk-managed path to transform Transformation Portal into a production-grade platform while protecting ongoing validation work. The 6-priority, PR-sequenced approach ensures each change is independently verifiable, revertible, and valuable.

**Key Success Factors**:
1. **Security First**: Eliminate risks before building new features
2. **Incremental Refactoring**: Each PR stands alone and delivers value
3. **Backward Compatibility**: Existing code continues to work
4. **Validation Integration**: Standardized reports enable quality claims
5. **Performance Focus**: GPU optimization with measurement
6. **Test-Driven**: Fill coverage gaps through architecture improvements

**Ready for Implementation**: This plan is approved for execution starting Week 1 with PR-1 (Security).

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-08  
**Next Review**: 2025-12-22 (2-week milestone check)
