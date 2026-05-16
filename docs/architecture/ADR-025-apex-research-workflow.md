# ADR-025: APEX Research Workflow Architecture

**Status:** Proposed
**Date:** 2026-02-10
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-019 (Depth Backend Unification), ADR-015 (DA3 1.1 Non-Commercial), ADR-018 (Depth Pro Integration), APEX Policy

---

## Executive Summary

This ADR defines a comprehensive **APEX Research** workflow tier that incorporates the highest-quality depth estimation, segmentation, and rendering tools available in the repository, regardless of commercial licensing restrictions.

**Key Decisions:**
- Introduce **APEX Research** as a distinct tier alongside existing APEX Commercial
- Leverage research-only tools: **Depth Pro** (Apple AMLR), **DA3 1.1 Nested Giant Large** (CC BY-NC 4.0), **Segment Anything (SAM)** (vit_h), **LLaVA quality validation**
- Enforce license compliance through multi-layer gating and preset markers
- Maintain strict separation from commercial workflows via preset taxonomy and CI enforcement
- Define quality benchmarking methodology to validate research-grade claims

**Design Principle:** Research workflows should achieve demonstrably superior quality over commercial APEX while maintaining repository architectural coherence, determinism, and enforceable license governance.

---

## Context

### Current APEX Implementation (Commercial)

The repository's existing APEX tier (introduced in v2.0.0) represents the highest quality achievable with commercially-licensed tools:

| Component | Tool | License | Quality Tier |
|-----------|------|---------|--------------|
| **Depth** | Depth Anything V3 Metric Large | MIT (Commercial) | Production |
| **Segmentation** | EfficientSAM v2 + CLIP | MIT (Commercial) | High Quality |
| **PBR** | NumPy/SciPy procedural | Custom | Standard |
| **Enhancement** | v2_enhance pipeline | MIT | Production |
| **Quality Control** | Quality Firewall + APEX gates | Custom | Enforced |

**Performance Characteristics (APEX Commercial):**
- Depth: ~800ms (DA3 Large, 4K input, M4 MPS)
- Segmentation: ~400ms (EfficientSAM)
- Quality: Statistical regression enforcement (APEX gates)
- License: Fully commercial, production-ready

### Research-Grade Tools Available

Several tools in the repository offer superior quality but carry non-commercial licensing:

| Tool | Quality Advantage | License Restriction | Status |
|------|-------------------|---------------------|--------|
| **Depth Pro** | Metric depth + focal length, superior boundary precision | Apple AMLR (research-only) | Integrated (ADR-019) |
| **DA3 1.1 Nested Giant Large** | Largest DA3 variant, enhanced detail | CC BY-NC 4.0 (non-commercial) | Available via HuggingFace |
| **Segment Anything (SAM vit_h)** | 2.4GB model, zero-shot segmentation | Apache 2.0 (commercial-OK) | Integrated (`SAMSegmenter`) |
| **LLaVA-1.5 (13B)** | Vision-language quality validation | MIT (commercial-OK) | Integrated (`LLaVAProcessor`) |
| **SkyGAN** | 14EV HDR skies, parametric control | Proprietary architecture | Integrated |

**Research Hypothesis:** Combining Depth Pro + SAM vit_h + enhanced PBR tuning should yield measurably superior results over APEX Commercial, validating a research workflow tier.

### Problem Statement

1. **Untapped Quality Potential:** Research-only tools exist but lack coherent workflow integration
2. **License Ambiguity:** No clear separation between commercial and research tiers
3. **Accidental Misuse Risk:** Researchers could accidentally use research tools in commercial contexts
4. **Benchmarking Gap:** No systematic quality comparison methodology between commercial and research workflows
5. **Governance Inconsistency:** Depth Pro has license enforcement (ADR-019), but no holistic research tier policy

### Requirements

1. **Quality Maximization:** Use objectively best tools regardless of commercial licensing
2. **License Compliance:** Multi-layer enforcement preventing commercial misuse
3. **Architectural Coherence:** Reuse existing patterns (protocol-based backends, preset taxonomy, Quality Firewall)
4. **Reproducibility:** Deterministic workflow with pinned model versions
5. **Validation:** Quantitative benchmarking proving research tier superiority
6. **Coexistence:** Research and commercial workflows must coexist without interference

---

## Decision

### 1. APEX Research Tier Definition

**DECISION: Introduce `tier: apex_research` alongside existing `tier: apex`**

**Taxonomy:**

```yaml
# Tier Hierarchy (from highest to lowest quality)
tier: apex_research          # Research-only, highest quality, non-commercial license
tier: apex                   # Commercial, production-grade, MIT/Apache 2.0
tier: pro                    # Commercial, enhanced, balanced
tier: standard               # Commercial, baseline, lightweight
tier: experimental           # Unstable, testing-only
```

**Enforcement:**
- All `tier: apex_research` presets MUST include `license_restriction: research_only`
- All research presets MUST require `non_commercial_ok=True` in EnhanceConfig
- CI validation MUST block research presets without proper license markers

**Rationale:**
- Clear separation prevents commercial misuse
- Explicit hierarchy documents quality expectations
- Aligns with existing ADR-015 (DA3 1.1) governance pattern

---

### 2. APEX Research Backend Selection

**DECISION: Protocol-Based Selection with Quality-First Fallback Chain**

#### Depth Backend Strategy

**Primary: Depth Pro (Apple AMLR Research License)**

```python
# Enhanced depth backend priority for APEX Research
APEX_RESEARCH_DEPTH_PRIORITY = [
    ("depth_pro", LicenseType.RESEARCH_ONLY),           # Best: metric + focal length
    ("da3_1.1_nested_giant_large", LicenseType.RESEARCH_ONLY),  # Fallback: largest DA3
    ("da3_metric_large", LicenseType.COMMERCIAL),       # Commercial fallback
]
```

**Characteristics:**
- **Depth Pro:** Metric depth (meters), focal length estimation, superior boundary detail
- **DA3 1.1 Giant:** Largest Depth Anything variant (higher capacity, better detail)
- **DA3 Metric Large:** Commercial fallback (maintains minimum APEX quality)

**License Enforcement:** Requires BOTH:
- `EnhanceConfig.non_commercial_ok = True`
- `EnhanceConfig.accept_apple_depth_pro_research_license = True` (if Depth Pro selected)

#### Segmentation Backend Strategy

**Primary: Segment Anything (SAM) vit_h**

```python
# Segmentation backend priority for APEX Research
APEX_RESEARCH_SEGMENTATION_PRIORITY = [
    ("sam_vit_h", 2400),      # Best: 2.4GB model, zero-shot universal
    ("sam_vit_l", 1200),      # Good: 1.2GB model
    ("efficientsam_v2", 50),  # Lightweight: CLIP-based commercial fallback
]
```

**License:** Apache 2.0 (commercial-OK) — SAM is NOT license-restricted, but offers superior quality.

**Rationale:** SAM vit_h provides zero-shot universal segmentation superior to EfficientSAM's heuristic-based approach.

#### PBR Parameter Tuning

**Enhanced PBR Settings for Research Tier:**

```yaml
processing:
  pbr:
    enabled: true
    preset: RESEARCH_PREMIUM  # New preset tier

    normal:
      strength: 1.5           # vs 1.2 in commercial APEX
      edge_aware: true
      high_frequency_detail: 0.3  # Enhanced micro-detail

    roughness:
      base_value: 0.6
      depth_modulation: 0.4   # Stronger depth-based variation
      edge_sharpening: true

    ao:
      radius: 5.0             # vs 3.0 in commercial
      samples: 128            # vs 64 in commercial
      quality: "ultra"
```

**Quality Impact:** Enhanced AO sampling + higher normal strength → superior material fidelity.

---

### 3. Preset Configuration Architecture

**DECISION: Create `config/presets/apex_research.yaml` with Stable/Canary/Experimental Variants**

#### Primary Preset: `apex_research.yaml`

```yaml
name: apex-research-stable
description: "APEX Research: highest quality using research-licensed tools (Depth Pro + SAM vit_h)"
tier: apex_research
license_restriction: research_only
stability: stable

# Depth configuration
depth_backend: depth_pro
model:
  variant: depth-pro
  device: auto  # MPS > CUDA > CPU
  checkpoint_path: checkpoints/depth_pro.pt
  expected_sha256: "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"

# Segmentation configuration
segmentation_backend: sam_vit_h
segmentation:
  model_variant: vit_h
  checkpoint_path: checkpoints/sam_vit_h_4b8939.pth
  expected_sha256: "4b8939a88964f0f4ff5f5b2642c598a6fe38d7a21b7d6f1f8e8b3e9f3e9f3e9f"  # Placeholder
  confidence_threshold: 0.85  # Higher threshold for research quality

# PBR configuration
processing:
  apply_bilateral: true
  bilateral_sigma_spatial: 5.0
  bilateral_sigma_range: 0.1

  enable_zone_mapping: false  # Metric depth incompatible with relative zone mapping

  pbr:
    enabled: true
    preset: RESEARCH_PREMIUM
    normal_strength: 1.5
    roughness_base: 0.6
    ao_radius: 5.0
    ao_samples: 128

# Quality enforcement
quality:
  strict_mode: true
  quality_firewall_active: true
  allow_8bit_output: false  # Enforce 16-bit output

  apex_gates:
    enabled: true
    mode: enforce
    min_samples: 30  # Higher bar for research validation
    regression_threshold: 0.10  # Stricter than commercial (0.15)

# Output configuration
io:
  cache_enabled: true
  output_format: npz  # Enhanced metadata caching
  depth_bit_depth: 32  # Float32 for metric depth
  output_bit_depth: 16  # 16-bit PNG/TIFF output

# License compliance
compliance:
  non_commercial_ok: true
  accept_apple_depth_pro_research_license: true

# Performance expectations (reference only, not enforced)
performance_targets:
  depth_inference_ms: 1200  # Depth Pro slower than DA3
  segmentation_ms: 800      # SAM vit_h slower than EfficientSAM
  total_pipeline_ms: 3500   # ~3.5s for 4K (research acceptable)
```

#### Canary Preset: `apex_research_canary.yaml`

**Purpose:** Test upcoming model versions before promotion to stable.

```yaml
name: apex-research-canary
description: "APEX Research (Canary): testing next-gen research models"
tier: apex_research
license_restriction: research_only
stability: canary

# Extends stable with overrides
extends: apex_research

# Override with DA3 1.1 Nested Giant Large (experimental)
depth_backend: da3_1.1_nested_giant_large
model:
  variant: depth-anything-v3-nested-giant-large-1.1
  source: huggingface
  hf_id: "depth-anything/DA3-NESTED-GIANT-LARGE-1.1"
  revision: "main"  # Pin after validation

# Experimental PBR tuning
processing:
  pbr:
    normal_strength: 2.0  # Aggressive tuning for quality testing
    ao_samples: 256       # Doubled sampling (slower but higher quality)
```

#### Experimental Preset: `apex_research_experimental.yaml`

**Purpose:** Bleeding-edge combinations, may be unstable.

```yaml
name: apex-research-experimental
description: "APEX Research (Experimental): unstable, testing-only"
tier: apex_research
license_restriction: research_only
stability: experimental

extends: apex_research

# Experimental: Multi-backend depth fusion
depth_fusion:
  enabled: true
  backends:
    - depth_pro
    - da3_1.1_nested_giant_large
  fusion_method: weighted_average
  weights: [0.6, 0.4]  # Favor Depth Pro

# Experimental: LLaVA quality validation
quality:
  llava_validation:
    enabled: true
    model: liuhaotian/llava-v1.5-13b
    validation_prompts:
      - "Assess architectural detail preservation quality (0-10)"
      - "Rate material boundary sharpness (0-10)"
    quality_threshold: 7.5  # Block output if LLaVA scores < 7.5
```

---

### 4. License Compliance Enforcement Architecture

**DECISION: Multi-Layer Enforcement (3 Layers + CI Gate)**

#### Layer 1: Config Validation (Entry Point)

```python
# src/transformation_portal/lux_depth_v3/config.py
@dataclass
class EnhanceConfig:
    # Existing fields...
    non_commercial_ok: bool = False
    accept_apple_depth_pro_research_license: bool = False

    def validate(self) -> None:
        """Validate configuration and license compliance."""

        # Validate preset licensing
        if self.preset_name and "apex_research" in self.preset_name:
            if not self.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"Preset '{self.preset_name}' is APEX Research tier.\n"
                    f"Requires: non_commercial_ok=True\n"
                    f"Reason: Uses research-licensed models (Depth Pro AMLR, DA3 1.1 CC BY-NC 4.0)\n"
                    f"See: docs/architecture/ADR-025-apex-research-workflow.md"
                )

        # Depth Pro specific validation
        if self.depth_backend == "depth_pro":
            if not self.accept_apple_depth_pro_research_license:
                raise LicenseRestrictionError(
                    "Depth Pro requires explicit license acceptance.\n"
                    "Set accept_apple_depth_pro_research_license=True to acknowledge:\n"
                    "  - Apple Machine Learning Research License (AMLR)\n"
                    "  - Research and non-commercial use only\n"
                    "  - See: https://github.com/apple/ml-depth-pro/blob/main/LICENSE"
                )

        # DA3 1.1 validation
        if "1.1" in str(self.depth_model_variant):
            if not self.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"DA3 1.1 models require non_commercial_ok=True (CC BY-NC 4.0)"
                )
```

#### Layer 2: Backend Registry (Factory)

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
                    f"Backend '{backend_name}' requires non_commercial_ok=True\n"
                    f"License: {backend_cls.license_type.value}"
                )

        logger.info(
            f"License compliance: backend={backend_name}, "
            f"license={backend_cls.license_type.value}, "
            f"non_commercial_ok={config.non_commercial_ok}"
        )

        return backend_cls(config)
```

#### Layer 3: Runtime Enforcement (Defense-in-Depth)

```python
# src/transformation_portal/depth/backends/depth_pro.py
class DepthProBackend:
    """Depth Pro backend with research license enforcement."""

    name = "depth_pro"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = True

    def compute(self, image, device=None) -> DepthResult:
        """Run Depth Pro inference with license check."""

        # Layer 3: Runtime gate (defense-in-depth)
        if not self._config.accept_apple_depth_pro_research_license:
            raise LicenseRestrictionError(
                "Depth Pro runtime check failed: license not accepted"
            )

        logger.info(
            "Depth Pro license compliance: AMLR accepted, "
            f"non_commercial_ok={self._config.non_commercial_ok}"
        )

        # ... inference logic
```

#### Layer 4: CI Enforcement (Automated Governance)

```yaml
# .github/workflows/apex_research_compliance.yml
name: APEX Research License Compliance

on:
  pull_request:
    paths:
      - 'config/presets/apex_research*.yaml'
      - 'src/transformation_portal/depth/backends/**'
      - 'src/transformation_portal/lux_depth_v3/config.py'

jobs:
  validate-licenses:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate APEX Research Presets
        run: |
          python -m transformation_portal.compliance.validate_apex_research \
            --check-presets config/presets/ \
            --check-code src/

      - name: Verify License Markers
        run: |
          # All apex_research presets MUST have license_restriction marker
          for preset in config/presets/apex_research*.yaml; do
            if ! grep -q "license_restriction: research_only" "$preset"; then
              echo "ERROR: $preset missing 'license_restriction: research_only'"
              exit 1
            fi
          done

      - name: Contract Tests (Research Backends)
        run: |
          pytest tests/unit/depth/backends/test_research_license_enforcement.py -v
```

**Validation Script:** `src/transformation_portal/compliance/validate_apex_research.py`

```python
#!/usr/bin/env python3
"""Validate APEX Research license compliance."""

import argparse
import yaml
from pathlib import Path
from typing import List

RESEARCH_TIER_REQUIREMENTS = {
    "license_restriction": "research_only",
    "compliance.non_commercial_ok": True,
}

RESEARCH_ONLY_MODELS = [
    "depth_pro",
    "da3_1.1",
    "DA3-NESTED-GIANT-LARGE-1.1",
]

def validate_preset(preset_path: Path) -> List[str]:
    """Validate APEX Research preset compliance."""
    errors = []

    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    # Check tier marker
    if preset.get("tier") == "apex_research":

        # Require license_restriction marker
        if preset.get("license_restriction") != "research_only":
            errors.append(
                f"{preset_path.name}: Missing 'license_restriction: research_only'"
            )

        # Require non_commercial_ok in compliance
        compliance = preset.get("compliance", {})
        if not compliance.get("non_commercial_ok"):
            errors.append(
                f"{preset_path.name}: Missing 'compliance.non_commercial_ok: true'"
            )

        # Check depth backend license
        depth_backend = preset.get("depth_backend", "")
        if depth_backend == "depth_pro":
            if not compliance.get("accept_apple_depth_pro_research_license"):
                errors.append(
                    f"{preset_path.name}: Depth Pro requires "
                    f"'compliance.accept_apple_depth_pro_research_license: true'"
                )

    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-presets", type=Path, required=True)
    args = parser.parse_args()

    all_errors = []
    for preset_file in args.check_presets.glob("apex_research*.yaml"):
        errors = validate_preset(preset_file)
        all_errors.extend(errors)

    if all_errors:
        print("❌ License compliance validation FAILED:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("✅ All APEX Research presets comply with license policy")
    return 0

if __name__ == "__main__":
    exit(main())
```

---

### 5. Integration with Existing APEX Infrastructure

**DECISION: Extend (Don't Replace) Existing APEX Components**

#### APEX Gates Integration

APEX Research presets inherit existing APEX gate enforcement:

```python
# Existing: src/transformation_portal/metrics/gate.py
# NO CHANGES REQUIRED — works as-is with research presets

class GateResult:
    """Statistical gate evaluation result."""
    should_block: bool
    regression_pct: float
    bucket_violations: List[str]
    worst_zone_p95: float
```

**Key Insight:** APEX gates are backend-agnostic — they validate output quality regardless of depth backend.

#### Quality Firewall Integration

```python
# Existing: src/transformation_portal/lux_depth_v3/v2_enhance.py
# NO CHANGES REQUIRED

def enhance_image_v2(
    config: EnhanceConfig,
    # ... existing args
) -> EnhancedImage:
    """V2 enhancement with Quality Firewall enforcement."""

    # Quality Firewall enforces bit-depth preservation
    # Works identically for APEX Research and APEX Commercial
    if not config.allow_8bit_output and input_bit_depth == 16:
        logger.info("Quality Firewall ACTIVE: enforcing 16-bit output")
```

**Reuse:** APEX Research benefits from existing Quality Firewall without changes.

#### Performance Ledger Integration

```python
# Enhanced: src/transformation_portal/metrics/performance_capsule.py
# Add tier marker to PerformanceCapsule

@dataclass(frozen=True)
class PerformanceCapsule:
    """Immutable performance measurement capsule."""

    # ... existing fields

    # NEW: tier marker for APEX Research workflows
    tier: str = "standard"  # apex_research | apex | pro | standard
    license_mode: str = "commercial"  # commercial | research_only
```

**Migration:** Backward compatible (defaults preserve existing behavior).

---

### 6. Quality Benchmarking Methodology

**DECISION: Define Reproducible Benchmarking Protocol for Research Validation**

#### Benchmark Dataset

**Synthetic Benchmark Suite:** `tests/fixtures/apex_research_benchmark/`

```
tests/fixtures/apex_research_benchmark/
├── architectural_exteriors/
│   ├── modern_glass_facade_4k.png      # Edge detail challenge
│   ├── historic_stone_detail_4k.png    # Texture preservation
│   └── mixed_materials_balcony_4k.png  # Multi-material boundaries
├── architectural_interiors/
│   ├── luxury_kitchen_hdr_16bit.tiff   # High dynamic range
│   ├── bathroom_reflective_4k.png      # Specular surfaces
│   └── living_room_depth_complex.png   # Depth complexity
└── ground_truth/
    ├── modern_glass_facade_depth.exr   # LiDAR-derived depth
    └── metrics_reference.json          # Expected quality scores
```

**Real-World Validation Set (Optional):** `data/apex_research_validation/` (not in repo, CI uses synthetic only)

#### Quality Metrics

```python
# src/transformation_portal/metrics/apex_research_quality.py

from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class APEXResearchQualityMetrics:
    """Quality metrics for APEX Research validation."""

    # Depth Quality
    depth_mae: float            # Mean Absolute Error vs ground truth
    depth_rmse: float           # Root Mean Squared Error
    edge_sharpness: float       # Edge preservation score (0-1)
    boundary_precision: float   # Material boundary accuracy (0-1)

    # Segmentation Quality
    material_iou: float         # Intersection-over-Union for materials
    segmentation_confidence: float  # Average confidence scores

    # PBR Quality
    normal_detail_score: float  # High-frequency detail preservation
    roughness_variance: float   # Material-specific roughness variation
    ao_realism: float          # Ambient occlusion plausibility (0-1)

    # Overall Quality
    composite_score: float      # Weighted average of all metrics

    def is_research_grade(self, commercial_baseline: "APEXResearchQualityMetrics") -> bool:
        """Validate that research tier exceeds commercial baseline."""

        # Research MUST exceed commercial in at least 3/4 categories
        improvements = 0

        if self.depth_mae < commercial_baseline.depth_mae * 0.9:
            improvements += 1
        if self.edge_sharpness > commercial_baseline.edge_sharpness * 1.1:
            improvements += 1
        if self.material_iou > commercial_baseline.material_iou * 1.1:
            improvements += 1
        if self.normal_detail_score > commercial_baseline.normal_detail_score * 1.1:
            improvements += 1

        return improvements >= 3
```

#### Benchmark Script

```python
# scripts/apex_research_benchmark.py

#!/usr/bin/env python3
"""Benchmark APEX Research vs APEX Commercial quality."""

import argparse
from pathlib import Path
from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
from transformation_portal.lux_depth_v3.orchestrator import OrchestrationEngine
from transformation_portal.metrics.apex_research_quality import (
    APEXResearchQualityMetrics,
    compute_quality_metrics,
)

def run_benchmark(
    input_dir: Path,
    output_dir: Path,
    ground_truth_dir: Path
) -> Dict[str, APEXResearchQualityMetrics]:
    """Run APEX Research vs Commercial benchmark."""

    results = {}

    # Run Commercial APEX
    commercial_config = EnhanceConfig(
        preset=Preset.APEX,  # Existing commercial APEX
        non_commercial_ok=False,
    )

    commercial_metrics = run_workflow_and_measure(
        config=commercial_config,
        input_dir=input_dir,
        output_dir=output_dir / "commercial",
        ground_truth_dir=ground_truth_dir,
    )
    results["apex_commercial"] = commercial_metrics

    # Run Research APEX
    research_config = EnhanceConfig(
        preset=Preset.APEX_RESEARCH,  # New research preset
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )

    research_metrics = run_workflow_and_measure(
        config=research_config,
        input_dir=input_dir,
        output_dir=output_dir / "research",
        ground_truth_dir=ground_truth_dir,
    )
    results["apex_research"] = research_metrics

    # Validate research tier superiority
    if research_metrics.is_research_grade(commercial_metrics):
        print("✅ APEX Research validated: superior to commercial baseline")
    else:
        print("❌ APEX Research failed validation: insufficient quality improvement")
        return None

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    results = run_benchmark(args.input, args.output, args.ground_truth)

    # Generate comparison report
    print("\n=== APEX Quality Comparison ===")
    print(f"Commercial MAE: {results['apex_commercial'].depth_mae:.4f}")
    print(f"Research MAE:   {results['apex_research'].depth_mae:.4f}")
    print(f"Improvement:    {(1 - results['apex_research'].depth_mae / results['apex_commercial'].depth_mae) * 100:.1f}%")
```

#### CI Benchmark Integration

```yaml
# .github/workflows/apex_research_benchmark.yml
name: APEX Research Quality Benchmark

on:
  pull_request:
    paths:
      - 'config/presets/apex_research*.yaml'
      - 'src/transformation_portal/depth/backends/depth_pro.py'

jobs:
  benchmark:
    runs-on: ubuntu-latest-m4  # Apple Silicon runner (if available)
    steps:
      - uses: actions/checkout@v4

      - name: Download Checkpoints
        run: |
          # Depth Pro checkpoint
          wget -O checkpoints/depth_pro.pt \
            https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

          # Verify SHA256
          echo "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce  checkpoints/depth_pro.pt" | \
            shasum -a 256 -c -

      - name: Run Benchmark
        run: |
          python scripts/apex_research_benchmark.py \
            --input tests/fixtures/apex_research_benchmark/architectural_exteriors \
            --output artifacts/benchmark_results \
            --ground-truth tests/fixtures/apex_research_benchmark/ground_truth

      - name: Validate Quality Improvement
        run: |
          # Fail if research tier doesn't exceed commercial
          python -c "
          import json
          with open('artifacts/benchmark_results/comparison.json') as f:
              data = json.load(f)

          improvement_pct = data['improvement_percentage']
          if improvement_pct < 10:  # Require at least 10% improvement
              print(f'❌ Research tier only {improvement_pct}% better (need ≥10%)')
              exit(1)
          print(f'✅ Research tier {improvement_pct}% better than commercial')
          "
```

---

## Consequences

### Positive

1. **Quality Leadership:** APEX Research demonstrates repository's highest achievable quality
2. **License Safety:** Multi-layer enforcement prevents accidental commercial misuse
3. **Research Enablement:** Academic/non-profit users gain access to premium workflows
4. **Architectural Coherence:** Reuses existing patterns (backends, presets, Quality Firewall, APEX gates)
5. **Reproducibility:** Pinned checkpoints + SHA256 validation ensure deterministic results
6. **Quantitative Validation:** Benchmark suite proves research tier superiority (not subjective claims)

### Negative

1. **Complexity:** Two APEX tiers (commercial + research) increases cognitive load
2. **Performance:** Research tier slower (Depth Pro ~1.2s vs DA3 Large ~800ms for 4K)
3. **Licensing Friction:** Multi-flag requirements (`non_commercial_ok` + `accept_apple_depth_pro_research_license`) verbose
4. **Maintenance Burden:** Must track two model ecosystems (commercial + research)
5. **CI Overhead:** Benchmark workflow requires expensive compute (M4 runners, large checkpoints)

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **License Bypass:** Research tools used commercially | Multi-layer enforcement (config + registry + runtime + CI) |
| **Performance Regression:** Research tier too slow for practical use | Document performance expectations in preset; optional for users |
| **Quality Claims Unfounded:** Research tier not measurably better | Mandatory benchmark validation in CI; fail if <10% improvement |
| **Supply Chain:** Depth Pro checkpoint compromised | SHA256 validation enforced in preset + CI |
| **Accidental Tier Confusion:** Users mix commercial/research | Explicit tier markers in preset names + CLI warnings |

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (1 PR)

**Goal:** Establish APEX Research preset + license enforcement.

**Tasks:**
- [ ] Create `config/presets/apex_research.yaml` (stable preset)
- [ ] Create `config/presets/apex_research_canary.yaml` (canary preset)
- [ ] Implement `validate_apex_research.py` compliance script
- [ ] Add `tier` and `license_mode` fields to `PerformanceCapsule` (backward compatible)
- [ ] Add CI workflow: `apex_research_compliance.yml`
- [ ] Unit tests: `tests/compliance/test_apex_research_enforcement.py`
- [ ] Documentation: Update `README.md` with APEX Research section

**Acceptance Criteria:**
- All `apex_research*.yaml` presets validated in CI
- License enforcement tests pass (config + registry + runtime layers)
- Preset loading with wrong license flags raises `LicenseRestrictionError`

### Phase 2: SAM vit_h Integration (1 PR)

**Goal:** Integrate Segment Anything (vit_h) as research-tier segmentation backend.

**Tasks:**
- [ ] Extend `SegmentationBackend` protocol to support SAM
- [ ] Implement `SAMVitHBackend` wrapper (reuse existing `SAMSegmenter`)
- [ ] Add checkpoint download + SHA256 validation
- [ ] Update `apex_research.yaml` to use `segmentation_backend: sam_vit_h`
- [ ] Performance profiling: SAM vit_h vs EfficientSAM
- [ ] Unit tests: `tests/materials/test_sam_vit_h_backend.py`

**Acceptance Criteria:**
- SAM vit_h backend selectable via preset
- Checkpoint validation enforced (SHA256 mismatch blocks execution)
- Segmentation quality measurably superior to EfficientSAM (≥10% IoU improvement)

### Phase 3: Quality Benchmarking (1 PR)

**Goal:** Implement reproducible benchmark suite validating research tier superiority.

**Tasks:**
- [ ] Create synthetic benchmark fixtures: `tests/fixtures/apex_research_benchmark/`
- [ ] Implement `apex_research_quality.py` metrics module
- [ ] Implement `scripts/apex_research_benchmark.py` comparison script
- [ ] Add CI workflow: `apex_research_benchmark.yml`
- [ ] Generate baseline metrics for APEX Commercial
- [ ] Validate APEX Research ≥10% improvement across ≥3/4 metrics

**Acceptance Criteria:**
- Benchmark runs in CI without manual intervention
- Research tier demonstrates quantifiable superiority (not subjective)
- Failing quality threshold blocks PR merge

### Phase 4: Enhanced PBR Research Preset (1 PR, Optional)

**Goal:** Tune PBR parameters specifically for research tier quality.

**Tasks:**
- [ ] Create `RESEARCH_PREMIUM` PBR preset (higher AO samples, normal strength)
- [ ] A/B testing: compare research PBR vs standard PBR on benchmark
- [ ] Document performance/quality trade-offs
- [ ] Add `apex_research_experimental.yaml` with aggressive PBR tuning

**Acceptance Criteria:**
- Research PBR preset demonstrates ≥5% improvement in normal detail score
- Performance overhead documented and acceptable (<1.5x slower)

---

## Testing Strategy

### Unit Tests

**Location:** `tests/compliance/test_apex_research_enforcement.py`

```python
import pytest
from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
from transformation_portal.compliance.licensing import LicenseRestrictionError

def test_apex_research_requires_non_commercial_flag():
    """APEX Research preset requires non_commercial_ok=True."""
    config = EnhanceConfig(
        preset=Preset.APEX_RESEARCH,
        non_commercial_ok=False,  # Should fail
    )

    with pytest.raises(LicenseRestrictionError, match="non_commercial_ok=True"):
        config.validate()

def test_depth_pro_requires_explicit_license():
    """Depth Pro requires accept_apple_depth_pro_research_license=True."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=False,  # Should fail
    )

    with pytest.raises(LicenseRestrictionError, match="Depth Pro requires explicit"):
        config.validate()

def test_commercial_apex_unaffected():
    """Commercial APEX workflows work without research flags."""
    config = EnhanceConfig(
        preset=Preset.APEX,
        non_commercial_ok=False,  # Should succeed
    )

    config.validate()  # No exception
```

### Integration Tests

**Location:** `tests/integration/test_apex_research_workflow.py`

```python
import pytest
from transformation_portal.lux_depth_v3.orchestrator import OrchestrationEngine
from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset

@pytest.mark.slow
@pytest.mark.requires_checkpoints
def test_apex_research_full_pipeline(synthetic_4k_image):
    """Full APEX Research pipeline with Depth Pro + SAM vit_h."""
    config = EnhanceConfig(
        preset=Preset.APEX_RESEARCH,
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )

    engine = OrchestrationEngine(config)
    result = engine.run(synthetic_4k_image)

    # Validate result quality
    assert result.depth_result.depth_units == "meters"
    assert result.depth_result.focal_length_px is not None
    assert result.segmentation_masks is not None
    assert len(result.segmentation_masks) >= 3  # Expect multiple materials
```

### Contract Tests

**Location:** `tests/unit/depth/backends/test_research_backends.py`

```python
def test_depth_pro_backend_protocol_compliance():
    """Depth Pro backend implements DepthBackend protocol."""
    from transformation_portal.depth.backends.depth_pro import DepthProBackend
    from transformation_portal.depth.backends.protocol import DepthBackend, LicenseType

    # Protocol compliance
    assert hasattr(DepthProBackend, 'name')
    assert hasattr(DepthProBackend, 'license_type')
    assert hasattr(DepthProBackend, 'compute')

    # License enforcement
    assert DepthProBackend.license_type == LicenseType.RESEARCH_ONLY
```

### Benchmark Validation Tests

**Location:** `tests/benchmark/test_apex_research_quality.py`

```python
@pytest.mark.benchmark
def test_apex_research_exceeds_commercial_baseline():
    """APEX Research must demonstrate ≥10% quality improvement."""
    from scripts.apex_research_benchmark import run_benchmark

    results = run_benchmark(
        input_dir=Path("tests/fixtures/apex_research_benchmark/architectural_exteriors"),
        output_dir=Path("artifacts/test_benchmark"),
        ground_truth_dir=Path("tests/fixtures/apex_research_benchmark/ground_truth"),
    )

    commercial = results["apex_commercial"]
    research = results["apex_research"]

    # Validate improvement
    depth_improvement = (commercial.depth_mae - research.depth_mae) / commercial.depth_mae
    assert depth_improvement >= 0.10, f"Research only {depth_improvement*100:.1f}% better (need ≥10%)"
```

---

## Required Documentation

### 1. README Update

**Location:** `/README.md`

**Section:** "Quality Tiers"

```markdown
## Quality Tiers

### APEX Research (Research/Non-Commercial Only)

**License:** Research-only (Apple AMLR + CC BY-NC 4.0)
**Purpose:** Highest quality achievable, academic/research workflows
**Tools:** Depth Pro, SAM vit_h, enhanced PBR tuning
**Performance:** ~3.5s for 4K (slower than commercial APEX)

**Enable:**
```python
config = EnhanceConfig(
    preset=Preset.APEX_RESEARCH,
    non_commercial_ok=True,
    accept_apple_depth_pro_research_license=True,
)
```

**Quality Expectations:**
- ≥10% depth MAE improvement over commercial APEX
- Superior material boundary precision (SAM vit_h zero-shot)
- Enhanced PBR detail (higher AO samples, normal strength)

**Restrictions:**
- ❌ Commercial use prohibited
- ❌ Revenue-generating services prohibited
- ✅ Academic research, benchmarking, non-profit projects

---

### APEX (Commercial)

**License:** MIT/Apache 2.0 (fully commercial)
**Purpose:** Production-grade highest quality
**Tools:** Depth Anything V3 Large, EfficientSAM v2, standard PBR
**Performance:** ~2.5s for 4K

**Enable:**
```python
config = EnhanceConfig(preset=Preset.APEX)
```

**Quality Expectations:**
- Statistical regression enforcement (APEX gates)
- Quality Firewall (16-bit preservation)
- Production-ready, enterprise-approved
```

### 2. ADR Cross-References

**Update:** `docs/architecture/ADR-019-depth-backend-unification.md`

Add cross-reference:
```markdown
**Related:** ADR-025 (APEX Research Workflow) — extends backend unification to research tier
```

**Update:** `docs/architecture/ADR-015-da3-1-1-non-commercial-research-tier.md`

Add cross-reference:
```markdown
**Related:** ADR-025 (APEX Research Workflow) — comprehensive research tier architecture
```

### 3. Preset Documentation

**Location:** `config/presets/README.md`

**Section:** "Preset Taxonomy"

```markdown
## Preset Taxonomy

### Stability Tiers

- **stable:** Production-ready, thoroughly validated
- **canary:** Testing next-gen features before stable promotion
- **experimental:** Unstable, research/testing only

### Quality Tiers

- **apex_research:** Highest quality (research-only licenses)
- **apex:** Highest quality (commercial licenses)
- **pro:** Enhanced quality (commercial)
- **standard:** Baseline quality (commercial)

### License Restrictions

- **research_only:** CC BY-NC 4.0, Apple AMLR, or similar non-commercial licenses
- **commercial:** MIT, Apache 2.0, or other permissive licenses
```

### 4. Compliance Documentation

**Location:** `docs/architecture/license_compliance_guide.md` (new document)

```markdown
# License Compliance Guide

## APEX Research Tier

### Legal Requirements

**You may use APEX Research tier if:**
- ✅ Academic research (university, institute)
- ✅ Non-profit projects (no revenue generation)
- ✅ Personal experimentation (non-commercial)
- ✅ Benchmarking and comparative studies

**You may NOT use APEX Research tier if:**
- ❌ Commercial products or services
- ❌ Revenue-generating applications
- ❌ Enterprise/business deployments
- ❌ Proprietary software distribution

### Technical Enforcement

APEX Research enforces compliance through:

1. **Config Flag:** `non_commercial_ok=True` (explicit opt-in)
2. **Depth Pro License:** `accept_apple_depth_pro_research_license=True`
3. **Preset Marker:** `license_restriction: research_only`
4. **Runtime Gate:** License checked before inference
5. **CI Validation:** Automated compliance checks

### Violation Consequences

Using research-licensed models commercially:
- Violates Apple AMLR license agreement
- Violates Creative Commons BY-NC 4.0
- May result in legal liability

**If you're unsure:** Use APEX Commercial tier (fully licensed for commercial use).
```

---

## Alternatives Considered

### Alternative 1: Single APEX Tier with License Flag

**Approach:** Keep one `apex` preset, use `non_commercial_ok` flag to enable research tools.

**Rejected:**
- Confusing UX (one preset, two quality levels)
- Harder to communicate quality expectations
- Increased risk of accidental commercial misuse

### Alternative 2: Separate Repository for Research

**Approach:** Fork APEX Research into `transformation-portal-research` repo.

**Rejected:**
- Fragments ecosystem
- Duplicates CI/infrastructure
- Harder to maintain integration with LuxRender pipeline
- Reduces visibility for academic users

### Alternative 3: No APEX Research (Ban Research Tools)

**Approach:** Only support commercially-licensed tools, ban Depth Pro/DA3 1.1.

**Rejected:**
- Blocks valuable academic/research use cases
- Reduces repository's quality leadership positioning
- Wastes existing Depth Pro integration effort (ADR-019)

### Alternative 4: Research Tier Without Benchmarking

**Approach:** Allow research tools but don't validate quality superiority.

**Rejected:**
- Unsubstantiated quality claims (lacks rigor)
- Could mislead users into thinking research tier is better without proof
- Violates "enforcement over documentation" principle

**Chosen:** Option in this ADR — **Separate APEX Research tier with mandatory benchmark validation**

---

## Migration Plan

### For Existing Commercial Users

**Impact:** NONE (backward compatible)

- APEX Commercial tier unchanged
- Existing presets work identically
- `non_commercial_ok` defaults to `False` (safe default)

### For Research Users

**Adoption Path:**

1. **Read License Documentation:** Review `docs/architecture/license_compliance_guide.md`
2. **Download Checkpoints:** Depth Pro (1.9GB), SAM vit_h (2.4GB)
3. **Enable Research Preset:**
   ```python
   config = EnhanceConfig(
       preset=Preset.APEX_RESEARCH,
       non_commercial_ok=True,
       accept_apple_depth_pro_research_license=True,
   )
   ```
4. **Run Benchmark (Optional):** Validate quality improvement on your data
5. **Acknowledge License:** Confirm non-commercial use only

### For CI/Deployment

**Required Updates:**

- Add checkpoint download step (if using APEX Research in CI)
- Add SHA256 validation for checkpoints
- Add compliance validation workflow (automatic)

**No Breaking Changes:** Commercial workflows unaffected.

---

## Governance & Approval

### Authority Chain

- ✅ **Architect:** Approved as ADR-025 (Transformation Portal Architect authority)
- **Specialist:** Implementation per ADR requirements
- **CI:** Automated enforcement via compliance + benchmark validation
- **Community:** Review and feedback via PR discussion

### Review Triggers

- **Any new research-licensed model:** Validate license, update preset, add test
- **APEX Research preset changes:** Trigger benchmark validation
- **License compliance failures:** Escalate to Architect immediately

### ADR Binding Rule

This ADR is binding. Deviations require:
- Explicit superseding ADR
- Clear migration plan
- Architect approval

---

## References

### Internal ADRs

- [ADR-019: Depth Backend Unification](ADR-019-depth-backend-unification.md)
- [ADR-015: DA3 1.1 Non-Commercial Research Tier](ADR-015-da3-1-1-non-commercial-research-tier.md)
- [ADR-018: Depth Pro Integration](ADR-018-depth-pro-integration.md)
- [ADR-001: PBR Integration Architecture](ADR-001-PBR-Integration-Architecture.md)
- [Agent Governance Policy](agent_governance.md)

### External Resources

- [Apple Depth Pro Repository](https://github.com/apple/ml-depth-pro)
- [Apple Machine Learning Research License (AMLR)](https://github.com/apple/ml-depth-pro/blob/main/LICENSE)
- [Segment Anything (SAM) Repository](https://github.com/facebookresearch/segment-anything)
- [Depth Anything V3 Repository](https://github.com/ByteDance/depth-anything-v3)
- [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)

### Code References

- `src/transformation_portal/depth/backends/` (Depth backend protocol + registry)
- `src/transformation_portal/lux_depth_v3/config.py` (EnhanceConfig)
- `src/transformation_portal/compliance/licensing.py` (License enforcement)
- `src/transformation_portal/metrics/gate.py` (APEX gates)
- `config/presets/apex_research.yaml` (Research preset)

---

**Document History**
- **2026-02-10:** Initial ADR-025 created (APEX Research Workflow Architecture)
