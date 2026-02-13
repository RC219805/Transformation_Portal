# APEX Research Testing & Validation Strategy

**Date:** 2026-02-10
**Authority:** Transformation Portal Architect
**Related:** ADR-025 (APEX Research Workflow Architecture)

---

## Overview

This document defines the comprehensive testing strategy for APEX Research, ensuring:
- License compliance enforcement at all layers
- Quality superiority over commercial APEX (validated quantitatively)
- Backward compatibility with existing workflows
- Deterministic and reproducible behavior

**Testing Philosophy:** Fast, deterministic unit tests + selective integration tests + mandatory benchmark validation.

---

## Testing Layers

### Layer 1: Unit Tests (Fast, No Models)

**Location:** `tests/compliance/`, `tests/unit/depth/backends/`, `tests/materials/`
**Execution:** Every commit, <30 seconds total
**Coverage Target:** ≥90% for license enforcement and backend protocol compliance

#### 1.1 License Enforcement Tests

**File:** `tests/compliance/test_apex_research_enforcement.py`

```python
"""APEX Research license enforcement unit tests."""

import pytest
from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
from transformation_portal.compliance.licensing import LicenseRestrictionError

class TestAPEXResearchLicenseEnforcement:
    """Test multi-layer license enforcement for APEX Research."""

    def test_apex_research_requires_non_commercial_flag(self):
        """APEX Research preset requires non_commercial_ok=True."""
        config = EnhanceConfig(
            preset=Preset.APEX_RESEARCH,
            non_commercial_ok=False,  # Should fail
        )

        with pytest.raises(LicenseRestrictionError) as exc_info:
            config.validate()

        assert "non_commercial_ok=True" in str(exc_info.value)
        assert "research-licensed models" in str(exc_info.value)

    def test_depth_pro_requires_explicit_license(self):
        """Depth Pro requires accept_apple_depth_pro_research_license=True."""
        config = EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=False,  # Should fail
        )

        with pytest.raises(LicenseRestrictionError) as exc_info:
            config.validate()

        assert "accept_apple_depth_pro_research_license=True" in str(exc_info.value)
        assert "Apple Machine Learning Research License" in str(exc_info.value)

    def test_commercial_apex_unaffected(self):
        """Commercial APEX workflows work without research flags."""
        config = EnhanceConfig(
            preset=Preset.APEX,  # Commercial APEX
            non_commercial_ok=False,  # Should succeed
        )

        config.validate()  # No exception

    def test_da3_1_1_requires_non_commercial_flag(self):
        """DA3 1.1 models require non_commercial_ok=True."""
        config = EnhanceConfig(
            depth_model_variant="depth-anything-v3-nested-giant-large-1.1",
            non_commercial_ok=False,  # Should fail
        )

        with pytest.raises(LicenseRestrictionError) as exc_info:
            config.validate()

        assert "CC BY-NC 4.0" in str(exc_info.value)

    def test_multi_layer_enforcement_config_layer(self):
        """Layer 1 (config validation) catches license violations."""
        # Test that config.validate() is called before backend instantiation
        config = EnhanceConfig(
            preset=Preset.APEX_RESEARCH,
            non_commercial_ok=False,
        )

        # Should fail at config layer (before reaching registry/runtime)
        with pytest.raises(LicenseRestrictionError):
            config.validate()

    def test_license_metadata_tracked_in_provenance(self):
        """License mode tracked in performance capsule."""
        from transformation_portal.metrics.performance_capsule import PerformanceCapsule

        capsule = PerformanceCapsule(
            tier="apex_research",
            license_mode="research_only",
            # ... other required fields
        )

        assert capsule.tier == "apex_research"
        assert capsule.license_mode == "research_only"
```

#### 1.2 Backend Protocol Compliance Tests

**File:** `tests/unit/depth/backends/test_depth_pro_backend.py`

```python
"""Depth Pro backend protocol compliance tests."""

import pytest
from unittest.mock import Mock, patch
from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.depth.backends.protocol import (
    DepthBackend,
    LicenseType,
    DepthResult,
)

class TestDepthProBackendProtocol:
    """Test Depth Pro backend implements DepthBackend protocol."""

    def test_protocol_compliance(self):
        """Depth Pro backend implements required protocol attributes."""
        assert hasattr(DepthProBackend, 'name')
        assert hasattr(DepthProBackend, 'license_type')
        assert hasattr(DepthProBackend, 'requires_checkpoint')
        assert hasattr(DepthProBackend, 'compute')
        assert hasattr(DepthProBackend, 'get_cache_key')

    def test_license_type_is_research_only(self):
        """Depth Pro license type is RESEARCH_ONLY."""
        assert DepthProBackend.license_type == LicenseType.RESEARCH_ONLY

    def test_requires_checkpoint_true(self):
        """Depth Pro requires checkpoint download."""
        assert DepthProBackend.requires_checkpoint is True

    @patch('transformation_portal.depth.backends.depth_pro.DepthProStage')
    def test_compute_returns_depth_result(self, mock_stage_cls):
        """compute() returns DepthResult with metric depth."""
        # Mock DepthProStage to avoid loading checkpoint
        mock_stage = Mock()
        mock_stage.compute.return_value = Mock(
            status='completed',
            artifacts={
                'depth_map': np.random.rand(100, 100),
                'depth_provenance': {'model': 'depth_pro'},
            },
            metadata={'focal_length_px': 500.0, 'fov_deg': 60.0},
        )
        mock_stage_cls.return_value = mock_stage

        # Create backend
        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )
        backend = DepthProBackend(config)

        # Test compute
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = backend.compute(image)

        # Validate result
        assert isinstance(result, DepthResult)
        assert result.depth_units == "meters"
        assert result.focal_length_px == 500.0
        assert result.field_of_view_deg == 60.0

    def test_runtime_license_check(self):
        """Layer 3 (runtime) license check enforced."""
        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=False,  # Missing
        )

        backend = DepthProBackend(config)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(LicenseRestrictionError):
            backend.compute(image)
```

#### 1.3 Preset Validation Tests

**File:** `tests/compliance/test_preset_validation.py`

```python
"""Preset validation tests."""

import pytest
import yaml
from pathlib import Path
from transformation_portal.compliance.validate_apex_research import validate_preset

class TestPresetValidation:
    """Test APEX Research preset validation."""

    def test_apex_research_preset_has_license_marker(self):
        """apex_research.yaml has license_restriction: research_only."""
        preset_path = Path("config/presets/apex_research.yaml")
        errors = validate_preset(preset_path)

        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_missing_license_marker_detected(self, tmp_path):
        """Preset without license_restriction marker fails validation."""
        # Create invalid preset
        invalid_preset = {
            "name": "invalid-research",
            "tier": "apex_research",
            # Missing: license_restriction: research_only
            "compliance": {"non_commercial_ok": True},
        }

        preset_path = tmp_path / "invalid.yaml"
        with open(preset_path, 'w') as f:
            yaml.dump(invalid_preset, f)

        errors = validate_preset(preset_path)
        assert len(errors) > 0
        assert "license_restriction" in errors[0]

    def test_depth_pro_preset_requires_explicit_license(self, tmp_path):
        """Depth Pro preset requires accept_apple_depth_pro_research_license."""
        invalid_preset = {
            "name": "invalid-depth-pro",
            "tier": "apex_research",
            "license_restriction": "research_only",
            "depth_backend": "depth_pro",
            "compliance": {
                "non_commercial_ok": True,
                # Missing: accept_apple_depth_pro_research_license
            },
        }

        preset_path = tmp_path / "invalid.yaml"
        with open(preset_path, 'w') as f:
            yaml.dump(invalid_preset, f)

        errors = validate_preset(preset_path)
        assert len(errors) > 0
        assert "accept_apple_depth_pro_research_license" in errors[0]
```

---

### Layer 2: Integration Tests (Selective, Mocked Models)

**Location:** `tests/integration/`
**Execution:** Pre-merge CI, ~2-5 minutes
**Coverage Target:** Critical paths (preset loading, backend selection, license gates)

#### 2.1 Orchestrator Integration Tests

**File:** `tests/integration/test_apex_research_orchestrator.py`

```python
"""APEX Research orchestrator integration tests (with mocked models)."""

import pytest
from unittest.mock import Mock, patch
from transformation_portal.lux_depth_v3.orchestrator import OrchestrationEngine
from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset

class TestAPEXResearchOrchestrator:
    """Test APEX Research workflow orchestration."""

    @patch('transformation_portal.depth.backends.depth_pro.DepthProBackend')
    @patch('transformation_portal.lux_depth_v3.segmentation_backend.SAMVitHBackend')
    def test_preset_loading_with_license_acceptance(
        self,
        mock_sam_cls,
        mock_depth_pro_cls,
        synthetic_4k_image,
    ):
        """APEX Research preset loads with correct license flags."""
        # Mock backends to avoid loading checkpoints
        mock_depth_pro_cls.return_value.compute.return_value = Mock(
            depth_map=np.random.rand(100, 100),
            original_image=synthetic_4k_image,
            metadata={},
            depth_units="meters",
            focal_length_px=500.0,
        )
        mock_sam_cls.return_value.segment.return_value = {}

        config = EnhanceConfig(
            preset=Preset.APEX_RESEARCH,
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )

        engine = OrchestrationEngine(config)
        result = engine.run(synthetic_4k_image)

        # Validate backend selection
        assert mock_depth_pro_cls.called
        assert mock_sam_cls.called

    def test_missing_license_flag_blocks_orchestration(self, synthetic_4k_image):
        """Missing license flag prevents orchestration."""
        config = EnhanceConfig(
            preset=Preset.APEX_RESEARCH,
            non_commercial_ok=False,  # Missing
        )

        with pytest.raises(LicenseRestrictionError):
            engine = OrchestrationEngine(config)
            engine.run(synthetic_4k_image)

    @patch('transformation_portal.depth.backends.registry.DepthBackendRegistry')
    def test_backend_registry_layer2_enforcement(self, mock_registry_cls):
        """Backend registry (Layer 2) enforces license."""
        mock_registry = mock_registry_cls.return_value
        mock_registry.get_backend.side_effect = LicenseRestrictionError(
            "Backend 'depth_pro' requires non_commercial_ok=True"
        )

        config = EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=False,
        )

        with pytest.raises(LicenseRestrictionError):
            engine = OrchestrationEngine(config)
```

---

### Layer 3: Benchmark Validation (Synthetic Fixtures)

**Location:** `tests/benchmark/`
**Execution:** Pre-merge CI (gated), ~10-15 minutes
**Coverage Target:** Quality validation (APEX Research > APEX Commercial)

#### 3.1 Quality Metrics Tests

**File:** `tests/benchmark/test_apex_research_quality_metrics.py`

```python
"""APEX Research quality metrics tests."""

import pytest
import numpy as np
from transformation_portal.metrics.apex_research_quality import (
    APEXResearchQualityMetrics,
    compute_depth_mae,
    compute_edge_sharpness,
    compute_material_iou,
)

class TestQualityMetrics:
    """Test quality metric computation functions."""

    def test_depth_mae_computation(self):
        """Depth MAE computed correctly."""
        predicted = np.array([[1.0, 2.0], [3.0, 4.0]])
        ground_truth = np.array([[1.1, 2.1], [3.1, 4.1]])

        mae = compute_depth_mae(predicted, ground_truth)
        assert np.isclose(mae, 0.1)

    def test_edge_sharpness_higher_for_sharper_images(self):
        """Edge sharpness score higher for sharper edges."""
        # Sharp image (high gradients)
        sharp = np.zeros((100, 100))
        sharp[:, 50:] = 1.0  # Hard edge

        # Blurry image (low gradients)
        blurry = np.zeros((100, 100))
        for i in range(100):
            blurry[i, :] = i / 100.0  # Soft gradient

        sharp_score = compute_edge_sharpness(sharp)
        blurry_score = compute_edge_sharpness(blurry)

        assert sharp_score > blurry_score

    def test_is_research_grade_validation(self):
        """is_research_grade() validates improvement threshold."""
        commercial = APEXResearchQualityMetrics(
            depth_mae=1.0,
            edge_sharpness=0.5,
            material_iou=0.7,
            normal_detail_score=0.6,
            composite_score=0.65,
        )

        # Research with insufficient improvement
        research_bad = APEXResearchQualityMetrics(
            depth_mae=0.95,  # Only 5% better
            edge_sharpness=0.52,
            material_iou=0.72,
            normal_detail_score=0.61,
            composite_score=0.67,
        )

        assert not research_bad.is_research_grade(commercial)

        # Research with sufficient improvement (≥10% in 3/4 metrics)
        research_good = APEXResearchQualityMetrics(
            depth_mae=0.85,  # 15% better
            edge_sharpness=0.60,  # 20% better
            material_iou=0.80,  # 14% better
            normal_detail_score=0.61,  # Only 2% better
            composite_score=0.72,
        )

        assert research_good.is_research_grade(commercial)
```

#### 3.2 End-to-End Benchmark Tests

**File:** `tests/benchmark/test_apex_research_benchmark.py`

```python
"""End-to-end APEX Research benchmark tests."""

import pytest
from pathlib import Path
from scripts.apex_research_benchmark import run_benchmark

@pytest.mark.benchmark
@pytest.mark.slow
class TestAPEXResearchBenchmark:
    """Test APEX Research benchmark suite."""

    def test_benchmark_reproducibility(self, benchmark_fixtures_dir):
        """Benchmark produces deterministic results across runs."""
        results1 = run_benchmark(
            input_dir=benchmark_fixtures_dir / "architectural_exteriors",
            output_dir=Path("artifacts/benchmark_run1"),
            ground_truth_dir=benchmark_fixtures_dir / "ground_truth",
        )

        results2 = run_benchmark(
            input_dir=benchmark_fixtures_dir / "architectural_exteriors",
            output_dir=Path("artifacts/benchmark_run2"),
            ground_truth_dir=benchmark_fixtures_dir / "ground_truth",
        )

        # Results should be identical (deterministic)
        assert np.isclose(
            results1["apex_research"].depth_mae,
            results2["apex_research"].depth_mae,
            rtol=1e-5,
        )

    def test_research_tier_exceeds_commercial(self, benchmark_fixtures_dir):
        """APEX Research demonstrates ≥10% improvement over commercial."""
        results = run_benchmark(
            input_dir=benchmark_fixtures_dir / "architectural_exteriors",
            output_dir=Path("artifacts/benchmark_validation"),
            ground_truth_dir=benchmark_fixtures_dir / "ground_truth",
        )

        commercial = results["apex_commercial"]
        research = results["apex_research"]

        # Validate research tier superiority
        assert research.is_research_grade(commercial), (
            f"Research tier failed quality validation:\n"
            f"  Depth MAE: {research.depth_mae:.4f} vs {commercial.depth_mae:.4f}\n"
            f"  Edge Sharpness: {research.edge_sharpness:.4f} vs {commercial.edge_sharpness:.4f}\n"
            f"  Material IoU: {research.material_iou:.4f} vs {commercial.material_iou:.4f}"
        )

    @pytest.mark.parametrize("fixture_name", [
        "modern_glass_facade_4k.png",
        "historic_stone_detail_4k.png",
        "mixed_materials_balcony_4k.png",
    ])
    def test_benchmark_per_fixture(self, benchmark_fixtures_dir, fixture_name):
        """Test benchmark on individual fixtures."""
        # Run benchmark on single fixture
        results = run_benchmark(
            input_dir=benchmark_fixtures_dir / "architectural_exteriors" / fixture_name,
            output_dir=Path(f"artifacts/benchmark_{fixture_name}"),
            ground_truth_dir=benchmark_fixtures_dir / "ground_truth",
        )

        # Validate metrics computed
        assert results["apex_research"].depth_mae > 0
        assert results["apex_commercial"].depth_mae > 0
```

---

### Layer 4: Contract Tests (Regression Prevention)

**Location:** `tests/contracts/`
**Execution:** Every commit, <10 seconds
**Coverage Target:** Public API stability, backward compatibility

#### 4.1 PerformanceCapsule Contract Tests

**File:** `tests/contracts/test_performance_capsule_contract.py`

```python
"""PerformanceCapsule contract tests (backward compatibility)."""

import pytest
from transformation_portal.metrics.performance_capsule import PerformanceCapsule

class TestPerformanceCapsuleContract:
    """Test PerformanceCapsule contract stability."""

    def test_new_fields_have_defaults(self):
        """New tier and license_mode fields have backward-compatible defaults."""
        # Old code should work without providing new fields
        capsule = PerformanceCapsule(
            # ... existing required fields only
        )

        assert capsule.tier == "standard"  # Default
        assert capsule.license_mode == "commercial"  # Default

    def test_serialization_round_trip_with_new_fields(self):
        """Capsules with new fields serialize/deserialize correctly."""
        capsule = PerformanceCapsule(
            tier="apex_research",
            license_mode="research_only",
            # ... other fields
        )

        # Serialize
        data = capsule.to_dict()

        # Deserialize
        restored = PerformanceCapsule.from_dict(data)

        assert restored.tier == "apex_research"
        assert restored.license_mode == "research_only"

    def test_old_capsules_deserialize_with_defaults(self):
        """Old capsules (without new fields) deserialize with defaults."""
        old_data = {
            # ... existing fields only (no tier, no license_mode)
        }

        capsule = PerformanceCapsule.from_dict(old_data)

        assert capsule.tier == "standard"
        assert capsule.license_mode == "commercial"
```

---

## Fixture Strategy

### Synthetic Fixtures (Required)

**Location:** `tests/fixtures/apex_research_benchmark/`

**Structure:**
```
tests/fixtures/apex_research_benchmark/
├── architectural_exteriors/
│   ├── modern_glass_facade_4k.png       # 3840×2160, RGB
│   ├── historic_stone_detail_4k.png
│   └── mixed_materials_balcony_4k.png
├── architectural_interiors/
│   ├── luxury_kitchen_hdr_16bit.tiff    # 16-bit HDR
│   ├── bathroom_reflective_4k.png
│   └── living_room_depth_complex.png
└── ground_truth/
    ├── modern_glass_facade_depth.exr    # Float32 depth map
    ├── modern_glass_facade_mask.png     # Material segmentation
    └── metrics_reference.json           # Expected quality scores
```

**Generation Approach:**
1. **Render synthetic scenes** with Blender (or similar)
   - Export RGB + depth maps (ground truth)
   - Material-specific rendering passes for segmentation ground truth
2. **Validate realism** (human review)
3. **Check into repository** (small file sizes, <50MB total)

**Acceptance Criteria:**
- Representative of real-world architectural photography
- Diverse material types (glass, stone, metal, wood)
- Edge/boundary challenges (depth discontinuities)
- HDR/16-bit variants (Quality Firewall validation)

### Real-World Validation Set (Optional)

**Location:** `data/apex_research_validation/` (not in repo, local only)

**Purpose:** Validate benchmark results on real-world data (CI uses synthetic only)

**Structure:**
```
data/apex_research_validation/
├── real_world_exteriors/
├── real_world_interiors/
└── README.md  # Setup instructions for researchers
```

**Note:** Not required for CI, optional for deep validation.

---

## CI Integration

### Workflow: License Compliance

**File:** `.github/workflows/apex_research_compliance.yml`

```yaml
name: APEX Research License Compliance

on:
  pull_request:
    paths:
      - 'config/presets/apex_research*.yaml'
      - 'src/transformation_portal/depth/backends/**'
      - 'src/transformation_portal/compliance/**'

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
          for preset in config/presets/apex_research*.yaml; do
            if ! grep -q "license_restriction: research_only" "$preset"; then
              echo "ERROR: $preset missing 'license_restriction: research_only'"
              exit 1
            fi
          done

      - name: Unit Tests (License Enforcement)
        run: |
          pytest tests/compliance/test_apex_research_enforcement.py -v
          pytest tests/compliance/test_preset_validation.py -v
```

### Workflow: Quality Benchmark

**File:** `.github/workflows/apex_research_benchmark.yml`

```yaml
name: APEX Research Quality Benchmark

on:
  pull_request:
    paths:
      - 'config/presets/apex_research*.yaml'
      - 'src/transformation_portal/depth/backends/depth_pro.py'
      - 'src/transformation_portal/metrics/apex_research_quality.py'

jobs:
  benchmark:
    runs-on: ubuntu-latest  # Future: Apple Silicon runner for MPS
    steps:
      - uses: actions/checkout@v4

      - name: Download Synthetic Fixtures (if not in repo)
        run: |
          # Download fixtures from artifact storage (if needed)
          # For now, assume fixtures checked into tests/fixtures/

      - name: Run Benchmark (Mocked Models)
        run: |
          # Use mocked backends for fast CI validation
          pytest tests/benchmark/test_apex_research_benchmark.py \
            --benchmark-only \
            -v

      - name: Validate Quality Improvement
        run: |
          python -c "
          import json
          with open('artifacts/benchmark_results/comparison.json') as f:
              data = json.load(f)

          improvement_pct = data['improvement_percentage']
          if improvement_pct < 10:
              print(f'❌ Research tier only {improvement_pct:.1f}% better (need ≥10%)')
              exit(1)
          print(f'✅ Research tier {improvement_pct:.1f}% better than commercial')
          "

      - name: Upload Benchmark Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: apex-research-benchmark-results
          path: artifacts/benchmark_results/
```

---

## Coverage Targets

### Unit Test Coverage

| Module | Target | Rationale |
|--------|--------|-----------|
| `compliance/licensing.py` | ≥95% | Critical security control |
| `compliance/validate_apex_research.py` | ≥90% | Governance enforcement |
| `depth/backends/depth_pro.py` | ≥85% | Research backend |
| `metrics/apex_research_quality.py` | ≥90% | Quality validation |

### Integration Test Coverage

| Scenario | Coverage | Rationale |
|----------|----------|-----------|
| Preset loading with correct flags | ✅ Required | Happy path |
| Preset loading with missing flags | ✅ Required | Error path |
| Backend selection (mocked) | ✅ Required | Registry integration |
| Orchestration end-to-end (mocked) | ⚠️ Optional | Full pipeline |

### Benchmark Coverage

| Metric Category | Fixtures | Target |
|-----------------|----------|--------|
| Depth quality | 3-5 exteriors + 3-5 interiors | ≥10% improvement |
| Segmentation quality | Same | ≥10% improvement |
| PBR quality | Same | ≥5% improvement |
| Composite score | Same | ≥10% improvement |

---

## Performance Testing

### Benchmark Performance Targets

| Workflow | Hardware | Target Time | Max Time |
|----------|----------|-------------|----------|
| APEX Commercial | M4 MPS | ~2.5s (4K) | 4s |
| APEX Research | M4 MPS | ~3.5s (4K) | 6s |

**Rationale:** Research tier can be slower (higher quality > speed).

### Memory Profiling

| Component | Peak Memory | Notes |
|-----------|-------------|-------|
| Depth Pro | ~4GB | Checkpoint + inference buffers |
| SAM vit_h | ~6GB | Large model |
| Total Pipeline | ~12GB | With PBR processing |

**Validation:** Profile with `memory_profiler` on benchmark run.

---

## Acceptance Criteria (Per Phase)

### Phase 1: Core Infrastructure

- [ ] All license enforcement unit tests pass (≥10 tests)
- [ ] Preset validation script catches all error cases
- [ ] CI workflow blocks PRs with invalid presets
- [ ] Documentation explains license restrictions clearly

### Phase 2: SAM vit_h Integration

- [ ] SAM vit_h backend passes protocol compliance tests
- [ ] Checkpoint validation enforced (SHA256 mismatch blocks)
- [ ] Integration test with mocked SAM backend passes
- [ ] Segmentation quality ≥10% IoU improvement (benchmark)

### Phase 3: Quality Benchmarking

- [ ] Benchmark runs in CI without manual steps
- [ ] Research tier ≥10% improvement in ≥3/4 metrics
- [ ] Benchmark reproducible (deterministic results)
- [ ] Comparison report generated and uploaded

### Phase 4: Enhanced PBR

- [ ] `RESEARCH_PREMIUM` PBR preset ≥5% normal detail improvement
- [ ] Performance overhead acceptable (<1.5x slower)
- [ ] No regression in other quality metrics

---

## Continuous Validation

### Weekly Checks (Automated)

- [ ] Re-run benchmark suite on latest commit
- [ ] Validate license compliance on all presets
- [ ] Check for new research-licensed dependencies

### Quarterly Audits (Manual)

- [ ] Review benchmark results for quality stability
- [ ] Update ground truth fixtures if needed
- [ ] Validate documentation accuracy

---

## Rollback Criteria

Rollback APEX Research implementation if:
- License enforcement bypassed in testing
- Research tier fails to exceed commercial baseline (≥10% improvement)
- Performance unacceptable (>6s for 4K on M4 MPS)
- Backward compatibility broken (commercial APEX affected)

---

**Document History**
- **2026-02-10:** Initial testing & validation strategy created
