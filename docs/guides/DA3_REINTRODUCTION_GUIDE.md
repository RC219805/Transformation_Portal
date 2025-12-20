# DA3 Reintroduction Guide

**Status**: DA3 evaluation complete, integration DEFERRED
**Decision Date**: December 19-20, 2025
**Decision Record**: `docs/decisions/DA3_EVALUATION_DECISION.md`

## Why DA3 Submodule Was Removed

The DA3 official repository was initially added as a Git submodule but was removed in PR #573 for the following reasons:

1. **Strategic Decision**: DA3 evaluation completed with a DEFER recommendation
   - DA3: 13.0% pass rate on production quality gates
   - DA2: 84.8% pass rate (validated, production-ready)
   - Metric incompatibility: DA3 optimized for geometry (AbsRel, RMSE, δ₁), production requires edge fidelity (Edge F1, chamfer)

2. **CI/CD Hygiene**: Submodule blocked continuous integration
   - GitHub Actions cannot access the submodule URL
   - Setup & Change Detection consistently failed
   - Contradicts "defer" decision by making DA3 a hard dependency

3. **Repository Consistency**: Code structure must reflect strategic decisions
   - Keeping deferred experimental code as required dependency creates technical debt
   - DA3 artifacts preserved in `lux_depth_v3/` for future work
   - Documentation complete and accessible

## DA3 Integration Artifacts (Preserved)

All DA3 evaluation work remains in the repository:

```
lux_depth_v3/                          # Complete DA3 integration module (62 files)
├── README.md                          # Module overview
├── SECURITY.md                        # Security guidelines
├── config.py                          # DA3 configuration
├── inference.py                       # DA3 inference engine
├── license.py                         # License management
├── service.py                         # FastAPI service
├── tests/                             # DA3 test suite
└── ...

docs/decisions/DA3_EVALUATION_DECISION.md  # Comprehensive decision record
scripts/run_da3_vs_da2_ab_test.py          # A/B validation script
outputs/da3_gate_fix_test/                 # DA3 validation results
```

## When to Reintroduce DA3

DA3 should be reconsidered when **ALL 5** of the following conditions are met:

### 1. Ground-Truth Depth Available
- LiDAR scans, multi-view stereo, or annotated depth for architectural datasets
- Enables alignment of production metrics with global geometry benchmarks

### 2. Business Needs Metric Depth
- 3D reconstruction, pose estimation, spatial measurements required
- Matches published DA3 benchmark strengths (AbsRel, RMSE, δ₁)

### 3. Time Available
- 2-3 week fine-tuning + calibration cycle acceptable
- Resources allocated for domain adaptation

### 4. Validation Expanded
- Standard depth metrics (AbsRel, δ₁, RMSE) added to production gates
- Composite scoring system balances geometry and edge fidelity

### 5. Edge-Aware Fine-Tuning
- Domain adaptation resources available
- Yields measurable improvements in composite scorecard relative to DA2

**Not before**: All 5 conditions met

## Reintroduction Options

### Option A: Direct Integration (Recommended for Production)

If DA3 meets all 5 conditions and passes validation:

```bash
# Install DA3 dependencies
pip install -e ".[da3]"

# Run validation against expanded metrics
python scripts/run_da3_vs_da2_ab_test.py \
  --metrics standard,edge,composite \
  --baseline v1.0-validation-baseline

# Update production config
# config/production.yaml
depth_model: DA3-LARGE-1.1
quality_gates:
  - edge_fidelity
  - geometry_accuracy
  - composite_score
```

### Option B: Research Branch (Recommended for Experimentation)

For exploratory work before production decision:

```bash
# Create dedicated research branch
git checkout -b research/da3-fine-tuning

# Add DA3 as optional submodule
git submodule add https://github.com/DepthAnything/Depth-Anything-V3.git \
  depth_anything_3_official

# Gate CI to skip on missing submodule
# .github/workflows/build.yml
- name: Check DA3 availability
  run: |
    if [ ! -d "depth_anything_3_official" ]; then
      echo "DA3_AVAILABLE=false" >> $GITHUB_ENV
    fi

# Run experiments
python lux_depth_v3/experiments/domain_adaptation.py
```

### Option C: External Dependency (Recommended for CI)

Avoid submodule complexity by treating DA3 as a Python package:

```bash
# Add to requirements-da3.txt
git+https://github.com/DepthAnything/Depth-Anything-V3.git@main#egg=depth_anything_v3

# Install conditionally
pip install -r requirements-da3.txt || echo "DA3 optional"

# Use in code
try:
    from depth_anything_v3 import DepthAnythingV3
except ImportError:
    logger.warning("DA3 not available, using DA2")
```

## Decision Velocity Reminder

The DA3 deferral was based on **decision velocity principles**:

- ✅ Ship proven solution (DA2: 84.8% validated)
- ✅ Avoid speculative improvements with uncertain ROI
- ✅ Defer complex adaptations until clear business need
- ✅ Maintain engineering efficiency and focus

When reconsidering DA3, apply the same principles:

1. **Evidence-based**: Quantitative improvement over DA2 on production metrics
2. **Time-bounded**: Clear deadline and resource allocation
3. **Risk-managed**: Fallback to DA2 if fine-tuning doesn't converge
4. **Business-aligned**: Addresses specific production requirement

## References

- **Decision Record**: `docs/decisions/DA3_EVALUATION_DECISION.md`
- **Baseline Report**: `validation_v1_baseline_pack/BASELINE_REPORT.md`
- **DA3 Module**: `lux_depth_v3/README.md`
- **A/B Results**: `outputs/da3_gate_fix_test/`
- **PR #573**: Validation baseline freeze + DA3 evaluation

---

**Last Updated**: December 20, 2025
**Status**: Active guidance for future DA3 work
