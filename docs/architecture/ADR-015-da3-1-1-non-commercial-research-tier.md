# ADR-015: Depth Anything V3.1 (DA3 1.1) Non-Commercial Research Tier

**Status:** Adopted
**Date:** 2026-02-01
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** PR #774, Security Policy (SECURITY.md)

> **2026-05-16 renumbering note:** Originally filed as `ADR-0015` (4-digit prefix). Renumbered to `ADR-015` to match the 3-digit convention used by every other ADR in this series. No content changes.

---

## Decision

**Transformation Portal will support Depth Anything V3.1 (DA3 1.1) exclusively for non-commercial research purposes, with explicit runtime and CI enforcement.**

- **Commercial Production:** DA3 V2 (commercially licensed) remains the sole approved depth model for production and commercial use cases.
- **Research/Academic:** DA3 1.1 (CC BY-NC 4.0) is available for non-commercial research, academic benchmarking, and non-profit projects only.
- **Enforcement:** Compliance gates will prevent accidental commercial use through preset markers, environment variables, and CI warnings.

---

## Context

### Why DA3 1.1?

Depth Anything V3.1 (M4 / 1.1 variant) offers compelling characteristics for research:
- **Apple Silicon Optimization:** Efficient M4 inference on MacBook Pro/Air
- **Quality-Performance Trade-off:** Competitive depth accuracy for research scenarios (~420ms for 4K on M4)
- **Model Size:** Smaller ONNX footprint (3.2 GB vs. 6.8 GB for larger variants)

### Why NOT DA3 1.1 for Production?

1. **Licensing Restriction:** CC BY-NC 4.0 (non-commercial use only)
   - Prohibits commercial products, services, and revenue-generating use
   - Creates legal liability in enterprise/commercial pipelines
2. **v2.0.0 Golden Path:** Transformation Portal's v2.0.0 stable release guarantees commercial-ready contracts
   - Existing commercial pipelines depend on DA3 V2 or equivalent commercial-licensed models
   - Introducing non-commercial models creates compatibility risk for commercial users

### Integration Requirements

- **Research-Only Branch:** DA3 1.1 should be discoverable and usable for research workflows
- **Production Safety:** Commercial pipelines must never accidentally load DA3 1.1
- **Governance Clarity:** Documentation must unambiguously state licensing restrictions

---

## Constraints

1. **Backward Compatibility:** No changes to existing commercial v2.0.0 contracts or DA3 V2 behavior
2. **License Compliance:** Explicit tracking of non-commercial models in configuration and runtime
3. **Supply Chain:** CI must validate that commercial builds do not inadvertently include non-commercial models
4. **Audit Trail:** Preset configurations and code must clearly mark non-commercial usage

---

## Proposed Design

### 1. Configuration Governance

**Preset Marker Pattern:**
```yaml
# Example: research_da31_preset.yaml
name: depth-anything-v3.1-research
description: "DA3 1.1 for research/academic use only (CC BY-NC 4.0)"
tier: research
license_restriction: non_commercial
model:
  variant: depth-anything-v3-metric-large-1.1
  source: huggingface
  hf_id: "depth-anything/DA3-Large-1.1"
```

**EnhanceConfig Field (already exists):**
```python
@dataclass
class EnhanceConfig:
    non_commercial_ok: bool = False  # User must explicitly opt-in
```

### 2. Runtime Compliance Gate

**Location:** `src/transformation_portal/compliance/licensing.py`

```python
def require_non_commercial(reason: str = ""):
    """Decorator ensuring non-commercial usage is explicitly authorized.

    Args:
        reason: Human-readable explanation of licensing restriction

    Raises:
        LicenseRestrictionError: If non_commercial_ok is False
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(config: EnhanceConfig, *args, **kwargs):
            if not config.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"Function '{func.__name__}' requires non_commercial_ok=True.\n"
                    f"Reason: {reason}\n"
                    f"This model uses CC BY-NC 4.0 (non-commercial research only)."
                )
            return func(config, *args, **kwargs)
        return wrapper
    return decorator
```

### 3. Preset Discovery & Validation

**Validation Rules:**
- Presets marked `tier: research` or `license_restriction: non_commercial` require explicit `non_commercial_ok=True` in EnhanceConfig
- Loading such presets without the flag raises `LicenseRestrictionError`
- CLI shows warning: `⚠️  Research tier (non-commercial use only)`

**CLI Integration:**
```bash
lux-depth-v2 --list-presets
# Output includes tier annotation:
#   interior_luxury          [STABLE]
#   depth-anything-v3.1-research   [RESEARCH] ⚠️  Non-commercial use only
```

### 4. CI/Pipeline Enforcement

**Pre-commit Hook (optional, for development):**
```bash
# Warn if non-commercial presets appear in production config
grep -l "license_restriction: non_commercial" config/presets/*.yaml | \
  xargs -I {} git add -p {}
```

**CI Gate (GitHub Actions):**
```yaml
- name: Validate commercial compliance
  run: |
    python -m transformation_portal.compliance.validate_licenses \
      --check-presets config/presets/ \
      --check-code src/
    # Warns if non-commercial models found in production builds
```

### 5. Documentation & Disclaimers

**README Section:**
```markdown
## Depth Models: Commercial vs. Research

### Production (Commercial)
- **Depth Anything V3 (V2 commercial variant):** Fully supported, production-ready
- **Recommended for:** Commercial applications, products, revenue-generating services

### Research & Non-Commercial
- **Depth Anything V3.1 (DA3 1.1, CC BY-NC 4.0):** Available for research/academic use only
- **Enabled by:** Setting `non_commercial_ok=True` in EnhanceConfig
- **Restricted from:** Commercial, proprietary, or revenue-generating use
```

**Preset File Header (YAML):**
```yaml
# ⚠️  LICENSE: CC BY-NC 4.0 (Non-Commercial Use Only)
# This preset uses Depth Anything V3.1 which is licensed for non-commercial research only.
# For commercial applications, use the 'depth-anything-v3-commercial' preset instead.
```

---

## Alternatives Considered

### Alternative 1: Ban DA3 1.1 Entirely
- **Rationale:** Avoids licensing complexity
- **Consequence:** Blocks valuable research use cases and Apple Silicon optimization studies
- **Rejected:** Reduces flexibility for academic users and research-tier workflows

### Alternative 2: Unified Model with Conditional Licensing
- **Rationale:** Single codebase handles both commercial and non-commercial models
- **Consequence:** Requires complex runtime licensing checks; harder to audit and maintain
- **Rejected:** Less clear responsibility boundaries; higher maintenance cost

### Alternative 3: Separate Repository for Research
- **Rationale:** Complete isolation of non-commercial code
- **Consequence:** Duplicates infrastructure; harder for researchers to access; breaks CI unity
- **Rejected:** Fragments ecosystem; doesn't serve the goal of accessible research tools

**Chosen:** Option 1 (this ADR) — **Unified codebase with strict governance markers and runtime enforcement**

---

## Consequences

### Benefits
1. **Clarity:** Unambiguous separation of commercial and research tiers
2. **Compliance:** Explicit markers prevent accidental non-commercial usage in production
3. **Research Enablement:** Researchers can access DA3 1.1 with clear opt-in
4. **Maintainability:** Single codebase; governance enforced mechanically, not by process

### Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| **Licensing Liability:** Commercial user accidentally enables DA3 1.1 | `non_commercial_ok` defaults to False; CI warnings on non-commercial presets |
| **Audit Gaps:** Non-commercial usage leaks into production builds | Preset markers + compliance validation test in CI |
| **Documentation Drift:** README/docs become unclear | Automated docstring extraction; ADR as source of truth |

### Maintenance Burden
- **Initial:** Create compliance module, update config validation, add tests (~4h)
- **Ongoing:** Validate non-commercial presets on new DA3 releases; update documentation
- **Monitoring:** CI must flag any unmarked DA3 1.1 variants; manual audit quarterly

---

## Implementation Requirements

### Code Changes
1. Create `src/transformation_portal/compliance/licensing.py` with decorator and validator
2. Update `src/transformation_portal/lux_depth_v3/config.py` to document `non_commercial_ok` field
3. Add test `tests/compliance/test_licensing_gates.py` to verify compliance enforcement
4. Create/mark DA3 1.1 presets in `config/presets/` with `license_restriction: non_commercial`
5. Update `README.md` with tier clarification (section: "Depth Models: Commercial vs. Research")

### CI Integration
1. Add GitHub Actions check to validate preset licenses (`validate_licenses` script)
2. Pre-commit hook (optional) to warn on non-commercial preset changes
3. Ensure compliance tests run on every PR

### Documentation
1. ADR-015 (this document) as source of truth
2. README section: licensing tiers and use case guidance
3. Preset file headers: CC BY-NC 4.0 disclaimer
4. API docs: `EnhanceConfig.non_commercial_ok` field documentation

---

## Testing Strategy

### Unit Tests (tests/compliance/test_licensing_gates.py)
1. Verify `require_non_commercial` decorator blocks access without flag
2. Verify `non_commercial_ok=True` allows access
3. Verify preset loading respects `license_restriction` field
4. Verify CLI warns on research-tier presets

### Integration Tests
1. Full pipeline with DA3 1.1 and `non_commercial_ok=True` → succeeds
2. Full pipeline with DA3 1.1 and `non_commercial_ok=False` → raises LicenseRestrictionError
3. Commercial DA3 V2 pipeline always succeeds (backward compatibility)

### CI Gates
1. `pytest tests/compliance/test_licensing_gates.py` (required, fast, no models)
2. `python -m transformation_portal.compliance.validate_licenses --check-presets` (required)

---

## Migration Path

### For Existing Commercial Users
- **No changes required.** v2.0.0 Golden Path behavior unchanged.
- DA3 V2 remains default and only option in production builds.
- `non_commercial_ok` defaults to False (opt-in only).

### For Research Users
1. Install with research extras (if applicable): `pip install -e ".[research]"`
2. Create EnhanceConfig with `non_commercial_ok=True`
3. Use DA3 1.1 presets (marked with `tier: research`)
4. Acknowledge licensing: review CC BY-NC 4.0 terms before use

### For CI/Deployment
- Existing workflows unchanged
- Add optional validation step: `python -m transformation_portal.compliance.validate_licenses`
- No model downloads or breaking changes

---

## Governance & Audit

### Approval Chain
- ✅ Architect: Approved as ADR-015 (originally ADR-0015; see renumbering note above) (authority: Transformation Portal Architect)
- Specialist: Implementation per ADR requirements
- CI: Automated enforcement via preset validation + license compliance tests

### Review Triggers
- Any new DA3 1.1 variant: validate license, update preset, add test
- Any new non-commercial model: follow same ADR-015 pattern
- Quarterly audit: verify no unmarked non-commercial models in production config

### Escalation
If a PR introduces non-commercial models without `license_restriction` marker:
1. CI validation fails (non-negotiable)
2. Escalate to Architect for ADR update if needed
3. Merge only with explicit approval and test coverage

---

## References

### External
- [Depth Anything V3 Repository](https://github.com/ByteDance/depth-anything-v3)
- [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)
- [Transformation Portal Security Policy](SECURITY.md)

### Internal
- [Repository Organization](../governance/REPO_ORGANIZATION.md)
- [v2.0.0 Implementation Plan](V2_0_0_IMPLEMENTATION_PLAN.md)
- [Quality Firewall Quick Reference](../../QUALITY_FIREWALL_QUICK_REF.md)

---

## Appendix A: M4 / Apple Silicon Optimization Case Study

(Optional documentation for researchers)

### Performance Metrics (Reference Only)
- **Model:** Depth Anything V3.1 Large (M4 variant)
- **Hardware:** Apple MacBook Pro 14" M4 Max
- **Input:** 4K RGB (3840×2160)
- **Throughput:** ~420ms per frame (non-production reference)
- **Memory:** 6.8 GB VRAM

**Note:** Metrics are for research interest only and do not constitute a production recommendation. Commercial performance guarantees require DA3 V2 or equivalent commercially-licensed models.

---

## Appendix B: Configuration Examples

### Research Preset (Non-Commercial)
```yaml
# config/presets/research_da31_m4.yaml
name: depth-anything-v3.1-m4-research
description: "DA3 1.1 M4-optimized for research (CC BY-NC 4.0)"
tier: research
license_restriction: non_commercial
model:
  variant: depth-anything-v3-metric-large-1.1
  source: huggingface
  hf_id: "depth-anything/DA3-Large-1.1"
device:
  type: cpu  # or mps for Apple Silicon
  dtype: float16
postprocessing:
  apply_metric_scaling: true
  apply_bilateral_filter: true
```

### Production Preset (Commercial)
```yaml
# config/presets/commercial_da3_v2.yaml
name: depth-anything-v3-commercial
description: "DA3 V2 for commercial/production use"
tier: stable
model:
  variant: depth-anything-v3-metric-large
  source: huggingface
  hf_id: "depth-anything/Depth-Anything-V3-Metric-Large-hf"
```

### EnhanceConfig Examples

**Research Usage (Explicit Opt-In):**
```python
config = EnhanceConfig(
    preset=Preset.RESEARCH_DA31,
    non_commercial_ok=True,  # Explicitly enable
    depth_device="mps",      # Apple Silicon
)
```

**Commercial Usage (Default):**
```python
config = EnhanceConfig(
    preset=Preset.COMMERCIAL_DA3_V2,
    non_commercial_ok=False,  # Default, safe
    depth_device="cuda",
)
```

---

## Appendix C: Compliance Validation Script

Location: `src/transformation_portal/compliance/validate_licenses.py`

```python
#!/usr/bin/env python3
"""Validate license compliance of presets and models."""
import argparse
import yaml
from pathlib import Path

def validate_preset_file(preset_path: Path) -> bool:
    """Validate that non-commercial presets have required markers."""
    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    # Check if model uses known non-commercial licenses
    model_id = preset.get('model', {}).get('hf_id', '')
    if 'DA3-Large-1.1' in model_id or 'DA3' in model_id:
        # Require explicit license_restriction marker
        if preset.get('license_restriction') != 'non_commercial':
            print(f"⚠️  {preset_path}: DA3 1.1 preset missing license_restriction marker")
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-presets', type=Path)
    args = parser.parse_args()

    all_valid = True
    for preset_file in args.check_presets.glob('*.yaml'):
        if not validate_preset_file(preset_file):
            all_valid = False

    exit(0 if all_valid else 1)
```

---

**Document History**
- **2026-02-01:** Initial ADR-0015 created; Option A implementation
- **2026-05-16:** Renumbered from ADR-0015 → ADR-015 for prefix-format consistency (no content changes)
