# ADR-004: Documentation Quality Standards for ML Pipeline Documentation

**Status**: Proposed
**Date**: 2026-01-05
**Decision Makers**: Transformation Portal Architect
**Context**: PR #655 Depth Estimation Documentation Review
**Related**: [PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md](../PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md)

---

## Context

The `docs/DEPTH_ESTIMATION_ANALYSIS.md` documentation (PR #655) revealed systematic quality issues that threaten long-term maintainability and operational trustworthiness:

1. **Unprovable Performance Claims**: "1,348 tests passing", "127-400 images/hour" lack citations
2. **License Terminology Conflicts**: Documentation uses "COMMERCIAL USE" while code uses `is_commercial` based on Apache 2.0
3. **Missing Output Format Contracts**: No clear taxonomy for preview (8-bit), contract (uint16), research (float32) artifacts
4. **CLI Command Misalignment**: Documentation references commands that conflict with actual Typer registrations
5. **Validation Metrics as Absolutes**: Context-sensitive thresholds (edge alignment, seam energy) presented as universal
6. **Missing Failure Mode Documentation**: No explicit warnings about texture embossing, sky instability, reflection mirroring

These issues create a documentation pattern that will:
- **Rot over time** as claims become stale
- **Mislead users** into pipeline-breaking usage (8-bit preview → uint16 contract violation)
- **Create legal risk** (non-commercial models in commercial deployments)
- **Block runtime usage** (duplicate CLI commands)

---

## Decision

We establish **Documentation Quality Standards** for all ML pipeline documentation in the Transformation Portal repository:

### 1. Verifiable Claims Standard

**Rule**: All quantitative claims MUST link to verifiable evidence or be removed.

**Allowed**:
```markdown
✅ Production-validated (See [CI Test Results](../../.github/workflows/ci.yml))
✅ Security-hardened ([CVE-2024-27763 mitigated](../../SECURITY.md#cve-2024-27763))
✅ Comprehensive test coverage (44 test suites covering inference, tiling, edge snapping, I/O)
```

**Prohibited**:
```markdown
❌ Production-validated (1,348 tests passing)  # No link, ambiguous scope
❌ 127-400 images/hour throughput              # No methodology, 3x variance
❌ Security-hardened (CVE-2024-27763 mitigated) # No link to SECURITY.md
```

**Rationale**: Claims without citations become stale and undermine credibility. Users cannot verify assertions, documentation loses trust.

---

### 2. License Terminology Alignment Standard

**Rule**: License documentation MUST use terminology that aligns with codebase `is_commercial` / `license` properties.

**Code Contract** (`lux_depth_v3/config.py`):
```python
@property
def is_commercial(self) -> bool:
    """Check if model allows commercial use."""
    return self.license == ModelLicense.APACHE_2_0
```

**Documentation Standard**:
```markdown
**Non-Commercial Models (CC-BY-NC-4.0)** - Research/Personal Use Only:
- DA3NESTED-GIANT-LARGE-1.1 (1.40B params) - RECOMMENDED for research

**Commercial-Friendly Models (Apache 2.0)** - Production Deployments:
- DA3METRIC-LARGE (0.35B params) - Commercial use allowed
```

**Prohibited**:
```markdown
❌ DA3METRIC-LARGE (Apache 2.0) - COMMERCIAL USE
   # Ambiguous: Suggests other models are also commercial
   # Conflicts with is_commercial property (only Apache 2.0 returns True)
```

**Rationale**: License terminology confusion creates legal compliance risk. Documentation must align with code contracts.

---

### 3. Output Artifact Contract Standard

**Rule**: All pipeline documentation MUST include an artifact taxonomy that separates preview, contract, and research outputs.

**Required Table**:

| Output | Format | Precision | Use Case | Pipeline-Safe? |
|--------|--------|-----------|----------|----------------|
| `*_depth_preview.png` | 8-bit RGBA | 256 levels | Visualization only | ❌ **NO** |
| `*_depth.tif` | 16-bit Grayscale | 65,536 levels | V2 enhancement pipeline | ✅ **YES** |
| `*_depth.npz` | float32 | Full precision | Metric analysis | ⚠️ Research only |

**Required Warning**:
```markdown
⚠️ **DO NOT** use preview PNG files as pipeline inputs:
- 8-bit quantization loses 99.6% of depth precision (65,536 → 256 levels)
- Colormap application is irreversible
- RGBA channels contain visualization metadata, not raw depth
```

**Rationale**: Users generate preview artifacts and pipe them into production → silent pipeline poisoning. Clear contracts prevent this failure mode.

---

### 4. CLI Alignment Standard

**Rule**: Documentation MUST reference actual CLI commands as registered in Typer/Click.

**Verification Process**:
1. **Before documenting**: Verify command exists in CLI via `grep "@app.command()" <cli_file>`
2. **Detect conflicts**: Check for duplicate command names (Typer will clobber)
3. **Test invocation**: Run `<cli-name> --help` to confirm command is callable
4. **Update on rename**: If CLI commands are renamed, update all documentation references

**Example**:
```markdown
# WRONG (Duplicate command - Typer will clobber one):
$ lux-depth-v3 benchmark --model large-v1.1
  # ^ Line 440 and 620 both define `benchmark` - runtime error

# CORRECT (After renaming):
$ lux-depth-v3 benchmark-inference --model large-v1.1
$ lux-depth-v3 benchmark-quality --datasets eth3d
```

**Rationale**: Documentation referencing non-existent commands blocks user adoption and creates support burden.

---

### 5. Benchmark Methodology Standard

**Rule**: Performance numbers MUST include methodology or be removed.

**Required Context**:
- **Hardware**: CPU/GPU model, memory
- **Warmup**: Number of iterations excluded from timing
- **Measurement**: Number of iterations, statistic (mean/median), percentiles
- **Precision**: FP16/FP32/Mixed
- **Batch Size**: Single vs batched
- **Resolution**: Input image dimensions
- **Script**: Link to reproducible benchmark script

**Example**:
```markdown
**Methodology**: See [bench/README.md](../../bench/README.md)

**Configuration**:
- Hardware: M4 Max (16 GPU cores, 128GB unified memory)
- Precision: FP32
- Warmup: 5 iterations
- Measurement: 100 iterations, median time
- Batch size: 1
- Input size: 518×518

| Model | Device | Median | 95th %ile | Throughput* |
|-------|--------|--------|-----------|-------------|
| DA2-Large | MPS | 65ms | 72ms | ~55/min |

*Throughput = 60,000ms / median_time_ms (single-image batch)
```

**Rationale**: Performance numbers without context are meaningless and create false expectations.

---

### 6. Context-Sensitive Validation Metrics Standard

**Rule**: Validation metrics MUST include context-sensitive thresholds or calibration guidance.

**Prohibited**:
```python
❌ assert edge_alignment > 0.5, "Edge alignment too low"
   # Universal threshold fails on vegetation/aerial scenes
```

**Required**:
```python
✅ # Context-sensitive thresholds
if scene_type == "interior":
    assert edge_alignment > 0.6, "Edge alignment below interior baseline"
elif scene_type == "exterior":
    assert edge_alignment > 0.4, "Edge alignment below exterior baseline"
else:
    # Calibration required
    logger.warning(f"Edge alignment {edge_alignment:.3f} - validate against reference dataset")
```

**Documentation Template**:
```markdown
### Edge Alignment Score

**Context-Sensitive Thresholds**:
- Structured interiors: > 0.6 (excellent)
- Exteriors with vegetation: > 0.4 (acceptable)
- Aerial/texture-heavy: > 0.5 (good)

**Calibration**: Validate on reference dataset before setting project thresholds.
```

**Rationale**: Absolute thresholds cause false positives/negatives. Context-aware validation improves operational reliability.

---

### 7. Failure Mode Documentation Standard

**Rule**: All ML pipeline documentation MUST include a "Known Limitations" or "Failure Modes" section.

**Required Content**:
1. **Fundamental Limitations**: Monocular depth constraints, model architecture limits
2. **Configuration Pitfalls**: Double-application of transformations, insufficient context
3. **Input Assumptions**: Scene structure requirements, material constraints
4. **Mitigation Strategies**: When available, how to work around or detect failures

**Template**:
```markdown
## Known Failure Modes

| Failure Mode | Symptom | Mitigation |
|--------------|---------|------------|
| **Texture Embossing** | Flat walls show false depth variation | Use guided filter |
| **Sky Instability** | Sky shows gradient/noise | Mask sky, clamp to infinity |
| **Reflection Mirroring** | Water/glass shows mirrored scene | Detect reflective materials |
| **Double Edge-Snapping** | Over-sharpened edges, halos | Don't enable both flags |

### Configuration Pitfalls

⚠️ **Double Edge-Snapping**:
[Code example showing wrong vs correct configuration]
```

**Rationale**: Hiding failure modes creates operational risk. Explicit documentation enables informed decision-making.

---

## Consequences

### Positive

1. **Credibility**: Claims are verifiable → users can trust documentation
2. **Legal Safety**: License terminology aligns with code → no compliance risk
3. **Pipeline Safety**: Output contracts prevent format mismatches → no silent failures
4. **Runtime Reliability**: CLI alignment ensures commands work as documented
5. **Operational Transparency**: Failure mode documentation enables informed usage

### Negative

1. **Documentation Effort**: Standards increase initial writing time (~20-30% overhead)
2. **Maintenance Burden**: Links to CI, benchmarks, code must be kept in sync
3. **Complexity**: Context-sensitive thresholds require more sophisticated validation

### Neutral

1. **Refactoring**: Existing docs must be updated to meet standards (one-time cost)
2. **Review Process**: Architectural reviews required for ML pipeline docs

---

## Compliance

### Enforcement

1. **Pre-commit Hook**: Check for unprovable claims patterns (e.g., `\d+ tests passing` without link)
2. **PR Template**: ML pipeline docs must include checklist:
   - [ ] All performance claims link to benchmarks or methodology
   - [ ] License terminology aligns with code properties
   - [ ] Output artifact contract table included
   - [ ] CLI commands verified against actual registration
   - [ ] "Known Limitations" section present

3. **Architectural Review**: Required for all new ML pipeline documentation

### Migration Plan

**Phase 1** (Immediate):
- Update `docs/DEPTH_ESTIMATION_ANALYSIS.md` to meet standards (PR #655 remediation)

**Phase 2** (Q1 2026):
- Audit existing ML docs: `lux_depth_v2/README.md`, `lux_depth_v3/README.md`, `high_fidelity_depth/README.md`
- Apply standards to high-priority docs (user-facing guides)

**Phase 3** (Q2 2026):
- Migrate all remaining ML docs
- Add pre-commit hook for automatic enforcement

---

## Alternatives Considered

### Alternative 1: "Lightweight Documentation" (Minimal Standards)

**Approach**: Only require factual accuracy, no methodology or contracts.

**Rejected Because**:
- Still allows unprovable claims (credibility risk)
- No output contract enforcement (pipeline safety risk)
- License confusion persists (legal risk)

### Alternative 2: "Code Comments Only" (Eliminate Prose Docs)

**Approach**: Document everything in docstrings, no separate guides.

**Rejected Because**:
- Poor discoverability (users don't read source code)
- No cross-module guidance (pipeline integration patterns)
- Benchmark data doesn't fit in docstrings

### Alternative 3: "Wiki-Style Documentation" (User-Editable)

**Approach**: Move all docs to GitHub Wiki for community editing.

**Rejected Because**:
- No version control alignment (docs drift from code)
- No review process (quality degradation)
- No CI enforcement (broken links, stale claims)

---

## References

- [PR #655 Architectural Review](../PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md)
- [PR #655 Executive Summary](../PR_655_EXECUTIVE_SUMMARY.md)
- [SECURITY.md](../../SECURITY.md) - CVE-2024-27763 mitigation details
- [lux_depth_v3/config.py](../../lux_depth_v3/config.py) - License property contracts
- [Google Technical Writing Guide](https://developers.google.com/tech-writing)
- [Divio Documentation System](https://documentation.divio.com/)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-05 | Establish 7-part documentation standard | PR #655 revealed systematic quality issues |
| 2026-01-05 | Require output artifact contract tables | Prevent 8-bit preview → uint16 contract failures |
| 2026-01-05 | Mandate failure mode documentation | Hide-the-problems approach creates operational risk |

---

**Author**: Transformation Portal Architect
**Status**: Proposed (Pending Approval)
**Next Review**: After PR #655 remediation completion
