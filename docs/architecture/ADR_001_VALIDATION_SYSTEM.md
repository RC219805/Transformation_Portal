# ADR 001: Validation System Architecture

**Status**: Accepted  
**Date**: 2025-12-08  
**Decision Authority**: Transformation Portal Architect  
**Impact**: System-wide (Lux Depth V2 + CI/CD + Operations)  

---

## Context

Lux Depth V2 is production-ready from an engineering perspective:
- 66/66 tests passing (100% pass rate)
- 80%+ code coverage
- Zero security vulnerabilities (CVE-2024-27763 mitigated)
- Comprehensive documentation

However, **commercial proof is missing**:
1. No repeatable benchmarks against industry baselines (Topaz, Adobe)
2. No quality regression prevention in CI/CD pipeline
3. No performance profiling to validate "fast" claims
4. No validation that material segmentation actually improves quality
5. No production observability for operational excellence

**The Gap**: Framework exists, but proof doesn't. "Breakthrough quality" is hypothesis, not verified fact.

---

## Decision

We will implement a **5-Priority Validation Stack** integrated into Lux Depth V2 and surrounding infrastructure:

### Priority 1: Repeatable Benchmark Framework
**Purpose**: Provide quantitative proof of superiority vs commercial baselines  
**Components**:
- Versioned dataset registry (20 images, 5 categories)
- Baseline runner with caching (Topaz, Adobe, Real-ESRGAN)
- Metric engine (LPIPS, NIMA, SSIM, PSNR)
- Category-based scorer with configurable weights
- HTML/MD report generator

### Priority 2: CI/CD Quality Gates
**Purpose**: Prevent quality regression during development  
**Components**:
- Golden image registry (5-10 curated failure modes)
- Regression detector with per-metric thresholds
- GitHub Actions workflow (blocks PRs on regression)
- Trend tracker (quality metrics over time)

### Priority 3: Performance Profiler
**Purpose**: Enable targeted optimization with measurable impact  
**Components**:
- Stage-by-stage profiler (I/O, segmentation, upscaling, post-processing)
- GPU utilization monitor (CUDA/MPS/CPU)
- Bottleneck analyzer (heuristics-based)
- Optimization advisor (actionable recommendations)

### Priority 4: Segmentation Validator
**Purpose**: Establish material segmentation as trusted system primitive  
**Components**:
- Consistency evaluator (10-run IoU stability test)
- Impact analyzer (ablation: segmentation ON vs OFF)
- Per-surface metrics (glass, wood, metal, fabric quality)
- Optional ground truth annotation tool

### Priority 5: Production Observability
**Purpose**: Enable operational excellence and traceability  
**Components**:
- Prometheus metrics endpoint (/metrics)
- Request tracer (full audit trail with config hashes)
- Error categorization (structured logging)
- Grafana dashboard (latency, throughput, errors, GPU)

---

## Architectural Principles

### 1. Modularity
**Principle**: Each priority is a standalone module with clear API contracts  
**Rationale**: Independent development, testability, optional adoption  
**Trade-off**: More integration complexity vs monolithic simplicity  

### 2. Versioning
**Principle**: Datasets, models, and baselines versioned with SHA256 checksums  
**Rationale**: Reproducibility is non-negotiable for commercial proof  
**Trade-off**: Storage costs (~50GB) vs traceability  

### 3. Category-Based Scoring
**Principle**: Configurable weights per category (interior: 30%, exterior: 25%, etc.)  
**Rationale**: Different use cases prioritize different quality aspects  
**Trade-off**: Configuration complexity vs one-size-fits-all scoring  

### 4. Golden Images as Failure Mode Coverage
**Principle**: 5-10 curated images exposing known weaknesses (edges, textures, halos)  
**Rationale**: Targeted regression prevention more effective than exhaustive testing  
**Trade-off**: Manual curation effort vs automated random selection  

### 5. Profiler as Optional Feature
**Principle**: Enable profiling via `--profile` flag, not always-on  
**Rationale**: GPU sync overhead (<5%) acceptable in profiling mode, not production  
**Trade-off**: Separate codepaths vs unified instrumentation  

### 6. Consistency as Proxy for Ground Truth
**Principle**: Use 10-run IoU stability instead of perfect annotations  
**Rationale**: Ground truth expensive to generate; consistency signals quality  
**Trade-off**: Indirect measure vs direct validation  

### 7. Prometheus-Compatible Observability
**Principle**: Standard Prometheus client + Grafana dashboards  
**Rationale**: Broad ecosystem compatibility, ops team familiarity  
**Trade-off**: External dependencies vs custom solution  

---

## Consequences

### Positive

1. **Commercial Proof**: Publishable benchmarks with reproducible methodology
2. **Quality Leadership**: Automated prevention of quality regression
3. **Performance Transparency**: Data-driven optimization roadmap
4. **Segmentation Trust**: Validated as reliable system primitive, not experimental
5. **Operational Maturity**: 99.9% uptime target with full traceability

### Negative

1. **Complexity**: 5 new modules (benchmark, golden, profiling, segmentation/validation, observability)
2. **Storage**: ~50GB for datasets + baselines (Git LFS required)
3. **Maintenance**: 0.5x engineer ongoing (weekly reviews, threshold tuning)
4. **Dependencies**: Baseline tools (Topaz license or alternatives), GPU runners

### Risks

| Risk | Mitigation |
|------|-----------|
| Baseline tools unavailable | Fallback to open-source alternatives (Real-ESRGAN) |
| GPU runner access limited | Self-hosted or scheduled (weekly vs per-PR) |
| Segmentation consistency low | Root cause analysis, disable until fixed |
| Metric disagreement | Weighted scoring, manual review edge cases |
| False positives in gates | Configurable thresholds + manual override |

---

## Alternatives Considered

### Alternative 1: Manual Validation Only
**Approach**: Continue with human review for quality validation  
**Rejected**: Not scalable, not reproducible, no commercial proof  

### Alternative 2: External Validation Service
**Approach**: Use third-party benchmarking (e.g., PapersWithCode)  
**Rejected**: Slow iteration, limited control over metrics/datasets  

### Alternative 3: Exhaustive Testing
**Approach**: Test every image in every configuration  
**Rejected**: Infeasible (combinatorial explosion), slow CI/CD  

### Alternative 4: Metrics-Only (No Baselines)
**Approach**: Track internal metrics without external comparison  
**Rejected**: No commercial proof of superiority  

---

## Implementation Plan

### Timeline: 10 Weeks (Part-Time Senior Engineer)

**Week 1-2**: Priority 1 (Benchmark Framework)  
**Week 3-4**: Priority 2 (CI/CD Quality Gates)  
**Week 5-6**: Priority 3 (Performance Profiler)  
**Week 7-8**: Priority 4 (Segmentation Validator)  
**Week 9-10**: Priority 5 (Production Observability)  

### Critical Path
Dataset Acquisition → Baseline Generation → CI/CD Integration

### Success Criteria

**Phase 1-2** (Week 4):
- ✅ Benchmark: Lux Depth V2 wins ≥60% vs Topaz
- ✅ LPIPS: 10%+ improvement on glossy surfaces
- ✅ NIMA: Matches/exceeds Adobe on ≥70%
- ✅ CI/CD: Zero regressions for 30 days

**Phase 3** (Week 6):
- ✅ Latency: 30%+ reduction via GPU optimization
- ✅ Throughput: <30s for 4K @ 4x (A100)
- ✅ Memory: <8GB VRAM

**Phase 4** (Week 8):
- ✅ Consistency: IoU >0.95 across 10 runs
- ✅ Impact: LPIPS improves ≥5% with segmentation ON
- ✅ Surface: Glass +10%, wood +7%

**Phase 5** (Week 10):
- ✅ Uptime: 99.9% over 30 days
- ✅ Latency: p95 <45s
- ✅ Errors: <1% rate
- ✅ Traceability: 100% requests logged

---

## Integration Strategy

### Additive, Not Disruptive
- Existing 66 tests unchanged
- No breaking changes to Lux Depth V2 core API
- Optional features enabled via flags/config

### Rollback Safety
- Per-phase rollback procedures documented
- 3-tier response (Immediate <1hr, Root Cause <24hr, Selective Re-enable <1wk)
- Validation system can be completely disabled without affecting core functionality

### Backward Compatibility
- All existing functionality preserved
- Validation system is purely additive
- No migration required for existing users

---

## References

- **Primary**: [`docs/architecture/VALIDATION_ARCHITECTURE.md`](VALIDATION_ARCHITECTURE.md) (37KB, comprehensive design)
- **Implementation**: [`docs/architecture/VALIDATION_IMPLEMENTATION_PLAN.md`](VALIDATION_IMPLEMENTATION_PLAN.md) (35KB, API contracts)
- **Integration**: [`docs/architecture/VALIDATION_INTEGRATION_CHECKLIST.md`](VALIDATION_INTEGRATION_CHECKLIST.md) (22KB, step-by-step guide)
- **Summary**: [`docs/VALIDATION_SYSTEM_EXECUTIVE_SUMMARY.md`](../VALIDATION_SYSTEM_EXECUTIVE_SUMMARY.md) (13KB, stakeholder overview)
- **Index**: [`docs/VALIDATION_SYSTEM_INDEX.md`](../VALIDATION_SYSTEM_INDEX.md) (13KB, navigation guide)

---

## Approval

**Decision Authority**: Transformation Portal Architect  
**Date**: 2025-12-08  
**Status**: ✅ Accepted  

**Implementation Authorization**: Pending stakeholder sign-off  
**Week 1 Kickoff**: Conditional on:
1. Architect review complete
2. Dataset acquisition plan approved
3. GPU compute access confirmed

---

## Changelog

### 2025-12-08: Initial Decision
- Defined 5-priority validation stack
- Established architectural principles
- Documented success criteria and rollback procedures
- Approved for implementation planning

---

**ADR Status**: Active  
**Next Review**: After Phase 2 completion (Week 4)  
**Supersedes**: None  
**Superseded By**: None
