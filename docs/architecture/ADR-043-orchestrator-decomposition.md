# ADR-043: Orchestrator Decomposition Strategy

**Status:** PROPOSED  
**Date:** 2026-03-20  
**Decision Makers:** Architect  
**Replaces:** None  

---

## Context

The `EnhanceOrchestrator` class in `src/transformation_portal/lux_depth_v3/orchestrator.py` has grown to **6,108 lines of code**, with the single class spanning **5,491 LOC**. This violates fundamental software design principles and creates multiple operational problems:

### Current Problems

1. **Single Responsibility Violation:** One class handles configuration, pipeline execution, artifact management, validation, and result generation.

2. **Testing Friction:** Unit testing individual behaviors requires mocking the entire orchestrator context.

3. **Onboarding Barrier:** New contributors cannot understand the orchestrator without reading 5,000+ lines.

4. **Merge Conflicts:** High-traffic file with multiple developers working on different features.

5. **Code Review Burden:** Reviews of orchestrator changes require understanding full context.

### Metrics

| Metric | Current | Acceptable |
|--------|---------|------------|
| File LOC | 6,108 | <2,000 |
| Class LOC | 5,491 | <1,500 |
| Methods per class | 47 | <20 |
| Cyclomatic complexity | High | Moderate |

---

## Decision

Decompose `EnhanceOrchestrator` into **five focused modules** with clear responsibility boundaries:

### Target Architecture

```
src/transformation_portal/lux_depth_v3/
├── orchestrator.py          # Facade (~500 LOC)
│   └── EnhanceOrchestrator   # Thin wrapper, delegates to modules
│
├── config_resolver.py       # Configuration (~1,000 LOC)
│   └── ConfigResolver        # Preset loading, validation, merging
│
├── pipeline_coordinator.py  # Execution (~1,500 LOC)
│   └── PipelineCoordinator   # Stage sequencing, backend selection
│
├── artifact_manager.py      # Artifacts (~1,500 LOC)
│   └── ArtifactManager       # Indexing, merkle, output management
│
├── execution_engine.py      # Runtime (~2,000 LOC)
│   └── ExecutionEngine       # Image processing, depth, v2, PBR
│
└── validators/
    └── run_card_validator.py # Validation (~500 LOC)
```

### Responsibility Matrix

| Module | Responsibilities | Does NOT Handle |
|--------|-----------------|-----------------|
| `orchestrator.py` | Public API, facade pattern | Any actual work |
| `config_resolver.py` | Preset discovery, config merging, defaults | Execution |
| `pipeline_coordinator.py` | Stage ordering, backend selection, fallback | Image processing |
| `artifact_manager.py` | Output paths, merkle roots, artifact indexing | Config |
| `execution_engine.py` | Actual image processing, model inference | Orchestration |
| `run_card_validator.py` | Run card schema, backend semantics validation | Execution |

### Interface Contracts

Each module exposes a single primary class with a minimal public interface:

```python
# config_resolver.py
class ConfigResolver:
    def resolve(self, config: EnhanceConfig) -> ResolvedConfig: ...
    def discover_presets(self, pipeline: str) -> list[PresetInfo]: ...

# pipeline_coordinator.py  
class PipelineCoordinator:
    def plan(self, config: ResolvedConfig) -> ExecutionPlan: ...
    def select_backend(self, requested: str) -> BackendSelection: ...

# artifact_manager.py
class ArtifactManager:
    def index_artifacts(self, output_dir: Path) -> ArtifactIndex: ...
    def compute_merkle_root(self, artifacts: list[Path]) -> str: ...
    def generate_output_key(self, input_path: Path) -> str: ...

# execution_engine.py
class ExecutionEngine:
    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult: ...
    def process_single(self, image: np.ndarray, config: ResolvedConfig) -> ProcessedImage: ...

# run_card_validator.py
class RunCardValidator:
    def validate_payload(self, payload: dict) -> ValidationResult: ...
    def validate_backend_semantics(self, run_card: RunCard) -> ValidationResult: ...
```

---

## Alternatives Considered

### Alternative 1: Do Nothing
- **Pros:** No migration risk
- **Cons:** Continued degradation, testing impossible
- **Decision:** Rejected—technical debt is accumulating

### Alternative 2: Partial Extraction (Validators Only)
- **Pros:** Lower risk, faster
- **Cons:** Doesn't address core monolith problem
- **Decision:** Rejected—insufficient impact

### Alternative 3: Complete Rewrite
- **Pros:** Clean architecture from scratch
- **Cons:** High risk, breaks existing contracts
- **Decision:** Rejected—too disruptive

### Alternative 4: Incremental Decomposition (Selected)
- **Pros:** Lower risk, preserves behavior, testable at each step
- **Cons:** Requires careful planning, longer timeline
- **Decision:** Selected

---

## Consequences

### Positive
- Unit testable modules with clear boundaries
- Reduced merge conflicts
- Faster onboarding (understand one module at a time)
- Enables parallel development
- Aligns with documented architecture principles

### Negative
- Short-term migration effort (40-60 hours)
- Temporary code duplication during transition
- Need to update imports across codebase

### Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Regression in behavior | Medium | High | Comprehensive integration tests before decomposition |
| Circular imports between modules | Low | Medium | Clear dependency direction enforced |
| Performance regression | Low | Low | Benchmark critical paths before/after |

---

## Implementation Plan

### Phase 1: Preparation (Week 1)
1. Add comprehensive integration tests for current orchestrator
2. Document current method dependencies
3. Create module stub files

### Phase 2: Extract Validators (Week 1-2)
1. Extract `_validate_run_card_*` methods to `validators/run_card_validator.py`
2. Update orchestrator to use new validator
3. Add unit tests for validator

### Phase 3: Extract ArtifactManager (Week 2)
1. Extract `_infer_artifact_type`, `_build_artifact_index`, `_compute_artifact_merkle_root`
2. Move output key generation logic
3. Add unit tests

### Phase 4: Extract ConfigResolver (Week 3)
1. Extract preset discovery and config merging
2. Create `ResolvedConfig` data class
3. Add unit tests

### Phase 5: Extract PipelineCoordinator (Week 3-4)
1. Extract stage planning and backend selection
2. Create `ExecutionPlan` data class
3. Add unit tests

### Phase 6: Extract ExecutionEngine (Week 4-5)
1. Extract core processing logic
2. Create `ProcessedImage` result type
3. Add unit tests

### Phase 7: Finalize Facade (Week 5)
1. Reduce orchestrator to facade pattern
2. Verify all integration tests pass
3. Update documentation

---

## Enforcement

### CI Gates
- [ ] No single file in `lux_depth_v3/` exceeds 2,000 LOC (pylint check)
- [ ] New modules have >80% coverage (coverage gate per module)
- [ ] No circular imports (import-linter or similar)

### Code Review
- Any PR touching `orchestrator.py` must be reviewed by Architect
- New modules require ADR reference in PR description

---

## Success Criteria

| Criteria | Measurement | Target |
|----------|-------------|--------|
| Orchestrator LOC | `wc -l orchestrator.py` | <1,000 |
| Largest module | `wc -l *.py | sort -n` | <2,000 |
| Unit test coverage | pytest-cov per module | >80% |
| Integration tests | Current tests pass | 100% |

---

## References

- [Q2 2026 Development Roadmap](DEVELOPMENT_ROADMAP_2026_Q2.md)
- [Q1 2026 Codebase Audit](CODEBASE_AUDIT_2026_Q1.md) — Code Architecture: 6.2/10
- [Architecture Philosophy](ARCHITECTURE_PHILOSOPHY.md)

---

**Author:** Transformation Portal Architect  
**Review Required:** Yes (Specialist implementation approval)  
**Effective Date:** Upon merge
