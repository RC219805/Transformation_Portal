# Architecture Improvements - Implementation Roadmap

**Date**: December 7, 2025  
**Status**: Ready for Implementation  
**Owner**: Transformation Portal Architecture Team

## Overview

This document provides a practical implementation roadmap for the architectural improvements identified in the comprehensive architecture review (`ARCHITECTURE_REVIEW_2025.md`).

## Quick Reference

| Phase | Timeline | Effort | Priority | Status |
|-------|----------|--------|----------|--------|
| Phase 1: Foundation | Q1 2026 | 4 weeks | Critical | ✅ Planned |
| Phase 2: Refactoring | Q2 2026 | 6 weeks | High | 📋 Pending |
| Phase 3: Extensibility | Q3 2026 | 4 weeks | Medium | 📋 Pending |

**Total Estimated Effort**: 14 weeks  
**Expected ROI**: Break-even at 1.75 years, positive 5-year NPV

---

## Phase 1: Foundation (Q1 2026 - 4 weeks)

### Week 1: Module Interface Contracts ✅ In Progress

**Goal**: Establish explicit API contracts for all architectural layers

**Deliverables**:
- [x] Create `src/transformation_portal/interfaces/` package
- [x] Define `ImageProcessor` interface with validation
- [x] Define `Pipeline` and `PipelineStage` interfaces
- [x] Document interface contracts in ADR-001
- [ ] Write interface contract tests
- [ ] Add CI validation for interface conformance

**Files Created**:
```
src/transformation_portal/interfaces/
├── __init__.py
├── processor.py (ImageProcessor, VideoProcessor)
└── pipeline.py (Pipeline, PipelineStage, BatchPipeline)
```

**Testing**:
```bash
# Run interface contract tests
pytest tests/interfaces/ -v

# Validate all processors implement interfaces
python scripts/validation/check_interface_conformance.py
```

**Success Criteria**:
- [ ] All 3 interfaces defined with comprehensive docstrings
- [ ] Contract tests pass (90%+ coverage)
- [ ] CI check prevents interface violations

---

### Week 2: Security Hardening ✅ In Progress

**Goal**: Eliminate path traversal and command injection vulnerabilities

**Deliverables**:
- [x] Create `src/transformation_portal/utils/security.py`
- [x] Implement `validate_filepath()` with comprehensive checks
- [x] Implement `build_safe_command()` for subprocess safety
- [x] Document security validation in ADR-002
- [ ] Migrate image processing pipelines to use validation
- [ ] Migrate video processing to use safe commands
- [ ] Add security tests (30+ test cases)

**Files to Update**:
```
Priority 1 (Critical):
- src/transformation_portal/pipelines/lux_render_pipeline.py
- src/transformation_portal/processors/luxury_video_master_grader.py
- lux_depth_v2/pipeline.py

Priority 2 (High):
- src/transformation_portal/processors/material_response/core.py
- All files in src/transformation_portal/enhancers/
- scripts/pipelines/*.py
```

**Migration Pattern**:
```python
# Before (unsafe)
def process_image(filename: str):
    img = Image.open(filename)
    return transform(img)

# After (safe)
from transformation_portal.utils.security import validate_image_path

def process_image(filename: str, base_dir: Path = Path.cwd()):
    safe_path = validate_image_path(
        Path(filename),
        allowed_dirs=[base_dir],
        max_size=MAX_IMAGE_SIZE
    )
    img = Image.open(safe_path)
    return transform(img)
```

**Testing**:
```bash
# Run security tests
pytest tests/security/ -v

# Audit for remaining shell=True usage
grep -r "shell=True" src/transformation_portal/
# Expected: 0 matches

# Validate path handling
python scripts/security/audit_file_operations.py
```

**Success Criteria**:
- [ ] 100% of file operations use `validate_filepath()`
- [ ] 0 FFmpeg commands use `shell=True`
- [ ] Security test suite passes (30+ tests)
- [ ] CodeQL scans show no path traversal vulnerabilities

---

### Week 3: Dependency Consolidation

**Goal**: Establish single source of truth for dependency management

**Deliverables**:
- [ ] Adopt `pyproject.toml` as primary dependency source
- [ ] Generate requirements/*.txt from pyproject.toml
- [ ] Add hash verification for ML dependencies
- [ ] Update CI/CD to use pyproject.toml
- [ ] Document dependency management process

**Tasks**:

1. **Update pyproject.toml**:
   ```toml
   [project]
   dependencies = [
       "numpy>=1.24,<2.3.0",
       "Pillow>=10.0.0,<12",
       # ... all base dependencies
   ]
   
   [project.optional-dependencies]
   ml-minimal = ["torch>=2.0,<3", "torchvision>=0.15.0,<1"]
   ml-depth = ["transformation-portal[ml-minimal]", "transformers>=4.35.0,<5"]
   ml-full = ["transformation-portal[ml-depth]", "diffusers>=0.20,<1"]
   ```

2. **Generate requirements files**:
   ```bash
   pip-compile pyproject.toml -o requirements/base.txt
   pip-compile --extra ml-full pyproject.toml -o requirements/ml.txt
   ```

3. **Add hash verification**:
   ```bash
   pip-compile --generate-hashes pyproject.toml -o requirements/base.lock.txt
   ```

4. **Update CI**:
   ```yaml
   - name: Install dependencies
     run: pip install -e ".[ml-full]"
   ```

**Success Criteria**:
- [ ] pyproject.toml is single source of truth
- [ ] All requirements/*.txt generated from pyproject.toml
- [ ] CI uses pyproject.toml for installation
- [ ] Hash verification enabled for production dependencies

---

### Week 4: Test Coverage Baseline

**Goal**: Establish visibility into testing gaps

**Deliverables**:
- [ ] Add pytest-cov to CI
- [ ] Generate coverage reports
- [ ] Add coverage badges to README
- [ ] Identify under-tested modules
- [ ] Document coverage targets

**Implementation**:

1. **Update CI workflow**:
   ```yaml
   - name: Test with Coverage
     run: |
       pytest --cov=src/transformation_portal \
              --cov-report=xml \
              --cov-report=term \
              --cov-report=html
   
   - name: Upload Coverage
     uses: codecov/codecov-action@v3
     with:
       files: ./coverage.xml
   ```

2. **Set coverage targets**:
   ```ini
   # pytest.ini or pyproject.toml
   [tool.pytest.ini_options]
   addopts = "--cov-fail-under=75"
   
   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "if __name__ == .__main__.:",
   ]
   ```

3. **Add badges**:
   ```markdown
   [![Coverage](https://codecov.io/gh/RC219805/Transformation_Portal/branch/main/graph/badge.svg)](https://codecov.io/gh/RC219805/Transformation_Portal)
   ```

**Success Criteria**:
- [ ] Coverage reports in CI
- [ ] Baseline coverage >75% overall
- [ ] Coverage badges in README
- [ ] Coverage tracked over time

---

## Phase 2: Refactoring (Q2 2026 - 6 weeks)

### Weeks 5-6: Refactor Monolithic Files

**Goal**: Break down files >1000 LOC into manageable modules

**Target Files**:
1. `rendering_4k_pipeline.py` (2274 LOC) → 4-5 modules
2. `unified_luxury_pipeline.py` (1484 LOC) → 3-4 modules
3. `lux_render_pipeline.py` (1344 LOC) → 3-4 modules
4. `material_response/core.py` (1274 LOC) → core.py + optimizer.py
5. `async_pipeline.py` (1239 LOC) → orchestrator.py + workers.py

**Refactoring Strategy**:

**Example**: `rendering_4k_pipeline.py` (2274 LOC)

**Split into**:
```
src/transformation_portal/pipelines/rendering_4k/
├── __init__.py           # Public API
├── pipeline.py           # Main orchestration (400 LOC)
├── stages.py             # Pipeline stages (500 LOC)
├── config.py             # Configuration classes (300 LOC)
├── validators.py         # Input validation (200 LOC)
└── utils.py              # Helper functions (200 LOC)
```

**Migration approach**:
1. Create new module directory
2. Extract classes/functions to new files
3. Add imports to `__init__.py` for backwards compatibility
4. Update tests
5. Deprecate old file with migration guide

**Success Criteria**:
- [ ] No files >800 LOC
- [ ] All refactored modules have 80%+ test coverage
- [ ] Backwards compatibility maintained
- [ ] Documentation updated

---

### Week 7: Centralize Duplicated Logic

**Goal**: Eliminate 3 major duplication patterns

**Pattern 1: Image I/O**
```
Before: 15 files with custom image loading
After: src/transformation_portal/utils/io.py
Functions: load_image(), save_image(), load_batch()
```

**Pattern 2: Depth Processing**
```
Before: 8 files with depth normalization/inversion
After: src/transformation_portal/depth/processing.py
Class: DepthProcessor with normalize(), invert(), detect_edges()
```

**Pattern 3: Material Segmentation**
```
Before: 3 implementations (material_response/, lux_depth_v2/, enhancers/)
After: src/transformation_portal/segmentation/material.py
Interface: MaterialSegmenter with HeuristicSegmenter, ONNXSegmenter
```

**Success Criteria**:
- [ ] Image I/O centralized (15 duplicates → 1 implementation)
- [ ] Depth processing centralized (8 duplicates → 1 class)
- [ ] Material segmentation unified (3 implementations → 1 interface + 3 backends)

---

### Weeks 8-9: Add Integration Tests

**Goal**: Catch cross-module issues with comprehensive integration tests

**Test Categories**:

1. **End-to-End Pipeline Tests**:
   ```python
   def test_full_luxury_pipeline():
       """Test complete processing: load → depth → material → color → save."""
       pass
   ```

2. **Contract Tests**:
   ```python
   def test_material_response_implements_interface():
       """Verify MaterialResponse conforms to ImageProcessor."""
       pass
   ```

3. **Performance Regression Tests**:
   ```python
   @pytest.mark.benchmark
   def test_depth_estimation_performance(benchmark):
       """Ensure depth estimation < 100ms per image."""
       pass
   ```

**Target**: 50+ integration tests

**Success Criteria**:
- [ ] Integration test suite with 50+ tests
- [ ] All interfaces have contract tests
- [ ] Performance benchmarks tracked in CI

---

### Week 10: API Documentation

**Goal**: Comprehensive documentation for all modules

**Deliverables**:
- [ ] Sphinx documentation setup
- [ ] Auto-generated API reference
- [ ] Usage examples for each module
- [ ] Migration guides for deprecated features

**Structure**:
```
docs/
├── api/
│   ├── interfaces.rst
│   ├── processors.rst
│   ├── pipelines.rst
│   └── utils.rst
├── guides/
│   ├── getting-started.md
│   ├── security.md
│   └── migration.md
└── examples/
    ├── basic-pipeline.md
    └── custom-processor.md
```

**Success Criteria**:
- [ ] Sphinx documentation published
- [ ] API reference for all public modules
- [ ] 10+ usage examples

---

## Phase 3: Extensibility (Q3 2026 - 4 weeks)

### Weeks 11-12: Plugin Architecture

**Goal**: Enable third-party extensions via plugin system

**Deliverables**:
- [ ] Plugin interface definition
- [ ] Plugin discovery mechanism
- [ ] Plugin lifecycle management
- [ ] Example plugins

**Implementation**:
```python
# src/transformation_portal/plugins/interface.py
class Plugin(ABC):
    @abstractmethod
    def get_name(self) -> str: pass
    
    @abstractmethod
    def get_version(self) -> str: pass
    
    @abstractmethod
    def initialize(self, config: Dict) -> None: pass
    
    @abstractmethod
    def register_processors(self) -> List[ImageProcessor]: pass
```

**Success Criteria**:
- [ ] Plugin interface defined
- [ ] 3+ example plugins
- [ ] Plugin documentation

---

### Week 13: Event System

**Goal**: Decouple cross-cutting concerns via pub/sub

**Deliverables**:
- [ ] Event system implementation
- [ ] Analytics via events (no direct imports)
- [ ] Event documentation

**Success Criteria**:
- [ ] Event system functional
- [ ] Analyzers use events only

---

### Week 14: Configuration Management

**Goal**: Unified configuration system

**Deliverables**:
- [ ] Configuration loader
- [ ] Environment-specific configs
- [ ] Configuration validation

**Success Criteria**:
- [ ] Unified config system
- [ ] Documentation complete

---

## Tracking & Metrics

### Weekly Progress Tracking

Update this section weekly:

**Week 1 (Dec 9-13, 2025)**:
- [x] Created interface contracts
- [x] Created security utilities
- [ ] Interface contract tests
- [ ] CI validation

**Week 2 (Dec 16-20, 2025)**:
- [ ] Migrate image pipelines
- [ ] Migrate video processing
- [ ] Security tests

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking changes | High | High | Gradual rollout with deprecation warnings |
| Performance regression | Medium | Medium | Benchmark tests in CI |
| Developer resistance | Medium | Low | Comprehensive documentation |
| Scope creep | Medium | High | Strict adherence to roadmap |

---

## Success Metrics

**Phase 1 Completion Criteria**:
- [ ] All interfaces defined and tested
- [ ] Security vulnerabilities eliminated
- [ ] Test coverage >75%
- [ ] Single source of truth for dependencies

**Phase 2 Completion Criteria**:
- [ ] No files >800 LOC
- [ ] 3 duplication patterns eliminated
- [ ] Integration test suite complete
- [ ] API documentation published

**Phase 3 Completion Criteria**:
- [ ] Plugin system functional
- [ ] Event-driven analytics
- [ ] Configuration management complete

---

## Appendix: Quick Commands

**Security Audit**:
```bash
# Find unsafe file operations
grep -r "open(" src/transformation_portal/ | grep -v "validate_filepath"

# Find shell=True usage
grep -r "shell=True" src/transformation_portal/
```

**Coverage Check**:
```bash
pytest --cov=src/transformation_portal --cov-report=html
open htmlcov/index.html
```

**Interface Validation**:
```bash
python scripts/validation/check_interface_conformance.py
```

---

**Last Updated**: December 7, 2025  
**Next Review**: January 7, 2026  
**Owner**: Transformation Portal Architecture Team
