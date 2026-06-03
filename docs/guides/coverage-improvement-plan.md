# Coverage Improvement Plan

**Current Status (as of 2026-02-04)**
- **Combined Coverage**: 25.44%
- **Total Statements**: 24,820
- **Covered Statements**: ~6,314 (24,820 × 0.2544)
- **Baseline Gate**: 25% (prevents regression below this floor)

## Historical Context

The coverage baseline was previously set at 33% but this was aspirational rather than actual. After PR #832 fixed coverage artifact consolidation, we now have accurate combined coverage data from both core and ML test suites.

**Why 25.44%?**
- Core tests focus on depth processing, material response, and rendering pipelines
- ML tests cover model backends and inference paths
- Many modules (plugins, streaming, style transfer, VLM) have 0% coverage
- This is legacy code—the team prioritized working features over tests during initial development

## Roadmap to 33% Coverage

To reach 33% coverage, we need to cover **~1,877 additional statements** (from 6,314 to 8,191).

**Calculation:**
- Target statements @ 33%: 24,820 × 0.33 ≈ 8,191
- Current covered: 6,314
- Gap: 8,191 - 6,314 = ~1,877 statements

### Phase 1: Low-Hanging Fruit (Q1 2026) – Target: 28%
**Goal**: +636 covered statements (6,314 → 6,950)

**Calculation:**
- Target @ 28%: 24,820 × 0.28 ≈ 6,950
- Increment: 6,950 - 6,314 = +636

Priority modules (currently 0% coverage):
1. **CLI & Entrypoints** (~250 statements)
   - `src/transformation_portal/__main__.py` (141 stmts)
   - `src/transformation_portal/cli/__init__.py` (partial, ~100 stmts)

2. **Configuration & Validation** (~200 statements)
   - `src/transformation_portal/config_loader.py` (148 stmts, currently 39.75%)
   - `src/transformation_portal/utils/recipe_validator.py` (63 stmts)

3. **Error Handling & Security** (~180 statements)
   - `src/transformation_portal/utils/input_validation.py` (195 stmts)
   - Already high: `utils/error_handling.py` (89%), focus on edge cases

### Phase 2: Core Pipelines (Q2 2026) – Target: 33%
**Goal**: +1,241 more covered statements (6,950 → 8,191)

**Calculation:**
- Target @ 33%: 24,820 × 0.33 ≈ 8,191
- Increment from Phase 1: 8,191 - 6,950 = +1,241

Priority modules:
1. **Depth Processing Extensions** (~400 statements)
   - `depth/processors/depth_aware_denoise.py` (90 stmts, currently 18%)
   - `depth/processors/depth_guided_filters.py` (136 stmts, currently 13%)
   - `depth/processors/zone_tone_mapping.py` (123 stmts, currently 13%)

2. **Pipeline Orchestration** (~400 statements)
   - `pipelines/lux_render_pipeline.py` (523 stmts, currently 32%)
   - `streaming/async_pipeline.py` (432 stmts, currently 24%)

3. **Plugins System** (~370 statements)
   - `plugins/loader.py` (278 stmts, 0%)
   - `plugins/manager.py` (218 stmts, 0%)
   - `plugins/registry.py` (118 stmts, 0%)

### Phase 3: Advanced Features (Q3 2026+) – Target: 40%+
**Goal**: +1,730 more covered statements (to reach 40%)

Low-priority modules (can defer):
- Neuroaesthetics (618 stmts, 0%)
- Style Transfer (388 stmts, 0%)
- VLM (423 stmts, 0%)
- ComfyUI Integration (451 stmts, 0%)
- Atmosphere/Sky (577 stmts, 0%)

## Implementation Strategy

### Test Writing Guidelines
1. **Start with smoke tests**: Does the function run without crashing?
2. **Add happy-path tests**: Does it work with valid inputs?
3. **Add edge cases**: Empty inputs, None, boundary values
4. **Mock expensive operations**: File I/O, model inference, FFmpeg calls

### CI Integration
- **Baseline gate** (`--fail-under=25`): Prevents regression
- **Diff coverage target** (`make coverage-diff`): Local PR-review signal for new code;
  not a required PR CI gate until `build.yml` wires in `diff-cover`
- **Ratchet mechanism**: As coverage improves, baseline gate increases

### Ownership
- **Core team**: Responsible for Phase 1 & 2 modules
- **Contributors**: Can help with Phase 3 specialized modules
- **Quarterly reviews**: Adjust targets based on velocity

## Tracking Progress

Update this document quarterly with:
- Current coverage percentage
- Modules completed
- Baseline gate threshold
- Blockers/challenges

**Next Update**: 2026-05-01 (end of Q1)

---

## FAQ

**Q: Why not 80% coverage like other projects?**
A: This is a legacy codebase with ~25K statements. Reaching 80% would require ~12,000 new test assertions. We're taking an incremental, pragmatic approach.

**Q: Can I merge code with <85% diff coverage?**
A: Required PR CI may still pass today, but expect review scrutiny. Target 85%
diff coverage locally with `make coverage-diff`; wiring this into `build.yml`
is the tracked step that would make it a blocking PR gate.

**Q: What if coverage drops unexpectedly?**
A: The CI gate will fail. Investigate whether:
1. Tests were accidentally deleted
2. New code diluted the percentage (expected)
3. The baseline needs adjustment (rare, requires team approval)

**Q: How do I run coverage locally?**
```bash
# Comprehensive coverage report (HTML + XML + terminal)
make coverage-report
open htmlcov/index.html

# Diff coverage check against main branch (85% local target for new code)
make coverage-diff

# Package-level baseline report for ratcheting
make coverage-package

# Scoped coverage for specific paths
make coverage-fast-scope
```

**Q: What are the coverage targets?**
- **Diff coverage**: 85% on new/changed lines (local target / future PR gate)
- **Global floor**: 25% (required PR gate; prevents regression)
- **Long-horizon target**: 70% overall

See `docs/testing/test_coverage_improvement_plan.md` for the full phased plan.
