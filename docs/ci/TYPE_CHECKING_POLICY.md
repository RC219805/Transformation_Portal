# Type Checking Policy

**Status**: DRAFT (requires architect approval)
**Owner**: Transformation Portal Architect
**Last Updated**: 2026-05-24

---

## Actively Enforced mypy Whitelist (current)

This is the authoritative list of paths that the **blocking** mypy gate in
`.github/workflows/build.yml` (`Type check with mypy (critical modules)`)
enforces. Each path passes `mypy --config-file=mypy.ini <path>` cleanly;
additions must do the same before being appended to the workflow list.

**Enforced as of 2026-05-24:**

- `src/transformation_portal/api/`
- `src/transformation_portal/lux_depth_v3/`
- `src/transformation_portal/orchestrator/queue/` *(added by N-1, 2026-05-24)*
- `src/transformation_portal/orchestrator/storage/` *(added by N-1, 2026-05-24)*
- `src/transformation_portal/orchestrator/artifact_store/` *(added by N-1, 2026-05-24)*
- `src/transformation_portal/core/geometry/`
- `src/transformation_portal/core/processing/`
- `src/transformation_portal/core/ml_dependency_health.py`
- `src/transformation_portal/core/da3_runtime.py`

**N-1 tranche notes (2026-05-24):** the three `orchestrator/` paid-pilot
Protocol packages were added together. `queue/` was already clean;
`storage/` required correcting the `JobEventStore.events_since` abstract
signature from `async def` to a plain `def` returning `AsyncIterator`
(every backend is an async generator consumed via `async for`, so the
`async def` form mis-typed it as a coroutine); `artifact_store/` required
tightening one `Optional[int]` local that is always `len(body)` to `int`.

**Remaining backlog (not yet enforced — candidates for the next tranche):**

- `src/transformation_portal/spatial_ai/segmentation/` (16 files; not yet triaged)
- `src/transformation_portal/core/` as a whole — blocked on ~21 currently-failing files (see workflow comment); enable per-file or per-subpackage, not wholesale.
- `src/transformation_portal/events/`, `storage/` (top-level), `hardening/`, `rendering/` — untriaged.

The phased strategy below predates the whitelist mechanism and is retained
for historical context; the list above is the live source of truth.

> ⚠️ **Historical snapshot.** Everything from the "Current State" section
> onward is a dated 2026-02-04 snapshot describing the original gradual-typing
> plan. Several of its bullets (e.g. "no explicit mypy enforcement in CI")
> are no longer accurate and are contradicted by the enforced whitelist
> above. Treat the section above as current policy; read what follows as
> background only.

---

## Current State

**Type checking status as of 2026-02-04** *(historical snapshot — see warning above; superseded by the enforced whitelist)*:
- ✅ Type hints exist in many modules (gradual typing)
- ⚠️ No explicit mypy or pyright enforcement in CI workflows
- ⚠️ `mypy.ini` exists in repository root but unclear if actively enforced
- ❌ No documented policy on type coverage requirements

**Audit findings**:
- `build.yml` (CI gate) does NOT run mypy or pyright
- `quality-gate.yml` does NOT run type checking
- Local `mypy.ini` suggests historical intent but no active enforcement

---

## Problem Statement

**Without enforced type checking**:
1. Type hints drift out of sync with actual code
2. Refactorings introduce type errors that pass CI
3. New contributors don't know what type coverage is expected

**With strict type checking**:
1. Existing code may have hundreds of type errors
2. Large-scale cleanup required before enforcement
3. Risk of breaking changes to fix type issues

---

## Recommended Strategy: Gradual Enforcement

### Phase 1: Establish Baseline (Gate 0 — Current)

**Goal**: Understand current type coverage and error count

**Actions**:
1. Run `mypy src/ tests/ --ignore-missing-imports --show-error-codes` locally
2. Document:
   - Total error count
   - Modules with most errors
   - Common error patterns (e.g., `Any` overuse, missing return types)
3. Decision point: baseline formatting first (black/isort), then type checking

**Timeline**: 1 week (after Gate 0 formatting baseline)

---

### Phase 2: Non-Blocking Enforcement (Tranche 2)

**Goal**: Add type checking to CI without blocking merges

**Actions**:
1. Add mypy job to `build.yml`:
   ```yaml
   type-check:
     runs-on: ubuntu-latest
     continue-on-error: true  # Non-blocking
     steps:
       - uses: actions/checkout@v6
       - uses: actions/setup-python@v6
         with:
           python-version: '3.12'
       - name: Install mypy
         run: pip install mypy
       - name: Run mypy
         run: |
           mypy src/ tests/ \
             --ignore-missing-imports \
             --show-error-codes \
             --pretty \
             --no-error-summary
   ```

2. Monitor for 2-4 weeks:
   - Track error count trends
   - Identify low-hanging fruit (easy fixes)

**Timeline**: 2 weeks setup + 2-4 weeks monitoring

---

### Phase 3: Module-Level Blocking (Tranche 3+)

**Goal**: Enforce type checking on critical modules

**Actions**:
1. Identify "tier 1" modules (core interfaces, public APIs):
   - `src/transformation_portal/core/`
   - `src/transformation_portal/config_loader.py`
   - CLI entrypoints

2. Fix type errors in tier 1 modules

3. Add strict checking for tier 1 only:
   ```yaml
   - name: Run mypy (strict on core modules)
     run: |
       mypy src/transformation_portal/core/ \
         --strict \
         --show-error-codes
   ```

4. Gradually expand tier 1 coverage

**Timeline**: 4-8 weeks (iterative)

---

### Phase 4: Repo-Wide Enforcement (Long-term)

**Goal**: Full type checking across all modules

**Actions**:
1. Remove `continue-on-error: true`
2. Enforce strict mypy across `src/`
3. Document type coverage in README

**Timeline**: 3-6 months (aspirational)

---

## Configuration Standards

### `mypy.ini` (Recommended Baseline)

```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False  # Phase 2: allow untyped for now
ignore_missing_imports = True  # Phase 2: ignore 3rd party stubs

# Phase 3: Enable strict for core modules
[mypy-transformation_portal.core.*]
disallow_untyped_defs = True
check_untyped_defs = True
warn_redundant_casts = True
warn_unused_ignores = True

# Phase 3: Enable strict for config
[mypy-transformation_portal.config_loader]
disallow_untyped_defs = True
check_untyped_defs = True
```

---

## Type Hint Standards

### Required Type Hints

**Always type-hint**:
- Public API functions and methods
- Function parameters (especially in core modules)
- Return types (except simple property getters)

**Example**:
```python
from pathlib import Path
from typing import Optional

def process_image(
    input_path: Path,
    output_dir: Path,
    quality: int = 95,
    preset: Optional[str] = None
) -> Path:
    """Process image with given settings."""
    ...
```

### Acceptable `Any` Usage

**Use `Any` sparingly**:
- External library returns without stubs
- Dynamic configuration objects (document why)
- Legacy code during gradual migration

**Prefer alternatives**:
- `object` for truly unknown types
- `TypedDict` for structured dicts
- `Protocol` for duck-typed interfaces

---

## Migration Guidance

### Fixing Common Type Errors

**1. Missing return type**:
```python
# Before
def get_config():
    return load_yaml("config.yaml")

# After
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    return load_yaml("config.yaml")
```

**2. Untyped parameters**:
```python
# Before
def resize_image(img, width, height):
    ...

# After
from PIL import Image

def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    ...
```

**3. Optional handling**:
```python
# Before
def get_preset(name):  # Could return None
    return presets.get(name)

# After
from typing import Optional

def get_preset(name: str) -> Optional[PresetConfig]:
    return presets.get(name)
```

---

## Enforcement in CI

### Phase 2 (Non-Blocking)

```yaml
# .github/workflows/build.yml
type-check:
  runs-on: ubuntu-latest
  continue-on-error: true
  steps:
    - uses: actions/checkout@v6
    - uses: actions/setup-python@v6
      with:
        python-version: '3.12'
    - run: pip install mypy
    - run: mypy src/ tests/ --config-file mypy.ini
```

### Phase 3 (Blocking for Core Modules)

```yaml
type-check-core:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
    - uses: actions/setup-python@v6
      with:
        python-version: '3.12'
    - run: pip install mypy
    - run: |
        # Strict on core modules (blocking)
        mypy src/transformation_portal/core/ --strict

        # Warning-only on other modules
        mypy src/ tests/ --config-file mypy.ini || true
```

---

## Metrics and Goals

### Baseline Metrics (To Be Measured)
- Total mypy errors: `[TBD]`
- Modules with >10 errors: `[TBD]`
- Percentage of functions with return type hints: `[TBD]`

### Phase Goals

| Phase | Timeline | Error Count | Coverage | Blocking |
|-------|----------|-------------|----------|----------|
| 1: Baseline | Week 1 | Measure | N/A | No |
| 2: Non-blocking | Weeks 2-6 | <50% baseline | Core modules | No |
| 3: Core blocking | Weeks 7-14 | 0 in core modules | 60%+ core | Yes (core only) |
| 4: Repo-wide | Months 4-6 | 0 repo-wide | 80%+ | Yes (all) |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-04 | Draft policy created | Gate 0 requires type checking strategy |
| [TBD] | Phase 2 approved | After baseline formatting complete |
| 2026-05-24 | N-1: added `orchestrator/{queue,storage,artifact_store}/` to the blocking whitelist | Audit finding #1 (whitelist-based type gating); paid-pilot Protocol packages were clean or needed only minor type fixes |

---

## References

- `mypy.ini`: Mypy configuration file (repository root)
- `docs/architecture/TRANCHE_EXECUTION_PLAN.md`: Execution timeline
- [Mypy documentation](https://mypy.readthedocs.io/)
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)

---

**Next Actions**:
1. Complete Gate 0 (baseline formatting)
2. Run baseline mypy audit (measure current state)
3. Update this document with baseline metrics
4. Seek architect approval for Phase 2 timeline

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: After each phase completion
