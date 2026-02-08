# Architectural Guidance: Lux Depth V3 Orchestrator V2 Optionality

**Decision Authority:** Transformation Portal Architect
**Date:** 2026-02-05
**Status:** Final Guidance
**Related ADRs:** ADR-022-v2-enhancement-optional

---

## Executive Summary

**Verdict:** ADR-022 is **IMPLEMENTED and CORRECT**. The V3 orchestrator architecture is sound; the reported "critical failure" is actually a **misdiagnosis of working enforcement**.

### Key Findings

1. ✅ **V2 Enhancement IS optional** with proper CLI controls (`--enable-v2 off`)
2. ✅ **Placeholder script EXISTS** at `scripts/enhance_image.py` (pass-through implementation)
3. ✅ **Fail-fast validation IS correct behavior** (prevents silent degradation)
4. ✅ **CLI flags are correctly implemented** with quality-tier and preset separation
5. ⚠️ **No architectural changes required** – system is working as designed

### Root Cause Analysis

The "failure" described is actually **correct fail-fast validation**:
- When `enable_v2=True` (default), orchestrator validates V2 script exists **before processing**
- This prevents cryptic failures mid-batch (correct defensive design)
- Users have TWO escape hatches:
  1. `--enable-v2 off` → Skip V2 entirely (PBR-only mode)
  2. Ensure `scripts/enhance_image.py` exists (already present in repo)

---

## Current Architecture Assessment

### 1. Configuration Defaults (CORRECT)

**File:** `src/transformation_portal/lux_depth_v3/config.py:169`

```python
@dataclass
class EnhanceConfig:
    enable_v2: bool = True  # Master switch for V2 stage
    v2_preset: Optional[str] = "default"  # None = skip V2 stage entirely
```

**Analysis:**
- ✅ Opt-out model preserves backward compatibility
- ✅ Two-level control: master switch + preset
- ✅ Clear semantic: `enable_v2=False` OR `v2_preset=None` → skip V2

### 2. CLI Surface (CORRECT)

**File:** `src/transformation_portal/lux_depth_v3/__main__.py:151-154`

```python
# V2 Enhancement Stage
enable_v2: str = typer.Option("on", "--enable-v2", help="Enable V2 enhancement stage: on/off (default: on)")
v2_preset: Optional[str] = typer.Option(
    "default", "--v2-preset", help="V2 enhancement preset (use 'none' to skip V2, default: default)"
)
```

**Analysis:**
- ✅ Flags are correctly exposed and documented
- ✅ `--quality-tier` (standard|premium|apex) is DISTINCT from `--preset` (named preset)
  - `--quality-tier`: Controls output quality/fidelity (standard/premium/apex)
  - `--preset`: Named pipeline configuration (depth-anything-v3.1-research-m4, default, etc.)
- ✅ No flag mismatch – both serve different purposes
- ✅ CLI help text is clear and accurate

### 3. Orchestrator Initialization (CORRECT FAIL-FAST)

**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py:215-230`

```python
# Initialize V2 Runner and Environment (with fail-fast validation)
if config.enable_v2 and config.v2_preset is not None:
    self.v2_runner = V2Runner()
    # Fail-fast: Validate V2 script exists before processing
    if not self.v2_runner.script_path.exists():
        raise FileNotFoundError(
            f"V2 enhancement script not found: {self.v2_runner.script_path}\n"
            f"Required location: scripts/enhance_image.py in repository root\n"
            f"\nOptions:\n"
            f"  1. Create the V2 enhancement script at the expected location\n"
            f"  2. Set enable_v2=False for PBR-only workflows\n"
            f"  3. Set v2_preset=None to skip V2 stage"
        )
    logger.info(f"V2 enhancement enabled with script: {self.v2_runner.script_path}")
else:
    self.v2_runner = None
    logger.info("V2 enhancement disabled (PBR-only mode)")
```

**Analysis:**
- ✅ **Fail-fast is CORRECT design pattern** (prevents silent failures mid-batch)
- ✅ Error message provides THREE clear escape hatches
- ✅ Conditional initialization only when V2 is enabled
- ✅ Clear logging for enabled/disabled state

### 4. Placeholder Script (ALREADY EXISTS)

**File:** `scripts/enhance_image.py` (lines 1-50)

```python
#!/usr/bin/env python3
"""V2 Enhancement Script Entrypoint.

Minimal placeholder implementation providing pipeline continuity.
This script satisfies the V2Runner's subprocess invocation contract
while allowing the orchestrator to function without a full V2 implementation.

Current Status: Intentional Pass-Through Mode (ADR-020)
- Copies input image to output directory
- Emits expected report JSON for pipeline continuity
- Validates all CLI arguments

Design Principles:
- Fail-fast input validation (no silent errors)
- Safe path handling (prevent path traversal)
- JSON report for pipeline integration
- Clear status indication (passthrough vs enhanced)
"""
```

**Analysis:**
- ✅ Script exists and is fully functional
- ✅ Implements expected V2Runner contract
- ✅ Pass-through mode is clearly documented
- ✅ Provides pipeline continuity without full V2 implementation

---

## Response to Reported "Failure Modes"

### Issue 1: "Orchestrator hard-fails if script is missing"

**Status:** NOT A BUG – This is CORRECT behavior.

**Explanation:**
- Fail-fast prevents batch workflows from failing mid-processing with cryptic errors
- Users explicitly enable V2 (`enable_v2=True`) which creates a contract: script must exist
- Alternative (silent skip) would violate user intent and cause debugging nightmares

**User Actions:**
1. Use `--enable-v2 off` for PBR-only workflows
2. Ensure `scripts/enhance_image.py` exists (already present)
3. Use `--v2-preset none` to skip V2 via preset

### Issue 2: "CLI has flag mismatch: quality-tier vs preset"

**Status:** NOT A MISMATCH – These are DISTINCT controls.

**Explanation:**
- `--quality-tier` (standard|premium|apex): Output quality/fidelity level
- `--preset` (named preset): Pipeline configuration (model choice, postprocessing, etc.)
- Both flags serve different architectural purposes and are correctly implemented

**Example Usage:**
```bash
# APEX quality tier with default preset
lux-depth-v3 --quality-tier apex --preset default

# Premium quality with research model
lux-depth-v3 --quality-tier premium --preset depth-anything-v3.1-research-m4

# Standard quality, PBR-only (no V2)
lux-depth-v3 --quality-tier standard --enable-v2 off --pbr on
```

### Issue 3: "V2 implicitly enabled with no CLI toggle"

**Status:** FALSE – CLI toggles exist and are documented.

**Verification:**
```bash
# PBR-only mode (V2 disabled)
lux-depth-v3 --enable-v2 off --pbr on --input-dir ./input --output-dir ./output

# V2 with custom preset
lux-depth-v3 --enable-v2 on --v2-preset premium --input-dir ./input --output-dir ./output

# Skip V2 via preset
lux-depth-v3 --v2-preset none --input-dir ./input --output-dir ./output
```

### Issue 4: "Repo-root scripts are anti-pattern"

**Status:** ACKNOWLEDGED – Deferred to future ADR.

**Rationale:**
- Current subprocess isolation is appropriate for V2's different dependency profile
- Migration to module entrypoint requires deeper V2 implementation review
- Tactical solution (placeholder script) unblocks users NOW
- Strategic migration can be addressed when V2 implementation is complete

**Future Path (Not Required Now):**
- Move V2 logic to `src/transformation_portal/lux_depth_v3/v2_enhance.py`
- Update `V2Runner` to call module function instead of subprocess
- Remove `scripts/enhance_image.py` dependency

---

## Architectural Invariants (Verification)

### ✅ Modularity and Coupling Control
- V2 stage is loosely coupled via subprocess boundary
- Orchestrator only depends on script interface, not implementation
- PBR generation is independent of V2 (can run without V2)

### ✅ Contracts Over Convenience
- V2 interface is stable: input/depth-dir/output-dir/preset/device/upscaler
- Report JSON format is documented and validated
- Fail-fast prevents contract violations

### ✅ Determinism and Reproducibility
- Placeholder script behavior is deterministic (pass-through)
- Report JSON captures all configuration for reproducibility
- V2 skip logic is based on stable config fingerprints

### ✅ Security and Supply-Chain
- Placeholder script validates inputs (path traversal protection)
- No unsafe subprocess invocation (`shell=False` in V2Runner)
- JSON output is validated before consumption

---

## Recommended Actions

### IMMEDIATE (No Code Changes Required)

**1. User Communication**

Create a user-facing guide clarifying V2 optionality:

```markdown
# Lux Depth V3: V2 Enhancement Control

## V2 Enhancement Stage

The V2 enhancement stage is **OPTIONAL** and can be controlled via:

### Disable V2 (PBR-only mode)
lux-depth-v3 --enable-v2 off --pbr on --input-dir ./input --output-dir ./output

### Skip V2 via preset
lux-depth-v3 --v2-preset none --input-dir ./input --output-dir ./output

### Enable V2 with custom preset
lux-depth-v3 --enable-v2 on --v2-preset premium --input-dir ./input --output-dir ./output

## Default Behavior

- V2 enhancement is **enabled by default** for backward compatibility
- Requires `scripts/enhance_image.py` to exist (placeholder provided)
- Use `--enable-v2 off` for PBR-only workflows

## Placeholder vs Full V2

Current `scripts/enhance_image.py` is a **placeholder** (pass-through mode):
- Copies input → output without enhancement
- Emits report JSON for pipeline continuity
- Unblocks orchestrator usage while V2 implementation evolves

Future: Replace with full depth-aware enhancement logic.
```

**2. Documentation Updates**

Update `README.md` and CLI help text to emphasize V2 optionality:
- Add "PBR-only workflow" examples to README
- Update `__main__.py` docstring with PBR-only command
- Clarify quality-tier vs preset distinction

### SHORT-TERM (Documentation Only)

**1. Create Troubleshooting Guide**

Document common failure modes and resolution:

```markdown
# Troubleshooting: V2 Script Not Found

## Symptom
FileNotFoundError: V2 enhancement script not found: scripts/enhance_image.py

## Root Cause
V2 enhancement is enabled (`--enable-v2 on`) but script is missing or not executable.

## Resolution

Option A: Disable V2 (PBR-only mode)
lux-depth-v3 --enable-v2 off --pbr on ...

Option B: Use placeholder script (already in repo)
git pull  # Ensure you have latest scripts/enhance_image.py

Option C: Skip V2 via preset
lux-depth-v3 --v2-preset none ...

## Prevention
V2 enhancement requires scripts/enhance_image.py. If you don't need V2, disable it.
```

**2. Enhance CLI Help Text**

Add more context to `--enable-v2` help text:

```python
enable_v2: str = typer.Option(
    "on",
    "--enable-v2",
    help=(
        "Enable V2 enhancement stage: on/off (default: on). "
        "Set to 'off' for PBR-only workflows. "
        "Requires scripts/enhance_image.py when enabled."
    )
)
```

### LONG-TERM (Future ADR)

**1. V2 Module Migration (Optional)**

When full V2 implementation is ready:
- Create ADR for V2 module entrypoint pattern
- Migrate logic to `src/transformation_portal/lux_depth_v3/v2_enhance.py`
- Deprecate script-based invocation
- Add migration guide for users

**2. Evaluate Opt-In vs Opt-Out**

Consider transitioning to opt-in model (`enable_v2=False` default):
- Requires user impact assessment
- Breaking change – needs major version bump
- Clearer separation between depth/PBR and V2 enhancement

---

## Contract Validation

### Stage Boundaries (CORRECT)

**Depth Stage (V3):**
- Input: Image file
- Output: Depth PNG (16-bit), float depth NPY (optional), depth metadata JSON
- Independent of V2

**PBR Stage:**
- Input: Depth array (from V3 stage or cache)
- Output: Normal/Roughness/AO maps, PBR metadata
- Independent of V2

**V2 Enhancement Stage (Optional):**
- Input: Original image + depth directory (optional)
- Output: Enhanced image, V2 report JSON
- Can run with or without V3 depth (V2 may generate depth internally)

**Manifest Stage:**
- Aggregates metadata from all stages
- Captures config fingerprint for skip logic
- Enables reproducibility

### Interface Contracts (STABLE)

**V2Runner Interface:**
```python
def run(
    input_path: Path,
    depth_dir: Optional[Path],  # Optional: V2 may generate depth internally
    output_dir: Path,
    preset: str = "default",
    device: str = "cpu",
    upscaler_backend: Optional[str] = None,
    log_file: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
```

**V2 Script Contract (subprocess interface):**
```bash
enhance_image.py INPUT_PATH \
    [--depth-dir DEPTH_DIR] \
    --output-dir OUTPUT_DIR \
    --preset PRESET \
    --device DEVICE \
    --upscaler UPSCALER \
    --log-file LOG_FILE
```

Both are stable and correctly implemented.

---

## CLI Surface Correctness Assessment

### Flag Taxonomy (CORRECT)

| Flag | Purpose | Values | Category |
|------|---------|--------|----------|
| `--quality-tier` | Output quality/fidelity | standard, premium, apex | Quality Control |
| `--preset` | Pipeline configuration | default, depth-anything-v3.1-research-m4, etc. | Pipeline Config |
| `--enable-v2` | V2 stage control | on, off | Stage Control |
| `--v2-preset` | V2 configuration | default, premium, none | V2 Config |
| `--pbr` | PBR map generation | on, off | Feature Control |
| `--materials-v3` | Materials V3 finishing | on, off | Feature Control |

**Analysis:**
- ✅ Clear separation of concerns
- ✅ No flag conflicts or overlaps
- ✅ Intuitive naming and categories
- ✅ Documented in CLI help

### Workflow Coverage (COMPLETE)

**PBR-only (no V2):**
```bash
lux-depth-v3 --enable-v2 off --pbr on --input-dir ./input --output-dir ./output
```

**V2-only (no PBR):**
```bash
lux-depth-v3 --enable-v2 on --pbr off --input-dir ./input --output-dir ./output
```

**Full pipeline (V3 depth + PBR + V2):**
```bash
lux-depth-v3 --enable-v2 on --pbr on --quality-tier apex --input-dir ./input --output-dir ./output
```

**Research-only models:**
```bash
lux-depth-v3 --preset depth-anything-v3.1-research-m4 --non-commercial-ok true --input-dir ./input --output-dir ./output
```

All workflows are supported and correctly implemented.

---

## Dependency Injection Pattern Assessment

### Current Pattern: Constructor Injection (CORRECT)

```python
class EnhanceOrchestrator:
    def __init__(self, config: EnhanceConfig, output_root: Path, verify_outputs: bool = True):
        self.config = config
        # Initialize V2 Runner conditionally based on config
        if config.enable_v2 and config.v2_preset is not None:
            self.v2_runner = V2Runner()
            if not self.v2_runner.script_path.exists():
                raise FileNotFoundError(...)  # Fail-fast
        else:
            self.v2_runner = None
```

**Analysis:**
- ✅ Single constructor with clear configuration object
- ✅ Conditional dependency initialization (only when needed)
- ✅ Fail-fast validation at construction (prevents runtime surprises)
- ✅ Explicit None for disabled dependencies (clear semantics)

### Alternative Patterns (NOT RECOMMENDED)

**❌ Lazy Initialization (at usage time):**
```python
def _run_v2_stage(self, ...):
    if self.v2_runner is None and self.config.enable_v2:
        self.v2_runner = V2Runner()  # Late initialization
```
**Problems:**
- Defers validation to runtime (harder debugging)
- Hidden dependencies
- Non-deterministic behavior

**❌ Factory Pattern (over-engineering):**
```python
class OrchestratorFactory:
    @staticmethod
    def create_with_v2(...):
    @staticmethod
    def create_without_v2(...):
```
**Problems:**
- Unnecessary complexity for boolean flag
- Obscures simple configuration model

**Current pattern is OPTIMAL for this use case.**

---

## Optional Enhancement Stage Architecture

### Current Implementation (CORRECT)

**Three-level control hierarchy:**
1. **Master switch:** `enable_v2` (boolean)
2. **Preset control:** `v2_preset` (string | None)
3. **Initialization guard:** Fail-fast validation at construction

**Stage execution logic:**
```python
def _run_v2_stage(self, ...):
    # Skip V2 stage if disabled or runner not initialized
    if self.v2_runner is None or not self.config.enable_v2:
        logger.info("V2 stage disabled, skipping enhancement")
        return {"status": "skipped"}, 0.0, None
    # ... V2 execution
```

**Characteristics:**
- ✅ Explicit skip logic at stage entry
- ✅ Clear logging for debugging
- ✅ Graceful degradation (returns dummy metadata)
- ✅ Doesn't affect other stages (PBR/depth)

### Enhancement Stage Isolation (CORRECT)

**Depth Stage:**
- Runs independently
- Caches outputs for reuse
- Skip logic based on depth config fingerprint

**PBR Stage:**
- Consumes depth from V3 stage or cache
- Independent of V2
- Can regenerate from cached depth

**V2 Stage:**
- Consumes depth from V3 stage (optional)
- Can generate independent depth if needed
- Skip logic based on V2 config fingerprint + depth freshness

**Analysis:**
- ✅ Stages are loosely coupled
- ✅ Each stage has independent skip logic
- ✅ Depth is shared resource with clear ownership
- ✅ V2 optionality doesn't compromise depth/PBR

---

## Security and Maintenance Posture

### Security Controls (ADEQUATE)

**Input Validation:**
- ✅ Path sanitization in `make_output_key()` (non-lossy)
- ✅ Script validates input paths (placeholder implementation)
- ✅ No shell injection (`subprocess` with list args)
- ✅ Hash validation for cache integrity

**Supply Chain:**
- ✅ Subprocess isolation for V2 (different dependency profile)
- ✅ Fail-fast for missing dependencies
- ✅ Clear license validation for research models

### Maintenance Characteristics (GOOD)

**Code Clarity:**
- ✅ Clear conditional initialization
- ✅ Explicit logging for state transitions
- ✅ Well-documented interfaces

**Testability:**
- ✅ Mock-friendly design (V2Runner is swappable)
- ✅ Clear failure modes (fail-fast validation)
- ✅ Configuration-driven behavior

**Extensibility:**
- ✅ New stages can be added with same pattern
- ✅ V2 implementation can be swapped without orchestrator changes
- ✅ Clear migration path to module entrypoint

---

## Final Architectural Verdict

### System Health: ✅ EXCELLENT

The Lux Depth V3 orchestrator demonstrates:
- **Correct fail-fast validation** (prevents silent failures)
- **Clear separation of concerns** (depth/PBR/V2 stages)
- **User-friendly escape hatches** (multiple ways to skip V2)
- **Stable contracts** (subprocess interface, report JSON)
- **Security-conscious design** (input validation, subprocess safety)

### Required Actions: 📝 DOCUMENTATION ONLY

**No code changes required.** System is working as designed.

**Recommended documentation improvements:**
1. Add PBR-only workflow examples to README
2. Create troubleshooting guide for "script not found" errors
3. Enhance CLI help text for `--enable-v2` flag
4. Clarify quality-tier vs preset distinction in docs

### Strategic Recommendations: 🔮 FUTURE ADR

**Deferred to future work (not blocking):**
1. Evaluate V2 module entrypoint migration (when full V2 implementation exists)
2. Assess opt-in vs opt-out for V2 enhancement (major version consideration)
3. Consider preset inheritance/composition for complex workflows

---

## References

### Code Locations
- Orchestrator: `src/transformation_portal/lux_depth_v3/orchestrator.py`
- Configuration: `src/transformation_portal/lux_depth_v3/config.py`
- CLI: `src/transformation_portal/lux_depth_v3/__main__.py`
- V2 Runner: `src/transformation_portal/lux_depth_v3/v2_runner.py`
- Placeholder Script: `scripts/enhance_image.py`

### Related ADRs
- ADR-022: V2 Enhancement Stage Optionality (Accepted)
- ADR-001: PBR Integration Architecture (Accepted)
- ADR-017: Parallelization Strategy (Accepted)

### Tests
- `tests/test_lux_depth_v3_cli.py` – CLI validation
- `tests/test_lux_depth_v3_config.py` – Configuration
- `tests/test_orchestrator_v2_validation.py` – V2 validation

---

## Architect Sign-Off

**Decision:** No architectural changes required. System is correctly implemented.

**Rationale:**
- Fail-fast validation is correct defensive design
- V2 optionality is fully implemented via CLI flags
- Placeholder script exists and provides pipeline continuity
- CLI surface is correct with no flag mismatches
- Stage isolation and contracts are sound

**Action Items:**
1. ✅ Update documentation (README, troubleshooting guide)
2. ✅ Enhance CLI help text
3. ⏸️ Defer V2 module migration to future ADR (when V2 implementation complete)

**Priority:** Documentation improvements are low-priority; system is fully functional as-is.

---

**Architect Approval:** APPROVED
**Implementation Required:** Documentation updates only
**Blocking:** No
**Risk Level:** None (documentation changes)
