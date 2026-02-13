# ADR-022: V2 Enhancement Stage Optionality

**Status:** Accepted
**Date:** 2026-02-04
**Authors:** Transformation Portal Architect
**Context:** Lux Depth V3 orchestrator hard-fails when V2 script missing

---

## Context

The Lux Depth V3 orchestrator (`EnhanceOrchestrator`) has a hard dependency on `scripts/enhance_image.py` for the V2 enhancement stage, which is enabled by default (`enable_v2=True`, `v2_preset="default"`).

**Current Behavior:**
- Orchestrator initialization performs fail-fast validation: if V2 is enabled and the script doesn't exist, it raises `FileNotFoundError`
- Users cannot run PBR-only workflows without either creating the V2 script or modifying config programmatically
- CLI does not expose `enable_v2` or `v2_preset` controls, making V2 always-on from the user perspective

**Problem:**
1. **Fragile dependency:** Repo-root scripts are a packaging footgun and make the pipeline fragile
2. **No CLI escape hatch:** Users cannot disable V2 via command-line flags
3. **Poor separation of concerns:** V2 enhancement couples depth pipeline to legacy script infrastructure

---

## Decision

**Make V2 enhancement stage optional with proper CLI controls and graceful degradation.**

### 1. Tactical Fix: Create V2 Enhancement Script Entrypoint

Create `scripts/enhance_image.py` with a minimal implementation that:
- Accepts all arguments expected by `V2Runner` (input path, depth-dir, output-dir, preset, device, upscaler, log-file)
- Provides pass-through behavior (copy input → output) as a placeholder
- Emits expected report JSON for pipeline continuity
- Unblocks immediate pipeline usage while allowing iterative V2 implementation

**Rationale:** Preserves current orchestrator behavior while providing immediate unblocking for users.

### 2. Strategic Fix: Expose V2 Controls via CLI

Add CLI flags to expose existing config knobs:
- `--enable-v2 on|off` → maps to `EnhanceConfig.enable_v2`
- `--v2-preset PRESET` → maps to `EnhanceConfig.v2_preset` (optional string)

**Default behavior (backward compatible):**
- `enable_v2=True` (V2 enabled by default)
- `v2_preset="default"` (uses default V2 preset)

**New workflows enabled:**
- `--enable-v2 off` → Skip V2 entirely (PBR-only mode)
- `--v2-preset none` → Skip V2 via preset control
- `--enable-v2 on --v2-preset premium` → Explicit V2 control

### 3. Future Migration Path (Not Implemented in This ADR)

**Recommended but deferred:**
- Replace repo-root script dependency with module entrypoint
- Move V2 enhancement logic into `src/transformation_portal/lux_depth_v3/v2_enhance.py`
- Update `V2Runner` to call module function instead of subprocess script
- Remove `scripts/enhance_image.py` dependency entirely

**Why defer:** Requires deeper V2 implementation review and refactoring. Current fix provides immediate unblocking while preserving options for future improvement.

---

## Constraints

### Backward Compatibility
- Default behavior remains unchanged: V2 enabled by default
- Existing test suite validates fail-fast behavior by explicitly simulating a missing script (e.g., by patching `V2Runner.script_path.exists()` / `Path.exists()` or temporarily renaming `scripts/enhance_image.py` during the test run)
- New CLI flags are additive (no breaking changes)

### Security
- Script must validate all inputs (paths, arguments)
- No shell injection via subprocess (already mitigated by `V2Runner` using list-based subprocess invocation)
- Report JSON should be validated before consumption

### Maintainability
- Script should be minimal and self-contained
- Clear separation between "placeholder" and "full implementation"
- Documentation must explain migration path

---

## Proposed Design

### `scripts/enhance_image.py` Structure

```python
#!/usr/bin/env python3
"""V2 Enhancement Script Entrypoint.

Minimal implementation providing pipeline continuity.
Replace with full enhancement logic as needed.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 enhancement entrypoint")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--depth-dir", type=Path, required=False, default=None)  # Optional; V2 may generate depth internally or proceed without it
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upscaler", default="default")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    # Validate inputs
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input_path}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pass-through: copy input → output
    output_path = args.output_dir / args.input_path.name
    shutil.copy2(args.input_path, output_path)

    # Emit report JSON
    report = {
        "input": str(args.input_path),
        "output": str(output_path),
        "preset": args.preset,
        "device": args.device,
        "upscaler": args.upscaler,
        "depth_dir": str(args.depth_dir),
        "status": "passthrough",
        "timestamp": time.time(),
    }
    report_path = args.output_dir / f"{args.input_path.stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### CLI Flag Additions (`__main__.py`)

```python
# V2 Enhancement Stage
enable_v2: str = typer.Option(
    "on",
    "--enable-v2",
    help="Enable V2 enhancement stage: on/off (default: on)"
)
v2_preset: Optional[str] = typer.Option(
    "default",
    "--v2-preset",
    help="V2 enhancement preset (use 'none' to skip V2)"
)
```

**Parsing:**
```python
# Parse V2 controls
enable_v2_bool = _parse_bool_flag(enable_v2)
v2_preset_value = None if v2_preset == "none" else v2_preset

config = EnhanceConfig(
    enable_v2=enable_v2_bool,
    v2_preset=v2_preset_value,
    # ... other fields
)
```

---

## Alternatives Considered

### Alternative 1: Remove V2 Entirely
**Rejected:** V2 enhancement provides value for depth-aware upscaling workflows. Removing it would break existing users who depend on this functionality.

### Alternative 2: Make V2 Opt-In (enable_v2=False by default)
**Rejected:** Breaking change for existing users who expect V2 to run by default. Backward compatibility requires opt-out model.

### Alternative 3: Auto-Detect Script and Skip if Missing
**Rejected:** Silent degradation makes debugging harder. Fail-fast is correct behavior when explicitly enabled; optionality is the right lever.

### Alternative 4: Inline V2 Logic into Orchestrator
**Rejected:** Increases coupling and package size. V2 enhancement has different dependency profile than depth inference. Subprocess isolation is appropriate for now.

---

## Consequences / Risks

### Positive
- **Immediate unblocking:** Users can run orchestrator without V2 script
- **PBR-only workflows:** Clear path for users who only want depth + PBR
- **Progressive enhancement:** Placeholder script can be replaced incrementally
- **CLI discoverability:** Flags make V2 control explicit and documented

### Negative
- **Technical debt:** Script-based subprocess invocation remains (deferred migration)
- **Placeholder logic:** Pass-through implementation provides no actual enhancement (documented limitation)
- **Additional flags:** CLI surface area increases (mitigated by clear help text)

### Risks
- **User confusion:** Placeholder behavior may surprise users expecting full V2 enhancement
  - **Mitigation:** Clear documentation, report JSON includes `status: "passthrough"`
- **Breaking future changes:** Script interface becomes de-facto contract
  - **Mitigation:** ADR documents intent to migrate to module entrypoint

---

## Required Enforcement

### Tests
1. Existing V2 validation tests must continue to pass:
   - `test_orchestrator_v2_validation.py::TestV2ValidationFailFast`
   - `test_v2_runner.py::TestV2RunnerExecution`

2. New CLI tests:
   - `--enable-v2 off` skips V2 runner initialization
   - `--v2-preset none` skips V2 runner initialization
   - Default behavior matches current (V2 enabled)

### CI Gates
- No changes to CI required (script creation unblocks existing tests)
- Linting: Ensure script passes `flake8` and `pylint`
- Security: Validate input paths in script (prevent path traversal)

### Documentation
- Update `README.md` with V2 control examples
- Update `__main__.py` docstring with corrected APEX commands
- Add migration guidance for users transitioning to PBR-only mode

---

## Migration Plan

### Phase 1: Immediate (This ADR)
1. Create `scripts/enhance_image.py` with pass-through implementation
2. Add `--enable-v2` and `--v2-preset` CLI flags
3. Update CLI documentation and examples
4. Validate all existing tests pass

### Phase 2: Future Enhancement (Separate ADR)
1. Implement full V2 enhancement logic in script or module
2. Consider migration to module entrypoint pattern
3. Evaluate deprecation of script-based invocation
4. Assess whether V2 should remain opt-out or transition to opt-in

---

## References

- Issue: V2 enhancement orchestrator crashes when script missing
- Code: `src/transformation_portal/lux_depth_v3/orchestrator.py` (lines 215-226)
- Code: `src/transformation_portal/lux_depth_v3/v2_runner.py`
- Tests: `tests/test_orchestrator_v2_validation.py`
- CLI: `src/transformation_portal/lux_depth_v3/__main__.py`

---

## Approval

**Architect Decision:** Accepted
**Rationale:** Tactical fix provides immediate unblocking while preserving strategic options for future refactoring. Backward compatible, security-safe, and maintainable.

**Implementation Priority:** High (blocking user workflows)
**Complexity:** Low (script creation + 2 CLI flags)
**Risk:** Low (backward compatible, well-tested)
