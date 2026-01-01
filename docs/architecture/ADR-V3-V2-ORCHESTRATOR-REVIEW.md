# Architecture Decision Record: V3+V2 Orchestrator Integration Review

**Status**: Under Review
**Date**: 2026-01-01
**Reviewer**: Transformation Portal Architect
**PR**: #633 - V3 orchestrator: integrate DA3 depth generation with V2 enhancement pipeline

## Executive Summary

This ADR documents the architectural review of the V3+V2 enhancement orchestrator integration. The implementation successfully maintains clean boundaries between V2 and V3, but requires **critical security hardening** before merge approval.

**Overall Assessment**: ⚠️ **APPROVE WITH REQUIRED CHANGES**

## Architecture Analysis

### ✅ Strengths

1. **Clean Separation of Concerns**
   - V3 handles depth generation only (Stage A)
   - V2 remains canonical for enhancement (Stage B)
   - No code duplication between V3 and V2
   - Subprocess isolation prevents V3/V2 coupling

2. **Contract Compliance**
   - Depth output strictly enforced: uint16 PNG, (H, W) shape
   - Filename convention: `{stem}_depth.png`
   - Robust quantization methods (p1p99, p0.5p99.5, minmax)
   - Shape validation and NaN/Inf detection

3. **Comprehensive Provenance**
   - Combined manifest links all stages
   - Git hashes tracked for reproducibility
   - Timing breakdowns for performance analysis
   - SHA256 hashing for input verification

4. **Robust Error Handling**
   - Three failure policies: `fail`, `skip`, `v2-auto`
   - Timeout handling for subprocess
   - Graceful degradation options
   - Comprehensive logging

5. **Production-Ready Features**
   - Resume support (skip existing outputs)
   - Force flags for regeneration
   - Device pinning for multi-GPU setups
   - License enforcement mechanism

### ⚠️ Security Concerns (MUST FIX)

#### 1. **CRITICAL: Command Injection Risk in V2Runner** 🔴

**Location**: `lux_depth_v3/enhance/v2_runner.py:91-104`

**Issue**: While `subprocess.run()` is used correctly with a list (not shell=True), the `extra_args` parameter allows arbitrary command injection if user-controlled data reaches it.

```python
if extra_args:
    cmd.extend(extra_args)  # ❌ No validation
```

**Risk**: High - An attacker could inject malicious arguments like `--config /etc/passwd` or exploit V2 CLI vulnerabilities.

**Recommendation**:
```python
# Add input validation for extra_args
ALLOWED_EXTRA_ARGS = {'--verbose', '--quiet', '--force'}  # Whitelist
if extra_args:
    for arg in extra_args:
        if not any(arg.startswith(allowed) for allowed in ALLOWED_EXTRA_ARGS):
            raise ValueError(f"Disallowed extra argument: {arg}")
    cmd.extend(extra_args)
```

#### 2. **CRITICAL: Path Traversal Vulnerability** 🔴

**Location**: `lux_depth_v3/enhance/orchestrator.py:124-127`

**Issue**: File paths are constructed from `image_input.path.stem` without validation. A malicious image filename like `../../etc/passwd.jpg` could escape the output directory.

```python
depth_path = self.depth_dir / f"{stem}_depth.png"  # ❌ No sanitization
```

**Risk**: High - Directory traversal could overwrite critical files.

**Recommendation**:
```python
# Add path sanitization
def sanitize_stem(stem: str) -> str:
    """Sanitize file stem to prevent path traversal."""
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[^\w\-.]', '_', stem)
    # Prevent hidden files
    sanitized = sanitized.lstrip('.')
    # Limit length
    return sanitized[:255]

stem = sanitize_stem(image_input.path.stem)
```

#### 3. **HIGH: Subprocess Timeout Exhaustion** 🟡

**Location**: `lux_depth_v3/enhance/v2_runner.py:111-117`

**Issue**: Default timeout is 600 seconds, but no process group cleanup on timeout. A hanging V2 subprocess could leave zombie processes.

**Risk**: Medium - Resource exhaustion in production.

**Recommendation**:
```python
# Add process group management
import signal
import os

# On Unix, use process groups
if sys.platform != 'win32':
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=self.v2_module_path,
        start_new_session=True,  # Create new process group
    )
```

#### 4. **MEDIUM: Git Command Execution** 🟡

**Location**: `lux_depth_v3/enhance/manifest.py:156-169`

**Issue**: `subprocess.run()` executes `git rev-parse HEAD` without validating `repo_path`. An attacker controlling the repository path could inject malicious git hooks.

**Risk**: Medium - Limited attack surface, but defense-in-depth needed.

**Recommendation**:
```python
def get_git_revision(repo_path: Path) -> Optional[str]:
    """Get current git revision for reproducibility."""
    # Validate repo_path is within expected boundaries
    try:
        repo_path = repo_path.resolve()
        # Ensure it's a real git repo
        git_dir = repo_path / '.git'
        if not git_dir.exists():
            return None

        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
            env={'GIT_DIR': str(git_dir)},  # Explicit git directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
```

#### 5. **LOW: Incomplete Input Validation** 🟢

**Location**: `lux_depth_v3/cli.py:1049-1062`

**Issue**: String parameters (`v2_preset`, `depth_device`, `depth_quantization`) are not validated against allowed values.

**Risk**: Low - Could cause confusing errors downstream.

**Recommendation**:
```python
# Add validation in EnhanceConfig dataclass
@dataclass
class EnhanceConfig:
    ALLOWED_DEVICES = {'auto', 'cuda', 'cpu', 'mps'}
    ALLOWED_QUANTIZATIONS = {'p1p99', 'p0.5p99.5', 'minmax'}
    ALLOWED_FALLBACKS = {'fail', 'skip', 'v2-auto'}

    def __post_init__(self):
        if self.depth_device not in self.ALLOWED_DEVICES:
            raise ValueError(f"Invalid depth_device: {self.depth_device}")
        if self.depth_quantization not in self.ALLOWED_QUANTIZATIONS:
            raise ValueError(f"Invalid depth_quantization: {self.depth_quantization}")
        if self.depth_fallback not in self.ALLOWED_FALLBACKS:
            raise ValueError(f"Invalid depth_fallback: {self.depth_fallback}")
```

### ⚠️ Architectural Concerns

#### 1. **MEDIUM: Tight Coupling to File System Structure** 🟡

**Issue**: The orchestrator assumes a rigid directory structure (`depth/`, `v2/`, `manifests/`, `logs/`). Changes to this structure would break resume logic and manifest paths.

**Recommendation**:
- Extract directory paths into a `DirectoryLayout` dataclass
- Make layout configurable for future flexibility
- Document the contract explicitly

#### 2. **MEDIUM: No Atomic Resume Operations** 🟡

**Issue**: Resume logic checks file existence but doesn't verify integrity (no checksums). Partial writes from previous failures will be reused.

**Recommendation**:
- Add checksum validation for depth files before resume
- Write `.partial` files and rename on success
- Consider manifest-based resume instead of file existence

#### 3. **LOW: V2 Subprocess Discovery is Fragile** 🟢

**Location**: `lux_depth_v3/enhance/v2_runner.py:42-57`

**Issue**: Auto-detection assumes `lux_depth_v2` is a sibling directory. This breaks if installed via pip or in monorepo structures.

**Recommendation**:
- Prefer importlib.util.find_spec() for module detection
- Fall back to explicit path only if import fails
- Document expected installation patterns

## Code Quality Assessment

### ✅ Positive Aspects

1. **Excellent Documentation**
   - Comprehensive READMEs, quick start guide, architecture doc
   - Clear docstrings with types and examples
   - Integration status document tracks requirements

2. **Good Test Coverage** (Unit Tests)
   - Depth writer: 8 tests covering edge cases
   - Manifest: 4 tests for serialization
   - V2 runner: Basic initialization tests
   - ⚠️ Integration tests marked skip (acceptable for new feature)

3. **Consistent Code Style**
   - Type hints throughout
   - Dataclasses for configuration
   - Logging at appropriate levels
   - Pathlib usage (not string paths)

### ⚠️ Technical Debt

1. **Missing Type Validation**
   - Runtime validation needed for string enums
   - No mypy/pyright type checking evident

2. **Incomplete Error Context**
   - Exception messages don't always include full context
   - Stack traces not captured in manifests for debugging

3. **No Performance Monitoring**
   - No metrics for memory usage
   - No tracking of batch throughput degradation
   - Missing profiling hooks

## Integration Concerns

### V2 Contract Compliance ✅

**Verified:**
- Depth output format: uint16 PNG ✅
- Filename convention: `{stem}_depth.png` ✅
- Shape: (H, W) exactly ✅
- Single-channel (no RGB depth) ✅
- Subprocess isolation prevents V2 modification ✅

**Risks:**
- V2 CLI changes could break V2Runner
- No versioning/compatibility check for V2

**Recommendation**: Add V2 version detection and compatibility check

### Manifest Schema Versioning ✅

**Good:**
- Schema version: `lux-depth-v3.enhance.v1`
- Forward-compatible deserialization (optional fields)
- Git hashes for provenance

**Missing:**
- No migration path for schema v2
- No deprecation strategy documented

## Deployment Readiness

### ✅ Production-Ready Aspects

1. Comprehensive logging infrastructure
2. Timeout handling for long-running processes
3. Resume support for interrupted workflows
4. License enforcement mechanism
5. Clear error messages and failure modes

### ⚠️ Pre-Production Requirements

1. **Security hardening** (MUST FIX before merge)
2. Integration tests (can defer to follow-up PR)
3. Performance benchmarking (recommended)
4. Load testing for batch processing (recommended)

## Decision

**APPROVE WITH REQUIRED CHANGES**

### Mandatory Changes (Blocking)

1. ✅ Fix command injection risk in `V2Runner.run()` (whitelist extra_args)
2. ✅ Add path sanitization in `EnhanceOrchestrator.enhance_image()`
3. ✅ Improve subprocess cleanup on timeout
4. ✅ Validate git repository paths before execution
5. ✅ Add input validation for CLI string parameters

### Recommended Changes (Non-Blocking)

1. Extract directory layout into configurable dataclass
2. Add checksum-based resume validation
3. Improve V2 module discovery robustness
4. Add V2 version compatibility check
5. Document schema migration strategy

### Follow-Up Work (Future PRs)

1. Integration test infrastructure
2. Performance profiling and optimization
3. Multi-GPU pipelined execution mode
4. Distributed processing support
5. Real-time monitoring dashboard

## References

- **PR**: #633
- **Specialist Implementation**: Transformation Portal Specialist
- **Security Standards**: OWASP Python Security Best Practices
- **Related ADRs**: None (first orchestrator integration)

## Approval Criteria

- [ ] All mandatory security fixes implemented
- [ ] Code review by architect completed
- [ ] Unit tests passing
- [ ] Documentation reviewed
- [ ] Security scan (CodeQL) passing

---

**Next Steps**: Create security hardening tickets and implement fixes.
