# ADR-002: Security Input Validation

**Date**: December 7, 2025  
**Status**: Proposed  
**Priority**: Critical  
**Architect**: Transformation Portal Security Team

## Context

The Transformation Portal processes user-supplied files (images, videos, configuration files) across multiple pipelines. Current implementations lack consistent input validation, creating potential security vulnerabilities:

**Identified Risks:**

1. **Path Traversal** (CWE-22): Unvalidated file paths allow access outside intended directories
   ```python
   # Current vulnerable pattern
   def process_image(filename: str):
       img = Image.open(filename)  # No validation
   ```

2. **Command Injection** (CWE-78): FFmpeg commands constructed with string formatting
   ```python
   # Current vulnerable pattern
   cmd = f"ffmpeg -i {input_file} -vf {filter_graph} {output_file}"
   subprocess.run(cmd, shell=True)  # Allows shell metacharacter injection
   ```

3. **Resource Exhaustion** (CWE-400): No file size limits or processing timeouts
   - A malicious 10GB TIFF could exhaust memory
   - Infinite loops in custom video filters

**Real-World Attack Scenarios:**

**Scenario 1: Path Traversal**
```python
# Attacker input
malicious_path = "../../../etc/passwd"
process_image(malicious_path)  # Reads sensitive file
```

**Scenario 2: FFmpeg Command Injection**
```python
# Attacker-controlled filter
malicious_filter = "scale=1920:1080; $(rm -rf /)"
# Results in: ffmpeg -i input.mp4 -vf "scale=1920:1080; $(rm -rf /)" output.mp4
```

**Scope**: 
- 15+ files accepting user file paths
- 3 files constructing FFmpeg commands
- 0 files with comprehensive input validation

## Decision

Implement a **defense-in-depth security validation layer** with three components:

### 1. Centralized Path Validation

**Create**: `src/transformation_portal/utils/security.py`

```python
from pathlib import Path
from typing import List, Optional

class SecurityError(Exception):
    """Raised when security validation fails."""
    pass

def validate_filepath(
    filepath: Path,
    allowed_dirs: List[Path],
    max_file_size: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate file path against security constraints.
    
    Args:
        filepath: Path to validate
        allowed_dirs: List of allowed parent directories
        max_file_size: Maximum file size in bytes (None = no limit)
        allowed_extensions: Whitelist of allowed extensions
        
    Returns:
        Resolved, validated Path
        
    Raises:
        SecurityError: If validation fails
        
    Security Guarantees:
    - Prevents path traversal (../../)
    - Validates file exists within allowed directories
    - Checks file size limits
    - Validates file extension whitelist
    - Resolves symlinks to real path
    """
    # Resolve to absolute path (follows symlinks)
    try:
        resolved = filepath.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Cannot resolve path {filepath}: {e}")
    
    # Check file exists
    if not resolved.exists():
        raise SecurityError(f"File does not exist: {filepath}")
    
    # Validate within allowed directories
    if not any(resolved.is_relative_to(d) for d in allowed_dirs):
        raise SecurityError(
            f"Path {filepath} outside allowed directories: {allowed_dirs}"
        )
    
    # Check file size
    if max_file_size and resolved.stat().st_size > max_file_size:
        raise SecurityError(
            f"File {filepath} exceeds size limit: "
            f"{resolved.stat().st_size} > {max_file_size}"
        )
    
    # Check extension
    if allowed_extensions and resolved.suffix.lower() not in allowed_extensions:
        raise SecurityError(
            f"File extension {resolved.suffix} not in whitelist: {allowed_extensions}"
        )
    
    return resolved
```

### 2. Safe FFmpeg Command Construction

```python
from typing import List
import shlex

def build_ffmpeg_command(
    input_file: Path,
    output_file: Path,
    filters: List[str],
    codec: str = "libx264",
    additional_args: Optional[List[str]] = None
) -> List[str]:
    """
    Build FFmpeg command with injection-safe argument list.
    
    Args:
        input_file: Validated input path
        output_file: Validated output path
        filters: List of filter strings (will be validated)
        codec: Output codec
        additional_args: Additional FFmpeg arguments
        
    Returns:
        List of command arguments (no shell required)
        
    Security:
    - No shell=True subprocess execution
    - Arguments passed as list (no string parsing)
    - Filter graph validated against injection patterns
    """
    # Validate filter strings don't contain shell metacharacters
    dangerous_chars = set(";&|`$()<>")
    for filter_str in filters:
        if any(char in filter_str for char in dangerous_chars):
            raise SecurityError(
                f"Filter contains dangerous characters: {filter_str}"
            )
    
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", codec,
        str(output_file)
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd

# Usage
cmd = build_ffmpeg_command(input_path, output_path, ["scale=1920:1080"])
subprocess.run(cmd, check=True)  # Safe: no shell=True
```

### 3. Resource Limits

```python
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""
    pass

@contextmanager
def timeout(seconds: int):
    """
    Context manager for operation timeout.
    
    Usage:
        with timeout(30):
            slow_operation()  # Raises TimeoutError after 30s
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds}s timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# File size constants
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_VIDEO_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
PROCESSING_TIMEOUT = 300  # 5 minutes
```

## Consequences

### Positive

1. **Eliminates path traversal vulnerabilities** across all file operations
2. **Prevents command injection** in FFmpeg/subprocess calls
3. **Resource exhaustion protection** via size/time limits
4. **Consistent security model** - one validation layer for all modules
5. **Audit trail** - security errors logged centrally
6. **Compliance ready** - meets OWASP Top 10 requirements

### Negative

1. **Performance overhead** - Path resolution adds ~1ms per file
2. **Breaking changes** - All file-accepting functions need signature updates
3. **Configuration complexity** - Must define allowed directories per environment
4. **False positives possible** - Legitimate files may be rejected if config wrong

### Mitigation Strategies

**For performance**: Cache validated paths with LRU cache
**For breaking changes**: Gradual rollout with deprecation warnings
**For configuration**: Provide sensible defaults per module

## Implementation Plan

### Week 1: Core Security Module
1. Implement `utils/security.py` with validation functions
2. Add comprehensive unit tests (90%+ coverage)
3. Document security guarantees

### Week 2: Pipeline Migration
1. Update image processing pipelines
   - `lux_render_pipeline.py`
   - `material_response/core.py`
2. Update video processing
   - `luxury_video_master_grader.py`
3. Add integration tests

### Week 3: FFmpeg Hardening
1. Replace all `shell=True` with list arguments
2. Implement `build_ffmpeg_command()`
3. Audit all subprocess calls

### Week 4: Resource Limits & Monitoring
1. Add timeout decorators
2. Implement file size checking
3. Add security event logging

## Validation Criteria

**Security Tests**:
```python
def test_path_traversal_blocked():
    """Ensure path traversal attempts are blocked."""
    malicious_paths = [
        Path("../../etc/passwd"),
        Path("/etc/passwd"),
        Path("../../../../root/.ssh/id_rsa"),
    ]
    
    for path in malicious_paths:
        with pytest.raises(SecurityError):
            validate_filepath(path, [Path("/allowed/dir")])

def test_ffmpeg_injection_blocked():
    """Ensure FFmpeg command injection is blocked."""
    malicious_filters = [
        "scale=1920:1080; $(rm -rf /)",
        "scale=1920:1080 | cat /etc/passwd",
        "scale=1920:1080 && echo pwned",
    ]
    
    for filter_str in malicious_filters:
        with pytest.raises(SecurityError):
            build_ffmpeg_command(
                Path("input.mp4"),
                Path("output.mp4"),
                [filter_str]
            )

def test_file_size_limit_enforced():
    """Ensure file size limits are enforced."""
    # Create 200MB test file
    large_file = create_test_file(200 * 1024 * 1024)
    
    with pytest.raises(SecurityError, match="exceeds size limit"):
        validate_filepath(
            large_file,
            [Path("/tmp")],
            max_file_size=100 * 1024 * 1024
        )
```

**Success Criteria**:
- [ ] All 15+ file-accepting functions use `validate_filepath()`
- [ ] Zero `subprocess.run(..., shell=True)` calls remain
- [ ] Security test suite passes (30+ tests)
- [ ] No new path traversal vulnerabilities in CodeQL scans
- [ ] Documentation updated with security guidelines

## Migration Guide

**Before**:
```python
def process_image(filename: str):
    img = Image.open(filename)
    return transform(img)
```

**After**:
```python
from transformation_portal.utils.security import validate_filepath

def process_image(filename: str, base_dir: Path = Path.cwd()):
    safe_path = validate_filepath(
        Path(filename),
        allowed_dirs=[base_dir],
        max_file_size=MAX_IMAGE_SIZE,
        allowed_extensions=['.jpg', '.png', '.tif', '.tiff']
    )
    img = Image.open(safe_path)
    return transform(img)
```

## Related ADRs

- ADR-001: Module Interface Contracts (security as cross-cutting concern)
- ADR-003: Dependency Management (supply chain security)

## References

- **OWASP Top 10**: A01:2021 - Broken Access Control
- **CWE-22**: Improper Limitation of a Pathname to a Restricted Directory
- **CWE-78**: OS Command Injection
- **Python Security Best Practices**: https://python.readthedocs.io/en/stable/library/subprocess.html#security-considerations

---

**Approval**: Requires immediate implementation (Critical Priority)  
**Implementation**: Week of December 9, 2025  
**Review Date**: January 7, 2026
