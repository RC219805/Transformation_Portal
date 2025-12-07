# Security Hardening Migration Guide

**Version**: 1.0  
**Date**: December 7, 2025  
**Status**: Active  
**Priority**: Critical

This guide provides practical examples for migrating code to use secure validation utilities defined in ADR-002.

## Table of Contents

1. [Overview](#overview)
2. [Security Utilities](#security-utilities)
3. [Migration Examples](#migration-examples)
4. [FFmpeg Security](#ffmpeg-security)
5. [Testing Security](#testing-security)
6. [Common Pitfalls](#common-pitfalls)

---

## Overview

The Transformation Portal provides comprehensive security utilities to prevent:

- **Path Traversal** (CWE-22): Unauthorized file access via `../` attacks
- **Command Injection** (CWE-78): Shell metacharacter exploitation
- **Resource Exhaustion** (CWE-400): File size and timeout limits

### Security Principles

1. **Validate Early**: Check inputs before processing
2. **Fail Safely**: Raise `SecurityError` on violations
3. **No Shell=True**: Use argument lists for subprocess
4. **Explicit Allowlists**: Use whitelists, not blacklists
5. **Defense in Depth**: Multiple validation layers

---

## Security Utilities

### Available Functions

| Function | Purpose | Module |
|----------|---------|--------|
| `validate_filepath()` | General path validation | `utils.security` |
| `validate_image_path()` | Image-specific validation | `utils.security` |
| `validate_video_path()` | Video-specific validation | `utils.security` |
| `validate_config_path()` | Config file validation | `utils.security` |
| `sanitize_filename()` | Remove dangerous characters | `utils.security` |
| `build_safe_command()` | Generic command builder | `utils.security` |
| `build_ffmpeg_command()` | FFmpeg-specific builder | `utils.security` |
| `validate_filter_graph()` | FFmpeg filter validation | `utils.security` |
| `timeout()` | Operation timeout (Unix) | `utils.security` |

### Import Pattern

```python
from transformation_portal.utils.security import (
    SecurityError,
    validate_filepath,
    validate_image_path,
    build_ffmpeg_command,
    timeout,
)
```

---

## Migration Examples

### Example 1: Basic File Path Validation

**Before** (vulnerable to path traversal):
```python
from PIL import Image

def load_image(filepath: str):
    """Load image from path."""
    img = Image.open(filepath)  # ⚠️ No validation!
    return np.array(img)
```

**After** (secure):
```python
from PIL import Image
from pathlib import Path
from transformation_portal.utils.security import validate_image_path, SecurityError

def load_image(filepath: str, base_dir: Path = Path.cwd()):
    """
    Load image from validated path.
    
    Args:
        filepath: Path to image file
        base_dir: Base directory (must contain filepath)
        
    Returns:
        Image as numpy array
        
    Raises:
        SecurityError: If path validation fails
    """
    try:
        safe_path = validate_image_path(
            Path(filepath),
            allowed_dirs=[base_dir]
        )
        img = Image.open(safe_path)
        return np.array(img)
    except SecurityError as e:
        print(f"❌ Security violation: {e}")
        raise
```

**Attack Prevented**:
```python
# Attacker tries path traversal
load_image("../../../etc/passwd")
# ✅ Raises: SecurityError: Path outside allowed directories
```

---

### Example 2: Batch File Processing

**Before**:
```python
def process_directory(input_dir: str, output_dir: str):
    """Process all images in directory."""
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path)
```

**After** (secure):
```python
from pathlib import Path
from transformation_portal.utils.security import (
    validate_filepath,
    sanitize_filename,
    SecurityError,
    IMAGE_EXTENSIONS
)

def process_directory(input_dir: str, output_dir: str):
    """
    Process all images in directory with security validation.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Raises:
        SecurityError: If directory validation fails
    """
    # Validate directories
    input_path = validate_filepath(
        Path(input_dir),
        allowed_dirs=[Path.cwd()],
        must_exist=True
    )
    
    output_path = validate_filepath(
        Path(output_dir),
        allowed_dirs=[Path.cwd()],
        must_exist=False
    )
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files
    for file in input_path.iterdir():
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        
        try:
            # Validate input file
            safe_input = validate_filepath(
                file,
                allowed_dirs=[input_path]
            )
            
            # Sanitize output filename
            safe_name = sanitize_filename(file.name)
            safe_output = output_path / safe_name
            
            # Process
            process_image(safe_input, safe_output)
            
        except SecurityError as e:
            print(f"⚠️  Skipping {file.name}: {e}")
            continue
```

---

### Example 3: User-Supplied Filenames

**Before** (vulnerable):
```python
def save_result(data, user_filename: str):
    """Save result with user-supplied name."""
    with open(user_filename, 'wb') as f:  # ⚠️ Dangerous!
        f.write(data)
```

**After** (secure):
```python
from transformation_portal.utils.security import sanitize_filename, validate_filepath

def save_result(data, user_filename: str, output_dir: Path):
    """
    Save result with sanitized user-supplied name.
    
    Args:
        data: Data to save
        user_filename: User-provided filename (will be sanitized)
        output_dir: Output directory (must be validated)
        
    Returns:
        Path to saved file
    """
    # Sanitize filename
    safe_name = sanitize_filename(user_filename)
    
    # Validate output path
    output_path = validate_filepath(
        output_dir / safe_name,
        allowed_dirs=[output_dir],
        must_exist=False
    )
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(data)
    
    return output_path
```

**Attack Prevented**:
```python
# Attacker tries directory traversal
save_result(data, "../../../tmp/malicious.sh", Path("output/"))
# ✅ Sanitized to: "output/tmp_malicious.sh"
```

---

## FFmpeg Security

### Example 4: Safe FFmpeg Command Construction

**Before** (vulnerable to command injection):
```python
def apply_video_filter(input_file: str, output_file: str, filter_str: str):
    """Apply FFmpeg filter."""
    cmd = f"ffmpeg -i {input_file} -vf {filter_str} {output_file}"
    subprocess.run(cmd, shell=True)  # ⚠️ Command injection risk!
```

**After** (secure):
```python
from pathlib import Path
from transformation_portal.utils.security import build_ffmpeg_command, SecurityError

def apply_video_filter(
    input_file: str,
    output_file: str,
    filter_str: str,
    base_dir: Path = Path.cwd()
):
    """
    Apply FFmpeg filter with security validation.
    
    Args:
        input_file: Input video path
        output_file: Output video path
        filter_str: FFmpeg filter string
        base_dir: Base directory for validation
        
    Raises:
        SecurityError: If validation fails
    """
    try:
        # Build safe command (validates paths and filters)
        cmd = build_ffmpeg_command(
            Path(input_file),
            Path(output_file),
            filters=[filter_str],
            codec="libx264",
            validate_paths=True,
            allowed_dirs=[base_dir]
        )
        
        # Execute safely (no shell=True)
        result = subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Video processed successfully")
        
    except SecurityError as e:
        print(f"❌ Security violation: {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()}")
        raise
```

**Attack Prevented**:
```python
# Attacker tries command injection
apply_video_filter("input.mp4", "output.mp4", "scale=1920:1080; rm -rf /")
# ✅ Raises: SecurityError: Filter contains dangerous characters
```

---

### Example 5: Complex FFmpeg Pipeline

**Before**:
```python
def create_video_master(input_file: str, output_file: str, lut_path: str):
    """Apply LUT and color grading."""
    filter_complex = f"lut3d={lut_path},colorbalance=rs=0.1"
    cmd = f"ffmpeg -i {input_file} -vf '{filter_complex}' -c:v libx264 -preset slow {output_file}"
    subprocess.run(cmd, shell=True)
```

**After** (secure):
```python
from transformation_portal.utils.security import (
    build_ffmpeg_command,
    validate_filepath,
    validate_filter_graph,
    SecurityError
)

def create_video_master(
    input_file: str,
    output_file: str,
    lut_path: str,
    base_dir: Path = Path.cwd()
):
    """
    Apply LUT and color grading with security validation.
    
    Args:
        input_file: Input video path
        output_file: Output video path
        lut_path: Path to LUT file
        base_dir: Base directory for validation
        
    Raises:
        SecurityError: If validation fails
    """
    # Validate LUT path
    safe_lut = validate_filepath(
        Path(lut_path),
        allowed_dirs=[base_dir / "assets" / "luts"],
        allowed_extensions=['.cube']
    )
    
    # Build filter graph
    filter_graph = f"lut3d={safe_lut},colorbalance=rs=0.1"
    
    # Validate filter (checks for shell metacharacters)
    validate_filter_graph(filter_graph)
    
    # Build safe command
    cmd = build_ffmpeg_command(
        Path(input_file),
        Path(output_file),
        filters=[filter_graph],
        codec="libx264",
        additional_args=["-preset", "slow"],
        validate_paths=True,
        allowed_dirs=[base_dir]
    )
    
    # Execute
    subprocess.run(cmd, check=True)
```

---

### Example 6: Timeout for Long Operations

**Before** (no timeout):
```python
def process_video(input_file: str):
    """Process video (may hang forever)."""
    result = subprocess.run(["ffmpeg", "-i", input_file, "output.mp4"])
```

**After** (with timeout):
```python
from transformation_portal.utils.security import timeout, TimeoutError

def process_video(input_file: str, max_duration: int = 300):
    """
    Process video with timeout.
    
    Args:
        input_file: Input video path
        max_duration: Maximum processing time in seconds (default 5 minutes)
        
    Raises:
        TimeoutError: If processing exceeds max_duration
    """
    try:
        with timeout(max_duration):
            result = subprocess.run(
                ["ffmpeg", "-i", input_file, "output.mp4"],
                check=True
            )
            print(f"✅ Processing complete")
            
    except TimeoutError:
        print(f"❌ Processing timed out after {max_duration}s")
        raise
    except NotImplementedError:
        # Windows platform - use alternative timeout
        print("⚠️  timeout() not available on Windows, skipping")
```

**Note**: `timeout()` only works on Unix-like systems (Linux, macOS). On Windows, use `subprocess.run(..., timeout=seconds)`.

---

## Testing Security

### Unit Test Example

```python
# tests/test_secure_processing.py
import pytest
from pathlib import Path
from transformation_portal.utils.security import SecurityError
from my_module import load_image

def test_path_traversal_blocked(tmp_path):
    """Test that path traversal attacks are blocked."""
    # Create a file outside allowed directory
    outside = tmp_path.parent / "outside.jpg"
    outside.touch()
    
    # Try to access via traversal
    with pytest.raises(SecurityError, match="outside allowed"):
        load_image(str(outside), base_dir=tmp_path)

def test_command_injection_blocked():
    """Test that command injection is blocked."""
    from my_module import apply_video_filter
    
    with pytest.raises(SecurityError, match="dangerous characters"):
        apply_video_filter(
            "input.mp4",
            "output.mp4",
            "scale=1920:1080; rm -rf /"
        )

def test_valid_path_accepted(tmp_path):
    """Test that valid paths are accepted."""
    valid_file = tmp_path / "valid.jpg"
    valid_file.write_bytes(b"fake image")
    
    # Should not raise
    result = load_image(str(valid_file), base_dir=tmp_path)
    assert result is not None
```

---

## Common Pitfalls

### Pitfall 1: Using shell=True

**❌ Bad**:
```python
subprocess.run(f"ffmpeg -i {input_file} {output_file}", shell=True)
```

**✅ Good**:
```python
subprocess.run(["ffmpeg", "-i", input_file, output_file], check=True)
```

### Pitfall 2: Trusting User Input

**❌ Bad**:
```python
def save_to_path(data, path: str):
    with open(path, 'wb') as f:  # User controls path!
        f.write(data)
```

**✅ Good**:
```python
def save_to_path(data, filename: str, output_dir: Path):
    safe_name = sanitize_filename(filename)
    safe_path = validate_filepath(
        output_dir / safe_name,
        allowed_dirs=[output_dir],
        must_exist=False
    )
    with open(safe_path, 'wb') as f:
        f.write(data)
```

### Pitfall 3: No File Size Limits

**❌ Bad**:
```python
img = Image.open(user_file)  # Could be 10GB!
```

**✅ Good**:
```python
safe_path = validate_image_path(
    Path(user_file),
    allowed_dirs=[upload_dir],
    max_size=100 * 1024 * 1024  # 100MB limit
)
img = Image.open(safe_path)
```

### Pitfall 4: Blacklisting vs. Whitelisting

**❌ Bad** (blacklist - incomplete):
```python
if '..' in filename or '/' in filename:
    raise ValueError("Invalid filename")
```

**✅ Good** (whitelist):
```python
safe_name = sanitize_filename(filename)  # Removes all dangerous chars
```

---

## Migration Checklist

### For File Operations
- [ ] Replace all `open(filepath)` with `validate_filepath()` + `open()`
- [ ] Use `sanitize_filename()` for user-supplied names
- [ ] Add file size limits via `max_file_size` parameter
- [ ] Use extension whitelists via `allowed_extensions`
- [ ] Define `allowed_dirs` explicitly (no wildcards)

### For FFmpeg Operations
- [ ] Replace `shell=True` with argument lists
- [ ] Use `build_ffmpeg_command()` for safety
- [ ] Validate filter strings with `validate_filter_graph()`
- [ ] Add timeouts for long-running operations
- [ ] Validate LUT/asset paths

### For Testing
- [ ] Add path traversal tests
- [ ] Add command injection tests
- [ ] Add file size limit tests
- [ ] Add extension validation tests
- [ ] Test error handling

---

## Security Audit Tool

Run the security audit to find remaining issues:

```bash
# Find shell=True usages
grep -r "shell=True" --include="*.py" src/ lux_depth_v2/

# Find unvalidated file operations
grep -r "open(" --include="*.py" src/ | grep -v "validate_filepath"

# Run automated security checks
python scripts/security/continuous_security.py
```

---

## References

- [ADR-002: Security Input Validation](adr/ADR-002-security-input-validation.md)
- [Security Module Source](../../src/transformation_portal/utils/security.py)
- [Security Tests](../../tests/test_security_validation.py)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

---

**Questions?** Contact the security team or open an issue.
