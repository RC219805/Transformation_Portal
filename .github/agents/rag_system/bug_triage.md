# Bug Triage Template

**Use this template for**: Debugging pipeline errors, import issues, FFmpeg problems, runtime errors, performance issues

---

## Bug Report

**Bug Title**: `{BUG_TITLE}`

**Severity**: 
- [ ] Critical (system crash, data loss)
- [ ] High (feature broken, no workaround)
- [ ] Medium (feature degraded, workaround exists)
- [ ] Low (cosmetic, minor inconvenience)

**Affected Component**:
- [ ] Depth Pipeline (`depth_pipeline/`)
- [ ] Lux Render Pipeline (`lux_render_pipeline.py`)
- [ ] Video Master Grader (`luxury_video_master_grader.py`)
- [ ] TIFF Batch Processor (`luxury_tiff_batch_processor.py`)
- [ ] Material Response (`material_response.py`)
- [ ] CI/CD Workflows (`.github/workflows/`)
- [ ] Other: `{SPECIFY}`

---

## Error Information

### Error Log
```
{PASTE_FULL_ERROR_LOG_HERE}
```

### Stack Trace
```python
{PASTE_STACK_TRACE_IF_AVAILABLE}
```

### Reproduction Steps
1. {Step 1}
2. {Step 2}
3. {Step 3}
4. Observe error: {ERROR_DESCRIPTION}

**Minimal Reproducible Example**:
```python
# Minimal code to reproduce the issue
{MINIMAL_CODE_EXAMPLE}
```

### Environment
- **OS**: `{macOS/Linux/Windows}` 
- **OS Version**: `{14.1/Ubuntu 22.04/Windows 11}`
- **Python Version**: `{3.10/3.11/3.12}`
- **Package Versions**:
  ```
  {PASTE_OUTPUT_OF: pip list | grep -E "(torch|PIL|numpy|ffmpeg)"}
  ```
- **Hardware**: 
  - CPU: `{CPU_MODEL}`
  - GPU: `{GPU_MODEL}` (or None)
  - RAM: `{TOTAL_RAM_GB}GB`
  - Apple Silicon: `{M1/M2/M3/M4}` (or N/A)

### Expected Behavior
```
{WHAT_SHOULD_HAPPEN}
```

### Actual Behavior
```
{WHAT_ACTUALLY_HAPPENS}
```

### Additional Context
- Does it work in previous version? `{YES/NO/UNKNOWN}`
- Previous working commit: `{COMMIT_SHA}` (if known)
- Frequency: `{ALWAYS/SOMETIMES/RARE}`
- File size/type: `{e.g., 4K TIFF 16-bit, 1080p MP4}`
- Recent changes: `{ANY_RECENT_CHANGES_TO_CODE_OR_ENVIRONMENT}`

---

## Error Classification

### Error Type
- [ ] **ImportError** - Missing module/package
- [ ] **RuntimeError** - Error during execution
- [ ] **ValueError** - Invalid parameter value
- [ ] **TypeError** - Incorrect type passed
- [ ] **FileNotFoundError** - Missing input file or LUT
- [ ] **MemoryError** - Out of memory (OOM)
- [ ] **FFmpegError** - FFmpeg command failed
- [ ] **ModelLoadError** - ML model failed to load
- [ ] **DepthEstimationError** - Depth pipeline failure
- [ ] **MetadataError** - Metadata preservation issue
- [ ] **PerformanceIssue** - Too slow or high memory usage
- [ ] **Other**: `{SPECIFY}`

### Probable Cause Categories
- [ ] **Dependency Issue** - Missing or incompatible package
- [ ] **Configuration Error** - Invalid config or preset
- [ ] **Resource Limitation** - Insufficient memory/GPU
- [ ] **Input Validation** - Invalid input file or parameters
- [ ] **Logic Bug** - Incorrect algorithm implementation
- [ ] **Integration Issue** - Components not working together
- [ ] **Environment Issue** - OS/hardware specific problem

---

## Root Cause Analysis

### Hypothesis
```
{PRIMARY_HYPOTHESIS_FOR_ROOT_CAUSE}
```

### Evidence Supporting Hypothesis
1. {Evidence 1}
2. {Evidence 2}
3. {Evidence 3}

### Files Likely Involved
- [ ] `{FILE_PATH_1}` - Function: `{FUNCTION_NAME}` - Line: `{LINE_NUMBER}`
- [ ] `{FILE_PATH_2}` - Class: `{CLASS_NAME}` - Method: `{METHOD_NAME}`
- [ ] `{FILE_PATH_3}` - Configuration: `{CONFIG_SECTION}`

### Why the Error Occurs
```
{DETAILED_EXPLANATION_OF_WHY_ERROR_HAPPENS}

Example:
The error occurs because the depth estimation model tries to load CoreML 
weights on a non-Apple Silicon system. The code doesn't check for MPS 
availability before attempting CoreML initialization, causing a runtime error.
```

### Related Issues/PRs
- Similar issue: `#{ISSUE_NUMBER}` - {BRIEF_DESCRIPTION}
- Related PR: `#{PR_NUMBER}` - {BRIEF_DESCRIPTION}

---

## Fix Strategy

### Approach 1 (Recommended): {APPROACH_NAME}
**Pros**:
- ✓ {Advantage 1}
- ✓ {Advantage 2}

**Cons**:
- ✗ {Disadvantage 1}
- ✗ {Disadvantage 2}

**Implementation**:
```python
# Pseudocode or actual code for fix
{FIX_CODE_APPROACH_1}
```

**Files to Modify**:
- [ ] `{FILE_PATH}` - {MODIFICATION_DESCRIPTION}

### Approach 2 (Alternative): {APPROACH_NAME}
**Pros**:
- ✓ {Advantage 1}

**Cons**:
- ✗ {Disadvantage 1}

**Implementation**:
```python
{FIX_CODE_APPROACH_2}
```

### Recommended Approach
**Choose**: Approach `{1/2}` because `{REASONING}`

---

## Fix Implementation

### Code Changes

**File 1**: `{FILE_PATH}`

**Current Code** (buggy):
```python
# Line {START_LINE}-{END_LINE}
{CURRENT_BUGGY_CODE}
```

**Fixed Code**:
```python
# Line {START_LINE}-{END_LINE}
{FIXED_CODE}
```

**Explanation**: 
```
{WHY_THIS_FIX_WORKS}
```

---

**File 2**: `{FILE_PATH}` (if applicable)

**Current Code**:
```python
{CURRENT_CODE}
```

**Fixed Code**:
```python
{FIXED_CODE}
```

---

### Unified Diff Patch

```diff
--- a/{FILE_PATH}
+++ b/{FILE_PATH}
@@ -{OLD_LINE_START},{OLD_LINE_COUNT} +{NEW_LINE_START},{NEW_LINE_COUNT} @@
-{REMOVED_LINE_1}
-{REMOVED_LINE_2}
+{ADDED_LINE_1}
+{ADDED_LINE_2}
```

---

## Testing Strategy

### Manual Testing
**Test 1: Reproduce the bug**
```bash
# Commands to reproduce original bug
{REPRO_COMMANDS}
```
**Expected**: Error occurs ✗

**Test 2: Verify fix**
```bash
# Commands to verify fix works
{VERIFICATION_COMMANDS}
```
**Expected**: No error, correct output ✓

**Test 3: Edge cases**
```bash
# Test edge cases to ensure robustness
{EDGE_CASE_COMMANDS}
```

### Automated Tests

**Regression Test**:
```python
# tests/test_regression_{bug_id}.py
import pytest
from {module} import {function_or_class}

def test_bug_{bug_id}_fixed():
    """
    Regression test for bug #{BUG_ID}: {BUG_TITLE}
    
    Previously failed with: {ERROR_MESSAGE}
    Now should: {EXPECTED_BEHAVIOR}
    """
    # Setup
    {TEST_SETUP}
    
    # Execute (should not raise)
    result = {function_or_class}({PARAMETERS})
    
    # Verify
    assert result is not None
    assert {SPECIFIC_ASSERTION}
```

**Edge Case Tests**:
```python
def test_{bug_name}_edge_case_1():
    """Test edge case: {EDGE_CASE_DESCRIPTION}"""
    # Test code
    pass

def test_{bug_name}_edge_case_2():
    """Test edge case: {EDGE_CASE_DESCRIPTION}"""
    # Test code
    pass
```

### Tests to Run
- [ ] `pytest tests/test_regression_{bug_id}.py -v`
- [ ] `pytest tests/test_{affected_module}.py -v`
- [ ] `pytest tests/integration/ -k {relevant_keyword} -v`
- [ ] Full test suite: `pytest tests/ -v`
- [ ] Linting: `flake8 {modified_files}`

---

## Potential Side Effects

### Files That May Be Affected
- [ ] `{FILE_1}` - {REASON_WHY_AFFECTED}
- [ ] `{FILE_2}` - {REASON_WHY_AFFECTED}

### Compatibility Concerns
- [ ] Breaks backward compatibility? `{YES/NO}` - {EXPLANATION}
- [ ] Requires migration? `{YES/NO}` - {MIGRATION_STEPS}
- [ ] Affects existing presets? `{YES/NO}` - {WHICH_PRESETS}

### Performance Impact
- [ ] Processing time change: `{+X%/-X%/NEUTRAL}`
- [ ] Memory usage change: `{+XMB/-XMB/NEUTRAL}`
- [ ] Requires re-profiling? `{YES/NO}`

---

## Validation Checklist

### Before Fix
- [ ] Confirmed bug reproduction locally
- [ ] Identified root cause
- [ ] Reviewed related code for similar issues
- [ ] Checked if bug exists in other pipelines

### After Fix
- [ ] Bug no longer reproducible
- [ ] All existing tests pass
- [ ] New regression test added and passes
- [ ] Edge cases tested
- [ ] No new flake8/pylint errors
- [ ] Performance not degraded
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated with bug fix note

### CI/CD
- [ ] All CI checks pass
- [ ] Tests pass on Python 3.10, 3.11, 3.12
- [ ] Tests pass on Ubuntu and macOS (if applicable)
- [ ] No new security vulnerabilities

---

## Response Format (JSON Schema)

```json
{
  "summary": "Fix {BUG_TITLE}: {ONE_LINE_FIX_DESCRIPTION}",
  "files": [
    {
      "path": "{FILE_PATH}",
      "patch": "@@ -10,5 +10,5 @@\n-old code\n+new code",
      "description": "Add input validation to prevent {ERROR_TYPE}"
    }
  ],
  "tests": [
    "tests/test_regression_{bug_id}.py",
    "tests/test_{affected_module}.py::test_edge_case"
  ],
  "explanation": "Root cause was {ROOT_CAUSE}. Fixed by {FIX_EXPLANATION}. Added input validation and graceful fallback to prevent future occurrences.",
  "confidence": 0.90,
  "citations": [
    {
      "file_path": "{RELATED_FILE}",
      "snippet": "{SIMILAR_ERROR_HANDLING_PATTERN}",
      "relevance": "Shows correct error handling pattern already in use"
    }
  ]
}
```

---

## Common Bug Patterns & Fixes

### Pattern 1: ImportError - Missing Optional Dependency

**Error**:
```python
ImportError: No module named 'tifffile'
```

**Root Cause**: Optional dependency not installed, code assumes it's available

**Fix**:
```python
# Before (broken)
import tifffile
image = tifffile.imread(path)

# After (fixed with graceful fallback)
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    import warnings
    warnings.warn("tifffile not available, using Pillow (8-bit only)")

# Usage
if TIFFFILE_AVAILABLE:
    image = tifffile.imread(path)
else:
    from PIL import Image
    image = np.array(Image.open(path))
```

**Tests**:
```python
def test_works_without_tifffile(monkeypatch):
    """Test that code works when tifffile is not available."""
    # Mock tifffile import to raise ImportError
    import sys
    monkeypatch.setitem(sys.modules, 'tifffile', None)
    
    # Should fall back to Pillow without crashing
    result = process_image(test_image_path)
    assert result is not None
```

### Pattern 2: FFmpeg Command Failed

**Error**:
```
FFmpegError: Command failed with exit code 1
ffmpeg ... Invalid filter graph: ...
```

**Root Cause**: Invalid FFmpeg filter syntax, missing file, or incompatible format

**Fix**:
```python
# Before (no validation)
cmd = f"ffmpeg -i {input_path} -vf {filter_graph} {output_path}"
subprocess.run(cmd, shell=True, check=True)

# After (with validation and error handling)
import shlex

def validate_filter_graph(filter_graph: str) -> bool:
    """Validate FFmpeg filter graph syntax."""
    # Use ffmpeg -h filter=<name> to validate
    return True  # Simplified

def build_safe_command(input_path: Path, output_path: Path, filters: str) -> List[str]:
    """Build safe FFmpeg command with proper escaping."""
    return [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', filters,
        '-y',  # Overwrite output
        str(output_path)
    ]

# Usage with error handling
try:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not validate_filter_graph(filter_graph):
        raise ValueError(f"Invalid filter graph: {filter_graph}")
    
    cmd = build_safe_command(input_path, output_path, filter_graph)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
except subprocess.CalledProcessError as e:
    logger.error(f"FFmpeg failed: {e.stderr}")
    raise FFmpegError(f"Processing failed: {e.stderr}")
```

### Pattern 3: Out of Memory (OOM)

**Error**:
```python
MemoryError: Unable to allocate array with shape (8192, 8192, 3)
```

**Root Cause**: Processing very large images without batching or streaming

**Fix**:
```python
# Before (loads entire image into memory)
image = np.array(Image.open(large_image_path))
result = apply_effect(image)  # OOM on 8K+ images

# After (tile-based processing for large images)
def process_large_image(image_path: Path, tile_size: int = 2048) -> np.ndarray:
    """Process large images using tiling to reduce memory usage."""
    from PIL import Image
    
    with Image.open(image_path) as img:
        width, height = img.size
        
        # If image is small enough, process normally
        if width <= tile_size and height <= tile_size:
            return apply_effect(np.array(img))
        
        # Otherwise, process in tiles
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract tile
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                tile = img.crop((x, y, x + tile_width, y + tile_height))
                tile_array = np.array(tile)
                
                # Process tile
                processed_tile = apply_effect(tile_array)
                
                # Store result
                result[y:y+tile_height, x:x+tile_width] = processed_tile
                
                # Free memory
                del tile, tile_array, processed_tile
        
        return result
```

### Pattern 4: Depth Model Loading on Non-Apple Hardware

**Error**:
```python
RuntimeError: CoreML not available on this platform
```

**Root Cause**: Code tries to load CoreML model on non-macOS system

**Fix**:
```python
# Before (assumes CoreML available)
from depth_pipeline.models import load_coreml_model
model = load_coreml_model("depth_anything_v2")

# After (with platform detection)
import platform
import torch

def load_depth_model(model_name: str = "depth_anything_v2"):
    """Load depth model with platform-appropriate backend."""
    
    # Check for Apple Silicon with MPS
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        try:
            from depth_pipeline.models import load_coreml_model
            return load_coreml_model(model_name)
        except Exception as e:
            logger.warning(f"CoreML loading failed: {e}, falling back to PyTorch")
    
    # Check for CUDA
    if torch.cuda.is_available():
        from depth_pipeline.models import load_cuda_model
        return load_cuda_model(model_name)
    
    # CPU fallback
    logger.info("Using CPU backend (slower)")
    from depth_pipeline.models import load_cpu_model
    return load_cpu_model(model_name)

# Usage
model = load_depth_model()  # Automatically selects best backend
```

### Pattern 5: Metadata Loss After Processing

**Error**: GPS coordinates or IPTC data missing in output files

**Root Cause**: Processing pipeline doesn't preserve Image.info dict

**Fix**:
```python
# Before (metadata lost)
image = Image.open(input_path)
processed = apply_filters(image)
processed.save(output_path)

# After (metadata preserved)
def process_with_metadata(input_path: Path, output_path: Path):
    """Process image while preserving all metadata."""
    from PIL import Image
    import piexif
    
    # Load with metadata
    image = Image.open(input_path)
    original_info = image.info.copy()
    
    # Extract EXIF if available
    exif_dict = None
    if 'exif' in original_info:
        exif_dict = piexif.load(original_info['exif'])
    
    # Process
    processed = apply_filters(image)
    
    # Restore metadata
    processed.info = original_info
    
    # Save with EXIF
    if exif_dict:
        exif_bytes = piexif.dump(exif_dict)
        processed.save(output_path, exif=exif_bytes)
    else:
        processed.save(output_path)
```

---

## Few-Shot Examples from Repository

### Example 1: ImportError - Missing tifffile

**Error Log**:
```
Traceback (most recent call last):
  File "luxury_tiff_batch_processor.py", line 15, in <module>
    import tifffile
ImportError: No module named 'tifffile'
```

**Environment**: Python 3.10, Ubuntu 20.04

**Output**:
```json
{
  "summary": "Fix ImportError: Add graceful fallback for missing tifffile dependency",
  "files": [
    {
      "path": "luxury_tiff_batch_processor.py",
      "patch": "@@ -12,7 +12,14 @@\n import numpy as np\n from PIL import Image\n-import tifffile\n+\n+# Optional dependency with fallback\n+try:\n+    import tifffile\n+    TIFFFILE_AVAILABLE = True\n+except ImportError:\n+    TIFFFILE_AVAILABLE = False\n+    warnings.warn('tifffile not available, using Pillow (8-bit only)')",
      "description": "Make tifffile optional with graceful fallback to Pillow"
    }
  ],
  "tests": [
    "tests/test_luxury_tiff_processor.py::test_works_without_tifffile"
  ],
  "explanation": "tifffile is an optional dependency for 16-bit TIFF support. When not available, the code should fall back to Pillow with a warning message. This matches the pattern used in other scripts like lux_render_pipeline.py.",
  "confidence": 0.90,
  "citations": [
    {
      "file_path": "lux_render_pipeline.py",
      "snippet": "try:\n    import tifffile\nexcept ImportError:\n    tifffile = None",
      "relevance": "Shows existing optional dependency pattern in repository"
    }
  ]
}
```

### Example 2: FFmpeg Filter Syntax Error

**Error Log**:
```
[Parsed_lut3d_0 @ 0x7f8a1c004000] Cannot find file 'missing_lut.cube'
Error while opening filter 'lut3d'
```

**Reproduction Steps**:
1. Run: `python luxury_video_master_grader.py input.mp4 output/ --preset custom`
2. Preset references non-existent LUT file
3. FFmpeg command fails

**Output**:
```json
{
  "summary": "Fix FFmpegError: Add LUT file existence validation before processing",
  "files": [
    {
      "path": "luxury_video_master_grader.py",
      "patch": "@@ -145,6 +145,11 @@\n def build_filter_graph(preset: PresetConfig) -> str:\n     \"\"\"Build FFmpeg filter graph from preset.\"\"\"\n     filters = []\n+    \n+    # Validate LUT file exists\n+    if preset.lut and not Path(preset.lut).exists():\n+        raise FileNotFoundError(f\"LUT file not found: {preset.lut}\")\n+    \n     if preset.lut:\n         filters.append(f\"lut3d='{preset.lut}':interp=trilinear\")",
      "description": "Add validation to check LUT file exists before building FFmpeg command"
    }
  ],
  "tests": [
    "tests/test_luxury_video_master_grader.py::test_missing_lut_raises_error",
    "tests/test_luxury_video_master_grader.py::test_build_filter_graph_validates_lut"
  ],
  "explanation": "Root cause: FFmpeg command is built without validating that referenced LUT files exist. By adding early validation before command construction, we provide clear error messages and fail fast. This prevents cryptic FFmpeg errors and saves processing time.",
  "confidence": 0.95,
  "citations": [
    {
      "file_path": "tests/test_luxury_video_master_grader.py",
      "snippet": "def test_preset_validation():\n    # Existing validation pattern",
      "relevance": "Shows testing pattern for validation logic"
    }
  ]
}
```

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
