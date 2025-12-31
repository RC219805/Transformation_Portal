# DA3 Wrapper Critical Fixes - Production Contract Alignment

**Created**: 2025-12-31
**Status**: BLOCKING - Must fix before 750 Picacho R&D evaluation
**Severity**: High - Current wrapper will fail or silently misbehave

## Executive Summary

Current `da3_wrapper.py` has **6 critical mismatches** with the official DA3 API contract that will cause:
- Silent failures (wrong signatures, unloaded weights)
- Subprocess deadlocks (pipe buffer exhaustion)
- CLI command construction errors (positional vs flag arguments)
- License enforcement gaps (non-commercial constraint not enforced)

All fixes are surgical and do not affect production `lux-depth-v2` APEX pipeline.

---

## Critical Issue #1: Python API Model Loading (HIGHEST PRIORITY)

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:490
model = self.DepthAnything3(model_name=api_model_name)  # ❌ Wrong constructor
```

### Official DA3 Contract
```python
# From official DA3 docs
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
```

### Why This Breaks
- `DepthAnything3(model_name=...)` likely instantiates an **uninitialized** model (config-only)
- No pretrained weights are loaded
- Inference will produce garbage outputs or crash

### Fix (SURGICAL)
```python
def _resolve_hf_id(self) -> str:
    """Map model name to HuggingFace ID."""
    # If already an HF ID (contains slash), use as-is
    if "/" in self.model_name:
        return self.model_name

    # Map our names to HF IDs
    api_name = self.VARIANT_TO_API_NAME.get(self.model_name, self.model_name)

    if api_name in self.AVAILABLE_MODELS:
        return self.AVAILABLE_MODELS[api_name]["hf_id"]

    # Fallback: assume it's a valid API name
    return f"depth-anything/{self.model_name.upper()}"

def _load_model(self):
    """Load DA3 model using official from_pretrained() path."""
    hf_id = self._resolve_hf_id()

    logger.info(f"Loading DA3 model from HuggingFace: {hf_id}")
    model = self.DepthAnything3.from_pretrained(hf_id)
    model = model.to(self.device)
    logger.info(f"Model loaded on {self.device}")
    return model
```

**Impact**: Fixes silent weight loading failure. **REQUIRED** for valid DA3 inference.

---

## Critical Issue #2: Inference Call Signature Mismatch

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:640-650 (approximately)
prediction = self.model.inference(
    image=image_prepared,  # ❌ Keyword argument may not match
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    ...
)
```

### Official DA3 Contract
```python
# Official docs show positional first argument
model.inference(images, ...)  # 'images' not 'image', positional
```

### Fix (SURGICAL)
```python
# Pass images as positional first argument for signature safety
prediction = self.model.inference(
    image_prepared,  # Positional
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir=export_dir,
    export_format=export_format,
)
```

**Impact**: Prevents `TypeError: got unexpected keyword argument 'image'`

---

## Critical Issue #3: CLI Command Construction (POSITIONAL VS FLAGS)

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:194-221
def _build_base_cmd(self, subcommand: str, **kwargs) -> List[str]:
    cmd = ["da3", subcommand]
    # ❌ Converts input_path to --input-path (wrong for DA3 CLI)
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(value)])
```

This produces:
```bash
da3 auto --input-path /path/to/images --export-dir /output  # ❌ WRONG
```

### Official DA3 Contract
```bash
# Official CLI usage (positional input path)
da3 auto <path> --export-dir <dir> --export-format <fmt>
```

### Fix (SURGICAL)
```python
def process_auto(self, input_path: Path, export_dir: Path, export_format: str = "mini_npz", **kwargs) -> Dict[str, Any]:
    """Auto-detect input type and process."""
    # Build command with POSITIONAL input path
    cmd = ["da3", "auto", str(input_path)]

    # Add required flags
    cmd.extend(["--export-dir", str(export_dir)])
    cmd.extend(["--export-format", export_format])

    # Add backend flag if configured
    if self.backend is not None:
        cmd.append("--use-backend")  # Boolean flag, no URL argument

    # Add optional kwargs
    for key, value in kwargs.items():
        if value is not None:
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

    return self._run_command(cmd)
```

**Impact**: Fixes CLI command syntax errors. CLI mode will actually work.

---

## Critical Issue #4: Backend Health Check (UNDOCUMENTED ENDPOINT)

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:163-169
def is_running(self) -> bool:
    """Check if backend is running and healthy."""
    try:
        response = requests.get(f"{self.get_url()}/status", timeout=1)  # ❌ /status not documented
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False
```

### Problem
- DA3 backend does **not** document a `/status` endpoint
- This will always fail with 404, incorrectly reporting "backend not running"

### Fix (SURGICAL - Minimal Viable Check)
```python
def is_running(self) -> bool:
    """Check if backend process is alive.

    Note: DA3 backend does not expose a documented health endpoint.
    We only check if the process is still running.
    """
    if self._process is None:
        return False

    # Check if process is still alive
    return self._process.poll() is None
```

**Alternative (More Robust)**:
```python
def start(self, timeout: int = 30) -> None:
    """Start backend service."""
    if self.is_running():
        return

    # Start backend WITHOUT pipe buffering
    cmd = ["da3", "backend", "--model-dir", self.model_dir]

    print(f"Starting DA3 backend: {' '.join(cmd)}")
    self._process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,  # Don't buffer (prevents deadlock)
        stderr=subprocess.DEVNULL,
    )

    # Simple wait - no health endpoint check
    time.sleep(5)  # Give backend time to initialize

    if not self.is_running():
        raise RuntimeError("Backend process died during startup")

    print("Backend started (process running)")
```

**Impact**: Prevents false negative health checks and subprocess deadlocks.

---

## Critical Issue #5: Subprocess Pipe Buffer Deadlock

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:131-136
self._process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,  # ❌ PIPE without reading = deadlock risk
    stderr=subprocess.PIPE,
    text=True,
)
```

### Problem
- Backend service may write to stdout/stderr continuously
- If buffers fill (typically 64KB), subprocess **blocks forever**
- Parent never reads the pipes → classic deadlock

### Fix (SURGICAL)
```python
# Option 1: Don't buffer at all
self._process = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# Option 2: Drain pipes asynchronously (more complex, only if logs needed)
import threading

def _drain_pipe(pipe, prefix):
    for line in pipe:
        logger.debug(f"[DA3 Backend {prefix}] {line.rstrip()}")

self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
threading.Thread(target=_drain_pipe, args=(self._process.stdout, "OUT"), daemon=True).start()
threading.Thread(target=_drain_pipe, args=(self._process.stderr, "ERR"), daemon=True).start()
```

**Recommendation**: Use Option 1 (DEVNULL) unless backend logs are critical for debugging.

**Impact**: Prevents random hangs during long batch processing.

---

## Critical Issue #6: Placeholder Model Name Collision

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:792
class DepthAnything3(nn.Module):  # ❌ Name collision with official API
    """Placeholder for Depth Anything 3 model."""
```

### Problem
- When official `DepthAnything3` is imported, this class name **shadows** it in this module's namespace
- Confusing for code readers and future maintainers
- Risk of accidental usage of placeholder instead of real model

### Fix (SURGICAL)
```python
class DepthAnything3Placeholder(nn.Module):
    """Placeholder for testing when official DA3 API not available.

    DO NOT USE IN PRODUCTION. Install official API:
    pip install depth-anything-3
    """

    def __init__(self, model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        logger.warning(f"Using DepthAnything3Placeholder for {model_name} - NOT REAL DA3 MODEL")
        # ... rest of placeholder implementation
```

Update wrapper fallback:
```python
# In DepthAnything3Wrapper.__init__
except ImportError:
    self.DepthAnything3 = DepthAnything3Placeholder  # Clear that it's a placeholder
    self.available = False
```

**Impact**: Eliminates name shadowing confusion, prevents accidental placeholder usage.

---

## Critical Issue #7: License Enforcement (NON-COMMERCIAL R&D)

### Current Code (WRONG)
```python
# lux_depth_v3/da3_wrapper.py:439-457
def __init__(
    self,
    model_name: str = "da3-large",
    device: str = "cuda",
    commercial_use: bool = False,  # ❌ Accepted but never enforced
    validate_license_strict: bool = False,
):
    # ... no actual validation code
```

### Problem
- `commercial_use` and `validate_license_strict` parameters exist but **do nothing**
- No protection against accidentally using NC (non-commercial) models in production

### Fix (SURGICAL - Hard Guardrail)
```python
# Non-commercial models (CC-BY-NC-4.0)
NC_MODELS = {
    "da3nested-giant-large",
    "da3-giant",
    "da3-large",
}

def __init__(
    self,
    model_name: str = "da3-large",
    device: str = "cuda",
    commercial_use: bool = False,
    validate_license_strict: bool = False,
):
    """Initialize DA3 wrapper.

    Raises:
        RuntimeError: If commercial_use=True with NC-licensed model in strict mode
    """
    self.model_name = model_name
    self.device = device

    # LICENSE ENFORCEMENT
    api_name = self.VARIANT_TO_API_NAME.get(model_name, model_name)
    is_nc_model = api_name in self.NC_MODELS

    if is_nc_model:
        logger.warning(f"Model {model_name} is CC-BY-NC-4.0 (NON-COMMERCIAL ONLY)")

        if commercial_use:
            msg = (
                f"License violation: {model_name} is non-commercial (CC-BY-NC-4.0)\n"
                f"For commercial use, switch to Apache-2.0 models:\n"
                f"  - da3metric-large (recommended)\n"
                f"  - da3-base\n"
                f"  - da3-small"
            )
            if validate_license_strict:
                raise RuntimeError(msg)
            else:
                logger.error(msg)

    # ... rest of init
```

**Impact**: **Hard stop** on accidental commercial use of NC models. Critical for compliance.

---

## Implementation Priority

1. **CRITICAL (Do First)**:
   - Fix #1: Model loading (`from_pretrained()`)
   - Fix #7: License enforcement
   - Fix #2: Inference signature

2. **HIGH (Before Batch Processing)**:
   - Fix #3: CLI positional arguments
   - Fix #5: Subprocess deadlock

3. **MEDIUM (Code Quality)**:
   - Fix #4: Backend health check
   - Fix #6: Placeholder rename

---

## Testing Protocol After Fixes

```python
# Test 1: Verify real DA3 import and loading
python -c "
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained('depth-anything/DA3-BASE')
print('✓ DA3 API loads correctly')
"

# Test 2: Wrapper with real model
python -c "
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper
wrapper = DepthAnything3Wrapper(model_name='da3-base')
print(f'✓ Wrapper initialized: {wrapper.available}')
"

# Test 3: License enforcement
python -c "
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper
try:
    wrapper = DepthAnything3Wrapper(
        model_name='da3nested-giant-large',
        commercial_use=True,
        validate_license_strict=True
    )
    print('✗ FAIL: Should have raised RuntimeError')
except RuntimeError as e:
    print(f'✓ License guard works: {e}')
"

# Test 4: CLI positional argument
python -c "
from lux_depth_v3.da3_wrapper import DA3CLI
cli = DA3CLI()
# Should build: da3 auto /path/to/imgs --export-dir /out --export-format mini_npz
# NOT: da3 auto --input-path /path/to/imgs ...
"
```

---

## 750 Picacho R&D Workflow (POST-FIX)

Once fixes are applied:

```bash
# 1. Verify DA3 is actually installed
pip list | grep depth-anything

# 2. Create R&D-only directory structure
mkdir -p 750Picacho_DA3_RND_ONLY/{staging_16bit_png,da3_output,comparison_crops}
echo "NON-COMMERCIAL R&D ONLY — DO NOT SHIP TO CLIENTS" > 750Picacho_DA3_RND_ONLY/README_LICENSE_NOTICE.txt

# 3. Convert TIFFs to 16-bit PNG (DA3 compatible)
for tiff in 750Picacho_Source_TIFFs/*.tif*; do
    base=$(basename "$tiff" | sed 's/\.[^.]*$//')
    magick "$tiff" -colorspace RGB -depth 16 "750Picacho_DA3_RND_ONLY/staging_16bit_png/${base}.png"
done

# 4. Run DA3 inference (with fixed wrapper)
lux-depth-v3 process \
    --input-dir 750Picacho_DA3_RND_ONLY/staging_16bit_png \
    --output-dir 750Picacho_DA3_RND_ONLY/da3_output \
    --model da3nested-giant-large \
    --export-format mini_npz \
    --preset interior_luxury \
    --commercial-use false \
    --validate-license-strict true

# 5. Compare against APEX-100 outputs
# (Manual visual inspection + RMSE on known-good depth maps)
```

---

## Non-Negotiable Boundaries

1. **Production pipeline remains lux-depth-v2 APEX-100** (SegFormer MaterialsV2)
2. **All DA3 outputs stay in** `750Picacho_DA3_RND_ONLY/` directory
3. **Never mix DA3 artifacts** into client deliverables
4. **Always set** `commercial_use=False` and `validate_license_strict=True`
5. **Document all comparisons** in `750Picacho_DA3_RND_ONLY/evaluation_report.md`

---

## Sign-Off Checklist

Before running DA3 on 750 Picacho:

- [ ] Fix #1 applied: `from_pretrained()` model loading
- [ ] Fix #7 applied: License enforcement with hard error
- [ ] Fix #2 applied: Positional `images` argument
- [ ] Fix #3 applied: CLI positional paths
- [ ] Fix #5 applied: Subprocess DEVNULL or async drain
- [ ] Verified DA3 package installed: `pip show depth-anything-3`
- [ ] R&D directory created with LICENSE notice
- [ ] `.gitignore` updated to exclude R&D outputs
- [ ] APEX-100 production pipeline unchanged and tested

---

**End of Critical Fixes Document**
