# Transformation Portal - Setup Guide

This guide walks you through setting up the Transformation Portal with all required dependencies and optional features.

## Table of Contents

- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Model Downloads](#model-downloads)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Create the repo virtual environment
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install pinned core dependencies and CLI
make install-core

# 4. Verify the environment
make check-environment
```

---

## Detailed Installation

### Prerequisites

- **Core Python**: 3.11+; Python 3.12 is supported and used by the lint and core-test lanes. Preserve an existing supported `.venv`.
- **DA3 cache-authorizing runtime**: separate Python 3.11 on native Darwin arm64; this does not require downgrading the core environment.
- **Git**: For cloning the repository
- **pip**: Python package manager (included with Python)
- **Optional**: native Apple Silicon for the checked-in ML lock; see the target restrictions below.

### Step 1: Core Dependencies

Install the core packages required for basic image processing:

```bash
make venv
source .venv/bin/activate
make install-core
```

This installs:
- **numpy**: Numerical processing
- **Pillow**: Image I/O and manipulation
- **scipy**: Scientific computing
- **typer**: CLI framework and other pinned core utilities
- the local project in editable mode without re-resolving dependencies outside the checked-in lockfiles

### Step 2: ML Dependencies (Optional)

For AI-powered features such as DA3 depth inference, use a trusted target-specific ML profile rather than the disabled umbrella install path. The currently checked-in ML core lock is target-owned for macOS Apple Silicon (`darwin-arm64`) only; Linux and macOS Intel ML lanes are retired unsupported lanes and fail closed until a governed lane is re-established.

```bash
# Supported checked-in Apple Silicon baseline
make install-ml-core

# Advanced Apple Silicon bootstrap-profile work only:
./scripts/bootstrap/install_ml_stack.sh --profile core-cpu
./scripts/bootstrap/install_ml_stack.sh --profile core-mps    # macOS Apple Silicon only
```

Optional profiles for `raw`, `coreml`, `research`, and `full` are currently fail-closed until trusted target-correct lockfile contracts exist again.

The `core-cuda` profile and all Linux ML lock lanes are retired unsupported lanes and fail closed. On Windows, use WSL2 only for non-ML/core workflows unless a governed Windows ML lane is established.

**Platform-specific notes:**

- **Apple Silicon macOS (M1/M2/M3/M4):** use `make install-ml-core`; reserve `./scripts/bootstrap/install_ml_stack.sh --profile core-mps` for bootstrap-profile work
- **Linux with NVIDIA GPU:** retired unsupported ML lane; `core-cuda` fails closed until a governed Linux lockfile contract exists
- **CPU only:** supported through the checked-in Apple Silicon baseline; Linux/macOS Intel ML lanes are retired unsupported lanes

### Step 3: Depth Processing (Optional)

For depth-aware processing, use the governed isolated runtime installers instead of installing model packages directly into the repo `.venv`:

```bash
# Default DA3 runtime used by Lux Depth V3 (installer may replace its own venv)
./scripts/setup/install_da3_runtime.sh --profile baseline

# Research-only Apple Depth Pro runtime; requires a separately obtained checkpoint
./scripts/setup/install_depth_pro_runtime.sh
```

DA3 depth-cache authority additionally requires the governed baseline runtime and
the exact source revision, dependency lock, and authority marker in the
[runtime identity contract](../../config/da3_runtime_identity_contract.json).
The pinned upstream `depth_anything_3` package is a namespace package without an
`__init__.py`; revision discovery uses its single canonical
`<checkout>/src/depth_anything_3` directory. Ambiguous or misplaced namespace
roots remain non-authorizing. A `source_revision_mismatch` requires inspecting
the imported source and checkout revision, rather than changing the marker.
Import-root identity uses canonical filesystem paths, so a library changing
the same root from `str` to `pathlib.Path` does not invalidate it. A different
root or changed bytes still fails verification.

Worker preparation retains per-file and per-directory observations so later
cache access can detect runtime changes. The bounded response permits up to
32 MiB for this verification token plus 4 MiB for evidence and other metadata;
both component limits are checked separately within the 36 MiB transport limit.
Inference result JSON retains its separate 4 MiB limit; the expanded transport
budget applies to preparation handshakes, not inference metadata.
An oversized response remains non-authorizing. Successful inference alone does
not establish cache reuse; confirm a cache hit on an identical repeat run.

---

## Model Downloads

### CoreML Instructions and Artifact Check

This utility prints CoreML conversion/setup instructions and checks local artifacts; it does not download a DA3 runtime or establish inference readiness:

```bash
.venv/bin/python scripts/setup/download_depth_models.py --model depth
```

Options:
- `--model depth`: Depth model setup/verification workflow
- `--verify-only`: Verify local model artifact status without setup steps
- `--output-dir PATH`: Custom output directory (default: ./weights)

### Legacy Transformers Depth Anything Example

The following is a separate Transformers V2 example, requiring an appropriate ML environment and network/cache access. It is not the Lux DA3 subprocess installer:

```python
from transformers import pipeline

# Transformers-compatible IDs use the "-hf" suffix.
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
```

The current Lux `da3` backend uses its isolated `depth_anything_3` runtime.
The default model selector is `da3-metric`; do not assume a V3-to-V2 fallback.
Depth Pro requires `--depth-backend depth_pro`, an installed isolated runtime,
its checkpoint, and both research-license acknowledgements described in the
[CLI guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md).

### Depth Anything (CoreML - Apple Silicon)

The checked-in Apple Silicon ML core lock includes `coremltools` for governed macOS arm64 workflows. Use `make install-ml-core` or `./scripts/bootstrap/install_ml_stack.sh --profile core-mps` on native Apple Silicon rather than installing CoreML tooling ad hoc into the repo `.venv`.

**Note**: CoreML conversion remains a macOS-specific workflow and is not the default DA3 runtime path. Use `./scripts/setup/install_da3_runtime.sh` for the governed DA3 subprocess runtime.

---

## Troubleshooting

### Issue #1: Image or Tensor Dimension Errors

Record the backend, input dimensions, and first exception. Dimension constraints
belong to the selected model or pipeline; a Stable Diffusion sizing rule is not
a universal Lux or DA3 contract. Do not resize source assets solely to satisfy
an unrelated model example.

### Issue #2: Slow Model Downloads

**Error:**
```
ZoeD_M12_N.pt: 0% | 703k/1.44G [00:30<10:48:36, 37.1kB/s]
```

**Solutions:**
- Use a faster internet connection
- Use `.venv/bin/python scripts/setup/download_depth_models.py --verify-only` to verify local artifacts
- HuggingFace models are still downloaded automatically on first model use
- Use cached models: Set `TRANSFORMERS_CACHE` environment variable
  ```bash
  export TRANSFORMERS_CACHE=/path/to/cache
  ```

### Issue #3: Missing Accelerate Warnings

**Warning:**
```
Cannot initialize model with low cpu memory usage because `accelerate` was not found
```

**Resolution**: Identify the interpreter reporting the warning. For Lux DA3,
inspect and repair the isolated DA3 runtime with its governed installer;
installing `accelerate` in the core `.venv` does not repair another process.
For supported in-process ML features, use the target-owned ML installation
contract in [requirements/README.md](../../requirements/README.md). Do not infer
a loading-speed or memory guarantee from the presence of one package.

### Issue #4: Depth Pipeline Module Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'depth_pipeline/pipeline.py'
```

**Explanation**: The depth pipeline is now part of the Lux Depth V3 module. Use the current implementation:

**Current usage:**
```bash
# Use the lux-depth-v3 CLI
lux-depth-v3 --input-dir ./input_images --output-dir ./output
```

```python
# Or use the Python API
from transformation_portal.lux_depth_v3 import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
```

See [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) for detailed usage.

---

## Verification

### Run Verification Script

```bash
.venv/bin/python scripts/verification/verify_core.py
```

Or for ML dependencies:

```bash
.venv/bin/python scripts/verification/verify_ml_deps.py
```

`verify_core.py` exercises an atmosphere/SkyBlender smoke on a synthetic gray
image; it does not verify the entire application, DA3, or produced Lux artifacts.
`verify_ml_deps.py` imports packages in the interpreter that runs it and reports
hardware availability there. It does not audit isolated DA3/Depth Pro/FastVLM
runtimes or prove successful inference, regardless of its summary wording.

Use separate evidence for each stage: `--help` proves parser availability;
`--plan` resolves a canonical execution plan without model loading or output
creation; runtime readiness checks cover their documented scope; an actual
successful job and indexed artifacts establish processing completion.

### Test CLI

```bash
# Test lux-depth-v3 CLI
lux-depth-v3 --help

# Inspect supported CLI options (there is no --list-stable option)
# Presets are documented in docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md
```

---

## Performance Tips

### Apple Silicon (M1/M2/M3/M4)

1. **Use the checked-in Apple Silicon ML baseline**: `make install-ml-core`
2. **Reserve MPS bootstrap for native arm64 macOS bootstrap-profile work**: `./scripts/bootstrap/install_ml_stack.sh --profile core-mps`
3. **Enable Metal**: Ensure macOS 13+ for best performance

Measure elapsed time and memory on your selected model, input size, device,
and enabled stages. No universal latency or throughput is established by setup.

### NVIDIA GPU

The Linux CUDA ML lane is retired unsupported and fails closed until a governed Linux lockfile contract is re-established. Do not install CUDA PyTorch packages ad hoc into the repo `.venv`; track any future CUDA enablement through `requirements/README.md` and the target-owned ML lock workflow.

### CPU Only

1. **Use smaller models**: Depth-Anything-V2-Small-hf vs Large-hf
2. **Reduce dimensions**: 512×512 instead of 1024×768
3. **Measure the selected runtime**: CPU/GPU ratios vary with model and workload.

---

## Next Steps

After setup:

1. **Read the main README**: `README.md` for usage examples
2. **Explore examples**: `examples/` directory
3. **Run tests**: `make test-fast` or `pytest tests/`
4. **Try the CLI**: `lux-depth-v3 --help` for the main processing pipeline

---

## Getting Help

- **Issues**: https://github.com/RC219805/Transformation_Portal/issues
- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory

---

**Last Updated**: 2026-09-12
**Version**: 2.0.0
