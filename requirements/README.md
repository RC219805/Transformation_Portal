# Layered Dependency Management

This directory contains the layered dependency management system for Transformation Portal, using [pip-tools](https://github.com/jazzband/pip-tools) for reproducible builds.

## 📁 Directory Structure

```
requirements/
├── README.md               # This file
├── Makefile                # Automation for compiling requirements
├── base.in                 # Core runtime dependencies (abstract)
├── base.txt                # Core runtime dependencies (pinned)
├── ml.in                   # ML umbrella (references ml-*.in layers)
├── ml.txt                  # Removed umbrella lock (not part of checked-in contract)
├── ml-core.in              # ML core layer - cross-platform (legacy)
├── ml-core-darwin-x86_64.in  # ML core layer - macOS Intel conservative baseline
├── ml-core-darwin-x86_64.txt # ML core layer - macOS Intel pinned lock
├── ml-core-darwin-arm64.in   # ML core layer - macOS Apple Silicon baseline
├── ml-core-darwin-arm64.txt  # ML core layer - macOS Apple Silicon pinned lock
├── ml-core-linux.in          # ML core layer - Linux (torch 2.2.2)
├── ml-core-linux.txt         # ML core layer - Linux (pinned)
├── ml-cpu.in               # ML CPU acceleration layer
├── ml-cpu.txt              # Deprecated optional ML lock (not checked in)
├── ml-mps.in               # ML MPS acceleration layer (Apple Silicon)
├── ml-mps.txt              # Deprecated optional ML lock (not checked in)
├── ml-cuda.in              # ML CUDA acceleration layer (Linux + NVIDIA)
├── ml-cuda.txt             # Deprecated optional ML lock (not checked in)
├── ml-raw.in               # ML RAW ingest layer - rawpy
├── ml-raw.txt              # Deprecated optional ML lock (not checked in)
├── ml-sam2.in              # ML SAM2 layer - Meta Segment Anything 2
├── ml-sam2.txt             # ML SAM2 layer (scripted-only)
├── ml-coreml.in            # ML CoreML layer - macOS only
├── ml-coreml.txt           # Deprecated optional ML lock (not checked in)
├── ml-research.in          # ML research/experimental layer
├── ml-research.txt         # Deprecated optional ML lock (not checked in)
├── dev.in                  # Development tools (abstract)
├── dev.txt                 # Development tools (pinned)
├── ci.in                   # CI/CD tools (abstract)
├── ci.txt                  # CI/CD tools (pinned)
├── tools-archive.in        # Archive reporting tool deps (abstract)
├── tools-archive.txt       # Archive reporting tool deps (pinned)
├── all.in                  # Aggregate of checked-in contract dependencies
└── all.txt                 # Aggregate pinned checked-in contract
```

## 🎯 Design Principles

### Platform Matrix (ADR-032)

ML dependencies use an explicit platform matrix with three orthogonal axes:

| Axis   | Values           | Detection           |
|--------|------------------|---------------------|
| OS     | Darwin / Linux   | `platform_system`   |
| ISA    | arm64 / x86_64   | `platform_machine`  |
| Accel  | cpu / mps / cuda | **Explicit profile** |

**Canonical platform targets:**
- `darwin-x86_64-cpu` (macOS Intel)
- `darwin-arm64-cpu` (macOS Apple Silicon, CPU-only)
- `darwin-arm64-mps` (macOS Apple Silicon, Metal)
- `linux-x86_64-cpu` (Linux Intel/AMD, CPU baseline)
- `linux-x86_64-cuda` (Linux Intel/AMD, NVIDIA GPU)
- `linux-arm64-cpu` (Linux ARM)

### Platform-Specific Lockfiles

**IMPORTANT:** pip-compile cannot resolve multi-platform conditional dependencies in a single graph.

To ensure deterministic builds, the checked-in ML contract is limited to platform-specific core lockfiles:

| Platform | Lockfile | Governed baseline |
|----------|----------|-------------------|
| macOS Intel (`x86_64`) | `ml-core-darwin-x86_64.txt` | `torch==2.2.2` + `numpy<2` + `transformers<5` |
| macOS Apple Silicon (`arm64`) | `ml-core-darwin-arm64.txt` | `torch==2.2.2` + `transformers<5` + pinned `coremltools` |
| Linux | `ml-core-linux.txt` | validated independently against the runtime and lock contract checks |

**Contract notes:**
- All platform core locks still anchor on torch 2.2.2 for deterministic CAS identity.
- The Intel Darwin lock is the conservative compatibility lane and must keep `numpy<2` and `transformers<5`.
- The Apple Silicon Darwin lock must keep pinned `coremltools` and must remain free of Linux/CUDA-only packages.
- Darwin platform-core lockfiles must never contain `nvidia-*` or `triton`.
- Linux lock state may evolve independently, but that drift must not be propagated into Darwin lockfiles.
- Platform core locks must not collapse to identical dependency graphs.

**Important:** Acceleration is NEVER inferred from OS—it must be explicitly specified via profile.

### Layered Dependencies

Dependencies are organized into logical layers:

- **base**: Core runtime essentials needed for the application to function
- **ml**: Optional machine learning and deep learning dependencies (umbrella, no checked-in lockfile contract)
- **ml-core**: Cross-platform ML baseline (torch with platform-aware pins, diffusers, transformers, etc.)
- **ml-cpu**: CPU acceleration layer (cross-platform capability, not a checked-in lock artifact)
- **ml-mps**: MPS acceleration layer (Apple Silicon, Metal Performance Shaders; installed via bootstrap/profile flow)
- **ml-cuda**: CUDA acceleration layer (Linux + NVIDIA GPU; installed via bootstrap/profile flow)
- **ml-raw**: RAW camera file ingest (rawpy) - no trusted checked-in lockfile contract
- **ml-sam2**: SAM2 segmentation backend - scripted-only (non-standard install)
- **ml-coreml**: Apple CoreML acceleration - macOS only
- **ml-research**: Research/experimental extras - reserved for future use
- **dev**: Developer tools for testing, linting, and formatting
- **ci**: CI/CD pipeline tools for builds, security scanning, and releases (NOT test runners)
- **tools-archive**: dependencies for `tools/archive_manifest_reports.py`
- **all**: Aggregate of the checked-in contract (does not include optional ML layers)

### Relationship to Root Requirements Files

The repository has **root-level** requirements files that reference this layered system:

| Root File | Purpose | Structure |
|-----------|---------|-----------|
| `requirements.txt` | Core runtime | References `requirements/base.txt` |
| `requirements-ci.txt` | CI test runs | References `requirements.txt` + inline test deps |
| `requirements-dev.txt` | Development | References `requirements-ci.txt` + dev tools |

**Important distinctions:**
- `requirements-ci.txt` (root) contains **test runner** deps (pytest, hypothesis, etc.)
- `requirements/ci.in` contains **CI pipeline tools** (bandit, safety, build, twine, etc.)
- Core test runner deps in root `requirements-ci.txt` must match `requirements/dev.in`. The enforced set is defined as `CORE_TEST_DEPS` in `scripts/validation/check_ci_dep_sync.py` (currently: pytest, pytest-cov, pytest-asyncio, pytest-json-report, pytest-xdist, hypothesis, httpx)
- Run `make check-ci-sync` to verify no drift for this core test runner set between the root files

### Current Web Runtime Baseline

The repository's governed web stack currently resolves to:

| Dependency | Source of Truth | Current Version |
|-----------|-----------------|-----------------|
| FastAPI | `requirements/base.in` | `0.135.1` |
| Starlette | `requirements/base.in` + `pyproject.toml` bound | `1.0.0` |
| Uvicorn | `requirements/base.in` | `0.42.0` |

This baseline was validated and merged via the curated compatibility path on 2026-03-26. Do not treat future updates to these exact pins as routine dependency bumps; use the governance flow documented in `docs/governance/DEPENDABOT_PR_GOVERNANCE.md`.

### Layered ML Strategy

The ML dependencies are split into capability layers for:

1. **Platform-safe compilation**: Each layer is compiled independently, allowing platform-specific handling
2. **Deterministic installs**: Each layer has explicit contracts (CPU-only, platform markers, etc.)
3. **Capability-gated promotion**: Optional features can be added incrementally
4. **Better failure semantics**: Instead of "pip install fails halfway through," you get clear capability boundaries

### Abstract vs Pinned

- **`.in` files**: Abstract requirements with version ranges (e.g., `numpy>=1.24,<2.5.0`)
- **`.txt` files**: Pinned requirements with exact versions (e.g., `numpy==2.2.6`)

The `.in` files define the desired version constraints, while `.txt` files are auto-generated by `pip-compile` with resolved, pinned versions.

## Secure-Install Pilot

The repository includes a Phase 2 secure-install pilot for evaluating
hash-enforced installs without changing the checked-in dependency contract.

### Pilot scope

The pilot generates hash-enriched copies of the non-ML checked-in layered locks:

- `all.txt`
- `base.txt`
- `dev.txt`
- `ci.txt`
- `security.txt`
- `tools-archive.txt`

These files are emitted into `HASH_PILOT_OUT_DIR` (default:
`requirements/.hash-pilot/`) and are not checked in.

### Why the pilot excludes root wrappers

Root wrapper files such as `requirements.txt`, `requirements-ci.txt`, and
`requirements-dev.txt` remain hand-authored convenience entry points and are
used broadly across CI and local setup flows. The pilot deliberately avoids
changing those interfaces until the team decides whether hash enforcement is
worth the operational cost.

### Why ML platform locks are deferred

The pilot does not currently include:

- `ml-core-darwin-x86_64.txt`
- `ml-core-darwin-arm64.txt`
- `ml-core-linux.txt`

Those platform locks resolve through the PyTorch extra index and carry more
cross-platform maintenance complexity than the base non-ML contract. The pilot
keeps them out of scope until the team has a clear decision on the simpler
layered install surface first.

### Running the pilot locally

The local pilot currently expects the same toolchain as the advisory CI
workflow:

- `pip<26`
- `pip-tools==7.5.2`

That pairing avoids a known compatibility problem between `pip 26` and
`pip-tools 7.5.2` during hash generation. To match the workflow locally:

```bash
python -m pip install --upgrade "pip<26"
python -m pip install "pip-tools==7.5.2"
```

This avoids a known compatibility issue between `pip 26` and
`pip-tools 7.5.2` that causes `--generate-hashes` compilation to fail. The
workflow in `.github/workflows/secure-install-pilot.yml` applies this
toolchain automatically, so local runs should use the same versions to avoid
unexpected failures.

```bash
cd requirements
make compile-hash-pilot LOCK_PYTHON_VERSION=3.11
make check-hash-pilot LOCK_PYTHON_VERSION=3.11
```

To write the pilot artifacts somewhere else:

```bash
cd requirements
make compile-hash-pilot LOCK_PYTHON_VERSION=3.11 HASH_PILOT_OUT_DIR=/tmp/tp-hash-pilot
make check-hash-pilot LOCK_PYTHON_VERSION=3.11 HASH_PILOT_OUT_DIR=/tmp/tp-hash-pilot
```

`check-hash-pilot` validates the generated artifacts with
`pip install --dry-run --require-hashes`. This pilot is advisory and does not
replace the standard `make compile` / `make check` flow.

## 🚀 Usage

### For Users

#### Using Bootstrap Script (Recommended)

The bootstrap script provides profile-based installation with platform validation:

```bash
# Install cross-platform CPU baseline
./scripts/bootstrap/install_ml_stack.sh --profile core-cpu

# Install Apple Silicon MPS acceleration (macOS ARM64 only)
./scripts/bootstrap/install_ml_stack.sh --profile core-mps

# Install NVIDIA CUDA acceleration (Linux only)
PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda

# Install with SAM2 segmentation
./scripts/bootstrap/install_ml_stack.sh --profile core-mps,sam2

# Dry run to preview what would be installed
./scripts/bootstrap/install_ml_stack.sh --profile core-mps,sam2 --dry-run

# RAW/coreml/research/full profiles are disabled until trusted target-correct
# checked-in contracts exist again
./scripts/bootstrap/install_ml_stack.sh --profile full
```

#### Using pip directly

To install the package with specific dependency sets:

```bash
# Install core dependencies only
pip install -r requirements/base.txt

# Install ML core layer (platform-specific baseline)
pip install -r requirements/ml-core-darwin-x86_64.txt  # macOS Intel
# or
pip install -r requirements/ml-core-darwin-arm64.txt   # macOS Apple Silicon
# or
pip install -r requirements/ml-core-linux.txt    # Linux

# Optional non-core ML layers no longer have trusted checked-in lockfiles.
# Use target-specific bootstrap flows once those contracts exist again.

# Install everything (development environment)
pip install -r requirements/all.txt
```

Or use the Makefile targets:

```bash
# Install ML core layer
make install-ml-core

# Install ML RAW ingest layer
# Fails closed until a trusted target-correct lockfile contract exists
make install-ml-raw

# Install ML SAM2 layer
make install-ml-sam2

# Install ML CoreML layer (macOS only)
# Fails closed unless a trusted target-correct CoreML lockfile exists
make install-ml-coreml

# Umbrella install is disabled until a trusted checked-in contract exists again
make install-ml
```

Or use the package extras (installs latest allowed versions, not pinned):

```bash
# Install ML core support only
pip install -e ".[ml-core]"

# Install with SAM2 segmentation
pip install -e ".[sam2]"

# Install with RAW camera support
pip install -e ".[raw]"

# Install full ML support
pip install -e ".[ml]"

# Install full development environment
pip install -e ".[all]"
```

### For Contributors

#### Installing Development Environment

For reproducible builds, use the pinned requirements:

```bash
# Full development install with exact versions
pip install -r requirements/all.txt
pip install -e .
```

#### Adding New Dependencies

1. Edit the appropriate `.in` file (e.g., `base.in`, `ml-core.in`, `ml-raw.in`, `dev.in`, etc.)
2. Add your dependency with a version constraint (e.g., `requests>=2.28,<3`)
3. Recompile all requirements:

```bash
cd requirements/
make compile
```

4. Commit the `.in` file and any `.txt` files that are part of the checked-in contract for that layer

#### Updating Dependencies

To update all dependencies to their latest allowed versions:

```bash
cd requirements/
make update
```

This will respect the version constraints in the `.in` files but find the newest versions within those constraints.

#### Checking for Drift

To verify that `.txt` files are up-to-date with `.in` files:

```bash
cd requirements/
make check
```

## 🔧 Makefile Targets

Run `make help` in this directory to see all available targets:

```
Targets:
  compile           Compile all pinned requirements from .in files
  compile-all       Same as compile (for compatibility)
  compile-ml-layers Compile only ML layer lockfiles
  compile-accel     Compile acceleration layer lockfiles (ml-cpu, ml-mps, ml-cuda)
  update            Update all dependencies to latest versions
  check             Verify that .txt files are up-to-date with .in files
  clean             Remove all compiled .txt files

Checked-in ML layer targets (platform-core contract, CPU-only PyTorch index):
  ml-core-darwin-x86_64.txt  macOS Intel ML baseline
  ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline
  ml-core-linux.txt   Linux ML baseline

Forbidden checked-in optional ML lock targets:
  ml-core.txt       not part of checked-in contract
  ml-cpu.txt        not part of checked-in contract
  ml-mps.txt        not part of checked-in contract
  ml-cuda.txt       not part of checked-in contract
  ml-raw.txt        not part of checked-in contract
  ml-coreml.txt     not part of checked-in contract
  ml-research.txt   not part of checked-in contract
  ml.txt            not part of checked-in contract

Scripted-only layers (NOT compiled here):
  ml-sam2           SAM2 segmentation - use bootstrap script
```

## 📚 Technical Details

### Compilation Strategy

The system uses a two-phase compilation strategy:

1. **Checked-in Contract Resolution**: First, `all.in` is compiled to produce `all.txt` for the checked-in contract layers. Optional non-core ML layers are excluded so host-resolved outputs do not leak into repository state.

2. **Layer-Specific Outputs**: Then, each individual `.in` file is compiled using `all.txt` as a constraint file. This ensures that the subset of packages in each layer uses the same versions as in the global resolution.

3. **Target-correct compilation**: Platform-core lockfiles must be regenerated on their target lanes. In particular, Darwin lockfiles must remain free of Linux/CUDA-only packages such as `nvidia-*` and `triton`.

This approach prevents conflicts between layers and ensures reproducible builds.

### ML Layer Contracts

Each ML layer has a specific contract:

| Layer | Contract | Platform Target | Notes |
|-------|----------|-----------------|-------|
| ml-core | Platform-aware PyTorch (torch 2.2.2) | All | Base ML functionality |
| ml-cpu | CPU-only, cross-platform | darwin-*/linux-*-cpu | No GPU packages |
| ml-mps | Apple Silicon MPS | darwin-arm64-mps | Includes accelerate |
| ml-cuda | NVIDIA CUDA | linux-x86_64-cuda | GPU packages allowed |
| ml-raw | Platform-scoped | Linux, macOS | rawpy wheel availability varies |
| ml-sam2 | Scripted-only | All (may need --no-build-isolation) | Build-time torch dependency |
| ml-coreml | macOS only | Darwin | coremltools |
| ml-research | Reserved | Varies | Future experimental extras |

### Why pip-tools?

[pip-tools](https://github.com/jazzband/pip-tools) provides:

- **Deterministic builds**: Exact versions are pinned in `.txt` files
- **Conflict resolution**: pip's backtracking resolver finds compatible versions
- **Layered compilation**: Constraint files ensure consistency across layers
- **Update workflow**: Easy to update dependencies while respecting constraints

### Relationship to pyproject.toml

The `pyproject.toml` file contains the same dependencies as the `.in` files, with version ranges. This ensures:

- The package can be installed with `pip install -e .`
- Extras like `[ml-core]`, `[sam2]`, `[raw]`, `[ml]`, `[dev]`, `[ci]`, and `[all]` work as expected
- The `.in` files remain the single source of truth for version constraints
- The `.txt` files provide reproducible pinned versions for deployments

When developing, prefer installing from `.txt` files for reproducibility:

```bash
pip install -r requirements/all.txt
pip install -e .
```

## 🔒 CI/CD Integration

The CI/CD pipeline should:

1. **Verify consistency**: Check that `.txt` files are up-to-date with `.in` files
2. **Use pinned versions**: Install from `.txt` files for reproducible tests
3. **Validate contracts**: Run `check_requirements_lock_contract.py` to verify platform-core lockfile purity, compatibility guards, and lane-specific contract rules
4. **Layer-specific validation**: Each layer can be validated independently

Example CI workflow step:

```yaml
- name: Check requirements consistency
  run: |
    cd requirements/
    make check

- name: Validate lockfile contract
  run: |
    python3 scripts/validation/check_requirements_lock_contract.py
```

## 📝 Notes

- **ML dependencies**: Due to disk space constraints, ML lockfiles may need to be regenerated in environments with more resources. The CI environment should have sufficient space for PyTorch and related packages.

- **GPU vs CPU**: Darwin platform-core lockfiles are governed as non-CUDA baseline locks and must remain free of Linux/CUDA-only artifacts. Linux acceleration behavior is defined separately by the CUDA layer and runtime/bootstrap flow.

- **Optional non-core ML layers**: raw/coreml/research/full are fail-closed until target-correct trusted lockfile contracts are defined.

- **SAM2 build isolation**: sam2 may require `pip install --no-build-isolation sam2==1.1.0` on some platforms due to build-time torch requirements.

- **Python version**: All layered requirements are compiled with Python 3.11 (`requirements/Makefile: LOCK_PYTHON_VERSION`) and support Python 3.11+ as specified in `pyproject.toml`.

- **pip-tools cache**: `requirements/.pip-tools-cache/` is a local ephemeral cache directory and must never be tracked in git.

## 🆘 Troubleshooting

### "No matching distribution found"

Some packages may not be available on all platforms. Check:
- Python version compatibility
- Platform (Windows/macOS/Linux)
- Package availability on PyPI

### "Requirements conflict"

If you get conflicts during compilation:
1. Review version constraints in `.in` files
2. Check for incompatible package combinations
3. Consider relaxing version bounds
4. Consult package changelogs for compatibility

### Disk space errors

If compilation fails due to disk space:
- Run `pip cache purge` to clear pip's cache
- Compile layers individually instead of all at once
- Use a machine with more disk space (ML dependencies can require 5-10GB)

### SAM2 installation issues

If sam2 fails to install:
1. Ensure ml-core is installed first (provides torch)
2. Try `pip install --no-build-isolation sam2==1.1.0`
3. Check platform compatibility

## 📖 References

- [pip-tools documentation](https://github.com/jazzband/pip-tools)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPA specifications](https://packaging.python.org/specifications/)
