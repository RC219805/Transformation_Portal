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
├── ml-core.in              # Supported ML metadata baseline (no checked-in lock)
├── ml-core-darwin-arm64.in   # ML core layer - macOS Apple Silicon baseline
├── ml-core-darwin-arm64.txt  # ML core layer - macOS Apple Silicon pinned lock
├── ml-cpu.in               # ML CPU acceleration layer
├── ml-cpu.txt              # Deprecated optional ML lock (not checked in)
├── ml-mps.in               # ML MPS acceleration layer (Apple Silicon)
├── ml-mps.txt              # Deprecated optional ML lock (not checked in)
├── ml-cuda.in              # Retired unsupported CUDA lane stub (fails closed)
├── ml-cuda.txt             # Retired unsupported CUDA lock (not checked in)
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

**Platform target taxonomy (not install support):**
- `darwin-x86_64-cpu` (retired unsupported ML lane)
- `darwin-arm64-cpu` (macOS Apple Silicon, CPU fallback)
- `darwin-arm64-mps` (macOS Apple Silicon, Metal)
- `linux-x86_64-cpu` (retired unsupported ML lane)
- `linux-x86_64-cuda` (retired unsupported CUDA lane; `core-cuda` fails closed)
- `linux-arm64-cpu` (retired unsupported ML lane)

### Target-Owned Lockfiles

**IMPORTANT:** pip-compile cannot resolve multi-platform conditional dependencies in a single graph.

To ensure deterministic builds, the checked-in ML contract is limited to the supported target-owned core lockfile:

| Target | Lockfile | Governed baseline |
|----------|----------|-------------------|
| macOS Apple Silicon (`darwin-arm64`) | `ml-core-darwin-arm64.txt` | `torch==2.13.0` + `torchvision==0.28.0` + `diffusers>=0.38.0` + `transformers==5.5.4` + pinned `coremltools` |

**Contract notes:**
- Supported target-owned core locks anchor on torch 2.13.0 / torchvision 0.28.0, Diffusers 0.38.0+, and Transformers 5.5.x.
- The Apple Silicon Darwin lock must keep pinned `coremltools` and must remain free of Linux/CUDA-only packages.
- Darwin target-owned lockfiles must never contain `nvidia-*` or `triton`.
- Linux and macOS Intel ML lanes are retired unsupported lanes and are not kept as installable `requirements/*.in` or `requirements/*.txt` manifests.
- Historical Linux/macOS Intel lane rationale belongs in non-installable governance documentation, not scan-visible pip requirement files.

**Important:** Acceleration is NEVER inferred from OS—it must be explicitly specified via profile.

### Layered Dependencies

Dependencies are organized into logical layers:

- **base**: Core runtime essentials needed for the application to function
- **ml**: Optional machine learning and deep learning dependencies (umbrella, no checked-in lockfile contract)
- **ml-core**: Supported ML metadata baseline (torch, diffusers, transformers, etc.; no checked-in lock artifact)
- **ml-cpu**: CPU acceleration layer (cross-platform capability, not a checked-in lock artifact)
- **ml-mps**: MPS acceleration layer (Apple Silicon, Metal Performance Shaders; installed via bootstrap/profile flow)
- **ml-cuda**: Retired unsupported CUDA lane stub; `core-cuda` fails closed until a governed Linux lockfile contract exists
- **ml-raw**: RAW camera file ingest (rawpy) - no trusted checked-in lockfile contract
- **ml-sam2**: SAM2 segmentation backend - scripted-only (non-standard install)
- **ml-coreml**: Apple CoreML acceleration - macOS only
- **ml-research**: Research/experimental extras - reserved for future use
- **dev**: Developer tools for testing, linting, and formatting
- **ci**: CI/CD pipeline tools for builds, security scanning, and releases (NOT test runners)
- **tools-archive**: dependencies for `tools/archive_manifest_reports.py`
- **all**: Aggregate of the checked-in contract (does not include optional ML layers)

### Relationship to Root Requirements Files

The repository has **root-level** requirements files that reference or complement
this layered system:

| Root File | Purpose | Structure |
|-----------|---------|-----------|
| `requirements.txt` | Core runtime | References `requirements/base.txt` |
| `requirements-ci.txt` | CI test runs | References `requirements.txt` + inline test deps |
| `requirements-dev.txt` | Development | References `requirements-ci.txt` + dev tools |
| `requirements-lint.txt` | CI/local lint parity | Hand-authored lean lint toolchain consumed by `scripts/setup/run_lint_tool.sh` |

**Important distinctions:**
- `requirements-ci.txt` (root) contains **test runner and test-support** deps (pytest, hypothesis, moto, etc.)
- `requirements/ci.in` contains **CI pipeline tools** (bandit, safety, build, twine, etc.)
- `requirements-lint.txt` is a separate root lint-tool parity surface, not a generated layered lock
- Core test deps in root `requirements-ci.txt` must match `requirements/dev.in`. The enforced set is defined as `CORE_TEST_DEPS` in `scripts/validation/check_ci_dep_sync.py` (currently: pytest, pytest-cov, pytest-asyncio, pytest-json-report, pytest-xdist, hypothesis, httpx, moto)
- Dev-only test tools in `requirements/dev.in` must also be available from root `requirements-dev.txt` without entering lean CI installs. The enforced set is defined as `DEV_ONLY_DEPS` in `scripts/validation/check_ci_dep_sync.py` (currently: pytest-rerunfailures for ADR-033 flaky-test quarantine support)
- Run `make check-ci-sync` to verify no drift for the enforced core-test and dev-only dependency sets across the root and layered files

### Current Web Runtime Baseline

The repository's governed web stack currently resolves to:

| Dependency | Source of Truth | Current Version |
|-----------|-----------------|-----------------|
| FastAPI | `requirements/base.in` | `0.141.1` |
| Starlette | `requirements/base.in` + `pyproject.toml` bound | `1.3.1` |
| Uvicorn | `requirements/base.in` + `pyproject.toml` bound | `0.52.1` |

This baseline was validated through the curated compatibility path, most recently on 2026-08-19 for the FastAPI `0.141.1` update in PR #2042 and the already-governed Uvicorn `0.52.1` pin. Do not treat future updates to these exact pins as routine dependency bumps; use the governance flow documented in `docs/governance/DEPENDABOT_PR_GOVERNANCE.md`.

### Layered ML Strategy

The ML dependencies are split into capability layers for:

1. **Target-safe compilation**: Generic locks compile together, while target-owned ML locks require explicit authoritative-lane commands
2. **Deterministic installs**: Each layer has explicit contracts (CPU-only, platform markers, etc.)
3. **Capability-gated promotion**: Optional features can be added incrementally
4. **Better failure semantics**: Instead of "pip install fails halfway through," you get clear capability boundaries

### Abstract vs Pinned

- **`.in` files**: Abstract requirements with version ranges (e.g., `numpy>=1.24,<2.5.0`)
- **`.txt` files**: Pinned requirements with exact versions (e.g., `numpy==2.2.6`)

The `.in` files define the desired version constraints, while `.txt` files are auto-generated by `pip-compile` with resolved, pinned versions.

## Secure-Install Hash Policy

The repository policy is to keep hash-enriched lock generation as a CI-only
advisory control for the non-ML checked-in layered locks. The checked-in
dependency contract and standard local install flows remain
pinned-without-hashes unless a later policy decision promotes broader
`--require-hashes` enforcement.

### Current policy decision

- Hash-enriched lockfiles are generated only in the secure-install pilot flow
  and are treated as evidence, not as the default install surface.
- The checked-in layered locks remain the primary deployment contract and
  continue to be consumed without mandatory hash enforcement.
- `requirements.txt`, `requirements-ci.txt`, and
  `requirements-dev.txt` remain outside this hash-enforced policy decision.
  `requirements-lint.txt` is also outside the hash-enforced pilot scope.
- ML platform locks remain outside this policy decision until the simpler
  non-ML layered surface has a reason to absorb broader enforcement cost.
- Promotion to mandatory `--require-hashes` enforcement requires a separate
  policy decision.

### Advisory pilot implementation

The repository includes a Phase 2 secure-install pilot for exercising that
policy without changing the checked-in dependency contract.

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

### Why the policy excludes root wrappers

`requirements.txt`, `requirements-ci.txt`, and `requirements-dev.txt` remain
outside this hash-enforced policy decision. `requirements-lint.txt` is also a
hand-authored convenience entry point outside the pilot scope. The pilot
deliberately avoids changing those interfaces until the team decides whether
hash enforcement is worth the operational cost.

### Why ML platform locks remain deferred

The pilot does not currently include:

- `ml-core-darwin-arm64.txt`

The Apple Silicon platform lock resolves through the PyTorch extra index and
carries more maintenance complexity than the base non-ML contract. The pilot
keeps it out of scope until the team has a clear decision on the simpler
layered install surface first.

### Running the advisory pilot locally

The local pilot currently expects the same toolchain as the advisory CI
workflow:

- `pip==26.2.1`
- `pip-tools==7.6.1`

That pairing includes the CVE-2026-13346 correction and the subsequent
26.2.1 keyring fix while retaining the repository-validated pip-tools compiler.
To match the workflow locally:

```bash
python -m pip install --upgrade "pip==26.2.1"
python -m pip install "pip-tools==7.6.1"
```

`pip-tools 7.6.1` is the first compiler release compatible with pip 26.2 and
replaces the pip-26.1-only 7.6.0 baseline. The workflow in
`.github/workflows/secure-install-pilot.yml` applies this toolchain
automatically. The requirements Makefile fails closed when `pip-compile`
or its paired pip interpreter reports any other version, so local runs must use
the same versions.

Pip 26.2 separates normal constraints from isolated-build constraints. Live
`pip install` commands that use `requirements/constraints.txt` therefore pass
both `-c requirements/constraints.txt` and
`--build-constraint requirements/constraints.txt`; do not rely on
`PIP_CONSTRAINT` propagating into an isolated build environment. The `-c`
arguments on `pip-compile` commands are compiler resolution inputs and are not
affected by this pip install behavior change.

Pip 26.2 also caches Simple API index responses. The scheduled dependency
update lane sets `PIP_NO_CACHE_DIR=1` around its lock update transaction so
newly published releases remain visible. For an ad hoc pip operation that must
retain the rest of the cache, use pip's targeted
`--refresh-package <distribution>` option.

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
# Install Apple Silicon CPU baseline
./scripts/bootstrap/install_ml_stack.sh --profile core-cpu

# Install Apple Silicon MPS acceleration (macOS ARM64 only)
./scripts/bootstrap/install_ml_stack.sh --profile core-mps

# Linux and macOS Intel ML lanes are retired unsupported lanes and fail closed.
# Do not install CUDA PyTorch packages ad hoc into the repo .venv.

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

# Install ML core layer (supported target-owned baseline)
pip install -r requirements/ml-core-darwin-arm64.txt   # macOS Apple Silicon

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
# Uses the MPS profile on native Apple Silicon; fails closed elsewhere
make install-ml-sam2

# Install ML CoreML layer (macOS only)
# Fails closed unless a trusted target-correct CoreML lockfile exists
make install-ml-coreml

# Umbrella install is disabled until a trusted checked-in contract exists again
make install-ml
```

Or use the package extras (installs latest allowed versions, not pinned).
The ML extras require the supported PyTorch security baseline
(`torch>=2.13.0`, `torchvision>=0.28.0`); retired historical ML locks are
not remediation targets for Dependabot alerts.

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
3. Recompile the generic checked-in requirements:

```bash
cd requirements/
make compile
```

4. If the change affects a target-owned ML lock, run the explicit target command on the authoritative lane:

```bash
cd requirements/
make compile-ml-darwin-arm64      # native Darwin arm64 only
```

5. Commit the `.in` file and any `.txt` files that are part of the checked-in contract for that layer

#### Updating Dependencies

To update the generic checked-in lockfiles to their latest allowed versions:

```bash
cd requirements/
make update
```

Every public generic writer (`compile-generic`, `update-generic`, and the six
individual `.txt` targets) compiles all six governed locks into a temporary
staging directory, validates their Python header, platform-marker, Linux
keyring-chain, and exact-pin contracts, and then publishes the set while one
exclusive destination-keyed advisory writer lock covers compilation through
publication. Before validating the `pip-compile` toolchain, the writer runs a
lock-protected recovery-only pass, then reacquires the same lock for the entire
compile-and-publish lifecycle before reading live locks as marker-pin sources.
A stale mixed set is therefore restored even when toolchain validation or
compilation then fails, and an older slow compile cannot overwrite a newer
writer. Compile and staged validation failures leave the recovered live files
unchanged. A Python exception, `KeyboardInterrupt`, or handled `SIGTERM` during
publication attempts an immediate rollback. A durable journal lets the next
writer restore a publication interrupted by an unhandled process crash or
`SIGKILL`; rollback or cleanup failures report the retained recovery directory
explicitly. Target-owned ML locks are outside this transaction and remain
untouched.

The six public lock paths intentionally remain regular files and are replaced
with sequential per-file renames. The advisory lock serializes cooperating
writers, but lock-free readers cannot obtain an atomic multi-file snapshot and
may observe a mixed set during publication. After `SIGKILL`, a mixed set can
remain until the next writer performs journal recovery; the journal does not
make reader visibility atomic.

For target-owned ML locks, use the explicit authoritative-lane command instead:

```bash
cd requirements/
make update-ml-darwin-arm64
```

These commands respect the version constraints in the `.in` files but only the explicit target-owned commands may regenerate governed ML target locks.

#### Checking for Drift

To verify that the generic checked-in lockfiles are up-to-date with `.in` files:

```bash
cd requirements/
make check
```

`make clean` recovers any stale publication and removes the complete six-file
generic set while holding the same destination writer lock used by publication.
Target-owned ML lock cleanup remains a separate explicit step after the generic
lock operation.

For target-owned ML locks, use:

```bash
cd requirements/
make check-ml-darwin-arm64
```

## 🔧 Makefile Targets

Run `make help` in this directory to see all available targets:

```
Targets:
  compile           Compile generic checked-in lockfiles only
  compile-all       Same as compile (for compatibility)
  compile-ml-layers Refuse broad ML regeneration; use explicit target-owned commands
  compile-accel     Refuse broad ML regeneration; use explicit target-owned commands
  update            Update generic checked-in lockfiles only
  check             Verify generic checked-in lockfiles only
  clean             Remove all compiled .txt files

Target-owned ML commands:
  compile-ml-darwin-arm64    Compile the Darwin arm64 ML lock on native Darwin arm64 only
  update-ml-darwin-arm64     Update the Darwin arm64 ML lock on native Darwin arm64 only
  check-ml-darwin-arm64      Verify the Darwin arm64 ML lock on native Darwin arm64 only
  compile-ml-linux-x86_64    Retired unsupported lane - always fails closed
  update-ml-linux-x86_64     Retired unsupported lane - always fails closed
  check-ml-linux-x86_64      Retired unsupported lane - always fails closed
  compile-ml-darwin-x86_64   Retired unsupported lane - always fails closed
  update-ml-darwin-x86_64    Retired unsupported lane - always fails closed
  check-ml-darwin-x86_64     Retired unsupported lane - always fails closed

Target-owned ML lockfiles:
  ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline
  macOS Intel/Linux          retired unsupported lanes; no checked-in installable ML lock

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

3. **Target-correct compilation**: Target-owned ML lockfiles must be regenerated on their authoritative lanes. In particular, Darwin lockfiles must remain free of Linux/CUDA-only packages such as `nvidia-*` and `triton`.

This approach prevents conflicts between layers and ensures reproducible builds.

### ML Layer Contracts

Each ML layer has a specific contract:

| Layer | Contract | Platform Target | Notes |
|-------|----------|-----------------|-------|
| ml-core | Supported PyTorch baseline (torch 2.13.0, torchvision 0.28.0, Transformers 5.5.x) | darwin-arm64 | Base ML functionality |
| ml-cpu | CPU fallback for supported Apple Silicon baseline | darwin-arm64-cpu | No GPU packages |
| ml-mps | Apple Silicon MPS | darwin-arm64-mps | Includes accelerate |
| ml-cuda | Retired unsupported lane | linux-x86_64-cuda | Fails closed until a governed Linux lane exists |
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
