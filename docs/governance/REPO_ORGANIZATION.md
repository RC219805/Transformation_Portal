# Repository Organization System

**Transformation Portal** uses an automated file organization system to maintain a clean, structured directory hierarchy. This document describes the organization system, its purpose, and how to use it.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Automated Organization](#automated-organization)
- [Helper Scripts](#helper-scripts)
- [Pre-Commit Hook](#pre-commit-hook)
- [Installation](#installation)
- [Usage](#usage)
- [File Classification Rules](#file-classification-rules)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [Version History](#version-history)

## Overview

The repository organization system solves the recurring problem of files accumulating in the repository root by:

1. **Automatically organizing files** into appropriate directories based on their type and purpose
2. **Preventing misplaced files** through pre-commit hooks
3. **Maintaining consistency** across the entire codebase
4. **Making the repository easier to navigate** for both humans and tools

Current documentation navigation is intentionally narrow: `README.md`,
`docs/README.md`, and `docs/governance/DOCUMENTATION_MAP.md` define live
guidance. Older project reports and session artifacts stay in approved docs
directories as historical evidence, but they should not be treated as current
runbooks unless promoted in the documentation map.

### Key Features

- ✅ **Automatic file organization** with `.auto-organize.sh`
- ✅ **Pre-commit validation** to prevent misplaced files
- ✅ **Dry-run mode** for safe testing
- ✅ **Comprehensive logging** for transparency
- ✅ **Cross-platform compatibility** (Linux, macOS, Windows via WSL)

## Directory Structure

The repository follows this standardized structure:

```text
Transformation_Portal/
├── .github/                    # GitHub workflows and actions
│   ├── workflows/              # CI/CD workflow definitions
│   └── agents/                 # Custom GitHub Copilot agents
├── archive/                    # Retired code, legacy scripts, historical bundles
├── artifacts/                  # Tracked artifact metadata and governed evidence
├── assets/                     # Brand assets, LUTs, project assets, texture plates
├── cloudflare/                 # Cloudflare Worker source owned by the root shim
├── config/                     # Runtime config, presets, manifests, feature gates
├── data/                       # Tracked sample images and fixture-like data
├── docs/                       # Maintained docs plus historical records
│   ├── README.md               # Current documentation entry point
│   ├── governance/             # Documentation map, policy, organization
│   ├── guides/                 # Current guides plus older project guides
│   ├── architecture/           # Architecture docs and ADR history
│   ├── api/                    # API and machine-mode contracts
│   ├── ci/                     # CI governance and workflow matrix
│   ├── historical/             # Point-in-time records
│   └── _archive/               # Retired or consolidated documentation
├── evalsets/                   # Evaluation fixtures and governed benchmark inputs
├── examples/                   # Example inputs, configs, and demos
├── input_images/               # Local/sample input image root
├── migrations/                 # Alembic migrations for durable orchestrator state
├── policy/                     # Policy-as-data and governance policy assets
├── public/                     # FastAPI-served portal assets, shared assets, video
├── requirements/               # Layered .in/.txt lock sources and lock governance
├── schemas/                    # Contract schemas
├── scripts/                    # All scripts
│   ├── setup/                  # Installation and setup scripts
│   ├── pipelines/              # Pipeline runners/processors
│   ├── governance/             # Organization and policy validators
│   ├── validation/             # Runtime/contract validation checks
│   └── utilities/              # Focused utility scripts
├── src/                        # Source code (installable package)
│   ├── transformation_portal/  # Main package
│   ├── tp/                     # Separate public contract/fixity import surface
│   └── luxury_tiff_batch_processor/  # TIFF processing module
├── tests/                      # Contract, unit, integration, fixture, and smoke tests
├── tools/                      # Governed CLIs for archive/performance/evidence
├── web/                        # Managed Next.js frontdoor and shared web assets
│   ├── secure-landing/         # Node 22 managed browser entry point
│   └── shared/                 # Shared web tokens/assets
├── workflows/                  # Repo-level workflow metadata and supporting files
└── [root files]                # Configuration and build files only
    ├── README.md
    ├── Makefile
    ├── pyproject.toml
    ├── requirements*.txt
    ├── .gitignore
    ├── .gitattributes
    └── .auto-organize.sh
```

## Automated Organization

The `.auto-organize.sh` script is the canonical entry point for repository organization. It orchestrates modular helper scripts and validation tools.

### Usage

```bash
./.auto-organize.sh [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be done without making changes |
| `--check` | CI validation mode - exit 1 if violations found |
| `--verbose` | Show detailed output including skipped files |
| `--docs-only` | Only run documentation organization and docs topology validation |
| `--skip-root` | Skip root file placement validation |
| `-h, --help` | Show help message |

### Examples

```bash
./.auto-organize.sh --dry-run              # Preview organization changes
./.auto-organize.sh                        # Apply organization changes
./.auto-organize.sh --check                # CI validation mode (fail if violations)
./.auto-organize.sh --docs-only --dry-run  # Preview docs moves and validate docs topology
```

### What Gets Organized

| File Type              | Destination             | Examples                      |
|------------------------|-------------------------|-------------------------------|
| Strategy/Planning docs | `docs/guides/`          | `*_PLAN.md`, `*_SUMMARY.md`   |
| Architecture docs      | `docs/architecture/`    | `ARCHITECTURE.md`, `*_DESIGN.md` |
| Utility scripts        | `scripts/utilities/`    | `navigate.sh`, `verify_*.sh`  |
| Setup scripts          | `scripts/setup/`        | `install_*.sh`, `download_*.py` |
| Data files             | `data/`                | `*.json`, `*.csv`             |
| Sample images          | `data/sample_images/`   | `*.jpg`, `*.png`              |
| Debug artifacts        | `archive/`             | `debug_*.jpg`, `test_*.png`   |
| Deprecated code        | `archive/deprecated/`   | Old implementations           |

### What Stays in Root

Only these files should remain in the repository root:

- **Core documentation**: `README.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`
- **Build configuration**: `Makefile`, `pyproject.toml`, root Cloudflare Workers Builds shim files (`package.json`, `package-lock.json`, `wrangler.jsonc`)
- **Dependency management**: `requirements.txt`, `requirements-dev.txt`, `requirements-ci.txt`, `requirements-lint.txt`
- **Testing and linting configuration**: `pyproject.toml`, `.pylintrc`, `mypy.ini`
- **Docker and local environment templates**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`
- **Git and security metadata**: `.gitignore`, `.gitattributes`, `.gitmodules`, `.git-blame-ignore-revs`, `.gitleaks.toml`, `.pre-commit-config.yaml`
- **Governance metadata**: `.architect_directive_status.yml`, `AGENTS.md`, `CLAUDE.md`
- **Runtime entrypoints**: `app.py`, `portal.html`
- **Organization system**: `.auto-organize.sh`

Current allowed root files are:
`README.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`,
`AGENTS.md`, `CLAUDE.md`, `Makefile`, `package.json`, `package-lock.json`,
`pyproject.toml`, `requirements.txt`, `requirements-dev.txt`,
`requirements-ci.txt`, `requirements-lint.txt`, `.pylintrc`, `mypy.ini`,
`Dockerfile`, `docker-compose.yml`, `.gitignore`, `.gitleaks.toml`,
`.dockerignore`, `.gitattributes`, `.gitmodules`, `.git-blame-ignore-revs`,
`.pre-commit-config.yaml`, `wrangler.jsonc`, `.auto-organize.sh`,
`.architect_directive_status.yml`, `.env.example`, `app.py`, and
`portal.html`.

The root Cloudflare Workers files are not a general JavaScript application
root. They are a minimal Workers Builds deploy shim for
`cloudflare/transformationportal-worker`; keep scripts, Node engine metadata,
Wrangler version, and the delegated entrypoint aligned with that governed
worker package. The contract is enforced by
`tests/validation/test_cloudflare_worker_root_shim_contract.py`.

### Root Directory Limits

The repository root should remain minimal and operational. Documentation files that are not canonical root documents must live under `docs/` in approved subdirectories.
Retired root-level bundles must not be recreated. The former `productivity/`
bundle is archived under `docs/historical/productivity-suite-2025/`, with its
retired placeholder script under `archive/scripts/productivity/`.

Approved tracked top-level directories are enforced by
`scripts/setup/pre-commit-check.sh`.

Current allowed top-level directories are:
`.github/`, `archive/`, `artifacts/`, `assets/`, `cloudflare/`, `config/`,
`data/`, `docs/`, `evalsets/`, `examples/`, `input_images/`, `migrations/`,
`policy/`, `public/`, `requirements/`, `schemas/`, `scripts/`, `src/`,
`tests/`, `tools/`, `web/`, and `workflows/`.

Contract-sensitive nested roots such as `public/portal-assets`,
`public/video`, and `web/secure-landing` remain stable. Retired roots such as
`dashboard/`, `data/luts/`, `linear_ingest_demo/`, `projects/`,
`test_sky_fix/`, and `textures/` must not be recreated; use `assets/luts/`,
`assets/textures/board_materials/`, `assets/projects/`, `docs/projects/`, or
ignored `output/...` paths instead.

## Organization Steps

`.auto-organize.sh` executes these validation and organization steps in sequence:

1. **Documentation Organization** (`scripts/organize_docs.sh`)
   - Scans root-level documentation and `docs/` topology violations
   - Classifies each file into an approved destination
   - Supports both report-only (`--dry-run`) and mutating (`--apply`) modes

2. **Root File Placement Validation** (`scripts/setup/pre-commit-check.sh`)
   - Ensures only approved files and top-level directories live in the repository root
   - Flags violations with suggested destinations

3. **Misplaced Python Scripts Detection**
   - Dynamically detects Python scripts in root that should be elsewhere
   - Suggests appropriate destinations based on naming patterns

4. **Misplaced Shell Scripts Detection**
   - Detects shell scripts in root that should be under `scripts/`

5. **Script Topology Validation** (`scripts/governance/check_script_topology.py`)
   - Verifies governed canonical locations for setup, pipeline, utility, and reusable package code
   - Ensures compatibility wrappers delegate to canonical implementations
   - Prevents retired broad-mutating organization helpers and historical reports from returning to active script paths

6. **Documentation Structure Validation** (`scripts/governance/check_docs_structure.py`)
   - Validates that documentation follows the approved directory structure

`--docs-only` runs steps 1 and 6 only. It skips root placement and script
topology checks, but still validates documentation topology.

### Running in CI

Use `--check` mode for CI validation:

```bash
./.auto-organize.sh --check
```

This mode:
- Runs all validation steps in dry-run mode
- Exits with code 1 if any violations are detected
- Provides actionable error messages

### Execution Model

You should **not normally call helper scripts directly**. Instead, run:

```bash
./.auto-organize.sh --dry-run      # Inspect proposed moves
./.auto-organize.sh                # Apply moves once you're satisfied
```

This guarantees a consistent ordering of operations and a single point of control for organization behavior.

## Pre-Commit Hook

The repository hook set is installed through the pre-commit framework. Its
root-placement hook calls `scripts/setup/pre-commit-check.sh` to prevent commits
with misplaced files, and `.pre-commit-config.yaml` installs both `pre-commit`
and `pre-push` hook types so push-time checks such as gitleaks are active too.

Repo-wide audits use the same checker in `--all` mode with no grandfathered root-file baseline.

### How It Works

1. Runs automatically before each commit through the configured pre-commit hook set
2. Runs configured push-time checks before `git push`
3. Checks for files in the root directory that should be elsewhere
4. Provides helpful error messages with suggested destinations
5. Allows bypass with `git commit --no-verify` or `git push --no-verify` (not recommended)

## Installation

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Install the repo-managed hook tooling and organization system
make install-core
./scripts/setup/auto-organize-install.sh

# 3. Run initial organization (dry-run first)
./.auto-organize.sh --dry-run

# 4. Apply organization
./.auto-organize.sh
```

### Manual Installation

If you prefer manual setup:

```bash
# Install the repo-managed pre-commit and pre-push hooks
make install-core
make install-hooks

# Verify root placement directly when needed
./scripts/setup/pre-commit-check.sh --all
```

## Usage

### Running the Organization Script

```bash
# Dry-run mode (see what would happen without making changes)
./.auto-organize.sh --dry-run

# Apply organization
./.auto-organize.sh

# Verbose output
./.auto-organize.sh --verbose

# Dry-run with verbose output
./.auto-organize.sh --dry-run --verbose
```

### Testing the Pre-Commit Hook

```bash
# Create a test file in the wrong location
echo "test" > TEST_FILE.md

# Try to commit (should be blocked)
git add TEST_FILE.md
git commit -m "Test commit"

# Move file to correct location
mv TEST_FILE.md docs/guides/

# Commit should now succeed
git add docs/guides/TEST_FILE.md
git commit -m "Add test file"
```

### Bypassing the Hook (Emergency Only)

```bash
# Use --no-verify to skip pre-commit hook
git commit --no-verify -m "Emergency commit"

# Note: This should only be used in emergencies
# Always run .auto-organize.sh afterward to fix organization
```

## File Classification Rules

The organization system uses these rules to classify files:

### Documentation Files (.md, .txt)

- Root README: stays in root (main entry point)
- Current docs navigation: -> `docs/README.md` and `docs/governance/DOCUMENTATION_MAP.md`
- Guide/Tutorial: -> `docs/guides/`
- Architecture: -> `docs/architecture/`
- API docs: -> `docs/api/`
- Deployment: -> `docs/deployment/`
- Version history: -> `docs/version_history/`
- PR-specific or merge-event records: -> `docs/pr_archive/`
- Point-in-time status, session, or delivery artifacts: -> `docs/historical/`
- Retired or consolidated material: -> `docs/_archive/`

### Scripts (.sh, .py)

- Setup/Installation: → `scripts/setup/`
- Automation: → `scripts/automation/`
- Utilities: → `scripts/utilities/`
- Pipeline scripts: → `scripts/pipelines/`
- Reusable runtime/domain code: → `src/transformation_portal/`
- Public compatibility paths in `scripts/`: wrappers only, delegating to canonical implementations
- Retired or superseded scripts: → `archive/scripts/legacy-organization/` or another `archive/scripts/` subdirectory

### Data Files

- Configuration: → `config/`
- Input data: → `data/input/`
- Output data: → `data/output/`
- Cache: → `data/cache/`
- Sample images: → `data/sample_images/`

### Code Files

- Source code: → `src/transformation_portal/`
- Tests: → `tests/`
- Examples: → `examples/`

## Best Practices

### For Developers

1. Run organization regularly: execute `.auto-organize.sh` before committing.
2. Use dry-run first: always test with `--dry-run` before applying changes.
3. Commit organized changes: commit organization changes separately from feature changes.
4. Keep root clean: don’t create new files in root unless they belong there.
5. Use correct directories: place files in their proper directories from the start.

### For New Files

1. Ask yourself: “Does this file need to be in the root?”.
2. If no: place it in the appropriate subdirectory.
3. If yes: ensure it fits one of the allowed root file categories.
4. Add to the relevant validator: root placement belongs in `scripts/setup/pre-commit-check.sh`, script placement belongs in `scripts/governance/check_script_topology.py`, and docs topology belongs in `scripts/governance/check_docs_structure.py`.

### For CI/CD

1. Validate organization: add a CI check that runs `.auto-organize.sh --check`.
2. Fail on violations: fail the build if organization is needed.
3. Report violations: provide clear error messages about misplaced files.

## Troubleshooting

### Script Won’t Run

```bash
# Make sure script is executable
chmod +x .auto-organize.sh

# Check for line ending issues (Windows)
dos2unix .auto-organize.sh  # or
sed -i 's/\r$//' .auto-organize.sh
```

### Pre-Commit Hook Not Working

```bash
# Check if hooks exist
ls -la .git/hooks/pre-commit
ls -la .git/hooks/pre-push

# Reinstall hooks through the repo-managed target
make install-hooks

# Run the root-placement check directly
./scripts/setup/pre-commit-check.sh --all
```

### Files Not Being Moved

```bash
# Run with verbose output to see what's happening
./.auto-organize.sh --dry-run --verbose

# Check if file patterns match
# Edit .auto-organize.sh and add your file to the appropriate section
```

### Organization Breaking Tests

```bash
# If moving files breaks imports or tests:
# 1. Update import paths in code
# 2. Update test fixtures
# 3. Update CI/CD paths
# 4. Update documentation

# Run tests to verify
make test-fast
```

### Merge Conflicts

```bash
# If organization causes merge conflicts:
# 1. Resolve conflicts as usual
# 2. Re-run organization after merge
./.auto-organize.sh

# 3. Commit organization changes
git add .
git commit -m "Re-organize after merge"
```

## Contributing

When contributing to the organization system:

1. Test thoroughly: test on multiple file types and edge cases.
2. Update documentation: update this file with any new rules or patterns.
3. Maintain backward compatibility: don’t break existing organization.
4. Add tests: add test cases for new organization rules.
5. Get review: have changes reviewed by maintainers.

## Support

For questions or issues with the organization system:

- Create an issue in GitHub Issues.
- Review this file and `scripts/setup/README.md`.
- Use GitHub Discussions if enabled.

## Version History

- **v1.2.0** (June 2026): Script topology governance refresh
  - Added script-placement validation through `scripts/governance/check_script_topology.py`
  - Promoted active setup/pipeline/utility implementations into governed subdirectories with public compatibility wrappers
  - Archived retired broad-mutating organization helpers under `archive/scripts/legacy-organization/`
- **v1.1.0** (April 2026): Documentation governance refresh
  - Re-established `docs/README.md` and `DOCUMENTATION_MAP.md` as current navigation
  - Classified old project reports, depth-model notes, and pipeline v1.1.0 material as historical unless explicitly promoted
  - Added repo-wide documentation state audit through PR #1562
- **v1.0.0** (November 2025): Initial release of automated organization system
  - Created `.auto-organize.sh` main script
  - Created pre-commit hook system
  - Established directory structure standards
  - Added comprehensive documentation

---

**Last Updated**: 2026-06-03
**Maintained By**: Transformation Portal Team
**License**: Same as main repository
