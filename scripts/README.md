# Scripts Directory

Organized executable scripts for the Transformation Portal.

## Structure
- **pipelines/** - Pipeline execution scripts (process_*, run_*)
- **utilities/** - Utility scripts (convert_*, fix_*, verify_*)
- **analysis/** - Analysis and diagnostic scripts (analyze_*, diagnose_*)
- **setup/** - Setup and installation scripts (install_*, download_*)
- **validation/** - Validation and contract-check scripts
- **ci/** - Shared CI/local-quality runners
- **verification/** - Local verifier scripts for setup/runtime contracts
- **maintenance/** - Local cleanup and migration helpers
- **automation/** - Automation and workflow scripts

## Organization System

The repository uses an automated file organization system to maintain a clean structure.

### Key Scripts

- **`setup/auto-organize-install.sh`** - Install the organization system and current pre-commit hook
- **`pre_commit_hook.sh`** - Public wrapper for `maintenance/pre_commit_hook.sh`
- **`maintenance/organize_docs.sh`** - Canonical documentation organizer; public wrapper remains `organize_docs.sh`
- **`utilities/pre-commit-quality-check.py`** - Compatibility quality gate used by `pre_commit_hook.sh`
- **`setup/pre-commit-check.sh`** - Standalone root-file-placement check used by the quality gate
- **`governance/check_script_topology.py`** - Governed script-placement and wrapper-contract validator
- **`utilities/verify_organization.sh`** - Verify repository organization

### Quick Start

```bash
# Install organization system
./scripts/setup/auto-organize-install.sh

# Test organization (dry-run)
./.auto-organize.sh --dry-run

# Apply organization
./.auto-organize.sh
```

For detailed documentation, see [REPO_ORGANIZATION.md](../docs/governance/REPO_ORGANIZATION.md) and [setup/README.md](setup/README.md).

## Usage

Most scripts can be run directly:

```bash
# Shell scripts
./scripts/setup/install_da3_runtime.sh

# Python scripts
.venv/bin/python scripts/setup/install_models.py --dry-run
```

## Contributing

When adding new scripts:

1. Place in the appropriate subdirectory based on purpose
2. Make executable: `chmod +x your_script.sh`
3. Add header comments explaining purpose and usage
4. Update this README if adding a new category
5. Keep top-level `scripts/*.py` and `scripts/*.sh` paths as compatibility wrappers only unless explicitly grandfathered or contract-bound like `scripts/enhance_image.py`
6. Keep `scripts/pipelines/` for implementations and runners. Put validation helpers in `scripts/validation/`, diagnostics in `scripts/analysis/`, and usage examples in `examples/`.
7. Test with `./.auto-organize.sh --dry-run` to verify organization

See [REPO_ORGANIZATION.md](../docs/governance/REPO_ORGANIZATION.md) for complete guidelines.
