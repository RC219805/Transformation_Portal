# Scripts Directory

Organized executable scripts for the Transformation Portal.

## Structure
- **pipelines/** - Pipeline execution scripts (process_*, run_*)
- **utilities/** - Utility scripts (convert_*, fix_*, verify_*)
- **analysis/** - Analysis and diagnostic scripts (analyze_*, diagnose_*)
- **setup/** - Setup and installation scripts (install_*, download_*)
- **automation/** - Automation and workflow scripts

## Organization System

The repository uses an automated file organization system to maintain a clean structure.

### Key Scripts

- **`setup/auto-organize-install.sh`** - Install the organization system and current pre-commit hook
- **`pre_commit_hook.sh`** - Compatibility wrapper for running the unified quality gate directly
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
5. Keep top-level `scripts/*.py` paths as compatibility wrappers only
6. Test with `./.auto-organize.sh --dry-run` to verify organization

See [REPO_ORGANIZATION.md](../docs/governance/REPO_ORGANIZATION.md) for complete guidelines.
