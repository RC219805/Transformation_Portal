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

- **`setup/auto-organize-install.sh`** - Install the organization system and pre-commit hook
- **`setup/pre-commit-check.sh`** - Pre-commit hook that validates file organization
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

For detailed documentation, see [REPO_ORGANIZATION.md](../REPO_ORGANIZATION.md) and [setup/README.md](setup/README.md).

## Usage

Most scripts can be run directly:

```bash
# Shell scripts
./scripts/setup/install_models.sh

# Python scripts
python scripts/utilities/verify_setup.py
```

## Contributing

When adding new scripts:

1. Place in the appropriate subdirectory based on purpose
2. Make executable: `chmod +x your_script.sh`
3. Add header comments explaining purpose and usage
4. Update this README if adding a new category
5. Test with `./.auto-organize.sh --dry-run` to verify organization

See [REPO_ORGANIZATION.md](../REPO_ORGANIZATION.md) for complete guidelines.
