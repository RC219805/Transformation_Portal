# Repository Organization System

**Transformation Portal** uses an automated file organization system to maintain a clean, structured directory hierarchy. This document describes the organization system, its purpose, and how to use it.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Automated Organization](#automated-organization)
- [Pre-Commit Hook](#pre-commit-hook)
- [Installation](#installation)
- [Usage](#usage)
- [File Classification Rules](#file-classification-rules)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The repository organization system solves the recurring problem of files accumulating in the repository root by:

1. **Automatically organizing files** into appropriate directories based on their type and purpose
2. **Preventing misplaced files** through pre-commit hooks
3. **Maintaining consistency** across the entire codebase
4. **Making the repository easier to navigate** for both humans and tools

### Key Features

- ✅ **Automatic file organization** with `.auto-organize.sh`
- ✅ **Pre-commit validation** to prevent misplaced files
- ✅ **Dry-run mode** for safe testing
- ✅ **Comprehensive logging** for transparency
- ✅ **Cross-platform compatibility** (Linux, macOS, Windows with WSL)

## Directory Structure

The repository follows this standardized structure:

```
Transformation_Portal/
├── .github/                    # GitHub workflows and actions
│   ├── workflows/              # CI/CD workflow definitions
│   └── agents/                 # Custom GitHub Copilot agents
├── assets/                     # Media and static resources
│   ├── brand/                  # Brand assets (logos, colors)
│   ├── luts/                   # Color grading LUTs
│   │   ├── film_emulation/     # Film stock emulations
│   │   ├── location_aesthetic/ # Location-specific profiles
│   │   └── material_response/  # Material Response LUTs
│   ├── images/                 # Sample and reference images
│   ├── videos/                 # Sample videos
│   ├── renders/                # Sample renders
│   ├── models/                 # 3D models and textures
│   └── textures/               # Texture maps
├── config/                     # Configuration files
│   ├── presets/                # Processing presets (YAML)
│   └── profiles/               # User profiles
├── data/                       # Data files
│   ├── input/                  # Input data
│   ├── output/                 # Output data
│   ├── cache/                  # Cached results
│   └── sample_images/          # Sample images for testing
├── docs/                       # Documentation
│   ├── guides/                 # How-to guides and tutorials
│   ├── architecture/           # Architecture documentation
│   ├── api/                    # API documentation
│   ├── deployment/             # Deployment guides
│   └── version_history/        # Changelogs and version notes
├── scripts/                    # All scripts
│   ├── setup/                  # Installation and setup scripts
│   ├── automation/             # Automation scripts
│   └── utilities/              # Utility scripts
├── src/                        # Source code (installable package)
│   └── transformation_portal/  # Main package
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
├── projects/                   # Client/property-specific work
│   ├── 750_picacho_lane/       # Pool processing examples
│   ├── montecito_shores/       # Interior enhancement examples
│   └── README.md               # Project guidelines
├── archive/                    # Historical and temporary files
│   ├── deprecated/             # Deprecated code
│   ├── experiments/            # Experimental features
│   └── legacy/                 # Legacy code
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

The `.auto-organize.sh` script automatically organizes files based on their type and purpose:

### What Gets Organized

| File Type | Destination | Examples |
|-----------|-------------|----------|
| Strategy/Planning docs | `docs/guides/` | `*_PLAN.md`, `*_SUMMARY.md` |
| Architecture docs | `docs/architecture/` | `ARCHITECTURE.md`, `*_DESIGN.md` |
| Utility scripts | `scripts/utilities/` | `navigate.sh`, `verify_*.sh` |
| Setup scripts | `scripts/setup/` | `install_*.sh`, `download_*.py` |
| Data files | `data/` | `*.json`, `*.csv` |
| Sample images | `data/sample_images/` | `*.jpg`, `*.png` |
| Debug artifacts | `archive/` | `debug_*.jpg`, `test_*.png` |
| Deprecated code | `archive/deprecated/` | Old implementations |

### What Stays in Root

Only these files should remain in the repository root:

- **Core documentation**: `README.md`, `LICENSE`, `CONTRIBUTING.md`
- **Build configuration**: `Makefile`, `pyproject.toml`, `setup.py`
- **Dependency management**: `requirements*.txt`, `Pipfile`, `poetry.lock`
- **Testing configuration**: `pytest.ini`, `tox.ini`, `.coveragerc`
- **Linting configuration**: `.pylintrc`, `.flake8`, `mypy.ini`
- **Docker**: `Dockerfile`, `docker-compose.yml`
- **Git**: `.gitignore`, `.gitattributes`
- **CI/CD**: `.travis.yml`, `.circleci/` (but prefer `.github/workflows/`)
- **Organization system**: `.auto-organize.sh`, `REPO_ORGANIZATION.md`

## Pre-Commit Hook

The pre-commit hook (`scripts/setup/pre-commit-check.sh`) prevents commits with misplaced files:

### How It Works

1. Runs automatically before each commit
2. Checks for files in the root directory that should be elsewhere
3. Provides helpful error messages with suggested destinations
4. Allows bypass with `git commit --no-verify` (not recommended)

### Installation

```bash
# Install the pre-commit hook
./scripts/setup/auto-organize-install.sh
```

This creates a symbolic link from `.git/hooks/pre-commit` to `scripts/setup/pre-commit-check.sh`.

## Installation

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Install the organization system
./scripts/setup/auto-organize-install.sh

# 3. Run initial organization (dry-run first)
./.auto-organize.sh --dry-run

# 4. Apply organization
./.auto-organize.sh
```

### Manual Installation

If you prefer manual setup:

```bash
# Make scripts executable
chmod +x .auto-organize.sh
chmod +x scripts/setup/pre-commit-check.sh
chmod +x scripts/setup/auto-organize-install.sh

# Create symbolic link for pre-commit hook
ln -sf ../../scripts/setup/pre-commit-check.sh .git/hooks/pre-commit
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

- **Root README**: Stays in root (main entry point)
- **Guide/Tutorial**: → `docs/guides/`
- **Architecture**: → `docs/architecture/`
- **API docs**: → `docs/api/`
- **Deployment**: → `docs/deployment/`
- **Version history**: → `docs/version_history/`

### Scripts (.sh, .py)

- **Setup/Installation**: → `scripts/setup/`
- **Automation**: → `scripts/automation/`
- **Utilities**: → `scripts/utilities/`
- **Pipeline scripts**: Keep in project root or pipelines/

### Data Files

- **Configuration**: → `config/`
- **Input data**: → `data/input/`
- **Output data**: → `data/output/`
- **Cache**: → `data/cache/`
- **Sample images**: → `data/sample_images/`

### Code Files

- **Source code**: → `src/transformation_portal/`
- **Tests**: → `tests/`
- **Examples**: → `examples/`

### Project Files

- **Client/property work**: → `projects/<project_name>/`
- **Reusable examples**: Consider `examples/` instead
- **Experiments**: → `archive/experiments/`

## Best Practices

### For Developers

1. **Run organization regularly**: Execute `.auto-organize.sh` before committing
2. **Use dry-run first**: Always test with `--dry-run` before applying changes
3. **Commit organized changes**: Commit the organization changes separately from feature changes
4. **Keep root clean**: Don't create new files in root unless they belong there
5. **Use correct directories**: Place files in their proper directories from the start

### For New Files

1. **Ask yourself**: "Does this file need to be in the root?"
2. **If no**: Place it in the appropriate subdirectory
3. **If yes**: Ensure it fits one of the allowed root file categories
4. **Add to `.auto-organize.sh`**: If creating a new file type, add rules to the script

### For CI/CD

1. **Validate organization**: Add a CI check that runs `.auto-organize.sh --dry-run`
2. **Fail on violations**: Fail the build if organization is needed
3. **Report violations**: Provide clear error messages about misplaced files

## Troubleshooting

### Script Won't Run

```bash
# Make sure script is executable
chmod +x .auto-organize.sh

# Check for line ending issues (Windows)
dos2unix .auto-organize.sh  # or
sed -i 's/\r$//' .auto-organize.sh
```

### Pre-Commit Hook Not Working

```bash
# Check if hook exists
ls -la .git/hooks/pre-commit

# Reinstall hook
./scripts/setup/auto-organize-install.sh

# Check hook is executable
chmod +x .git/hooks/pre-commit
chmod +x scripts/setup/pre-commit-check.sh
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

1. **Test thoroughly**: Test on multiple file types and edge cases
2. **Update documentation**: Update this file with any new rules or patterns
3. **Maintain backward compatibility**: Don't break existing organization
4. **Add tests**: Add test cases for new organization rules
5. **Get review**: Have changes reviewed by maintainers

## Support

For questions or issues with the organization system:

- **Create an issue**: [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)
- **Check documentation**: Review this file and `scripts/setup/README.md`
- **Ask in discussions**: [GitHub Discussions](https://github.com/RC219805/Transformation_Portal/discussions)

## Version History

- **v1.0.0** (November 2025): Initial release of automated organization system
  - Created `.auto-organize.sh` main script
  - Created pre-commit hook system
  - Established directory structure standards
  - Added comprehensive documentation

---

**Last Updated**: November 2025  
**Maintained By**: Transformation Portal Team  
**License**: Same as main repository
