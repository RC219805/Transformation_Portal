# Directory Structure Optimization Plan

## Current Issues
1. **110 Python files in root** - Should be in organized subdirectories
2. **11 shell scripts in root** - Should be in scripts/ or tools/
3. **13+ output directories** - Should be consolidated under outputs/
4. **Mixed test files** - Some in root, some in tests/
5. **Scattered examples** - Multiple example files in different locations
6. **Legacy/experimental files** - No clear archive structure

## Proposed Structure

```
transformation_portal/
├── src/transformation_portal/     # Core package (already good)
├── scripts/                        # All executable scripts
│   ├── pipelines/                 # process_*.py, run_*.py
│   ├── utilities/                 # convert_*.py, fix_*.py, verify_*.py
│   ├── analysis/                  # analyze_*.py, diagnose_*.py
│   └── setup/                     # install_*.py, download_*.py
├── examples/                       # All example code
│   ├── pipelines/                 # Pipeline usage examples
│   ├── rag/                       # RAG system examples
│   └── workflows/                 # Workflow demonstrations
├── outputs/                        # All generated outputs (gitignored)
│   ├── 750_picacho/               # Project-specific outputs
│   ├── tests/                     # Test outputs
│   └── archive/                   # Old outputs to keep
├── tests/                          # All test files (already exists)
├── docs/                           # Documentation (already exists)
├── tools/                          # Build/dev tools (already exists)
├── .github/                        # GitHub config, actions, agents
└── archive/                        # Legacy/deprecated code
    ├── experiments/               # Experimental features
    ├── deprecated/                # Old implementations
    └── legacy/                    # Historical code
```

## Migration Strategy

### Phase 1: Create New Structure
- Create new directories: scripts/, outputs/, archive/
- Create subdirectories within each

### Phase 2: Move Files
- Move 110 root .py files to appropriate scripts/ subdirectories
- Move 11 shell scripts to scripts/ or tools/
- Move output directories to outputs/
- Move example files to examples/
- Archive experimental/legacy code

### Phase 3: Update References
- Update imports in moved files
- Update path references in scripts
- Update CI/CD workflows
- Update documentation

### Phase 4: Update .gitignore
- Add outputs/ directory
- Clean up old patterns
- Add archive exceptions if needed

### Phase 5: Cleanup
- Remove empty directories
- Update README.md with new structure
- Run tests to verify nothing broke
