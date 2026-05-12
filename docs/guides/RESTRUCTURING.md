# Repository Restructuring Guide

This document explains the major restructuring performed to align the Transformation Portal repository with modern Python packaging standards.

## Overview

The repository has been restructured to consolidate scattered scripts into a cohesive, installable package and reorganize data and documentation assets into standard, conventional directories.

## Major Changes

### 1. Documentation Consolidation

**Before:**
- Documentation split between `docs/` and `08_Documentation/`

**After:**
- All documentation consolidated into `docs/`
- Organized by topic:
  - `docs/version_history/` - Version history and changelogs
  - `docs/brand/` - Brand assets and specifications
  - `docs/depth_pipeline/` - Depth pipeline documentation
  - `docs/workflow/` - Workflow guides
  - `docs/processing/` - Processing logs

**Migration:**
- `08_Documentation/Version_History/` → `docs/version_history/`
- `08_Documentation/CHANGELOG_CLI_v1_3.md` -> `docs/historical/cli/CHANGELOG_CLI_v1_3.md`
- `08_Documentation/lantern_logo_component_spec.md` → `docs/brand/lantern_logo_component_spec.md`
- `08_Documentation/luxury_rendering_insights.md` → `docs/guides/luxury_rendering_insights.md`
- `08_Documentation/Palette_Assignment_Guide.md` → `docs/brand/Palette_Assignment_Guide.md`

### 2. Asset Organization

**Before:**
- LUTs in numbered directories: `01_Film_Emulation/`, `02_Location_Aesthetic/`, `03_Material_Response/`
- Client deliverables in `09_Client_Deliverables/`

**After:**
- Unified `assets/` directory for all non-code assets:
  - `assets/luts/film_emulation/` (formerly `01_Film_Emulation/`)
  - `assets/luts/location_aesthetic/` (formerly `02_Location_Aesthetic/`)
  - `assets/luts/material_response/` (formerly `03_Material_Response/`)
  - `assets/brand/lantern_logo/` (formerly `09_Client_Deliverables/Lantern_Logo_Implementation_Kit/`)
  - `assets/projects/` (formerly `09_Client_Deliverables/Custom_Projects/`)

**Impact:**
- All LUT path references in code and documentation have been updated
- More intuitive and standard directory structure
- Better programmatic access to assets

### 3. Depth Pipeline Integration

**Before:**
- Standalone `depth_pipeline/` directory at repository root

**After:**
- Integrated into main package: `src/transformation_portal/depth/`
- Maintains all functionality with improved organization

**Import Changes:**
```python
# Old
from depth_pipeline import ArchitecturalDepthPipeline
from depth_pipeline.models import DepthAnythingV2Model
from depth_pipeline.processors import ZoneToneMapping

# New
from transformation_portal.depth import ArchitecturalDepthPipeline
from transformation_portal.depth.models import DepthAnythingV2Model
from transformation_portal.depth.processors import ZoneToneMapping
```

### 4. Root Scripts Migration

**Status:**
- Core implementations moved to `src/transformation_portal/`
- Root scripts preserved as thin CLI wrappers for backward compatibility

**Files:**
- `lux_render_pipeline.py` - Now wraps `src/transformation_portal/pipelines/lux_render_pipeline.py`
- `luxury_video_master_grader.py` - Now wraps `src/transformation_portal/processors/luxury_video_master_grader.py`
- `material_response.py` - Core logic in `src/transformation_portal/processors/material_response/core.py`
- `depth_tools.py` - Now wraps `src/transformation_portal/depth/tools.py`

**Note:** `luxury_tiff_batch_processor` was already a proper package and remains unchanged.

### 5. Utility Scripts Organization

**Before:**
- Utility scripts scattered in repository root

**After:**
- Moved to `scripts/` directory for clarity:
  - `codebase_philosophy_auditor.py` → `scripts/codebase_philosophy_auditor.py`
  - `decision_decay_dashboard.py` → `scripts/decision_decay_dashboard.py`
  - `parse_workflows.py` → `scripts/parse_workflows.py`

**Note:** These tools are also available in `src/transformation_portal/analyzers/` for programmatic use.

## Updated File Structure

```
.
├── assets/                        # All data assets required by the app
│   ├── luts/
│   │   ├── film_emulation/
│   │   ├── location_aesthetic/
│   │   └── material_response/
│   ├── brand/
│   │   └── lantern_logo/
│   └── projects/
│
├── src/
│   └── transformation_portal/     # Main installable package
│       ├── analyzers/             # Code quality and analysis tools
│       ├── cli/                   # CLI entry points
│       ├── depth/                 # Depth pipeline (formerly depth_pipeline/)
│       │   ├── models/
│       │   ├── processors/
│       │   ├── utils/
│       │   ├── pipeline.py
│       │   └── tools.py
│       ├── enhancers/             # Image enhancement modules
│       ├── pipelines/             # High-level workflows
│       ├── processors/            # Core processing units
│       │   ├── material_response/
│       │   ├── luxury_video_master_grader.py
│       │   └── ...
│       ├── rendering/             # Rendering workflows
│       └── utils/                 # Common utilities
│
├── docs/                          # Single source for all documentation
│   ├── brand/
│   ├── depth_pipeline/
│   ├── processing/
│   ├── version_history/
│   ├── workflow/
│   └── ...
│
├── scripts/                       # Auxiliary, non-core scripts
├── tests/                         # All tests
├── examples/                      # Usage examples
├── config/                        # YAML configuration presets
├── luxury_tiff_batch_processor/  # TIFF processing package
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```

## Migration Guide for Users

### Updating Imports

If you have code that imports from the old structure, update as follows:

```python
# Depth pipeline
from depth_pipeline import ArchitecturalDepthPipeline
# →
from transformation_portal.depth import ArchitecturalDepthPipeline

# Material response
from material_response import MaterialResponseValidator
# →
from transformation_portal.processors.material_response.core import MaterialResponseValidator
```

### Updating LUT Paths

If you reference LUT files in your code:

```python
# Old
lut_path = "01_Film_Emulation/Kodak/Kodak_2393_D55.cube"

# New
lut_path = "assets/luts/film_emulation/Kodak/Kodak_2393_D55.cube"
```

### Updating Documentation References

```markdown
<!-- Old -->
See [Version History](08_Documentation/Version_History/changelog.md)

<!-- New -->
See [Version History](docs/version_history/changelog.md)
```

## Benefits

1. **Standard Python Package Structure**: Follows Python packaging best practices (PEP 517, PEP 518)
2. **Improved Organization**: Clear separation of code, assets, documentation, and utilities
3. **Better Maintainability**: Consolidated structure makes it easier to find and update code
4. **Easier Distribution**: Proper package structure enables easier installation and distribution
5. **Backward Compatibility**: Root scripts preserved as thin wrappers for existing workflows

## Testing

All existing tests continue to work with the new structure:

```bash
# Run fast tests
make test-fast

# Run full test suite
make test-full

# Run linting
make lint
```

## Rollout Plan

This restructuring was performed in stages:

1. ✅ Phase 1: Documentation consolidation (`08_Documentation/` → `docs/`)
2. ✅ Phase 2: Asset organization (numbered directories → `assets/`)
3. ✅ Phase 3: Depth pipeline integration (`depth_pipeline/` → `src/transformation_portal/depth/`)
4. ✅ Phase 4: Root scripts migration (thin CLI wrappers)
5. ✅ Phase 5: Utility scripts organization (`scripts/` directory)
6. ✅ Phase 6: Update all path references and documentation

## Notes

- All functionality remains intact - this is purely a structural reorganization
- Backward compatibility maintained through thin CLI wrappers
- Tests updated to work with new import paths
- CI/CD workflows updated to exclude new directories from linting where appropriate

## Questions?

If you encounter any issues with the restructuring, please:
1. Check this guide for migration instructions
2. Review the updated README.md for usage examples
3. Check the test files for examples of the new import structure
4. Open an issue with details about the problem

## Related Documentation

- [README.md](../../README.md) - Main project documentation
- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - System architecture
- [Depth Pipeline README](../depth_pipeline/DEPTH_PIPELINE_README.md) - Depth processing documentation
