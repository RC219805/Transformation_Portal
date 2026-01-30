# Changelog

All notable changes to the Transformation Portal project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.8.0] - 2026-02-01

### Added
- ✨ **New `depth_canonical` module** - Unified depth processing API consolidating three legacy modules
- ✨ **PBR Map Generation** - Generate Normal, Roughness, and Ambient Occlusion maps from depth
- ✨ **Two-Tier Caching System** - Memory + disk caching for 10-20x speedup in iterative workflows
- ✨ **Auto-Device Detection** - Automatically selects best device (CoreML/ANE, CUDA, MPS, CPU)
- ✨ **ModelRegistry** - Centralized model management with DA2/DA3 support
- ✨ **61 Comprehensive Tests** - 100% coverage of depth_canonical module
- 🛠️ **Migration Script** - `scripts/migrate_to_depth_canonical.py` for automated migration
- 📚 **Complete Migration Guide** - `docs/migration/depth_v2_migration.md` with examples and FAQ
- 🔍 **CI Deprecation Check** - New workflow to prevent deprecated API usage in new code

### Deprecated
- ⚠️ **`transformation_portal.depth`** - Use `depth_canonical` instead (removal in v2.0.0)
- ⚠️ **`transformation_portal.lux_depth_v3`** - Use `depth_canonical` instead (removal in v2.0.0)
- ⚠️ **`transformation_portal.depth_intelligence`** - Use `depth_canonical` instead (removal in v2.0.0)

**Deprecation Timeline:**
- v1.8.0 (Feb 2026): Deprecation warnings added
- v1.9.0 (Apr 2026): Final reminder warnings
- v2.0.0 (Aug 2026): Modules removed

### Changed
- 📝 **README.md** - Added prominent deprecation notice section
- 📝 **Module Docstrings** - All deprecated modules now show DEPRECATED notice with migration guide URL
- 🔧 **Import Warnings** - All deprecated modules issue FutureWarning on import with clear migration path
- 📊 **CI/CD** - New deprecation-check workflow prevents new code from using deprecated APIs

### Fixed
- 🐛 **Fixed .gitignore** - No longer blocks `models/` Python packages
- 🐛 **Fixed CI Tests** - Slow integration tests now properly skipped in CI
- 🐛 **Fixed DA3 Model IDs** - Now use correct HuggingFace model identifiers

### Performance
- ⚡ **Depth Estimation** - 24-65ms @ 4K resolution (CoreML/ANE on Apple Silicon)
- ⚡ **PBR Generation** - ~420ms @ 4K resolution
- ⚡ **Cache Hits** - 10-20x faster than cold start (15ms vs 500ms @ 4K)
- ⚡ **Batch Throughput** - 100-120 images/hour @ 4K resolution

### Migration
**Backward Compatibility:** ✅ All deprecated modules still work via compatibility shims. **Zero breaking changes** in this release.

**Automated Migration:**
```bash
# Scan for deprecated imports
python scripts/migrate_to_depth_canonical.py --scan src/

# Auto-migrate (creates .bak backups)
python scripts/migrate_to_depth_canonical.py --migrate src/
```

**Manual Migration:**
- `ArchitecturalDepthPipeline` → `DepthPipeline`
- `DepthConfig` → `UnifiedDepthConfig`
- `generate_pbr_maps` → Same name, different module (`depth_canonical`)
- `DepthEstimator` → `ModelRegistry`

See [Migration Guide](docs/migration/depth_v2_migration.md) for complete details.

### Testing
- ✅ **12 Deprecation Tests** - Verify warnings, shims, and migration script
- ✅ **61 Depth Canonical Tests** - Complete coverage of new module
- ✅ **CI Integration** - Automated deprecation checking on all PRs

---

## [1.7.0] - 2025-11-01

### Added
- 🏗️ **Context-Aware Rendering System** - Extracts architectural intelligence from construction documents
- 🧠 **Intelligent Strategy Derivation** - Automatically optimizes processing for each space type
- 🎯 **Room-Specific Processing** - Tailored treatment for kitchens, bedrooms, bathrooms, etc.
- 📐 **Dimension-Aware Processing** - Depth processing respects actual room proportions

### Changed
- 📝 **Repository Refactoring** - 92% smaller size (180MB → 15MB)
- ⚡ **60% Faster Imports** - Lazy loading for heavy dependencies
- 📁 **Clear Module Structure** - Organized packages with comprehensive documentation

---

## [1.6.0] - 2025-10-15

### Added
- 🎨 **Material Response System** - Physics-based surface enhancement for wood, metal, glass, textiles
- 📊 **Board Material Aerial Enhancer** - Specialized enhancement for aerial photography
- 🎬 **Luxury Video Master Grader** - Professional video color grading with LUTs

### Fixed
- 🐛 **Depth Pipeline Edge Cases** - Improved handling of extreme depth values
- 🐛 **TIFF Metadata Preservation** - Fixed GPS coordinate handling

---

## [1.5.0] - 2025-09-01

### Added
- 🖼️ **Luxury TIFF Batch Processor** - 16-bit TIFF processing with metadata preservation
- 🎨 **16+ Professional LUTs** - Film Emulation, Location Aesthetics, Material Response
- 📸 **Real-ESRGAN Integration** - 4x upscaling for high-resolution output

### Performance
- ⚡ **Batch Processing** - 400-600 images/hour throughput @ 4K

---

## [1.4.0] - 2025-08-01

### Added
- 🧪 **Depth Anything V2** - Monocular depth estimation with Apple Neural Engine optimization
- 🎯 **ControlNet Integration** - Edge-preserving AI enhancement
- 🎨 **SDXL Refinement** - Photorealistic architectural detail enhancement

### Performance
- ⚡ **CoreML Optimization** - 3-5x speedup on Apple Silicon (M1/M2/M3/M4)

---

## [1.0.0] - 2025-06-01

### Added
- 🚀 **Initial Release** - Core depth pipeline and processing tools
- 🎨 **Basic LUT Support** - Film emulation and color grading
- 🖼️ **Image Processing** - TIFF and JPEG support with metadata preservation

---

[Unreleased]: https://github.com/RC219805/Transformation_Portal/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/RC219805/Transformation_Portal/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/RC219805/Transformation_Portal/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/RC219805/Transformation_Portal/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/RC219805/Transformation_Portal/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/RC219805/Transformation_Portal/compare/v1.0.0...v1.4.0
[1.0.0]: https://github.com/RC219805/Transformation_Portal/releases/tag/v1.0.0
