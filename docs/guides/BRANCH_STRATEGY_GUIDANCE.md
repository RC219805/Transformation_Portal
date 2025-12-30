# Branch Strategy & Sprint Planning Guidance
**Date**: December 20, 2025
**Architect**: Transformation Portal Architect
**Status**: Strategic Recommendation

---

## Executive Summary

After analyzing the three unmerged branches and current repository state, I recommend a **conservative consolidation strategy** that respects the active feature freeze while preparing infrastructure for the next sprint.

**Key Findings**:
- ✅ `main` is clean, stable, and synced with origin
- ⚠️ All three branches contain **massive deletions** (98-157k lines removed) from aggressive cleanup
- ⚠️ Significant overlap exists between branches (same validation script deletions, workflow modifications)
- ✅ Feature freeze is active until Jan 10, 2026
- ✅ External backup exists (.local_backup + 16.5 GB external)

---

## Question 1: Should I Merge or Archive?

### **RECOMMENDATION: Archive All Three Branches**

**Rationale**:

1. **Feature Freeze Compliance**: All three branches introduce changes that violate the active feature freeze:
   - `materials-v3-prw1-w2-water-detection-integration`: Water detection fixes + stone pixel ops (118k lines deleted)
   - `phase2-validation`: Phase 2 validation infrastructure (157k lines deleted)
   - `pr-571`: Dependency fixes + CI fixes (98k lines deleted)

2. **Massive Cleanup Overlap**: All branches delete similar validation scripts, workflow files, and test infrastructure. This suggests they branched before the Golden Path consolidation (commit 82010ef) was completed on `main`.

3. **Risk Assessment**:
   - **High Risk**: Merging would require resolving massive conflicts and could destabilize main
   - **Medium Risk**: Changes may conflict with Golden Path architecture established in main
   - **Low Risk**: Most changes appear to be cleanup-oriented (same scripts deleted across branches)

4. **Backup Status**: External backup provides recovery safety net.

### **Action Plan**:

```bash
# 1. Create archive tags for all branches (preserve work)
git tag archive/materials-v3-water-detection feature/materials-v3-prw1-w2-water-detection-integration
git tag archive/phase2-validation phase2-validation
git tag archive/pr-571 pr-571

# 2. Push tags to remote (backup to GitHub)
git push origin archive/materials-v3-water-detection
git push origin archive/phase2-validation
git push origin archive/pr-571

# 3. Delete local branches (keep workspace clean)
git branch -D feature/materials-v3-prw1-w2-water-detection-integration
git branch -D phase2-validation
git branch -D pr-571

# 4. Verify tags exist
git tag -l "archive/*"
```

**Recovery Path**: If needed later, restore with `git checkout -b materials-v3-recovered archive/materials-v3-water-detection`

---

## Question 2: Implementation Sequencing During Feature Freeze

### **RECOMMENDATION: Infrastructure-Only Preparation**

The feature freeze prohibits **functional changes** but allows **infrastructure preparation** for the next sprint.

### **Permitted Actions (Dec 20 - Jan 10)**:

1. ✅ **Design Documentation** (no code changes):
   - Architecture design documents (ADRs) for edge refinement modules
   - API contracts for post-processing pipeline
   - Performance benchmarking methodology for input-size sweep
   - Integration specifications for bilateral filtering, guided filter

2. ✅ **Test Infrastructure** (test-only changes):
   - Create test harnesses for edge refinement modules
   - Prepare validation datasets for structure scene improvement
   - Design test cases for 518px → 1022px input sweep
   - Mock implementations for bilateral/guided filters (tests only)

3. ✅ **Dependency Analysis** (no installation):
   - Document required libraries for edge refinement (scipy, cv2 extended filters)
   - Create requirements-edge-refinement.txt (not installed yet)
   - Identify version constraints and conflicts

4. ✅ **Workspace Organization**:
   - Clean up temporary files and artifacts
   - Organize validation results and benchmarks
   - Update .gitignore for new experiment directories

### **Prohibited Actions (Until Jan 10)**:

1. ❌ **New Feature Implementation**:
   - No bilateral filtering implementation
   - No guided filter integration
   - No edge-guided enhancement modules
   - No input-size sweep execution

2. ❌ **Pipeline Modifications**:
   - No changes to lux_depth_v2, lux_depth_v3, or high_fidelity_depth
   - No modifications to existing processing workflows
   - No new CLI commands or API endpoints

3. ❌ **Dependency Changes**:
   - No pip install of new packages
   - No requirements.txt updates (except documentation files like requirements-edge-refinement.txt)

### **Proposed Timeline**:

| Phase | Dates | Activities |
|-------|-------|------------|
| **Freeze Period** | Dec 20 - Jan 10 | Design docs, test infrastructure, workspace cleanup |
| **Sprint Kickoff** | Jan 10 - Jan 13 | Dependency installation, module scaffolding |
| **Implementation** | Jan 13 - Jan 31 | Edge refinement modules, input-size sweep |
| **Validation** | Feb 1 - Feb 7 | Structure scene improvement validation (target: 25% → 60%) |

---

## Question 3: Post-Processing Module Architecture

### **RECOMMENDATION: Separate Refinement Module Aligned with Golden Path**

Create a new module that integrates cleanly with the existing Golden Path (`lux_depth_v2`).

### **Proposed Architecture**:

```
lux_depth_v2/
├── pipeline.py              # Existing production pipeline
├── config.py                # Existing configuration
├── upscaling.py             # Existing upscaling backends
├── material_segmentation.py # Existing material detection
└── refinement/              # NEW: Post-processing refinement
    ├── __init__.py
    ├── edge_refinement.py   # Bilateral filtering, edge-guided enhancement
    ├── guided_filter.py     # Guided filter implementation
    ├── structure_enhancer.py # Structure scene improvement orchestrator
    └── config.py            # Refinement-specific configuration
```

### **Integration Points**:

1. **Pipeline Hook**: Add optional refinement stage in `lux_depth_v2/pipeline.py`:
   ```python
   def process_image(self, image, config):
       # Existing stages
       depth_map = self.estimate_depth(image)
       enhanced = self.enhance_materials(image, depth_map)

       # NEW: Optional refinement stage
       if config.enable_edge_refinement:
           from lux_depth_v2.refinement import EdgeRefinement
           refiner = EdgeRefinement(config.refinement_config)
           enhanced = refiner.refine(enhanced, depth_map)

       return enhanced
   ```

2. **Configuration Extension**: Add refinement config to `lux_depth_v2/config.py`:
   ```python
   @dataclass
   class PipelineConfig:
       # Existing fields
       preset: str = "interior_luxury"

       # NEW: Refinement configuration
       enable_edge_refinement: bool = False
       refinement_config: Optional[RefinementConfig] = None
   ```

3. **CLI Flag**: Add `--enable-edge-refinement` flag to `lux-depth-v2` CLI.

### **Design Principles**:

1. ✅ **Opt-In**: Refinement disabled by default (maintains backward compatibility)
2. ✅ **Modular**: Refinement code isolated in separate submodule
3. ✅ **Testable**: Each refinement algorithm has dedicated unit tests
4. ✅ **Configurable**: Refinement parameters exposed via config.py presets

---

## Question 4: Backup Requirements

### **RECOMMENDATION: Create Incremental Backup Before Branch Cleanup**

While an external backup exists (16.5 GB), create a **lightweight incremental backup** before deleting branches.

### **Action Plan**:

```bash
# 1. Create timestamped backup directory
mkdir -p .local_backup/branch_cleanup_$(date +%Y%m%d)

# 2. Export branch diffs (lightweight - only changes, not binary assets)
git diff main..feature/materials-v3-prw1-w2-water-detection-integration > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/materials-v3.diff

git diff main..phase2-validation > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/phase2-validation.diff

git diff main..pr-571 > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/pr-571.diff

# 3. Export commit logs
git log main..feature/materials-v3-prw1-w2-water-detection-integration --oneline > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/materials-v3.log

git log main..phase2-validation --oneline > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/phase2-validation.log

git log main..pr-571 --oneline > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/pr-571.log

# 4. Document backup
echo "Branch cleanup backup created: $(date)" > \
  .local_backup/branch_cleanup_$(date +%Y%m%d)/README.txt
```

**Backup Size**: ~5-10 MB (diffs only, no binary assets)
**Recovery**: Apply diffs with `git apply <diff_file>` if needed

### **Why Not Full Backup?**

- External backup (16.5 GB) already exists
- Tags pushed to GitHub provide remote recovery
- Diffs capture all code changes without duplicating binary assets
- Faster to create and restore

---

## Implementation Checklist

### **Immediate Actions (Today - Dec 20)**:

- [ ] Create archive tags for all three branches
- [ ] Push tags to GitHub remote
- [ ] Create incremental backup (diffs + logs)
- [ ] Delete local branches
- [ ] Clean up workspace (remove temporary files, organize outputs)
- [ ] Update .gitignore for new experiment directories

### **Feature Freeze Period (Dec 20 - Jan 10)**:

- [ ] Write Architecture Decision Record (ADR) for edge refinement module
- [ ] Design API contracts for bilateral filtering, guided filter, edge-guided enhancement
- [ ] Create test infrastructure for refinement modules (test harnesses, mock data)
- [ ] Document input-size sweep methodology (518px → 1022px)
- [ ] Prepare validation datasets for structure scene improvement
- [ ] Draft requirements-edge-refinement.txt (no installation)

### **Sprint Kickoff (Jan 10 - Jan 13)**:

- [ ] Install edge refinement dependencies
- [ ] Create `lux_depth_v2/refinement/` module structure
- [ ] Implement module scaffolding and configuration classes
- [ ] Write unit tests for each refinement algorithm

### **Implementation Phase (Jan 13 - Jan 31)**:

- [ ] Implement bilateral filtering
- [ ] Implement guided filter
- [ ] Implement edge-guided enhancement
- [ ] Execute input-size sweep (518px → 1022px)
- [ ] Integrate refinement module into lux_depth_v2 pipeline

### **Validation Phase (Feb 1 - Feb 7)**:

- [ ] Run structure scene improvement validation
- [ ] Target: 25% → 60%+ improvement
- [ ] Generate validation reports
- [ ] Document results and lessons learned

---

## Risk Assessment

### **High Risk (Addressed)**:

1. ❌ **Merging Stale Branches**: Massive conflicts, destabilization risk
   - **Mitigation**: Archive branches, work from clean main

2. ❌ **Feature Freeze Violation**: Implementing features during freeze period
   - **Mitigation**: Strict infrastructure-only preparation during freeze

### **Medium Risk (Monitored)**:

1. ⚠️ **Architectural Misalignment**: New modules don't fit Golden Path
   - **Mitigation**: Design ADRs during freeze, align with lux_depth_v2 architecture

2. ⚠️ **Dependency Conflicts**: New edge refinement libraries conflict with existing packages
   - **Mitigation**: Draft requirements-edge-refinement.txt during freeze, test in isolated venv

### **Low Risk (Acceptable)**:

1. ✅ **Recovery Complexity**: Restoring archived branches if needed
   - **Mitigation**: Tags pushed to GitHub, diffs in local backup

2. ✅ **Timeline Pressure**: Feb 7 validation deadline
   - **Mitigation**: Phased timeline with buffer (3 weeks implementation + 1 week validation)

---

## Conclusion

**Strategic Recommendation**: Archive all three branches, respect the feature freeze by preparing infrastructure only, and design the edge refinement module to integrate cleanly with the Golden Path architecture.

**Key Benefits**:
- ✅ Maintains main branch stability
- ✅ Respects feature freeze commitments
- ✅ Prepares for efficient sprint execution in January
- ✅ Reduces merge conflict risk
- ✅ Aligns with Golden Path consolidation

**Next Steps**: Execute immediate actions checklist (archive, backup, cleanup), then proceed with design documentation during the freeze period.

---

**Architect Sign-Off**: This strategy prioritizes system stability and long-term maintainability over short-term feature velocity. The feature freeze is a commitment to quality, and we honor that commitment while preparing for a successful next sprint.
