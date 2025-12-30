# Golden Path Consolidation - 1-Week Implementation Checklist

**Period**: December 20-27, 2025
**Objective**: Reduce cognitive load while preserving all capabilities

---

## ✅ Day 1-2: Golden Path Declaration

### README.md Restructure
- [ ] Move lux_depth_v2 to lines 1-100 (currently buried at line 150+)
- [ ] Create "🎯 Golden Path: Production Image Processing" section at top
- [ ] Add 3-command quick start (install, process, deploy)
- [ ] Add visual decision tree: "Processing images in production? → lux-depth-v2"
- [ ] Move Phase 1/2/3 announcements below fold (or to CHANGELOG)
- [ ] Collapse multiple "Latest Update" sections into single CHANGELOG link

### Architecture Decision Record
- [ ] Create `docs/GOLDEN_PATH_DECISION.md` explaining:
  - Why lux_depth_v2 is canonical for production
  - When to use async pipeline (performance layer)
  - When to use unified/context-aware (advanced orchestration)
  - When to use training infrastructure (research/development)

### Validation
- [ ] Test 3-command workflow from scratch (clean environment)
- [ ] Time-to-first-result for new user (target: <5 minutes)
- [ ] Verify all existing links still work

---

## ✅ Day 3-4: Documentation Consolidation

### Training Section Collapse (README.md lines 536-599)
- [ ] **Replace with**:
  ```markdown
  ## Advanced: Model Training (Research)

  Neural network training infrastructure for researchers and advanced users.

  ⚠️ **Not part of default production workflow** - requires GPU, 10GB+ disk, 2-3 hours.

  See [docs/training/TRAINING_GUIDE.md](docs/training/TRAINING_GUIDE.md) for complete training documentation.
  ```
- [ ] Move comprehensive training content to `docs/training/`
- [ ] Update all training doc cross-references

### Edge Refinement Promotion
- [ ] Add `--edge-refinement` boolean flag to `lux_depth_v2/cli.py`
- [ ] Add `edge_refinement: bool = False` to `lux_depth_v2/config.py` presets
- [ ] Update `lux_depth_v2/pipeline.py` to use refinement when enabled
- [ ] Add docstrings explaining when to enable

### Copilot Instructions Update
- [ ] Add "Golden Path" section to `.github/copilot-instructions.md`
- [ ] Clarify lux_depth_v2 as primary production module
- [ ] Update "Common Tasks" to prioritize lux_depth_v2 examples
- [ ] Add training infrastructure location clarification

---

## ✅ Day 5-6: Validation & Testing

### Edge Refinement Validation
- [ ] Generate comparison report (with/without refinement)
  - [ ] Run on 10 test images (interior, exterior, mixed)
  - [ ] Measure: PSNR, SSIM, structure scene pass rate
  - [ ] Visual side-by-side artifacts
- [ ] Create `lux_depth_v2/docs/EDGE_REFINEMENT_VALIDATION.md`
- [ ] Decision: Enable by default OR keep opt-in?
- [ ] Update preset configs based on validation results

### Full Test Suite
- [ ] Run `make test-full` (all tests must pass)
- [ ] Run `make test-lux-depth-v2` (module-specific tests)
- [ ] Run `make lint` (no new linting errors)
- [ ] Verify CI passes on feature branch

### Documentation Review
- [ ] Verify all README links resolve
- [ ] Check that docs/training/ gate works
- [ ] Validate Docker quick start commands
- [ ] Test lux-depth-v2 CLI examples from docs

---

## ✅ Day 7: Freeze & Communication

### Feature Freeze Policy
- [ ] Create `.github/ISSUE_TEMPLATE/feature_freeze_check.md`:
  ```markdown
  ## Feature Freeze Check

  The repository is in a **narrative consolidation period** (Dec 20 - Jan 10).

  **Blocked**:
  - New pipelines/modules
  - New presets/stages/metrics
  - New feature surfaces

  **Allowed**:
  - Security fixes
  - Performance profiling
  - Validation improvements
  - CLI ergonomics (help text, error messages)
  - Bug fixes

  Does this issue add a new feature surface? → Requires architect approval.
  ```

- [ ] Update `CONTRIBUTING.md`:
  - [ ] Add "Feature Freeze Period" section
  - [ ] Explain consolidation objectives
  - [ ] List allowed/blocked changes
  - [ ] Expected end date (Jan 10, 2025)

### Validation Metrics Update
- [ ] Add edge refinement metrics to `lux_depth_v2/README.md`
- [ ] Update `docs/PHASE2_PERFORMANCE.md` with new benchmarks
- [ ] Create `docs/VALIDATION_REPORTS/edge_refinement_dec2025.md`

### Communication
- [ ] Create GitHub Discussion post: "Golden Path Consolidation - Dec 20-Jan 10"
- [ ] Pin discussion for visibility
- [ ] Update project README badges (if needed)

---

## Success Metrics (Before/After)

| Metric | Before | Target After | Measurement |
|--------|--------|--------------|-------------|
| Time to "How do I process 100 images?" | 15-30 min | <2 min | User testing |
| Decision branches presented | 6+ | 1 + "Advanced" | README section count |
| Lines to scroll before lux-depth-v2 | 150+ | <10 | README line numbers |
| Training prominence | Line 536 (high) | Gated (low) | Section position |
| Edge refinement discoverability | Hidden | CLI flag | `--help` output |

---

## Files Changed Summary

### Modified
1. `README.md` - Restructure (Golden Path first, collapse announcements)
2. `.github/copilot-instructions.md` - Add Golden Path guidance
3. `CONTRIBUTING.md` - Add feature freeze policy
4. `lux_depth_v2/cli.py` - Add `--edge-refinement` flag
5. `lux_depth_v2/config.py` - Add `edge_refinement` to presets
6. `lux_depth_v2/pipeline.py` - Integrate refinement logic
7. `lux_depth_v2/README.md` - Add validation metrics

### Created
1. `docs/GOLDEN_PATH_DECISION.md` - Architecture decision record
2. `docs/training/TRAINING_GUIDE.md` - Consolidated training docs
3. `.github/ISSUE_TEMPLATE/feature_freeze_check.md` - Feature freeze template
4. `lux_depth_v2/docs/EDGE_REFINEMENT_VALIDATION.md` - Validation report
5. `docs/VALIDATION_REPORTS/edge_refinement_dec2025.md` - Benchmark data

### Moved
1. Training content from README → `docs/training/`
2. Phase announcements below fold or to CHANGELOG

---

## Risk Mitigation

### If validation shows edge refinement degrades quality:
→ Keep opt-in, document use cases in validation report

### If users complain about training section removal:
→ Point to docs/training/TRAINING_GUIDE.md (all content preserved)

### If feature freeze blocks critical work:
→ Architect approval process for exceptions

### If README restructure breaks external links:
→ Add redirects in docs/README.md, GitHub handles anchor changes

---

## Post-Week Review (Day 8)

- [ ] Measure success metrics (table above)
- [ ] Collect feedback from 2-3 test users
- [ ] Document lessons learned
- [ ] Plan next phase (if needed)

---

**Owner**: Transformation Portal Architect
**Status**: Ready to execute
**Start Date**: December 20, 2025
**Target Completion**: December 27, 2025
