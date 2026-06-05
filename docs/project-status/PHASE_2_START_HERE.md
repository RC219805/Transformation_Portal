# 🚀 Phase 2: Start Here

**Last Updated**: 2024-02-14
**Time Required**: 2-3 hours
**Difficulty**: 🟢 LOW

---

## Quick Start (30 seconds)

```bash
# Option 1: Archived automation, retained for historical reference only
archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh

# Option 2: Manual execution
# Follow: docs/project-status/PHASE_2_QUICK_REFERENCE.md
```

---

## What is Phase 2?

**Goal**: Extract and organize Materials V3 investigation documentation and diagnostic scripts.

**What you'll extract**:
- 📚 Investigation reports (edge artifacts, color grading analysis)
- 🛠️ Diagnostic tools (depth analyzer, visual comparator)
- 📖 Methodology documentation (how to debug Materials V3)

**Why it matters**:
- Preserves critical debugging knowledge
- Provides reusable diagnostic tools
- Trains team on systematic debugging

---

## Three Ways to Execute

### 🏃 Fast Track (2-2.5 hours)
**For**: Quick execution with automated help

The historical automation now lives at
`archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh` and
is retained only as evidence. It performs branch checkout, pull, stash, commit,
and push operations from 2024 guidance. Prefer the manual execution path for any
current investigation recovery.

The script will:
- ✅ Handle steps 1-3 automatically (prep, search, reorganize)
- ⏸️ Pause for manual steps 4-5 (extract additional files, create docs)
- ✅ Handle steps 6-8 automatically (update refs, validate, commit)

**Documentation needed during pauses**:
- Use templates from `PHASE_2_EXECUTION_PLAN.md` (Part 5)

---

### 🎯 Guided Track (2.5-3 hours)
**For**: Step-by-step control with clear instructions

**Start with**: `PHASE_2_QUICK_REFERENCE.md`
- Read one-page overview
- Follow 8-step checklist
- Execute commands manually

**Reference**: `PHASE_2_EXECUTION_PLAN.md`
- Detailed instructions for each step
- Templates for documentation
- Troubleshooting guidance

---

### 📚 Deep Dive Track (3-3.5 hours)
**For**: Complete understanding before execution

**Read in order**:
1. `PHASE_2_QUICK_REFERENCE.md` (5 min)
2. `PHASE_2_EXECUTION_PLAN.md` Parts 1-3 (15 min)
3. Execute using detailed plan (2.5 hours)
4. Create PR with `PHASE_2_PR_TEMPLATE.md` (15 min)

---

## Document Quick Reference

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **START_HERE.md** | Quick start guide | Right now! |
| **QUICK_REFERENCE.md** | One-page cheat sheet | During execution |
| **EXECUTION_PLAN.md** | Comprehensive plan | Detailed instructions needed |
| **PR_TEMPLATE.md** | PR description | When creating PR |
| **execute_phase_2_extraction_legacy.sh** | Archived automation evidence | Historical reference only |

---

## Pre-Flight Checklist

Before you start (2 minutes):

- [ ] You have 2-3 hours available
- [ ] Working directory is clean (`git status`)
- [ ] On main branch (`git branch`)
- [ ] Up to date (`git pull origin main`)
- [ ] Read this document (you're here!)

---

## Expected Output

After completing Phase 2:

```
✅ Created branch: docs/materials-v3-investigations
✅ Organized files:
   - docs/investigations/materials_v3/ (investigation reports)
   - tools/investigations/materials_v3/ (diagnostic tools)
✅ Created 3 README files (index, methodology, tools)
✅ Updated file references
✅ Validated scripts and links
✅ Committed and pushed
✅ PR created and ready for review
```

---

## What Happens in Each Step?

### Step 1: Preparation (10 min)
**Automated** ✅
- Create branch
- Create directories
- Set up structure

### Step 2: Search Branches (20 min)
**Automated** ✅
- Search 3 development branches
- Document discovered files
- Generate findings report

### Step 3: Reorganize Files (30 min)
**Automated** ✅
- Move investigation reports to new structure
- Move diagnostic scripts from root
- Rename files for clarity

### Step 4: Extract Additional (40 min)
**Manual** ⏸️
- Review discovered files
- Cherry-pick any missing investigations
- Extract additional tools

### Step 5: Create Documentation (60 min)
**Manual** ⏸️
- Write `docs/investigations/materials_v3/README.md`
- Write `DIAGNOSTIC_METHODOLOGY.md`
- Write `tools/investigations/materials_v3/README.md`
- Templates provided in EXECUTION_PLAN.md Part 5

### Step 6: Update References (20 min)
**Automated** ✅
- Fix relative paths in moved files
- Update cross-references
- Clean up links

### Step 7: Validation (20 min)
**Automated** + Manual
- Test scripts execute `--help`
- Check markdown linting
- Verify links (manual check)

### Step 8: Commit & PR (15 min)
**Automated** ✅
- Commit in logical groups
- Push to remote
- Instructions for PR creation

---

## Time Budget

| Step | Auto/Manual | Time |
|------|-------------|------|
| 1. Preparation | ✅ Auto | 10 min |
| 2. Search | ✅ Auto | 20 min |
| 3. Reorganize | ✅ Auto | 30 min |
| 4. Extract | ⏸️ Manual | 40 min |
| 5. Documentation | ⏸️ Manual | 60 min |
| 6. Update Refs | ✅ Auto | 20 min |
| 7. Validation | ✅/⏸️ Mixed | 20 min |
| 8. Commit/PR | ✅ Auto | 15 min |
| **TOTAL** | | **~3.5 hours** |

**Automated**: ~1.5 hours
**Manual**: ~1.5 hours
**Contingency**: +30 min

---

## If Something Goes Wrong

### Common Issues

**"Script fails at branch search"**
- Some branches may not exist locally
- Script will skip and continue
- Not a blocker

**"Can't find files to move"**
- Files may already be in correct location
- Check `git status` for what changed
- Continue to next step

**"Import error in diagnostic scripts"**
- Check `requirements.txt` has PIL, numpy
- Run `pip install -r requirements.txt`
- Scripts use core deps only

**"Taking too long"**
- Skip optional documentation sections
- Focus on core README files
- Can enhance documentation in follow-up PR

### Getting Help

1. **Check troubleshooting** in EXECUTION_PLAN.md Part 6
2. **Review execution log** at `docs/project-status/phase_2_execution.log`
3. **Escalate** if governance/security decisions needed

---

## Success Looks Like

When Phase 2 is complete:

```
✅ Investigation Reports Organized
   - Primary Bedroom edge artifacts documented
   - Sky/water analysis documented
   - Clear file structure

✅ Diagnostic Tools Ready
   - Scripts work (`--help` executes)
   - Clear usage documentation
   - Integration examples provided

✅ Methodology Documented
   - 6-stage debugging approach written
   - Quantitative thresholds defined
   - Validation checklist created

✅ PR Merged
   - Comprehensive description
   - Linked to related PRs
   - CI passing
```

---

## Next Steps After Phase 2

1. **Merge PR** (wait for review approval)
2. **Announce** new documentation to team
3. **Update** onboarding materials
4. **Execute Phase 3**: Config presets (1-2 hours)

---

## Ready to Start?

### Choose Your Path:

**🏃 Fast Track (Recommended)**:
```bash
archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh
```

**🎯 Guided Track**:
1. Open `PHASE_2_QUICK_REFERENCE.md`
2. Follow 8-step checklist
3. Execute commands

**📚 Deep Dive**:
1. Read `PHASE_2_EXECUTION_PLAN.md`
2. Understand each step
3. Execute with full context

---

**Questions?** Check these documents:
- Quick reference: `PHASE_2_QUICK_REFERENCE.md`
- Detailed plan: `PHASE_2_EXECUTION_PLAN.md`
- PR template: `PHASE_2_PR_TEMPLATE.md`
- This summary: `phase_2_deliverables_summary.md`

**Good luck! 🚀**

---

**Phase 2 of 6** | **Extraction Strategy**: `PR934_EXTRACTION_STRATEGY.md`
