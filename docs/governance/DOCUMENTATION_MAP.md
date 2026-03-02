# Documentation Map

**Purpose:** Single source of truth for finding documentation in Transformation Portal.

**Last Updated:** 2026-03-01
**Maintainer:** Repository Architect

---

## 🎯 Start Here

| Topic | Canonical Document | Purpose |
|-------|-------------------|---------|
| **First Steps** | [README.md](../../README.md) | Project overview, installation, quick start |
| **Setup & Installation** | [docs/guides/SETUP_GUIDE.md](../guides/SETUP_GUIDE.md) | Detailed installation for all tiers |
| **Contributing** | [CONTRIBUTING.md](../../CONTRIBUTING.md) | How to contribute code, docs, issues |
| **Security** | [SECURITY.md](../../SECURITY.md) | Security policy, reporting vulnerabilities |
| **Security Hardening Report** | [security_best_practices_report.md](../../security_best_practices_report.md) | Security findings and remediation status |

---

## 📚 Core Documentation

### Development

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **Architecture Overview** | [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) | ✅ Stable |
| **Ingest Determinism Policy (ADR-030)** | [docs/architecture/ADR-030-phase2-deterministic-raw-ingest.md](../architecture/ADR-030-phase2-deterministic-raw-ingest.md) | ✅ Implemented (Phase II) |
| **Determinism Harness Spec (SPEC-DH-001)** | [docs/architecture/specifications/SPEC-DH-001.md](../architecture/specifications/SPEC-DH-001.md) | ✅ LOCKED (Phase II) |
| **Certified Bounded Determinism Analysis (ANALYSIS-DH-001)** | [docs/architecture/analysis/ANALYSIS-DH-001.md](../architecture/analysis/ANALYSIS-DH-001.md) | ✅ Informative (Phase II) |
| **API Reference** | [docs/api/](../api/) | ✅ Stable (Sphinx) |
| **Code Quality Standards** | [docs/CODE_QUALITY_STANDARDS.md](../CODE_QUALITY_STANDARDS.md) | ✅ Stable |
| **Custom Agents** | [docs/CUSTOM_AGENT_GUIDE.md](../CUSTOM_AGENT_GUIDE.md) | ✅ Stable |
| **TODO Inventory** | [docs/analysis/TODO_INVENTORY.md](../analysis/TODO_INVENTORY.md) | ✅ Stable (v2.0.0) |
| **TODO Quick Reference** | [docs/architecture/TODO_INVENTORY_QUICK_REF.md](../architecture/TODO_INVENTORY_QUICK_REF.md) | ✅ Stable (v2.0.0) |

### CI/CD & Operations

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CI/CD Workflows** | [docs/ci_cd/CI_CD_WORKFLOWS.md](../ci_cd/CI_CD_WORKFLOWS.md) | ✅ Stable |
| **Workflow Reference** | [.github/workflows/build.yml](../../.github/workflows/build.yml) | ✅ Stable (Primary CI) |
| **Branch Protection** | [docs/BRANCH_PROTECTION_SETUP.md](../BRANCH_PROTECTION_SETUP.md) | ✅ Stable |

### Pipelines & Processing

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **Luxury Estate Pipeline** | [docs/pipeline/LUXURY_ESTATE_PIPELINE_README.md](../pipeline/LUXURY_ESTATE_PIPELINE_README.md) | ✅ Stable |
| **PBR Processing** | [docs/PBR_PROCESSOR_QUICKSTART.md](../PBR_PROCESSOR_QUICKSTART.md) | ✅ Stable |
| **Lux Depth V3 CLI** | [docs/LUX_DEPTH_V3_CLI_GUIDE.md](../LUX_DEPTH_V3_CLI_GUIDE.md) | ✅ Stable |
| **Lux Depth V3 Troubleshooting** | [docs/LUX_DEPTH_V3_TROUBLESHOOTING.md](../LUX_DEPTH_V3_TROUBLESHOOTING.md) | ✅ Stable |
| **Elite Pipeline** | [docs/ELITE_PIPELINE_GUIDE.md](../ELITE_PIPELINE_GUIDE.md) | ✅ Stable |

### Quick References

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CLI Reference** | [docs/cli/CLI_REFERENCE.md](../cli/CLI_REFERENCE.md) | ✅ Stable |
| **PBR Presets** | [docs/PBR_PRESETS_QUICK_REFERENCE.md](../PBR_PRESETS_QUICK_REFERENCE.md) | ✅ Stable |
| **Agent Quick Ref** | [docs/AGENT_QUICK_REFERENCE.md](../AGENT_QUICK_REFERENCE.md) | ✅ Stable |

---

## 🗂️ Specialized Topics

### Format Support
- **File Formats:** [docs/SUPPORTED_FILE_FORMATS.md](../SUPPORTED_FILE_FORMATS.md)
- **TIFF Handling:** [docs/TIFF_FIX_QUICKREF.md](../TIFF_FIX_QUICKREF.md)

### Advanced Features
- **Temporal Architecture:** [docs/TEMPORAL_ARCHITECTURE_QUICKREF.md](../TEMPORAL_ARCHITECTURE_QUICKREF.md)
- **RAG System:** [docs/RAG_SYSTEM_COMPLETE_GUIDE.md](../RAG_SYSTEM_COMPLETE_GUIDE.md)
- **VFX Extensions:** [docs/VFX_EXTENSION_GUIDE.md](../VFX_EXTENSION_GUIDE.md)

### Troubleshooting
- **General Troubleshooting:** [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- **Known Issues:** [docs/incidents/](../incidents/)

---

## 📦 Directory Organization

```
docs/
├── README.md
├── governance/
│   ├── DOCUMENTATION_MAP.md  ← You are here
│   ├── DOCUMENTATION_POLICY.md
│   └── REPO_ORGANIZATION.md
├── architecture/             ← Architecture ADRs + specs + analysis
├── api/                      ← API documentation
├── ci/                       ← CI/CD governance docs
├── cli/                      ← CLI references
├── historical/               ← Historical execution artifacts
├── pipeline/                 ← Pipeline-specific documentation
├── pr_archive/               ← PR-specific archives
└── quick_references/         ← Cheat sheets and quick refs
```

---

## 🚫 Deprecated Documentation

The following docs are **deprecated** and will be removed:

### Duplicate Architecture Docs → Use [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
- ~~docs/ARCHITECTURE.md~~ (legacy root-level location)
- ~~docs/ARCHITECTURE_PHILOSOPHY.md~~ (merged into ARCHITECTURE.md)
- ~~docs/ARCHITECTURAL_CONTEXT_INTEGRATION.md~~ (superseded)
- ~~docs/ARCHITECTURAL_WORKFLOW.md~~ (moved to ci/)

### Duplicate Quality Docs → Use [docs/CODE_QUALITY_STANDARDS.md](../CODE_QUALITY_STANDARDS.md)
- ~~docs/CODEBASE_QUALITY_STANDARDS.md~~ (duplicate)
- ~~docs/CODE_QUALITY_BASELINE.md~~ (superseded)
- ~~docs/CODE_QUALITY_SYSTEM.md~~ (merged)
- ~~docs/QUALITY_CONTROL_SYSTEM.md~~ (merged)

### Duplicate CI Docs → Use [docs/ci/README.md](../ci/README.md)
- ~~docs/CI_FIXES_COMPLETED.md~~ (archived)
- ~~docs/CI_003_COMPLETION.md~~ (archived)
- ~~docs/CI_CD_FIXES_REPORT.md~~ (archived)
- ~~docs/CI_SECURITY_FIXES.md~~ (archived)

### Session/Status Reports → Moved to [docs/archive/](../archive/)
- ~~docs/COMMIT_SUMMARY.md~~
- ~~docs/FILES_CHANGED.md~~
- ~~docs/FILES_COMMITTED.md~~
- ~~docs/PUSH_SUMMARY.md~~
- ~~docs/STATUS.md~~
- ~~docs/SUMMARY.md~~
- ~~docs/TASK_COMPLETION_SUMMARY.md~~

### Project-Specific (750 Picacho) → Moved to [docs/projects/750_picacho/](../projects/750_picacho/)
- ~~docs/750_PICACHO_RESOLUTION.md~~
- ~~docs/ENHANCEMENT_PLAN_750Picacho_Kitchen.md~~

---

## 📋 Maintenance Protocol

### When Creating New Documentation:
1. Check this map first - does a canonical doc already exist?
2. If yes, update the canonical doc (don't create a new one)
3. If no, create in appropriate subdirectory
4. Update this map with new canonical doc

### When Updating Documentation:
1. Update the canonical doc only
2. Add deprecation notice to duplicates
3. Schedule removal after 30 days

### Deprecation Template:
```markdown
> ⚠️ **DEPRECATED**
>
> This document has been superseded by [CANONICAL_DOC.md](path/to/canonical.md).
> Please use that document instead. This file will be removed on YYYY-MM-DD.
```

---

## 🔍 Finding Documentation

**Quick Search:**
```bash
# Find doc by keyword
grep -R "keyword" docs/

# List all guides
ls docs/guides/

# View map
cat docs/governance/DOCUMENTATION_MAP.md
```

**GitHub Search:**
Use GitHub's search with `path:docs/` filter.

---

## 📝 Status Legend

- ✅ **Stable:** Complete, maintained, canonical
- 🔄 **In Progress:** Being actively updated
- ⚠️ **Deprecated:** Being phased out, see replacement
- 📦 **Archived:** Historical reference only

---

**Questions?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open an issue.
