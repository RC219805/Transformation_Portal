# Documentation Map

**Purpose:** Single source of truth for finding documentation in Transformation Portal.

**Last Updated:** 2026-02-20
**Maintainer:** Repository Architect

---

## 🎯 Start Here

| Topic | Canonical Document | Purpose |
|-------|-------------------|---------|
| **First Steps** | [README.md](README.md) | Project overview, installation, quick start |
| **Setup & Installation** | [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Detailed installation for all tiers |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute code, docs, issues |
| **Security** | [SECURITY.md](SECURITY.md) | Security policy, reporting vulnerabilities |

---

## 📚 Core Documentation

### Development

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **Architecture Overview** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | ✅ Stable |
| **Ingest Determinism Policy (ADR-030)** | [docs/architecture/ADR-030-phase2-deterministic-raw-ingest.md](docs/architecture/ADR-030-phase2-deterministic-raw-ingest.md) | 🔄 Proposed (Phase II) |
| **Determinism Harness Spec (SPEC-DH-001)** | [docs/architecture/specifications/SPEC-DH-001.md](docs/architecture/specifications/SPEC-DH-001.md) | ✅ **NEW** (Phase II, LOCKED) |
| **Certified Bounded Determinism Analysis (ANALYSIS-DH-001)** | [docs/architecture/analysis/ANALYSIS-DH-001.md](docs/architecture/analysis/ANALYSIS-DH-001.md) | ✅ **NEW** (Phase II, Informative) |
| **API Reference** | [docs/api/](docs/api/) | ✅ Stable (Sphinx) |
| **Code Quality Standards** | [docs/CODE_QUALITY_STANDARDS.md](docs/CODE_QUALITY_STANDARDS.md) | ✅ Stable |
| **Testing Guidelines** | [docs/development/TESTING_GUIDE.md](docs/development/TESTING_GUIDE.md) | 🔄 In Progress |
| **Custom Agents** | [docs/CUSTOM_AGENT_GUIDE.md](docs/CUSTOM_AGENT_GUIDE.md) | ✅ Stable |
| **TODO Inventory** | [docs/analysis/TODO_INVENTORY.md](docs/analysis/TODO_INVENTORY.md) | ✅ **NEW** (v2.0.0) |
| **TODO Executive Summary** | [docs/architecture/TODO_INVENTORY_EXECUTIVE_SUMMARY.md](docs/architecture/TODO_INVENTORY_EXECUTIVE_SUMMARY.md) | ✅ **NEW** (v2.0.0) |
| **TODO Quick Reference** | [docs/architecture/TODO_INVENTORY_QUICK_REF.md](docs/architecture/TODO_INVENTORY_QUICK_REF.md) | ✅ **NEW** (v2.0.0) |

### CI/CD & Operations

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CI/CD Overview** | [docs/ci_cd/README.md](docs/ci_cd/README.md) | ✅ Stable |
| **Workflow Reference** | [.github/workflows/build.yml](.github/workflows/build.yml) | ✅ Stable (Primary CI) |
| **Branch Protection** | [docs/BRANCH_PROTECTION_SETUP.md](docs/BRANCH_PROTECTION_SETUP.md) | ✅ Stable |

### Pipelines & Processing

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **Pipeline Overview** | [docs/pipeline/README.md](docs/pipeline/README.md) | 🔄 Needs Update |
| **PBR Processing** | [docs/PBR_PROCESSOR_QUICKSTART.md](docs/PBR_PROCESSOR_QUICKSTART.md) | ✅ Stable |
| **Lux Depth V3 CLI** | [docs/LUX_DEPTH_V3_CLI_GUIDE.md](docs/LUX_DEPTH_V3_CLI_GUIDE.md) | ✅ Stable |
| **Lux Depth V3 Troubleshooting** | [docs/LUX_DEPTH_V3_TROUBLESHOOTING.md](docs/LUX_DEPTH_V3_TROUBLESHOOTING.md) | ✅ Stable |
| **Elite Pipeline** | [docs/ELITE_PIPELINE_GUIDE.md](docs/ELITE_PIPELINE_GUIDE.md) | ✅ Stable |

### Quick References

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CLI Reference** | [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | ✅ Stable |
| **PBR Presets** | [docs/PBR_PRESETS_QUICK_REFERENCE.md](docs/PBR_PRESETS_QUICK_REFERENCE.md) | ✅ Stable |
| **Agent Quick Ref** | [docs/AGENT_QUICK_REFERENCE.md](docs/AGENT_QUICK_REFERENCE.md) | ✅ Stable |

---

## 🗂️ Specialized Topics

### Format Support
- **File Formats:** [docs/SUPPORTED_FILE_FORMATS.md](docs/SUPPORTED_FILE_FORMATS.md)
- **TIFF Handling:** [docs/TIFF_FIX_QUICKREF.md](docs/TIFF_FIX_QUICKREF.md)

### Advanced Features
- **Temporal Architecture:** [docs/TEMPORAL_ARCHITECTURE_QUICKREF.md](docs/TEMPORAL_ARCHITECTURE_QUICKREF.md)
- **RAG System:** [docs/RAG_SYSTEM_COMPLETE_GUIDE.md](docs/RAG_SYSTEM_COMPLETE_GUIDE.md)
- **VFX Extensions:** [docs/VFX_EXTENSION_GUIDE.md](docs/VFX_EXTENSION_GUIDE.md)

### Troubleshooting
- **General Troubleshooting:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Known Issues:** [docs/incidents/](docs/incidents/)

---

## 📦 Directory Organization

```
docs/
├── DOCUMENTATION_MAP.md      ← You are here
├── architecture/             ← Architecture ADRs + specs + analysis
│   ├── specifications/       ← Normative specs (LOCKED)
│   └── analysis/             ← Informative analysis
├── ci_cd/                    ← CI/CD documentation
├── fixes/                    ← Bug fix documentation and postmortems
├── guides/                   ← User guides and tutorials
├── pipeline/                 ← Pipeline-specific documentation
├── quick_references/         ← Cheat sheets and quick refs
├── reference/                ← Technical reference docs
├── archive/                  ← Deprecated/historical docs
└── [root level docs]         ← Active, canonical docs only
```

---

## 🚫 Deprecated Documentation

The following docs are **deprecated** and will be removed:

### Duplicate Architecture Docs → Use [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- ~~docs/ARCHITECTURE_PHILOSOPHY.md~~ (merged into ARCHITECTURE.md)
- ~~docs/ARCHITECTURAL_CONTEXT_INTEGRATION.md~~ (superseded)
- ~~docs/ARCHITECTURAL_WORKFLOW.md~~ (moved to ci/)

### Duplicate Quality Docs → Use [docs/CODE_QUALITY_STANDARDS.md](docs/CODE_QUALITY_STANDARDS.md)
- ~~docs/CODEBASE_QUALITY_STANDARDS.md~~ (duplicate)
- ~~docs/CODE_QUALITY_BASELINE.md~~ (superseded)
- ~~docs/CODE_QUALITY_SYSTEM.md~~ (merged)
- ~~docs/QUALITY_CONTROL_SYSTEM.md~~ (merged)

### Duplicate CI Docs → Use [docs/ci/README.md](docs/ci/README.md)
- ~~docs/CI_FIXES_COMPLETED.md~~ (archived)
- ~~docs/CI_003_COMPLETION.md~~ (archived)
- ~~docs/CI_CD_FIXES_REPORT.md~~ (archived)
- ~~docs/CI_SECURITY_FIXES.md~~ (archived)

### Session/Status Reports → Moved to [docs/archive/](docs/archive/)
- ~~docs/COMMIT_SUMMARY.md~~
- ~~docs/FILES_CHANGED.md~~
- ~~docs/FILES_COMMITTED.md~~
- ~~docs/PUSH_SUMMARY.md~~
- ~~docs/STATUS.md~~
- ~~docs/SUMMARY.md~~
- ~~docs/TASK_COMPLETION_SUMMARY.md~~

### Project-Specific (750 Picacho) → Moved to [docs/projects/750_picacho/](docs/projects/750_picacho/)
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
grep -r "keyword" docs/*.md

# List all guides
ls docs/guides/

# View map
cat DOCUMENTATION_MAP.md
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

**Questions?** See [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue.
