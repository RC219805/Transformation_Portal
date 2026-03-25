# Documentation Map

**Purpose:** Single source of truth for finding documentation in Transformation Portal.

**Last Updated:** 2026-03-25
**Maintainer:** Repository Architect

---

## 🎯 Start Here

| Topic | Canonical Document | Purpose |
|-------|-------------------|---------|
| **First Steps** | [README.md](../../README.md) | Project overview, installation, quick start |
| **Setup & Installation** | [SETUP_GUIDE.md](../guides/SETUP_GUIDE.md) | Detailed installation for all tiers |
| **Contributing** | [CONTRIBUTING.md](../../CONTRIBUTING.md) | How to contribute code, docs, issues |
| **Security** | [SECURITY.md](../../SECURITY.md) | Security policy, reporting vulnerabilities |
| **Security Hardening Report** | [security_best_practices_report.md](security_best_practices_report.md) | Security findings and remediation status |

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
| **Code Quality Standards** | [CODE_QUALITY_STANDARDS.md](../guides/CODE_QUALITY_STANDARDS.md) | ✅ Stable |
| **Custom Agents** | [CUSTOM_AGENT_GUIDE.md](../guides/CUSTOM_AGENT_GUIDE.md) | ✅ Stable |
| **TODO Inventory** | [docs/analysis/TODO_INVENTORY.md](../analysis/TODO_INVENTORY.md) | ✅ Stable (v2.4.0) |
| **TODO Quick Reference** | [docs/architecture/TODO_INVENTORY_QUICK_REF.md](../architecture/TODO_INVENTORY_QUICK_REF.md) | ✅ Stable (aligned with v2.4.0) |

### CI/CD & Operations

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CI/CD Workflows** | [docs/ci_cd/CI_CD_WORKFLOWS.md](../ci_cd/CI_CD_WORKFLOWS.md) | ✅ Stable |
| **Workflow Reference** | [.github/workflows/build.yml](../../.github/workflows/build.yml) | ✅ Stable (Primary CI) |
| **Branch Protection** | [BRANCH_PROTECTION_SETUP.md](../ci/BRANCH_PROTECTION_SETUP.md) | ✅ Stable |

### Pipelines & Processing

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **Luxury Estate Pipeline** | [docs/pipeline/LUXURY_ESTATE_PIPELINE_README.md](../pipeline/LUXURY_ESTATE_PIPELINE_README.md) | ✅ Stable |
| **PBR Processing** | [PBR_PROCESSOR_QUICKSTART.md](../guides/PBR_PROCESSOR_QUICKSTART.md) | ✅ Stable |
| **Lux Depth V3 CLI** | [LUX_DEPTH_V3_CLI_GUIDE.md](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) | ✅ Stable |
| **Lux Depth V3 Troubleshooting** | [LUX_DEPTH_V3_TROUBLESHOOTING.md](../guides/LUX_DEPTH_V3_TROUBLESHOOTING.md) | ✅ Stable |
| **Elite Pipeline** | [ELITE_PIPELINE_GUIDE.md](../guides/ELITE_PIPELINE_GUIDE.md) | ✅ Stable |

### Quick References

| Topic | Canonical Document | Status |
|-------|-------------------|--------|
| **CLI Reference** | [docs/cli/CLI_REFERENCE.md](../cli/CLI_REFERENCE.md) | ✅ Stable |
| **PBR Presets** | [PBR_PRESETS_QUICK_REFERENCE.md](../reference/PBR_PRESETS_QUICK_REFERENCE.md) | ✅ Stable |
| **Agent Quick Ref** | [AGENT_QUICK_REFERENCE.md](../reference/AGENT_QUICK_REFERENCE.md) | ✅ Stable |

---

## 🗂️ Specialized Topics

### Format Support
- **File Formats:** [SUPPORTED_FILE_FORMATS.md](../guides/SUPPORTED_FILE_FORMATS.md)
- **TIFF Handling:** [TIFF_FIX_QUICKREF.md](../reference/TIFF_FIX_QUICKREF.md)

### Advanced Features
- **Temporal Architecture:** [TEMPORAL_ARCHITECTURE_QUICKREF.md](../architecture/TEMPORAL_ARCHITECTURE_QUICKREF.md)
- **RAG System:** [RAG_SYSTEM_COMPLETE_GUIDE.md](../guides/RAG_SYSTEM_COMPLETE_GUIDE.md)
- **VFX Extensions:** [VFX_EXTENSION_GUIDE.md](../guides/VFX_EXTENSION_GUIDE.md)

### Troubleshooting
- **General Troubleshooting:** [TROUBLESHOOTING.md](../guides/TROUBLESHOOTING.md)
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
└── reference/                ← Cheat sheets and quick refs
```

---

## Canonical Topology

Legacy root-level duplicates have been retired. Canonical documentation now lives only in approved subdirectories such as:

- [architecture/](../architecture/)
- [ci/](../ci/)
- [cli/](../cli/)
- [guides/](../guides/)
- [historical/](../historical/)
- [performance/](../performance/)
- [reference/](../reference/)

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
