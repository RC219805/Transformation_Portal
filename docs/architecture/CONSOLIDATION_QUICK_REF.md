# Repository Consolidation - Quick Reference

**Date**: December 23, 2025  
**Status**: ✅ Complete

---

## What Just Happened?

The repository transitioned from **complexity growth** to **consolidation discipline**. No functionality changed—only governance and structure.

---

## Where to Start

### 👤 New Users
→ **[QUICKSTART.md](../../QUICKSTART.md)** - Get started in 2 minutes

### 🔧 Power Users
→ **[docs/advanced/](../advanced/)** - Advanced workflows (async, context-aware, material, video)

### 🧪 Researchers
→ **[docs/research/](../research/)** - Experimental features (NOT production-ready)

### 🏗️ Maintainers
→ This document + [CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md)

---

## Key Changes

### Feature Freeze ❄️
- **Scope**: `lux_depth_v2/` only
- **Policy**: [lux_depth_v2/FEATURE_FREEZE.md](../../lux_depth_v2/FEATURE_FREEZE.md)
- **Impact**: Golden Path is predictable and stable

### Boundary Enforcement 🛡️
- **Rule**: Production code cannot import experimental code
- **Enforcement**: CI ([experimental-boundary.yml](../../.github/workflows/experimental-boundary.yml))
- **Impact**: No production contamination

### Documentation Tiers 📚
- **Production**: QUICKSTART.md, README.md, lux_depth_v2/
- **Advanced**: [docs/advanced/](../advanced/)
- **Research**: [docs/research/](../research/)
- **Impact**: Clear decision tree, no confusion

### CLI Reorganization 🗂️
- **Before**: 10 scripts in root directory
- **After**: All in `scripts/` subdirectories
- **Impact**: Professional appearance, clear separation

### Mission Declared 🎯
- **Primary Identity**: Deployable Image Processing Service
- **Document**: [MISSION_STATEMENT.md](../../MISSION_STATEMENT.md)
- **Impact**: Strategic clarity for all decisions

---

## Governance Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| [MISSION_STATEMENT.md](../../MISSION_STATEMENT.md) | Strategic direction | All |
| [STABILITY_POLICY.md](STABILITY_POLICY.md) | Long-term stability guarantees | Maintainers |
| [lux_depth_v2/FEATURE_FREEZE.md](../../lux_depth_v2/FEATURE_FREEZE.md) | Feature freeze policy | Contributors |
| [CLI_AUDIT_REPORT.md](CLI_AUDIT_REPORT.md) | CLI classification | Maintainers |
| [CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md) | Full execution report | Maintainers |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Contribution guidelines | Contributors |

---

## Quick Navigation

### I want to...

**...process images** → [QUICKSTART.md](../../QUICKSTART.md)  
**...deploy to production** → [deployment/](../../deployment/)  
**...use async pipeline** → [docs/advanced/ASYNC_PIPELINE.md](../advanced/ASYNC_PIPELINE.md)  
**...understand stability tiers** → [STABILITY_POLICY.md](STABILITY_POLICY.md)  
**...contribute code** → [CONTRIBUTING.md](../../CONTRIBUTING.md)  
**...propose new feature** → Check tier: frozen (no), advanced (yes), research (yes)  
**...understand strategic direction** → [MISSION_STATEMENT.md](../../MISSION_STATEMENT.md)  
**...see full consolidation details** → [CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md)

---

## Success Metrics

**Immediately** (this session):
- ✅ 15 files created
- ✅ 3 files modified
- ✅ 16 files reorganized
- ✅ All 4 phases complete

**Next 30 days**:
- 🎯 Zero feature freeze violations
- 🎯 Zero experimental boundary violations
- 🎯 User feedback: "Clear entry point"
- 🎯 100% test pass rate maintained

**Next 90 days**:
- 🎯 Stability metrics validated
- 🎯 Exception process tested
- 🎯 Tier graduation evaluated

---

## Related Documentation

**Entry Points**:
- [QUICKSTART.md](../../QUICKSTART.md) - Start here (95% of users)
- [README.md](../../README.md) - Project overview

**Governance**:
- [MISSION_STATEMENT.md](../../MISSION_STATEMENT.md) - Strategic direction
- [STABILITY_POLICY.md](STABILITY_POLICY.md) - Long-term stability
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - How to contribute

**Technical**:
- [docs/advanced/](../advanced/) - Advanced workflows
- [docs/research/](../research/) - Experimental features
- [lux_depth_v2/](../../lux_depth_v2/) - Production module

---

**Last Updated**: December 23, 2025  
**Next Review**: March 2026 (quarterly)

---

*Governance is not bureaucracy—it's disciplined freedom.*
