# Scripts Directory

**Purpose**: Development, testing, validation, and project-specific tools

---

## Directory Structure

```
scripts/
├── validation/         # Validation and metrics extraction
├── testing/            # Development testing scripts
├── maintenance/        # Repository maintenance and cleanup
└── projects/           # Project-specific scripts
    ├── phase1/         # Phase 1 specific
    ├── 750_picacho/    # 750 Picacho project
    └── montecito/      # Montecito project
```

---

## Production vs Development

**⚠️ Scripts in this directory are NOT production CLIs.**

**Production CLIs** (user-facing):
- `lux-depth-v2` (installed via `pip install -e .`)
- `lux-depth-v2-service` (installed via `pip install -e .`)

**Development scripts** (maintainer-facing):
- Everything in `scripts/`
- Everything in `tools/`

---

## Related Documentation

- [CLI Audit Report](../docs/architecture/CLI_AUDIT_REPORT.md) - Full classification
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

---

*Scripts directory: where development happens before it becomes production.*
