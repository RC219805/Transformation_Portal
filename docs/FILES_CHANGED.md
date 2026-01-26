# Phase 1 Enforcement Foundation - Files Changed

## Summary

13 files created/modified to implement Phase 1 enforcement infrastructure.

---

## New Files Created

### CI/Security Scripts
1. **`scripts/ci/verify_action_pins.py`** (2949 bytes)
   - Staged enforcement for GitHub Actions pinning
   - FAIL: Security-critical workflows
   - WARN: Others (gradual migration)

2. **`scripts/security/verify_banned_dependencies.py`** (3295 bytes)
   - Structural TOML parsing via `tomli`
   - Blocks: `realesrgan`, `gfpgan`
   - Strict mode for CI enforcement

### Workflows
3. **`.github/workflows/enforcement.yml`** (3022 bytes)
   - Five CI jobs: action-pins, banned-deps, layer1-tests, layer2-ml, golden-regression
   - Path-based ML tier triggers
   - Pip caching for speed
   - Artifact boundary validation

### Dependencies
4. **`requirements/constraints.txt`** (205 bytes)
   - Hard blocks banned packages
   - Used with `-c` flag in installs

### Documentation
5. **`docs/PHASE1_ENFORCEMENT_COMPLETE.md`** (6028 bytes)
   - Main Phase 1 summary
   - Risk mitigations documented
   - Pre-push battery instructions
   - Next steps for Draft PR

6. **`artifacts/README.md`** (955 bytes)
   - Symlink portability warning
   - Migration guide
   - .gitignore coverage notes

7. **`tests/fixtures/golden/CURATION.md`** (2121 bytes)
   - Fixture selection rationale
   - Real-world failure modes
   - Growth path (3→10 fixtures)

8. **`tests/fixtures/golden/README.md`** (1301 bytes)
   - Storage guidelines
   - Usage instructions
   - External fixture handling

---

## Modified Files

### Test Configuration
9. **`pytest.ini`**
   - Added markers: `unit`, `regression`, `integration`, `ml`, `golden`
   - Enables tiered test execution

### Build System
10. **`Makefile`**
    - Added `install-core` target (with constraints)
    - Added `install-ml` target (with constraints)
    - Updated help text

---

## Directory Structure Created

```
scripts/
├── ci/
│   └── verify_action_pins.py
└── security/
    └── verify_banned_dependencies.py

requirements/
└── constraints.txt

tests/
└── fixtures/
    └── golden/
        ├── README.md
        └── CURATION.md

.github/
└── workflows/
    └── enforcement.yml

artifacts/
└── README.md

docs/
└── PHASE1_ENFORCEMENT_COMPLETE.md
```

---

## Verification

All files are non-breaking additions except:
- `requirements/constraints.txt` - Blocks banned packages (intentional)
- `.github/workflows/enforcement.yml` - New CI job (parallel, non-blocking)

Run pre-push battery to validate:
```bash
python scripts/ci/verify_action_pins.py
python scripts/security/verify_banned_dependencies.py --strict
pytest -m "unit or regression" --maxfail=3  # if tests exist
make install-core
```

---

*Generated: 2026-01-26*
