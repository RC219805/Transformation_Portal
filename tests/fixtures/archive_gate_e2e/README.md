# Archive Gate E2E Test Fixtures

This directory contains test fixtures for end-to-end testing of archive governance pipelines (Gates A, B, C).

## Structure

```
archive_gate_e2e/
├── README.md                           # This file
├── archive_index.csv                   # Plain text archive index
├── archive_index_normalized.csv.gz     # Compressed archive index (Gate A input)
├── manifest.jsonl                      # Rights manifest (Gate B/C input)
├── rights_flags.yml                    # Rights policy (Gate A rights-apply input)
├── archive_root/                       # Sample archive directory
│   └── DriveA/
│       └── Part1/
│           ├── alpha.txt               # Empty text file
│           └── bravo.dat               # 2-byte data file
└── golden/                             # Expected outputs for regression tests
```

## Gate Input/Output Dependencies

| Gate | Pipeline ID | Required Inputs | Produces |
|------|-------------|-----------------|----------|
| Gate A | `archive-gate-a` | `archive_index_normalized.csv.gz`, `archive_root/` | `hash_manifest.csv.gz`, `manifest.jsonl` |
| Gate B | `archive-gate-b` | `manifest.jsonl` | BagIt bag directory |
| Gate C | `archive-gate-c` | `manifest.jsonl` | METS XML, PROV-JSON, STAC catalog |

## Usage

These fixtures are used by:
- `tests/test_app_orchestrator_runtime.py` - Unit tests for argv generation and readiness
- `tests/test_app_orchestrator_contract_http.py` - HTTP API contract tests
- `scripts/validation/audit_pipeline_readiness.py` - Pipeline readiness audit
- `scripts/validation/validate_portal_browser_smoke.py` - Browser smoke tests
