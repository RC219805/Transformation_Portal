# Agent Repository Map

**Purpose**: Guide for autonomous agents working in the Transformation Portal repository. This document defines safe working boundaries, entry points, and change protocols.

**Last Updated**: 2026-01-01

---

## ⚠️ Critical Safety Rules

### Allowed Paths (Agent May Modify)

Agents are **permitted** to modify files in these directories:

- `src/` - Core transformation portal source code
- `lux_depth_v2/` - Production Golden Path pipeline (stable, feature-frozen)
- `lux_depth_v3/` - Next-generation depth processing (read carefully, experimental)
- `docs/` - Documentation (except session notes and WIP files)
- `deployment/` - Docker, docker-compose, deployment configs
- `tests/` - Test suites and fixtures
- `scripts/` - Utility scripts and automation
- `config/` - Pipeline configuration and presets
- Root-level config files: `.gitignore`, `pyproject.toml`, `requirements*.txt`, `Makefile`, etc.

### Forbidden Paths (Agent MUST NOT Modify)

Agents are **prohibited** from modifying these directories without explicit override:

- `phase*_outputs/` - Historical deliverables and validation outputs
- `.local_backup/` - Local backup directory
- `bench*/` - Benchmark results and performance baselines
- `experimental/` - Experimental code and R&D work
- `exploration/` - Exploratory analysis and prototypes
- `forensics/` - Forensic analysis outputs
- `archive/` - Archived legacy code
- `regression_baselines/` - Regression test baseline images
- `validation_images/` - Validation dataset images
- `input_images/` - Sample input images
- `data/` - Training and validation datasets
- `assets/` - LUTs, brand assets, large binary files
- `output*/` - Generated output directories
- `test_output*/` - Test output directories
- `*.log` - Log files
- `*_TEMP.md`, `*_WIP.md`, `SESSION_*.md` (except in docs/) - Temporary documentation

### Why These Restrictions Matter

**Problem**: The repository contains ~180MB of artifacts, outputs, and experimental work. An agent without guardrails will:
1. Edit large binary files or output directories, creating noise in PRs
2. Modify historical deliverables, breaking validation
3. Touch experimental code paths, creating production risks

**Solution**: Explicit allow/deny lists prevent agents from "helpfully" modifying the wrong things.

---

## 📍 Entry Points and CLIs

### Primary Production Entry Points

| Entry Point | Purpose | Location | Status |
|-------------|---------|----------|--------|
| `lux-depth-v2` | Production CLI (batch) | `lux_depth_v2/cli.py` | ✅ Feature-frozen |
| `lux-depth-v2-service` | Production API (FastAPI) | `lux_depth_v2/service.py` | ✅ Deployment-ready |
| `lux-depth-v3` | Next-gen depth (experimental) | `lux_depth_v3/cli.py` | ⚠️ Experimental |
| `lux-depth-v3-service` | Next-gen API | `lux_depth_v3/service.py` | ⚠️ Experimental |

### Legacy/Advanced Entry Points

| Entry Point | Purpose | Location | Status |
|-------------|---------|----------|--------|
| `luxury_video_master_grader.py` | Video color grading | Root | ⚠️ Standalone script |
| `luxury_tiff_batch_processor.py` | Batch TIFF processing | Root | ⚠️ Standalone script |
| `lux_render_pipeline.py` | AI-powered enhancement | Root | ⚠️ Standalone script |
| `material_response.py` | Material enhancement | Root | ⚠️ Standalone script |

### Configuration Files

| File | Purpose | Modify? |
|------|---------|---------|
| `config/*.yaml` | Pipeline presets | ✅ Yes (with care) |
| `lux_depth_v2/config.py` | Golden Path config schema | ✅ Yes (versioned) |
| `pyproject.toml` | Package metadata | ✅ Yes |
| `requirements*.txt` | Dependencies | ✅ Yes (security audit first) |
| `.gitignore` | Git exclusions | ✅ Yes |
| `Makefile` | Build automation | ✅ Yes |
| `docker-compose.yml` | Service orchestration | ✅ Yes |

---

## 🔧 Service Endpoints

### Lux Depth V2 Service (Port 8088)

**Base URL**: `http://localhost:8088`

| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/health` | GET | Health check (always returns `{"ok": true}`) | ✅ Live |
| `/ready` | GET | Readiness check (models loaded, returns 503 if not ready) | ✅ Live |
| `/v2/process` | POST | Process single image with preset | ✅ Live |
| `/metrics` | GET | Prometheus metrics (if observability enabled) | ⚠️ Conditional |

**Rate Limits**: 10 requests/minute per IP (configurable via `LUX_HARDEN_ENABLE_RATE_LIMIT`)

**Max Upload**: 100MB (configurable via `MAX_UPLOAD_SIZE`)

### Lux Depth V2 GPU Service (Port 8089)

Same as above, but with `DEVICE=cuda` for GPU acceleration.

---

## 🧪 Test Infrastructure

### Test Organization

```
tests/
├── core/                  # Core platform tests
├── foundation/            # Foundation layer tests
├── integration/           # Integration tests
├── perceptual/           # Perceptual quality tests
├── stage_graph/          # Pipeline stage tests
├── test_*.py             # Top-level test files
└── conftest.py           # Shared pytest fixtures
```

### Running Tests

```bash
# Fast subset (recommended during development)
make test-fast

# Lux Depth V2 tests only
make test-lux-depth-v2

# Full test suite (parallel if xdist available)
make test-full

# All module tests
make test-all-modules
```

### Test Coverage Requirements

- New features require tests (unit + integration)
- Security changes require security tests
- Performance changes require benchmark validation
- Golden Path changes require regression tests

---

## 🏗️ Architecture Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Golden Path (lux_depth_v2)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ Ingest   │→ │  Depth   │→ │Material  │→ │ Output  ││
│  │ Validate │  │Inference │  │Segment   │  │ Export  ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
│                       ↓                                  │
│              ┌────────────────┐                         │
│              │ Post-Processing│                         │
│              │ (Clarity, etc) │                         │
│              └────────────────┘                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Advanced Workflows (Optional)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Async        │  │ Context-Aware│  │ Material     │  │
│  │ Pipeline     │  │ Rendering    │  │ Response     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│       Next-Gen (lux_depth_v3 + DA3) - Experimental      │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ DA3 Wrapper  │  │ Metric Depth │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Pipeline Core | `lux_depth_v2/pipeline.py` | Main processing orchestration |
| CLI | `lux_depth_v2/cli.py` | Command-line interface |
| Service | `lux_depth_v2/service.py` | FastAPI web service |
| Config | `lux_depth_v2/config.py` | Configuration and presets |
| Orchestrator | `lux_depth_v2/orchestrator.py` | Batch processing with fault isolation |
| Materials V2/V3 | `lux_depth_v2/materials_v*.py` | Material-aware enhancement |
| Upscaling | `lux_depth_v2/upscaling.py` | Safe upscaling backends |

---

## 📋 Agent Change Protocol

### Before Making Changes

1. **Read the relevant code** in allowed paths
2. **Identify the blast radius** - what might break?
3. **Check for existing tests** - run them first
4. **Verify security implications** - especially for service/API changes

### Making Changes

1. **Make minimal edits** - surgical changes only
2. **Follow existing patterns** - don't reinvent the wheel
3. **Add tests** for new functionality
4. **Update documentation** if behavior changes
5. **Run linters** - `make lint`
6. **Run tests** - `make test-fast` or `make test-lux-depth-v2`

### After Making Changes

1. **Run security checks** - `make verify-security`
2. **Test the happy path** - does the Golden Path still work?
3. **Document the change** - update relevant docs
4. **Provide rollback notes** - how to undo if needed
5. **Report progress** - use `report_progress` tool

### Change Checklist Template

```
## Change Summary
- What: <one-line description>
- Why: <business/technical reason>
- Files modified: <count>
- Tests run: <which tests>
- Rollback: <how to undo>

## Before/After Behavior
- Before: <current behavior>
- After: <new behavior>

## Validation
- [ ] Linters pass (`make lint`)
- [ ] Tests pass (`make test-fast`)
- [ ] Security checks pass (`make verify-security`)
- [ ] Golden Path still works (manual smoke test)
- [ ] Documentation updated
```

---

## 🔒 Security Considerations

### Known Vulnerabilities

- **CVE-2024-27763**: Vulnerable `basicsr` / `realesrgan` (mitigated in lux_depth_v2)
  - **Status**: ✅ Mitigated (use `--upscaler-backend torch` instead)
  - **Documentation**: `lux_depth_v2/SECURITY.md`

### Security Best Practices

1. **Never add `basicsr` or `realesrgan`** as dependencies
2. **Validate file paths** before use (prevent path traversal)
3. **Sanitize user inputs** in service endpoints
4. **Use rate limiting** for public APIs
5. **Enforce file size limits** for uploads
6. **Pin dependency versions** with SHA256 hashes where possible

---

## 📊 Performance Characteristics

### Golden Path (lux_depth_v2)

| Metric | CPU | GPU (CUDA) | Apple Silicon (MPS) |
|--------|-----|------------|---------------------|
| Throughput | 127 img/hr | 400 img/hr | 300 img/hr |
| Memory | 2-4 GB | 4-8 GB | 3-6 GB |
| Latency (single) | 24-65 ms | 8-20 ms | 12-30 ms |

### Service Mode

| Metric | Value |
|--------|-------|
| Max concurrency | 1-4 (configurable) |
| Request timeout | 60s |
| Max upload size | 100 MB (default) |
| Rate limit | 10 req/min/IP |

---

## 🎯 Decision-Making Heuristics

### When to Modify Golden Path (lux_depth_v2)

✅ **DO** modify if:
- Security vulnerability discovered
- Bug fix (with regression test)
- Performance optimization (with benchmark proof)
- Documentation improvement
- Test coverage improvement

❌ **DON'T** modify if:
- Adding new experimental features (use lux_depth_v3 instead)
- Breaking backward compatibility
- Adding unproven dependencies
- "Just refactoring" without clear benefit

### When to Use lux_depth_v3

✅ **DO** use lux_depth_v3 for:
- Experimental depth models (DA3, etc.)
- Research and R&D work
- Proof-of-concept features
- Bleeding-edge algorithms

❌ **DON'T** use lux_depth_v3 for:
- Production workloads (use lux_depth_v2)
- Client deliverables
- Mission-critical processing

### When to Add Dependencies

✅ **DO** add dependencies if:
- Security audit passes (`make security-audit`)
- No known CVEs
- Actively maintained (commits in last 6 months)
- Minimal transitive dependencies
- Clear, single-purpose functionality

❌ **DON'T** add dependencies if:
- Duplicates existing functionality
- Adds bloat (>10 new transitive deps)
- Unmaintained (no commits in 12+ months)
- Known security issues
- "Nice to have" feature

---

## 📚 Additional Resources

- [QUICKSTART.md](../QUICKSTART.md) - Get started in 2 minutes
- [README.md](../README.md) - Full feature overview
- [lux_depth_v2/SECURITY.md](../lux_depth_v2/SECURITY.md) - Security guidelines
- [docs/DECISION_GUIDE.md](DECISION_GUIDE.md) - Workflow decision tree
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [docs/CUSTOM_AGENT_GUIDE.md](CUSTOM_AGENT_GUIDE.md) - Agent usage patterns

---

## 🤖 Agent Self-Check

Before submitting a PR, agents should verify:

- [ ] Only modified allowed paths (or have explicit override justification)
- [ ] Did not touch forbidden directories
- [ ] Followed change protocol (before/during/after)
- [ ] Tests pass (`make test-fast`)
- [ ] Linters pass (`make lint`)
- [ ] Security checks pass (`make verify-security`)
- [ ] Documentation updated if behavior changed
- [ ] Can explain rollback procedure
- [ ] PR description includes change checklist

---

**End of Agent Repository Map**
