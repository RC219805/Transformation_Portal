[![CI](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml)
[![APEX Performance](https://github.com/RC219805/Transformation_Portal/actions/workflows/apex_performance.yml/badge.svg?branch=main)](https://github.com/RC219805/Transformation_Portal/actions/workflows/apex_performance.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](#license)
[![Release](https://img.shields.io/github/v/release/RC219805/Transformation_Portal?sort=semver)](https://github.com/RC219805/Transformation_Portal/releases)

# Transformation Portal

Transformation Portal is a governed image and video processing platform for luxury real estate rendering, architectural visualization, and editorial finishing.

It combines orchestrated depth estimation, PBR map generation, material-aware finishing, optional AI enhancement, and portal/orchestrator surfaces behind explicit contracts, provenance, and reproducibility artifacts.

**📊 [Performance Dashboard](https://rc219805.github.io/Transformation_Portal/)** | **📈 [Latest Metrics](https://rc219805.github.io/Transformation_Portal/latest.html)**

---

## Repository Status

`main` tracks the active development branch for the repository.

Current documentation navigation starts at [docs/README.md](docs/README.md)
and [docs/governance/DOCUMENTATION_MAP.md](docs/governance/DOCUMENTATION_MAP.md).
The May 11, 2026 repo-wide refresh audit remains the inventory baseline for
historical classification decisions.
Recent merged work added typed API v1 envelopes, typed health/readiness and job
lifecycle response models, Docker health/env wiring, CI workflow hardening and
a 30-workflow matrix, archive-gate readiness evidence, APEX fallback /
failure-code hardening, optional FastVLM advisory captioning, portal/frontdoor
modularization, governed RUM/privacy controls, logout UX, dependency
maintenance, and the Portal UX/UI status snapshot.

For reproducible installs, pin a specific release tag from [GitHub Releases](https://github.com/RC219805/Transformation_Portal/releases) instead of relying on branch prose. The release badge above reflects the latest tagged GitHub release.

Core entry points:
- `lux-depth-v3` for orchestrated depth, PBR, materials, and enhancement workflows
- Portal/orchestrator HTTP surfaces with liveness (`/healthz`, `/ready`) plus operator-truth readiness at `/v1/readiness`
- Determinism, manifest, run-card, and provenance layers for governed execution

Quick discovery:
```bash
lux-depth-v3 --help

# If console scripts aren't on PATH, run as module:
python -m transformation_portal.lux_depth_v3 --help

# Portal/orchestrator contract gate:
make test-orchestrator-contract
```

Install a pinned release:
```bash
pip install "git+https://github.com/RC219805/Transformation_Portal.git@<release-tag>"
```

Replace `<release-tag>` with a tag from [GitHub Releases](https://github.com/RC219805/Transformation_Portal/releases).

Key docs:
- [Documentation Index](docs/README.md)
- [Documentation Map](docs/governance/DOCUMENTATION_MAP.md)
- [2026-05-11 Documentation Refresh Audit](docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
- [Portal + Orchestrator Quickstart](docs/guides/PORTAL_ORCHESTRATOR_QUICKSTART.md)
- [Portal Secure Front Door Quickstart](docs/guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md)
- [CI Workflow Matrix](docs/ci/WORKFLOW_MATRIX.md)
- [Archive Gates A/B/C Audit](docs/governance/audit/archive-gates-2026-04-27.md)
- [Custom Agent Guide](docs/guides/CUSTOM_AGENT_GUIDE.md)
- [Portal Orchestrator Roadmap (Re-Baselined)](docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md)
- [Portal Frontdoor Roadmap](docs/architecture/PORTAL_FRONTDOOR_ROADMAP.md)
- [Portal UX/UI Status Snapshot](docs/architecture/DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md)
- [Lux Depth V3 CLI Guide](docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Context-Aware Rendering Guide](docs/guides/CONTEXT_AWARE_RENDERING.md)

Portal surfaces:
- FastAPI on `127.0.0.1:8000` remains the system-of-record origin for the direct-debug portal HTML, `/ready`, and `/v1/*`.
- The secure front door in `web/secure-landing/` is the managed browser entry point and keeps the backend API key out of browser code.
- The managed front door now splits the browser experience into three surfaces:
  - `/` public Dynamic Neural Access homepage
  - `/login` operator login
  - `/portal` governed operator console
- `GET /healthz` is the managed front-door liveness contract, `GET /ready` is backend liveness, and `GET /v1/readiness` is the execution-readiness matrix for the four governed pipelines. The backend health/readiness routes now have typed OpenAPI response models while preserving their existing wire shapes.
- Shared public branding assets now live at `web/secure-landing/public/brand/dna-symbol-*.svg`, `web/secure-landing/public/brand/dna-lockup-*.svg`, and `web/secure-landing/public/video/dna-loop.mp4`.
- Direct FastAPI portal access is now a `direct_debug` workflow for local troubleshooting, not the preferred production browser path.
- The front door is a Node app. `web/secure-landing` now documents and enforces **Node 22.x only** for install, dev, test, build, and start flows.
- Docker Compose reads root `.env` values using the checked-in `.env.example` template and allows missing env files for local defaults; set `TP_API_KEY` for any non-throwaway orchestrator run.

---

## Flagship Capability: Context-Aware Rendering

Context-Aware Rendering extracts architectural intelligence from construction documents (floor plans, elevations, specifications) and uses that context to inform processing decisions.
- Architectural context extraction from PDFs (room types, dimensions, materials, design style)
- Room-specific strategy derivation (kitchen, bedroom, bath, living, outdoor)
- Dimension-aware depth decisions (proportion-respecting depth logic)
- Style-consistent color decisions aligned to design language
- Document provenance: explicit linkage from construction docs → final render decisions

Docs:
- [Context-Aware Rendering Guide](docs/guides/CONTEXT_AWARE_RENDERING.md)

---

## What this repository provides

Core capabilities:
- Context-aware rendering workflows (document-informed processing)
- Depth-aware enhancement (monocular depth + depth-guided processing)
- **PBR Map Generation** (Physically Based Rendering maps: normal, roughness, AO)
- AI-powered refinement (optional ML stack)
- Material Response technology (surface-aware finishing)
- Professional grading looks (LUT library for film/location/material aesthetics)
- TIFF workflows (high bit-depth + metadata preservation, where supported)
- Video grading workflows (FFmpeg-based pipelines)

---

## Depth Models: Production and Research Tiers

Transformation Portal supports depth models across two tiers with different licensing and use cases.

### Production Path
- **DA3 (`da3` backend):** Primary production backend for Lux Depth V3
- **Use for:** The governed depth workflow surface. Select `model_key="da3-metric"` for the Apache-2.0 DA3 path, or `model_key="da3"` / `model_key="da3-research"` for the research-default selector.
- **Requirement:** Install a trusted ML core profile for actual DA3 inference. The checked-in ML core lock is currently target-owned for macOS Apple Silicon (`darwin-arm64`) only; Linux and macOS Intel ML lanes are retired unsupported lanes that fail closed until a governed lane is re-established. For example: `make install-ml-core` or `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu`.
- **Default:** Standard CLI flows resolve here unless a research-only backend is explicitly requested

### Research & Non-Commercial
- **DA3 research selector:** `model_key="da3"` or `model_key="da3-research"`
- **Depth Pro backend:** `depth_pro`
- **Use for:** Explicitly acknowledged research and non-commercial evaluation paths
- **Requirements:** `non_commercial_ok=True`, plus Apple license acceptance for `depth_pro`

**Important:** The research-default DA3 selector is not part of the Apache-2.0 path. See [ADR-015: DA3 1.1 Non-Commercial Research Tier](docs/architecture/ADR-015-da3-1-1-non-commercial-research-tier.md) for governance details.

### Research Preset Example

```python
from transformation_portal.lux_depth_v3 import EnhanceConfig

# Non-commercial research (requires explicit opt-in)
config = EnhanceConfig(
    preset_requested="depth-anything-v3.1-research-m4",
    non_commercial_ok=True,  # Acknowledge CC BY-NC 4.0 restrictions
    depth_device="mps",      # Apple Silicon
)
```

---

## Backend Selection

Lux Depth V3 supports multiple depth estimation backends with automatic fallback for robustness.

### Primary User-Facing Backends

| Backend | Model | License | Focal Length | Metric Depth | Checkpoint Required |
|---------|-------|---------|--------------|--------------|---------------------|
| `da3` (default) | Depth Anything V3 | Commercial-safe production path | ❌ | ❌ | No (auto-download) |
| `depth_pro` | Apple Depth Pro | Research-only with explicit license acceptance | ✅ | ✅ | Yes (1.9 GB) |

The orchestrator also contains an internal `synthetic` fallback path used for explicit test/CI or fallback scenarios. It is not the primary production backend surfaced for normal CLI use.

### Usage

**Default (DA3):**
```bash
lux-depth-v3 --input-dir ./input --output-dir ./output
```

If `./.runtime/Depth-Anything-3/.venv-da3/bin/python` exists, Lux Depth V3
auto-discovers that repo-local DA3 runtime before falling back to the main repo
environment.

**Optional FastVLM advisory captions:** Lux Depth V3 can emit governed FastVLM
caption sidecars with `--vlm-captioning on`. This runtime stays isolated under
`.runtime/fastvlm/.venv-fastvlm`, is called by subprocess only, and is advisory:
it is not used for APEX pass/fail, Materials V3 segmentation, or any quality
gate. Install or verify the manifest-pinned local runtime with
`make install-fastvlm-runtime` and `make check-fastvlm-runtime`. See
[FastVLM Advisory Captioning Runtime](docs/runtimes/fastvlm.md).

**Recommended (DA3 via isolated Depth Anything 3 environment):**
```bash
./scripts/setup/install_da3_runtime.sh

lux-depth-v3 --input-dir ./input --output-dir ./output
```

The repo-local DA3 setup script pins the upstream checkout to a validated ref under
`.runtime/Depth-Anything-3`, keeps the interpreter contract at
`./.runtime/Depth-Anything-3/.venv-da3/bin/python`, captures a
`.runtime/da3-pip-freeze.txt` snapshot for debugging/provenance, and leaves the
main repo `.venv` unchanged. The default DA3 contract uses the PR #110-style
packaging split: NumPy 2, baseline `open3d`, and optional `pycolmap` / `xformers`
profiles rather than upstream `.[all]`. `pycolmap` is pinned by the setup script;
`xformers` is intentionally operator-managed by default because compatible wheels
vary by torch/platform. Set `DA3_XFORMERS_SPEC` to a pinned pip spec for an
environment with a known-good wheel. Use `--da3-python` only when you want to
override that repo-local runtime. Explicit `--depth-backend da3` requests are now
strict: if DA3 cannot be initialized, the command raises an actionable error
instead of silently downgrading to DA2.

**Depth Pro (requires license acceptance):**
```bash
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --depth-backend depth_pro \
  --depth-pro-python ./.venv-depth-pro/bin/python \
  --accept-apple-depth-pro-research-license true \
  --non-commercial-ok true
```

**Python API:**
```python
from transformation_portal.lux_depth_v3 import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from pathlib import Path

# Using Depth Pro
config = EnhanceConfig(
    depth_backend="depth_pro",
    depth_pro_checkpoint_path="checkpoints/depth_pro.pt",
    depth_pro_python_executable=".venv-depth-pro/bin/python",
    accept_apple_depth_pro_research_license=True,
    non_commercial_ok=True,
    depth_device="cpu",
    enable_v2=False,
)

orchestrator = EnhanceOrchestrator(config, Path("./output"))
```

**DA3 live model tests remain opt-in:** set `TP_RUN_HF_MODEL_TESTS=1` before running the
real Hugging Face DA3 integration tests.

### Fallback Behavior

If the requested backend is unavailable, the orchestrator records the resolution outcome in backend metadata and falls back through the configured operational chain. In explicit test or constrained environments, a synthetic fallback path can also be enabled.

### Backend Metadata

All processing manifests include backend selection metadata:
- `requested_backend`: User's requested backend
- `resolved_backend`: Actually used backend
- `resolution_status`: "success" or "fallback"
- `resolution_reason`: Explanation if fallback occurred

See [ADR-019: Depth Backend Unification](docs/architecture/ADR-019-depth-backend-unification.md) for architectural details.

---

## Optional Dependencies

### RAW Camera File Support

Enable direct ingestion of professional camera RAW formats such as CR2, NEF, ARW, and DNG.

```bash
pip install -e ".[raw]"
# or: pip install rawpy
```

RAW inputs are auto-detected and converted into pipeline-ready RGB using LibRaw via `rawpy`. If `./.venv-raw/bin/python` exists, Lux Depth V3 auto-discovers that repo-local RAW runtime before falling back to the main repo environment.

**Recommended (RAW via isolated runtime):**
```bash
./scripts/setup/install_raw_runtime.sh

lux-depth-v3 --input-dir ./input --output-dir ./output
```

Use `--raw-python` only when you want to override that repo-local runtime explicitly:

```bash
lux-depth-v3 --input-dir ./input --output-dir ./output --raw-python ~/venvs/raw/bin/python
```

See [SETUP_GUIDE.md](docs/guides/SETUP_GUIDE.md) for environment details.

---

### Depth Pro (Experimental)

Use `depth_pro` when you need metric depth and are operating in an explicit research-only workflow.

```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
./scripts/setup/install_depth_pro_runtime.sh
```

Keep `depth-pro` in its own environment. Its dependency constraints conflict with
the main repository stack, and the repo-owned setup script pins the known-good
Depth Pro runtime surface (`torch==2.7.1`, `torchvision==0.22.1`,
`numpy==1.26.4`) that restores MPS readiness on current macOS 26.x pip-wheel
hosts.

Required CLI wiring:
```bash
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --depth-backend depth_pro \
  --depth-pro-python ./.venv-depth-pro/bin/python \
  --non-commercial-ok true \
  --accept-apple-depth-pro-research-license true

# Or export once for repeated runs
export TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON=./.venv-depth-pro/bin/python
```

See the [Lux Depth V3 CLI Guide](docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md) for presets, Apple Silicon guidance, and license enforcement details.

---

## PBR Map Generation

Lux Depth V3 can generate physically based rendering maps directly from the full pipeline or from cached depth when you are tuning look-development workflows.

Fast PBR-only run:
```bash
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output/pbr \
  --quality-tier apex \
  --pbr "on" \
  --enable-v2 "off"
```

Typical outputs:
- `*_depth.png` and optional float depth artifacts
- `*_normal.png`
- `*_roughness.png`
- `*_ao.png`

For standalone depth-to-PBR iteration, see [PBR Processor Quick Start](docs/guides/PBR_PROCESSOR_QUICKSTART.md).

---

## Quick Start

Recommended local setup:

```bash
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal
make venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install-core
make check-environment
lux-depth-v3 --help
```

Core-only installs are useful for documentation, contract checks, portal/orchestrator surfaces, and non-ML utilities. Actual depth inference with the default `da3` backend requires the ML tier unless you are intentionally exercising synthetic fallback in a constrained test setup.

Add a trusted ML profile when you need DA3 depth inference, research backends, segmentation, or other model-heavy workflows:

```bash
make install-ml-core
# or use the target-specific bootstrap flow on macOS Apple Silicon:
./scripts/bootstrap/install_ml_stack.sh --profile core-cpu
make test-fast
```

The umbrella ML install path is intentionally disabled until a trusted checked-in umbrella lockfile contract exists again. Linux and macOS Intel ML install lanes are retired unsupported lanes and fail closed by design.

For a more guided environment bring-up, see [SETUP_GUIDE.md](docs/guides/SETUP_GUIDE.md).

---

## Machine-Readable JSON Output for Automation

The metadata extraction tooling supports deterministic machine-mode JSON for CI/CD and automation.

```bash
python scripts/test_metadata_extraction.py --json extract /input/image.CR2
```

Key properties:
- deterministic structure and stable keys
- explicit exit-code semantics
- schema-versioned payloads for automation

Docs:
- [Machine Mode JSON Quick Reference](docs/quick_references/MACHINE_MODE_JSON.md)
- [Machine Mode Contract](docs/api/MACHINE_MODE_CONTRACT.md)

---

## Dependency Management

- Root `requirements*.txt` files are convenience entry points.
- `requirements/` contains the layered source of truth for maintainers.
- If you change dependency inputs, regenerate and commit the matching lockfiles from `requirements/`.

See [AGENTS.md](AGENTS.md) and the `requirements/` Make targets for the supported lock/update workflow.

---

## Repository Layout (high level)

- `src/` installable package source
- `docs/` architecture, guides, contracts, and reports
- `scripts/` operational workflows, validation, and setup helpers
- `tests/` pytest suite and regression gates
- `config/` presets and pipeline configuration
- `tools/` developer and audit utilities

---

## Supported File Formats (summary)

- Images: PNG, JPEG, TIFF, WebP, BMP
- RAW stills: optional via `rawpy` for major camera ecosystems
- Video: MP4, MOV, AVI, MKV where FFmpeg codec support is available
- High-bit-depth and metadata-preserving workflows are supported where the selected pipeline path and dependencies allow

---

## System Requirements
- Python 3.11+
- CPU-only operation supported
- Apple Silicon (`mps`) and CUDA acceleration supported where the selected workflow can use them
- FFmpeg recommended for video workflows

---

## Testing

Use the Make targets first:
```bash
make test-fast
make test-full
make ci
make test-orchestrator-contract
make test-frontdoor-contract
make seed-frontdoor-user
make validate-orchestrator-http
make validate-portal-browser
make validate-frontdoor-browser
make run-frontdoor-local
make run-backend-local
make dev-start
make dev-stop
make audit-pipeline-readiness
```

Readiness and validation tiers:
- `make test-orchestrator-contract` keeps the portal/orchestrator contract suite local and deterministic.
- `make test-frontdoor-contract` keeps the managed front-door Node contract/build gate deterministic under Node 22.x.
- `make seed-frontdoor-user` writes the canonical local managed-frontdoor credential fixture to `/tmp/tp-frontdoor-users.json` using `smoke-admin` / `correct horse battery staple` unless you override the frontdoor credential env vars.
- `make validate-orchestrator-http` runs the live backend smoke against a running FastAPI origin.
- `make validate-portal-browser` launches an isolated local backend, then runs the live portal browser smoke and fails early if `/v1/config-preview` cannot authenticate or validate.
- `make validate-frontdoor-browser` launches isolated local backend and managed front-door runtimes, auto-seeds the canonical local smoke credentials for that ephemeral runtime, and then runs the live browser smoke against the managed proof setup.
- `make run-frontdoor-local` starts the canonical managed front door on `http://localhost:3000`, auto-seeds the same canonical local user fixture when no explicit frontdoor user source is configured, and refuses to fall back to `:3001`. The startup preflight (`web/secure-landing/scripts/preflight-backend-auth.mjs`) requires `TP_FASTAPI_ORIGIN`, validates the configured user source, probes `/v1/config-metadata` with `TP_BACKEND_API_KEY`, and refuses to start on non-OK protected-probe responses.
- `make run-backend-local` starts the FastAPI backend on `127.0.0.1:8000` with `--reload` boundaries that exclude `.runtime/`, `output/`, `tmp/`, `tests/`, and `node_modules/` so pipeline runtime writes do not trigger restarts mid-job.
- `make dev-start` / `make dev-stop` orchestrate the full local stack (env file → backend → readiness wait → frontdoor) and tear it down. See `docs/operations/local_dev.md`.
- `make audit-pipeline-readiness` runs the safe local four-pipeline readiness audit and reports `ready` / `degraded` / `blocked` outcomes, including separate `lux-depth-v3` base vs canary status.

Configuration coherency:
- Use `./scripts/dev/write_local_env.sh` to generate `/tmp/tp-local-http-all-on.env` with `TP_API_KEY` and `TP_BACKEND_API_KEY` bound to the same value; source it in any shell that runs the backend or the frontdoor.
- For Vercel/production: `make check-vercel-env TP_VERCEL_ENV_FILE=...` validates that all required environment variables are present per `docs/operations/frontdoor_vercel_env.md`.

Direct pytest examples:
```bash
pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
```

---

## Performance Monitoring

Performance is treated as a first-class signal in CI.

- APEX workflows publish performance summaries and dashboard updates on `main`
- Determinism and contract gates protect reproducibility, not just raw throughput
- Local baseline capture and comparison are available through the performance ledger tooling

For deeper performance workflows, see:
- [Performance Monitoring Guide](docs/performance/README.md)
- [APEX Real Pipeline Integration Guide](docs/guides/APEX_REAL_PIPELINE_INTEGRATION.md)
- [ADR-024](docs/decisions/ADR-024-performance-regression-authority-canonicalization.md)

---

## Documentation

Start here:
- [Documentation Index](docs/README.md)
- [Documentation Map](docs/governance/DOCUMENTATION_MAP.md)
- [Setup Guide](docs/guides/SETUP_GUIDE.md)
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [Lux Depth V3 CLI Guide](docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Lux Depth V3 Troubleshooting](docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- [API Documentation](docs/api/)

Historical project reports, PR summaries, and 2025 pipeline/depth-model notes are
retained under `docs/` for audit context. Use the documentation map and
[2026-05-11 documentation refresh audit](docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
to distinguish current guidance from archive-only material.

---

## License

This repository is distributed under a proprietary, limited non-commercial evaluation license. Public visibility does not make it open source.

Repository-level restrictions include:
- No commercial use without prior written authorization
- No redistribution or derivative works
- No ML training, benchmarking, extraction, or reuse of repository materials beyond permitted evaluation use

Component-specific terms may add further restrictions for optional integrations:
- DA3 research preset (`depth-anything-v3.1-research-m4`): CC BY-NC 4.0
- Apple Depth Pro (`depth_pro`): Apple AMLR research license with explicit acceptance flags

For exact legal terms, see [LICENSE](LICENSE).

---

## Support and Contact

Author: Richard Cheetham
Brand: Carolwood Estates · RACLuxe Division
Email: info@racluxe.com

Resources:
- GitHub Issues: bug reports and feature requests
- Documentation: docs/
- Examples: examples/

---

Last Updated: 2026-06-03
