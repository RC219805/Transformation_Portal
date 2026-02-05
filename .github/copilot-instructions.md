````markdown
# Copilot Instructions — Transformation Portal (RC219805)

You are working in a production-grade **Image + Video Processing Transformation Portal**.
This repo is a **Context-Aware Rendering engine** for luxury real estate / ArchViz post-production:
raw pixels enter; semantic + geometric intelligence (depth, materials, room context) drives deterministic,
repeatable enhancements; finished assets exit.

Optimize for:
- **Correctness + determinism**
- **Contract stability** (v2.0.0 “Golden Path” is the baseline)
- **Performance under the “Quality Firewall”** (no regressions)
- **Safe change management** (small PRs, reviewable diffs, docs + tests updated)

> This repository is NOT a government / health IT “portal”. Do not introduce assumptions,
> workflows, or terminology from those domains.

---

## Non‑Negotiables

1. **Preserve Golden Path behavior**
   - Do not change default outputs, preset defaults, schema semantics, or public CLI/API behavior
     unless the change is explicitly contract-scoped and versioned.
2. **Keep CI green**
   - No model downloads in tests.
   - Minimal disk usage.
   - Fast feedback loops: isolate heavy tests.
3. **Never “paper over” failures**
   - Fix root causes rather than broadly skipping tests or weakening assertions.
   - Comments must match reality (if CI runs a test, don’t claim it’s skipped).
4. **Precision over speed**
   - Be explicit. Validate inputs early. Handle optional deps safely. Avoid fragile mocks.

---

## Repository Map

Place changes in the correct zone. Avoid new root-level scripts unless truly necessary.

| Area | Purpose | Rules of thumb |
|------|---------|----------------|
| `src/` | Installable package code | Preferred for new production logic. Keep APIs stable. |
| `scripts/` | Thin orchestration runners | CLI glue only; delegate logic into `src/`. |
| `config/` | YAML presets and configuration | Stable/canary/experimental taxonomy enforced. |
| `assets/` | LUTs + look assets | Treat as brand-critical assets; keep versioned + reproducible. |
| `docs/` | Architecture + guides + governance docs | Update whenever workflows/behavior change. |
| `tools/` | DevOps utilities (Quality Firewall, ledgers, audits) | Deterministic, testable, safe file IO. |
| `tests/` | Pytest suite | Fast default, heavy isolated via markers. |

**Import discipline (important):**
- Use absolute imports from the installed package under `src/`.
- Never rely on “local folder imports” that pass in dev but fail in CI.

---

## Golden Path Contract Rules (v2.0.0)

The Golden Path is the stable contract baseline. Treat it like an API.

**You MUST:**
- Preserve schema field meanings and defaults.
- Preserve preset behavior and naming.
- Preserve stable output characteristics (quality/perf envelopes).

**If you need a contract-impacting change:**
- Provide a migration story.
- Add/extend contract tests.
- Bump versions in lockstep (see “Version Alignment”).

---

## Architecture Principles for a Context-Aware Rendering Engine

This system is a pipeline: it should remain **composable**, **testable**, and **backend-agnostic**.

### Preferred high-level structure
- **Orchestrator**: coordinates stages; owns run lifecycle and error handling.
- **Stages**: pure-ish transformations (load → infer → postprocess → export).
- **Backends**: device/model-specific implementations behind stable interfaces.
- **Config**: validated, typed, serializable; supports presets + overrides.

### Use design patterns intentionally (avoid “if/elif soup”)

**Strategy Pattern**
- Use when behavior varies by room/zone, preset tier, or rendering objective.
- Example: different tone mapping strategies for *Sunroom* vs *Home Cinema*.

**Adapter Pattern**
- Use when integrating multiple depth backends (Depth Anything V3, Apple Depth Pro, research models)
  with inconsistent APIs.
- Standardize on one internal interface.

**Factory / Registry Pattern**
- Use to create pipeline stages/backends from config without deep branching.
- Keep “selection logic” centralized and testable.

**Template Method**
- Use for stage skeletons where subclasses override only safe extension points.

> Anti-pattern: scattering backend selection and device checks throughout the codebase.

### Interface guidance
- Prefer small protocols / ABCs for stage and backend contracts.
- Keep contracts explicit: input types, output types, shape expectations, and units.

---

## Model + Licensing Governance (Depth + Materials)

This repo uses a **multi-backend depth strategy** and must enforce compliance.

### Backend policy (encode as “code-as-compliance”)
| Backend | Tier | Default? | Offline allowed? | Commercial allowed? | Notes |
|--------|------|----------|------------------|---------------------|------|
| Depth Anything V3 | Production | Yes | Yes | Yes | Default for production runs. |
| Apple Depth Pro | Hardware-optimized | No | Yes | Yes | Prefer on Apple Silicon when available and validated. |
| DA3 1.1 (research) | Experimental | No | Yes | **Only if `non_commercial_ok=True`** | Must hard-fail otherwise. |

**Hard rule:** If a model requires non-commercial gating, the code must raise a clear,
typed exception before any processing begins.

Example guard (pattern, not literal file paths):
```py
if backend == "da3_1_1" and not cfg.non_commercial_ok:
    raise ComplianceError(
        "DA3 1.1 is restricted. Set non_commercial_ok=True for non-commercial use only."
    )
````

### No-download policy in tests

* ML tests must honor offline patterns:

  * `TRANSFORMERS_OFFLINE=1`
  * `HF_HUB_OFFLINE=1` (if used)
* Tests must use tiny local fixtures or mocks—not network calls, not hub downloads.

---

## Hardware Acceleration: Apple Silicon (MPS) First-Class

This repo is optimized for **MPS on Apple Silicon**, with CUDA/CPU fallbacks.

### Device selection requirements

* Never hardcode `cuda:0`.
* Prefer a single, tested device selector utility used everywhere.
* Be defensive: torch might be missing in core test environments.

Pattern:

```py
def choose_device() -> "torch.device":
    import torch

    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### Determinism caveats

* Some ops differ across CPU/CUDA/MPS. When determinism matters:

  * seed RNGs (`random`, `numpy`, `torch`)
  * avoid nondeterministic kernels where possible
  * compare outputs using tolerant metrics (PSNR/SSIM thresholds) rather than exact bytes,
    unless the stage is known to be bitwise stable.

---

## Quality Firewall: Performance Is a Contract

Performance regressions are treated as correctness failures.

### Firewall rules (treat as blocking)

* Block if **p95 latency** increases by **> 10%**
* Block if **mean latency** increases by **> 15%**
* Block if **failure rate > 0%**
* Block if disk usage or temp artifacts scale unexpectedly

### Performance engineering rules

* Avoid Python loops over pixels. Prefer vectorized NumPy / Torch ops.
* Avoid re-loading models per frame/image. Cache backends safely.
* Avoid unnecessary copies (watch dtype conversions, `.cpu()`/`.numpy()` ping-pong).
* Stream large files; do not read entire videos into memory.
* Use atomic writes for outputs and ledgers.

### Benchmark tests

* Benchmark/performance regression tests must be **explicitly marked** and kept out of fast CI.
* Ensure test markers and CI selection logic match. Do not rely on comments like
  “CI skips these” unless markers/config truly enforce it.

---

## Tests: Taxonomy, Markers, Methodology

### Marker taxonomy

| Marker             | Typical command            | What it covers                                             | Constraints                                            |
| ------------------ | -------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| *(default / core)* | `-m "not ml and not slow"` | config parsing, schemas, IO utilities, orchestration logic | Must run fast; no torch/model loads.                   |
| `ml`               | `-m "ml and not slow"`     | backend wiring, inference shape rules, device placement    | Must be offline; small fixtures only.                  |
| `slow`             | *(excluded by default)*    | stress, large fixtures, full pipelines                     | Run manually or scheduled.                             |
| `benchmark`        | *(manual / gated)*         | performance ledger updates, regression thresholds          | Must not run in default CI unless explicitly intended. |

### CI matrix (do not break this)

| CI job     | Python      | Requirements            | Notes                                    |
| ---------- | ----------- | ----------------------- | ---------------------------------------- |
| Lint       | 3.12        | `requirements-lint.txt` | Fast static checks.                      |
| Core tests | 3.10 + 3.12 | `requirements-ci.txt`   | Offline; no heavy deps.                  |
| ML tests   | 3.11        | ML deps installed       | Offline; no downloads; minimal fixtures. |

### Testing methodology expectations

* Prefer **TDD** for bug fixes and non-trivial changes:

  1. write failing test
  2. implement minimal fix
  3. refactor with confidence
* Use **contract tests** for stable outputs:

  * validate schema invariants
  * validate preset parameter invariants
  * validate deterministic transforms (where applicable)
* Use **golden master tests** sparingly:

  * small fixtures only
  * store tiny reference outputs (or metrics) in-repo
* Use **property-based tests** (e.g., Hypothesis) when validating invariants (shapes, ranges,
  idempotence of config serialization, etc.)—but keep them bounded and deterministic.

### Mocking guidance (be precise)

Mock boundaries where external systems exist:

* FFmpeg subprocess calls
* file system IO (use temp dirs)
* model inference in core tests
* network/hub calls (should not exist)

Do **not** over-mock internal logic.
Prefer small real tensors for ML unit tests when feasible.

---

## Presets: Stability Taxonomy + Rules

### Taxonomy

| Tier           | Promise                              | Allowed change cadence | Tests required                          |
| -------------- | ------------------------------------ | ---------------------- | --------------------------------------- |
| `stable`       | production-safe, backward-compatible | rare, deliberate       | strong coverage, contract + perf checks |
| `canary`       | production trial                     | iterative              | tests for schema + key params           |
| `experimental` | research                             | flexible               | basic validation + clear docs           |

### When adding/modifying presets

* Document intent + expected impact in `docs/` (and/or changelog).
* Add tests validating:

  * required keys exist
  * parameter ranges
  * compatibility with Golden Path schema
* Keep names human-meaningful and consistent.

---

## Dependency Management: Layered, Reproducible, Lean

* Root requirement `.txt` files are convenience pins.
* **Source of truth** is `requirements/*.in`.

To change dependencies:

1. edit the correct `.in` file
2. recompile:

   ```bash
   cd requirements && make compile
   ```
3. ensure CI remains lean:

   * do not pull heavy ML dependencies into the core runtime unless strictly necessary
   * prefer optional extras (e.g., `.[ml]`, `.[rag]`) if the project supports it

---

## File IO, Safety, and Operational Rigor

### IO safety rules

* Use `pathlib.Path`
* Use atomic writes for outputs and ledgers:

  * write to temp file
  * fsync if needed
  * rename into place
* Never assume directories exist; create explicitly.
* Clean up temp artifacts deterministically.

### Security + privacy (luxury real estate context)

* Treat inputs/outputs as sensitive client assets.
* Do not log raw image paths that leak addresses/client names when avoidable.
* Avoid embedding metadata unless explicitly requested.

### Subprocess rules (FFmpeg)

* Prefer `subprocess.run([...], check=True, capture_output=True, text=True)`
* Never use `shell=True` unless there is a documented, reviewed reason.

---

## Version Alignment

For **contract-impacting changes**, keep versions aligned:

* contract schema version
* package version
* runtime `__version__`

Single PR should update all relevant version sites and tests.

---

## Documentation Expectations

Any workflow/behavior change must update:

* `README` (if user-facing)
* relevant `docs/` pages (architecture, Quality Firewall quick refs, etc.)
* examples (if present)

Docs are governance, not optional.

---

## PR Hygiene: Make Review Easy

### PR size and scope

* Prefer one feature/fix per PR.
* Avoid formatting-only churn.
* Explain the “why”, not just the “what”.

### Required preflight (local)

Run the same split CI expects:

```bash
pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
```

If ML deps are installed:

```bash
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
```

### Performance changes checklist (if touching pipeline/inference/postprocess)

* Explain expected perf impact.
* Add/adjust benchmark coverage if required.
* Confirm Quality Firewall thresholds won’t regress.

---

## Mistake-Proofing Checklist (Use This Before You Commit)

* [ ] Did you validate inputs early (before model load / heavy IO)?
* [ ] Did you keep behavior stable for Golden Path?
* [ ] Are optional deps handled safely (torch missing/mocked in core env)?
* [ ] Are pytest markers correct (core vs ml vs slow vs benchmark)?
* [ ] Does every comment match what CI actually does?
* [ ] Are outputs deterministic or appropriately tolerance-tested?
* [ ] Did you update docs/tests alongside behavior changes?
* [ ] Are you using the correct directory (logic in `src/`, wrappers in `scripts/`)?

---

## Concrete Examples (Patterns to Copy)

### 1) Backend Adapter interface (depth)

```py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

class DepthBackend(Protocol):
    def infer_depth(self, image: "np.ndarray") -> "np.ndarray": ...

@dataclass(frozen=True)
class DepthConfig:
    backend: str
    non_commercial_ok: bool = False

def create_depth_backend(cfg: DepthConfig) -> DepthBackend:
    if cfg.backend == "da3_1_1" and not cfg.non_commercial_ok:
        raise ComplianceError("Restricted backend requires non_commercial_ok=True")

    # factory/registry lookup goes here
    return backend_registry[cfg.backend]()
```

### 2) Strategy for room-specific rendering

```py
class RoomStrategy(Protocol):
    def apply(self, rgb: "np.ndarray", ctx: "RenderContext") -> "np.ndarray": ...

class SunroomStrategy:
    def apply(self, rgb, ctx):
        return preserve_highlights_and_warmth(rgb, ctx)

class CinemaStrategy:
    def apply(self, rgb, ctx):
        return deepen_blacks_and_boost_contrast(rgb, ctx)
```

### 3) Tests that enforce preset invariants

```py
def test_stable_preset_schema_invariants(stable_preset):
    assert stable_preset["tier"] == "stable"
    assert 0.0 <= stable_preset["tone"]["strength"] <= 1.0
```

---

## Default Decision Rules (When Unsure)

* Prefer **stability** over novelty.
* Prefer **explicitness** over cleverness.
* Prefer **small, testable units** over monolith functions.
* Prefer **offline, deterministic tests** over integration-by-accident.
* Prefer **vectorized operations** over Python loops.

If you must make a risky change (contract/perf), surface the risk clearly in code comments,
tests, docs, and PR notes.

```
```
