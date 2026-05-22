# Portal Audit Backlog - 2026-05-18

**Document Status:** Active backlog tracking [PORTAL_AUDIT_REPO_WIDE_2026-05-18.md](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md)
**Last Updated:** 2026-05-20
**Maintainer:** Repository Architect
**Tracks audit:** [PORTAL_AUDIT_REPO_WIDE_2026-05-18.md](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md)

---

## How to use

Each item carries severity, effort, the files to touch, observable acceptance criteria, and a back-link to the originating finding. Owners are deliberately left blank — assign during planning. Items move tiers only via the audit doc; the backlog mirrors the audit, it does not redefine it.

| Severity | Effort |
|---|---|
| High / Medium / Low (matches audit table) | S = ≤1 day, M = ≤1 week, L = multi-week |

## Tier 1 — Immediate (target window: 2026-05-19 → 2026-05-29)

> **Tier 1 status (2026-05-20):** all four items merged. One open follow-up remains on I-1 — promote the torch-load CI step from `continue-on-error: true` to blocking after the soak window (~2026-05-25).

### I-1. Wire `check_unsafe_torch_load.py` into pre-commit and security-unified.yml

**Status:** Done — merged in PR #1806 (`70d45dda074f6e2fa0a6c1b49a5cfb5bf0793c53`) on 2026-05-18. Follow-up: flip `continue-on-error: true` → blocking after one quiet week (~2026-05-25).

**Severity / Effort:** Medium / S
**Tracks finding:** [#10](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#63-security) — orphaned torch-load scanner
**Files to touch:** `.pre-commit-config.yaml`, `.github/workflows/security-unified.yml`
**Acceptance criteria:**
- `.pre-commit-config.yaml` contains a hook that invokes `scripts/validation/check_unsafe_torch_load.py` on staged Python files.
- `security-unified.yml` runs the scanner as a non-blocking step (`continue-on-error: true`) for one quiet week, then is promoted to blocking via a follow-up PR.
- Re-running `python3 scripts/validation/check_unsafe_torch_load.py` against the working tree exits 0 (no violations) or the allowlist is updated with an inline rationale.

### I-2. Align SAM2 benchmark docs, docstring, and assertion to one baseline

**Status:** Done — landed in PR #1815 (`eb7a0a80`) and refined in PR #1830 (`5b740cff`). Baselines re-measured on Apple Silicon (MPS 13.38s, CPU 42.66s, recorded 2026-05-19); assertion now derives the threshold from the recorded per-device baseline (`1.5×`) instead of the legacy `< 20.0`.

**Severity / Effort:** Medium / S
**Tracks finding:** [#3](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#62-runtime-and-performance) — benchmark inconsistency
**Files to touch:** `docs/performance/sam2_benchmarks.md`, `tests/spatial_ai/segmentation/test_sam2_backend_performance.py`
**Acceptance criteria:**
- Test docstring at `test_sam2_backend_performance.py:14` no longer claims "<2s on MPS"; it cites the measured MPS baseline recorded in `sam2_benchmarks.md` (currently ~13.4 s for 512×512 auto mode) and a corresponding CPU baseline re-measured during this fix.
- Assertion at `test_sam2_backend_performance.py:296` uses a threshold derived from the same recorded baseline (e.g., `1.5 × baseline`), not the legacy `< 20.0`.
- `docs/performance/sam2_benchmarks.md` records when this baseline was last re-measured and the hardware target.

### I-3. Add non-root `USER` to Dockerfile and harden compose defaults

**Status:** Done — landed in PR #1808 (`f54d3eea`), with follow-ups PR #1809 (`03b600f8`, restore non-root image smoke build) and PR #1813 (`24f9df88`, harden runtime image construction). Runtime stages (`cpu`/`gpu`/`apple-silicon`) end with `USER tp` via a multi-stage builder that drops compiler toolchains from runtime images; compose uses a `tp-init` bootstrap + `tp_state` named volume with read-only `./input`/`./config`. Contract enforced by `tests/validation/test_dockerfile_contract.py`.

**Severity / Effort:** Medium / S
**Tracks finding:** [#6](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#63-security) — container default is root
**Files to touch:** `Dockerfile`, `docker-compose.yml`
**Acceptance criteria:**
- Each `Dockerfile` stage that ships a runtime image creates a dedicated unprivileged user and ends with a `USER` directive for that user.
- Writable paths inside the image are `chown`ed to the new user; image still passes a smoke `docker run --rm <image> python -c "import transformation_portal"` as the unprivileged user.
- `docker-compose.yml` either inherits the image user or sets `user:` explicitly on each service; volumes are mounted with mode compatible with the new uid/gid.

### I-4. Update ADR-032 to remove `safety` reference and document pip-audit posture

**Status:** Done — merged in PR #1807 (`a699c161f3943006df9ef3a952d6e12498428152`) on 2026-05-18.

**Severity / Effort:** Low / S
**Tracks finding:** [#8](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#64-ci-governance-and-licensing) — ADR-032 drift
**Files to touch:** `docs/architecture/ADR-032-dependency-pinning-strategy.md`
**Acceptance criteria:**
- The "Security Monitoring Tools" section at `ADR-032:170` no longer lists `safety`.
- pip-audit is documented as the governed blocking scanner, with the current high/critical block policy and the `--ignore-vuln` exception process referenced (cross-link `requirements/security.in` and `.github/workflows/security-unified.yml`).
- An "Amendments" or change-log entry records the March 2026 Safety removal with rationale.

## Tier 2 — Near term (target window: 2026-05-26 → 2026-06-16)

### N-1. Expand mypy whitelist by tranche

**Severity / Effort:** High / M
**Tracks finding:** [#1](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#61-code-quality-typing-and-test-enforcement) — whitelist-based type gating
**Files to touch:** `.github/workflows/build.yml:567-573`, possibly `mypy.ini`, individual package `__init__.py` / typed helpers
**Acceptance criteria:**
- At least one new tranche (e.g., `src/transformation_portal/orchestrator/storage/`, `src/transformation_portal/orchestrator/queue/`, or a `spatial_ai` subpackage) is added to the mypy invocation and the build is green.
- Each newly typed module passes `mypy --config-file=mypy.ini <path>` cleanly per the in-comment policy at `build.yml:565`.
- `docs/ci/TYPE_CHECKING_POLICY.md` (referenced from the workflow comment) is updated with the new tranche and remaining backlog.

### N-2. Add lightweight ML sampled coverage and raise the 25% core floor

**Severity / Effort:** Medium / M
**Tracks finding:** [#2](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#61-code-quality-typing-and-test-enforcement) — fragmented coverage
**Files to touch:** `.github/workflows/build.yml:722-728`, `scripts/ci/check_per_package_coverage.py`, `docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md`
**Acceptance criteria:**
- The ML PR lane runs at least one sampled coverage slice (e.g., `--cov=src/transformation_portal/vlm --cov=src/transformation_portal/spatial_ai/segmentation` on a small marker subset), uploaded but not blocking.
- `--cov-fail-under` is ratcheted from 25 to 30 once the cold-zone baseline shows stable margin on the affected packages.
- The cold-zone program doc records the new floor and the date it took effect.

### N-3. Thread a single content digest through segmentation cache and integrity validation

**Severity / Effort:** Medium / M
**Tracks finding:** [#4](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#62-runtime-and-performance) — duplicate hashing
**Files to touch:** `src/transformation_portal/lux_depth_v3/segmentation/_cache.py`, `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py`, `src/transformation_portal/lux_depth_v3/config_resolver.py`
**Acceptance criteria:**
- Image and checkpoint digests are computed at most once per pipeline run (memoization keyed by inode+size+mtime is acceptable; do not weaken `_validate_checkpoint_sha256` correctness).
- The existing `SAM2CheckpointIntegrityError` fail-closed path remains observable in tests.
- A microbenchmark or perf test demonstrates measurable CPU-time reduction for a representative large-image batch.

## Tier 3 — Medium term (target window: 2026-06-16 → 2026-07-14)

### M-1. Sandbox or sign plugins before broader use

**Severity / Effort:** Medium / M
**Tracks finding:** [#7](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#63-security) — plugin trust boundary
**Files to touch:** `src/transformation_portal/plugins/loader.py`, possibly a new `src/transformation_portal/plugins/signing.py`, `docs/architecture/`
**Acceptance criteria:**
- One of: (a) plugin manifests carry a signature that is verified before instantiation against a configured trust set, or (b) plugins execute in a separate worker process with a documented small API surface.
- The opt-in environment toggle remains the gate; with it disabled, no plugin code runs.
- A new ADR records the chosen trust model and the threat scenarios it does and does not cover.

### M-2. Emit machine-readable runtime license manifest in run cards

**Severity / Effort:** Medium / M
**Tracks finding:** [#9](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#64-ci-governance-and-licensing) — mixed licensing
**Files to touch:** `src/transformation_portal/lux_depth_v3/run_card_contract.py`, `src/transformation_portal/lux_depth_v3/artifact_manager.py`, `src/transformation_portal/lux_depth_v3/manifest.py`, `docs/schemas/`
**Acceptance criteria:**
- Every run card / manifest carries a `licensing` block listing the software license tier, each model used (id + license), and a `non_commercial_active: bool` flag derived from `EnhanceConfig.non_commercial_ok` plus model selection.
- Schema docs and validators are updated in the same change (per CLAUDE.md "When you edit X, also update Y").
- Contract tests cover at least one commercial-only and one research-only configuration.

### M-3. Unify performance gate policy

**Severity / Effort:** Medium / M
**Tracks finding:** [#2 / #5 boundary](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#7-remediation-roadmap) — fragmented perf governance
**Files to touch:** `docs/performance/README.md`, `docs/performance/PHASE6_BUDGETS.md`, possibly `.github/workflows/build.yml` and nightly perf workflows
**Acceptance criteria:**
- `docs/performance/README.md` enumerates exactly which metrics are advisory, which are nightly-blocking, and which are PR-blocking.
- Each performance test under `tests/spatial_ai/` and `tests/test_pbr_processor*` carries a marker (`benchmark`, `slow`) that matches its enforcement tier.
- The reconstruction nightly job rejects regressions per the documented tier; the doc and the workflow agree.

## Tier 4 — Longer term (target window: 2026-07-14 → 2026-08-25)

### L-1. Replace the Python per-splat rasterizer loop with a vectorized or compiled path

**Severity / Effort:** High / L
**Tracks finding:** [#5](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#62-runtime-and-performance) — Gaussian rasterizer hotspot
**Files to touch:** `src/transformation_portal/spatial_ai/reconstruction/gaussian_rasterizer.py`, `tests/spatial_ai/reconstruction/test_performance_budgets.py`
**Acceptance criteria:**
- Per-splat Python loop at `gaussian_rasterizer.py:346-374` is replaced with a tiled or batched implementation (PyTorch vectorized, Triton, or compiled extension).
- Numerical equivalence to the current implementation is verified by a golden test on the existing 96×64×64 fixture and at least one larger fixture (e.g., 1024 Gaussians at 256×256).
- The new performance budget reflects the larger fixture and is at least 5× faster than the previous Python loop on the same hardware.

### L-2. Broaden strict typing toward repo-wide coverage

**Severity / Effort:** High / L
**Tracks finding:** [#1](../PORTAL_AUDIT_REPO_WIDE_2026-05-18.md#61-code-quality-typing-and-test-enforcement) — whitelist gating
**Files to touch:** `mypy.ini`, `.github/workflows/build.yml:567-573`, individual modules under `src/transformation_portal/`
**Acceptance criteria:**
- At least four additional packages typed-clean and added to the whitelist beyond N-1.
- `ignore_missing_imports` is replaced with per-module `[mypy-<package>.*]` ignore stanzas for true third-party gaps, so accidental local typing regressions are no longer masked.
- `docs/ci/TYPE_CHECKING_POLICY.md` updated to reflect the new policy and remaining gaps.

---

## Validation

- `make ci` — confirms governance checks remain green after each backlog item lands.
- `python3 scripts/validation/check_unsafe_torch_load.py` — confirms I-1 wiring stays effective.
- `make check-stale-docs` and `make check-doc-heading-links` — confirm this backlog stays linked from the audit and the map after any title/path edits, with all internal anchors resolving.
