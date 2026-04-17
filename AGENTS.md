# AGENTS.md

Quick reference for common workflows and commands in this repo.

## Canonical worktree discipline
- `origin/main` is the canonical repository baseline and the only standing source of truth on Desktop.
- Desktop siblings such as `Transformation_Portal__fastapi` or `Transformation_Portal__upload` are temporary git worktrees, not independent repos.
- When parallel work needs consolidation, first fast-forward local `main` to `origin/main`, then create a single integration branch from that updated base and replay the intended commits there.
- After validation lands, retire obsolete Desktop worktrees and prune their local-only branches so `Transformation_Portal/` remains the canonical checkout.

## Common commands (Makefile)
- `make venv` create local `.venv` with a Python 3.11+ interpreter, or fail closed if an existing `.venv` is unsupported.
- `make setup` install package in editable mode.
- `make install-core` install the pinned core runtime + dev tooling lockfiles into `.venv`, install the project editable with `--no-deps`, and run `pip check`.
- `make repair-core-venv` recreate `.venv`, reinstall the pinned core environment, and re-run `pip check`.
- `make install-ml` disabled; no trusted checked-in umbrella ML lockfile contract.
- `make install-ml-core` install the target-owned checked-in ML core baseline selected from the local OS/architecture.
- `make install-ml-raw` disabled; no trusted checked-in RAW lockfile contract.
- `make install-ml-sam2` install ML SAM2 layer (Meta Segment Anything 2, optional); it uses the Apple Silicon MPS path on native Darwin arm64 and the CPU path elsewhere.
- `make install-ml-coreml` install ML CoreML acceleration on macOS only when a trusted `requirements/ml-coreml.txt` is present.
- `make test-fast` run fast test subset plus the Phase 6 smoke coverage layer.
- `make test-novideo` run tests excluding luxury video master grader tests (filters out `video_master_grader`).
- `make test-full` run full test suite (parallel if xdist installed).
- `make test-integration` run DA3/HuggingFace model-loading integration (`tests/test_da3_inference_integration.py`) with `TP_RUN_HF_MODEL_TESTS=1` (downloads models from HF Hub unless offline; typically requires `HF_TOKEN`).
- `make test-structure` run codebase structure validation.
- `make test-utils` run performance/error utility tests.
- `make test-orchestrator-contract` run the full portal/orchestrator contract suite (`tests/test_app_orchestrator_runtime.py`, `tests/test_app_orchestrator_contract_http.py`, `tests/validation/test_portal_smoke_scripts.py`).
- `make test-orchestrator-http-contract` run HTTP-only orchestrator contract tests (`tests/test_app_orchestrator_contract_http.py`).
- `make test-portal-contract` run portal runtime/browser contract tests (`tests/test_app_orchestrator_runtime.py`, `tests/validation/test_portal_smoke_scripts.py`).
- `make test-frontdoor-contract` run managed frontdoor Node 22 contract/build checks (`./scripts/setup/ensure_node_version.sh && cd web/secure-landing && npm test && npm run build`).
- `make test-archive-gate-contract` run archive gate readiness + HTTP contract coverage for archive Gates A/B/C (`tests/test_app_orchestrator_runtime.py`, `tests/test_app_orchestrator_contract_http.py` with `-k "archive_gate"`).
- `make seed-frontdoor-user` write the canonical local managed-frontdoor credential fixture to `/tmp/tp-frontdoor-users.json` using `smoke-admin` / `correct horse battery staple` unless you override the env vars.
- `make run-frontdoor-local` start the canonical local managed frontdoor on `http://localhost:3000` after verifying backend readiness, auth env, and no silent fallback to `:3001`; it auto-seeds the canonical local user fixture when no explicit frontdoor user source is configured.
- `make validate-orchestrator-http` run the live orchestrator HTTP smoke against a running backend.
- `make validate-portal-browser` launch an isolated local backend, then run the live portal browser smoke; it seeds `TP_API_KEY=contract-secret` unless you override it.
- `make validate-frontdoor-browser` launch isolated local backend and managed frontdoor runtimes, then run the live browser smoke; it auto-seeds the canonical local smoke credentials when it creates the managed frontdoor runtime itself.
- `make validate-frontdoor-deployment-gate` run the manual shared-deployment frontdoor posture gate against a Cloudflare-fronted public hostname, a protected Vercel deployment URL, and either a public FastAPI probe URL or explicit non-public attestation.
- `make validate-full` run the full validation suite with all checks including browser smokes (`./scripts/validation/run_full_validation_suite.sh`).
- `make validate-quick` run quick validation skipping browser smokes (`./scripts/validation/run_full_validation_suite.sh --quick`).
- `make audit-pipeline-readiness` run the safe local four-pipeline readiness audit using checked-in archive fixtures.
- `make coverage-fast-scope` run branch coverage for the audited `core/config` and `streaming` paths with `term-missing` output.
- `make clean` remove Python caches and build/test artifacts.
- `make clean-frontdoor` remove frontdoor build artifacts (`.next`, `.next-build-verify`, `.next-smoke-*`, `.next-codex-*`).
- `make clean-all` remove all build artifacts (Python + Node).
- `make lint` run flake8 + pylint (non-blocking).
- `make lint-parity` run the GitHub lint job locally with the CI-pinned Python 3.12 lint environment.
- `make ci` run local CI checks (lint + check-json-serialization + check-python-headers + check-yaml-governance + check-piptools-cache + check-requirements-lock-contract + check-ci-sync + test-fast + test-orchestrator-contract + test-frontdoor-contract).
- `make ci-full` run comprehensive local CI (`./scripts/local_ci_check.sh`).
- `make ci-quick` run quick local CI (`./scripts/local_ci_check.sh --quick`).
- `make pre-commit` run pre-commit hooks with CI-aligned Black/isort versions.
- `make install-hooks` install git pre-commit hook.
- `make quality-check` run lint + workflow validation + the root-file placement check.
- `make fix-quality` auto-fix quality issues (`scripts/auto_fix_quality.py --fix-all`).
- `make check-quality` dry-run quality auto-fix checks (`scripts/auto_fix_quality.py --dry-run`).
- `make check-environment` run pre-flight environment validation through the resolved repo interpreter.
- `make check-worktree` check if git worktree is clean after builds (`scripts/validation/check_worktree_clean.sh`).
- `make validate-ci` validate GitHub Actions configs plus gitleaks, dependency-update, and Dependabot workflow contracts.
- `make check-json-serialization` fail when raw `json.dump`/`json.dumps` usage is detected outside approved modules.
- `make check-python-headers` fail when Python header lines 1-2 contain invalid encoding-cookie-like text outside valid PEP 263 declarations.
- `make check-yaml-governance` fail when raw `yaml.safe_load` usage appears outside the shared preset loader or explicitly exempt non-preset loaders.
- `make check-piptools-cache` fail if `requirements/.pip-tools-cache` is tracked in git.
- `make check-requirements-lock-contract` fail when layered lockfile headers, target-owned purity guards, or lane structure drift from contract.
- `make check` verify the generic layered requirements surface under `requirements/`.
- `make check-test-markers` audit test marker coverage (ADR-044) - reports unmarked test functions.
- `make check-ci-sync` verify CI dependency files are in sync (no drift between `requirements-ci.txt` and `requirements/ci.in`).
- `make check-todo-governance` scan repository for TODO patterns and fail if ungoverned TODOs (missing tracking references) are found.
- `make check-portal-asset-budgets` validate raw and gzipped portal asset size budgets against the checked-in budget contract.
- `make organize-docs` move markdown files into `docs/` (repo hygiene).
- `make check-docs` dry-run docs organization.
- `make check-stale-docs` detect changed-file references to deleted or moved docs root paths.
- `make lock` regenerate all requirements lockfiles.
- `make lock-prod` regenerate `requirements.lock.txt`.
- `make lock-ci` regenerate `requirements-ci.lock.txt`.
- `make lock-dev` regenerate `requirements-dev.lock.txt`.
- `cd requirements && make compile LOCK_PYTHON_VERSION=3.11` compile only the generic checked-in layered lockfiles (`all/base/dev/ci/security/tools-archive`).
- `cd requirements && make compile-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11` compile the Darwin arm64 target-owned ML lock on native Darwin arm64 only.
- `cd requirements && make compile-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11` compile the Linux x86_64 target-owned ML lock on native Linux x86_64 only.
- `cd requirements && make compile-ml-darwin-x86_64 LOCK_PYTHON_VERSION=3.11` fail closed because the Darwin x86_64 target-owned ML lock is frozen pending an authoritative lane decision.
- `cd requirements && make compile-ml-layers LOCK_PYTHON_VERSION=3.11` fail closed and direct operators to the explicit target-owned ML commands.
- `cd requirements && make compile-accel LOCK_PYTHON_VERSION=3.11` fail closed and direct operators to the explicit target-owned ML commands.
- `cd requirements && make compile-hash-pilot LOCK_PYTHON_VERSION=3.11` generate advisory hash-enforced pilot lockfiles into `requirements/.hash-pilot/`.
- `cd requirements && make update LOCK_PYTHON_VERSION=3.11` update only the generic checked-in layered lockfiles.
- `cd requirements && make update-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11` update the Darwin arm64 target-owned ML lock on native Darwin arm64 only.
- `cd requirements && make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11` update the Linux x86_64 target-owned ML lock on native Linux x86_64 only.
- `cd requirements && make check LOCK_PYTHON_VERSION=3.11` verify only the generic checked-in layered lockfiles are current.
- `cd requirements && make check-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11` verify the Darwin arm64 target-owned ML lock on native Darwin arm64 only.
- `cd requirements && make check-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11` verify the Linux x86_64 target-owned ML lock on native Linux x86_64 only.
- `cd requirements && make check-hash-pilot LOCK_PYTHON_VERSION=3.11` validate the pilot lockfiles with `pip install --dry-run --require-hashes`.
- `python3 scripts/validation/check_requirements_lock_contract.py` validate layered lock contract (headers + target-owned purity/compatibility guards + lock ownership manifest coverage).
- `make docs` build API docs with Sphinx.
- `make docs-clean` remove generated docs output.

## Environment Validation Scripts
- `make check-environment` run the canonical pre-flight validation flow.
- `./.venv/bin/python scripts/validation/check_local_environment.py` run the pre-flight validation script directly when you need a specific check.
- `./.venv/bin/python scripts/validation/check_local_environment.py --strict` treat soft failures as hard failures.
- `./.venv/bin/python scripts/validation/check_local_environment.py --check python` check only Python version.
- `./.venv/bin/python scripts/validation/check_local_environment.py --check node` check only Node.js version (22.x required).
- `./.venv/bin/python scripts/validation/check_local_environment.py --check dependency-health` run `pip check` for the active interpreter.
- `./scripts/setup/ensure_node_version.sh` Node version enforcement wrapper with version manager detection.
- `cd web/secure-landing && npm run build:portal` bundle the modularized portal sources back into the shipped `public/portal-assets/portal.js` asset and sync shared UI token primitives.
- `./scripts/validation/run_full_validation_suite.sh` all-in-one validation orchestrator.
- `./scripts/validation/run_full_validation_suite.sh --quick` skip browser smokes for faster iteration.
- `./scripts/validation/run_full_validation_suite.sh --skip-frontdoor` Python-only validation.
- `./scripts/validation/check_worktree_clean.sh` verify git worktree is clean after builds.

## ML Layer Bootstrap Script (ADR-032 Platform Matrix)
### Core profiles (mutually exclusive)
- `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu` install CPU baseline (darwin-*/linux-*-cpu).
- `./scripts/bootstrap/install_ml_stack.sh --profile core-mps` install Apple Silicon MPS (darwin-arm64-mps, macOS ARM64 only).
- `PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda` install NVIDIA CUDA (linux-x86_64-cuda, Linux only).

### Capability layers (stack on core profile)
- `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,raw` install ML baseline + RAW ingest.
- `./scripts/bootstrap/install_ml_stack.sh --profile core-mps,sam2` install MPS + SAM2 segmentation.
- `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,coreml` install ML baseline + CoreML (macOS only).
- `./scripts/bootstrap/install_ml_stack.sh --profile full` disabled until a trusted umbrella ML lockfile contract exists again.

### Utility options
- `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu --dry-run` show what would be installed without installing.
- `./scripts/bootstrap/install_ml_stack.sh --help` show all available profiles and options.

## Workflow scripts (bash)
- `./scripts/pipelines/run_montecito_apex_full.sh` run Montecito Shores APEX batch with all deliverables (interactive prompt).
- `./scripts/pipelines/run_montecito_apex_lean.sh` run Montecito Shores APEX batch (lean outputs, faster).
- `./scripts/pipelines/process_source_tiffs_apex.sh` batch APEX V2 enhancement for `input_images/source_tiffs` with optional depth generation.
- `./scripts/pipelines/process_source_tiffs_individual.sh` per-image APEX V2 enhancement commands (manual execution).
- `./scripts/pipelines/run_800_picacho_efficientsam_validation.sh` run the 800 Picacho EfficientSAM production validation pass with DA3, Materials V3, V2 tone mapping, and PBR enabled.
- `./scripts/pipelines/run_sealed_eval_72h.sh --archive-index <path> --archive-root <path>` run sealed pre/post fixity verification around an optional eval command and emit an audit package.
- `./scripts/pipelines/hdr_production_pipeline.sh` interactive HDR video mastering workflow that pairs source footage with a 3D LUT and writes web deliverables.
- `./scripts/setup/install_da3_runtime.sh` install the repo-local DA3 subprocess runtime (validated `.runtime/Depth-Anything-3` ref + auto-discovered `./.venv-da3/bin/python` contract + `.runtime/da3-pip-freeze.txt` snapshot).
- `./scripts/setup/install_depth_pro_runtime.sh` install the repo-local Depth Pro subprocess runtime (pinned `torch==2.7.1` / `torchvision==0.22.1` / `numpy==1.26.4` + pinned Apple `ml-depth-pro` ref + auto-discovered `./.venv-depth-pro/bin/python` contract + `.runtime/depth-pro-pip-freeze.txt` snapshot).
- `./scripts/setup/install_raw_runtime.sh` install the repo-local RAW subprocess runtime (auto-discovered `./.venv-raw/bin/python` contract + `.runtime/raw-pip-freeze.txt` snapshot).
- `./scripts/setup/run_frontdoor_local.sh` start the local managed frontdoor only when the backend is ready, auth env is set, and `localhost:3000` is free.
- `./scripts/test_v2_integration.sh` validate end-to-end lux-depth-v3 + V2 stage integration (`--verbose`, `--clean` available).
- `./scripts/validate_dependency_constraints.sh` enforce dependency pinning rules used by repo policy (`--verbose` available).
- `./scripts/pipelines/run_fixity_cycle.sh` run archive hash-manifest scan + verification cycle for fixity evidence (`--archive-index` and `--archive-root` required).
- `./scripts/diagnostics/full_chain_determinism_trial.sh` run Phase 4C/4D/4E determinism checks (`--input-root` or `--capture-metadata`).
- `./scripts/setup/auto-organize-install.sh` install repository file-organization guardrails and pre-commit hook.
- `./scripts/setup/pre-commit-check.sh` run root-file placement validation manually (also used by the hook).
- `./scripts/pre_commit_hook.sh` unified pre-commit quality gate wrapper; delegates to `scripts/utilities/pre-commit-quality-check.py` (`--all-files`, `--quick-tests` available).
- `./scripts/runbooks/merge_phase2_runbook.sh` temporary guarded merge runbook for the APEX Phase 2 branch; it checks tree cleanliness, syncs branches, runs fast-lane validation, and prompts before merge/push.

## ComfyUI workflows (`workflows/`)
- `python -c "from transformation_portal.comfyui import WorkflowTemplates; WorkflowTemplates.save_all_templates('workflows/templates')"` generate ComfyUI template workflows.
- `python -c "from transformation_portal.comfyui import WorkflowExecutor; from transformation_portal.comfyui.workflow_builder import Workflow; wf = Workflow.load('workflows/examples/simple_enhancement.json'); print(WorkflowExecutor(verbose=True).execute(wf)['success'])"` execute the `simple_enhancement` workflow example.

## AI Skills Usage Rules

These skills assist repository maintenance but must follow their individual skill contracts.
Skills that modify code must only be executed with explicit user approval.

### `gh-fix-ci`

Purpose:
Investigate failing GitHub Actions checks.

When to use:
- GitHub Actions workflow failures
- failing dependency-update CI runs
- structural/test failures reported by GitHub Actions

Important:
- The skill may analyze CI failures automatically.
- Code changes must **not** be implemented without explicit user approval.

Scope limitation:
- Operates on **GitHub Actions checks only**.
- External CI providers are reported by URL only and must be investigated manually.

### `gh-address-comments`

Purpose:
Address pull-request review comments.

When to use:
- reviewers request code or test changes
- follow-up revisions are required after review

Important:
- The user must **explicitly select which review comment threads should be addressed**.
- Do not automatically fix all review comments.

Avoid using when:
- comments are informational or discussion-only

### `security-best-practices`

Purpose:
Provide secure-by-default code review guidance.

Invocation policy:
- Invoke **only when explicitly requested by the user** or when a security review is requested.

Recommended use cases:
- dependency handling changes
- GitHub workflow modifications
- scripts interacting with filesystem or network
- external model downloads or integrations
- validation scripts or artifact generation logic

Do not run automatically on general code changes.

### `security-threat-model`

Purpose:
Perform threat modeling for new capabilities.

Invoke when:
- introducing external downloads or model integrations
- adding new pipeline stages
- adding artifact export mechanisms
- adding automation workflows with repository write permissions

Output should identify:
- trust boundaries
- attack surfaces
- privilege escalation paths
- mitigation strategies

If the skill is unavailable:
perform a manual threat-model analysis instead.

### `doc`

Purpose:
Maintain documentation accuracy and workflow consistency.

Invoke when changes affect:
- CLI flags
- Makefile commands
- validation scripts
- dependency lock workflows
- GitHub Actions workflows
- onboarding commands

Primary targets:
- `AGENTS.md`
- `README.md`
- `docs/`
- CLI help text

If the skill is unavailable:
update documentation manually.

### `screenshot` (optional)

Purpose:
Capture visual evidence for review and regression verification.

Use when:
- pipeline output images change
- visual artifact comparison is required
- before/after results assist PR review
