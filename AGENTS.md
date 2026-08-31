# AGENTS.md

Coding-agent maintainer guide for this repository. Keep this file short and
actionable; use the linked docs and `Makefile` for exhaustive inventories.

## Operating Contract

- Preserve public contracts: route paths and names, response envelopes,
  selectors, auth behavior, CLI flags, Make targets, schema semantics, and
  test meaning. Contract changes require explicit scope plus matching
  tests/docs/schemas/version handling in the same change.
- Make the smallest safe patch that solves the observed problem. Do not turn a
  narrow fix into adjacent refactors, governance rewrites, or cleanup work
  without evidence.
- Pair behavior changes with focused tests in the same pass. Prefer tests that
  prove a real contract or regression risk over tests that only raise counts.
- Keep `src/tp` as the separate public import surface for contract, fixity, and
  Phase 4 tooling (`tp.crypto`, `tp.merkle`, `tp.phase4`). Do not collapse it
  into `transformation_portal`; tracked file and package names stay lowercase
  snake_case Python source.
- Treat Postgres, Redis, Docker, browser, and external-model failures as
  environment/tooling blockers until evidence shows a product regression. Do
  not weaken product contracts to hide missing services.
- Preserve unrelated local dirt. Stage explicit paths, never `git add -A`, and
  do not revert changes you did not make unless the user asks.
- Avoid destructive commands (`git reset --hard`, `git checkout --`, broad
  `rm`) unless the user clearly requested that operation.
- Keep generated artifacts out of commits. If a normal workflow creates a new
  generated path, update ignore rules or clean it before closeout.

## Worktree And PR Hygiene

- `origin/main` is the canonical Desktop baseline. Before branching for new
  work, fetch/prune and fast-forward local `main`:
  `git fetch origin --prune` then `git switch main` then
  `git merge --ff-only origin/main`.
- Desktop siblings such as `Transformation_Portal__fastapi` are temporary
  worktrees, not independent repositories. Retire obsolete siblings after their
  work lands.
- If the canonical checkout has unrelated dirt, use a temp worktree or stage
  only exact intended paths. Do not mix local dirt into PR fixes.
- For review-thread remediation, fetch the current PR head, fix only the
  actionable thread, rerun the focused gate, then reply/resolve only after the
  fix is pushed.
- For merges, prefer exact-head safety:
  `gh pr merge <id> --squash --delete-branch --match-head-commit <sha>`.
  After merge, fetch/prune and `--ff-only` local `main`.
- Branch hygiene closeout: confirm PR state, local `main...origin/main`, no
  stale local branch for the PR, and no stale remote-tracking branch after
  prune.
- Hook setup is `make install-hooks`; it installs both pre-commit and pre-push
  hooks through the repo-managed `.venv` pre-commit binary.

## Authority And Navigation

- Current docs navigation starts at [README.md](README.md),
  [docs/README.md](docs/README.md), and
  [docs/governance/DOCUMENTATION_MAP.md](docs/governance/DOCUMENTATION_MAP.md).
- The active cleanup ledger is
  [docs/architecture/ARCHITECTURE_CLEANUP_BOARD.md](docs/architecture/ARCHITECTURE_CLEANUP_BOARD.md).
  Do not reopen landed Tier 1/Tier 2 audit work or landed monolith seams unless
  new evidence shows a regression.
- Historical docs may keep old dates and facts. They are not current operator
  guidance unless the documentation map promotes them.
- Live Copilot/custom-agent instructions are:
  `.github/copilot-instructions.md`, `.github/agents/README.md`,
  `.github/agents/QUICK_START_v2.md`,
  `.github/agents/transformation-portal-architect.md`,
  `.github/agents/portal-app-steward.md`,
  `.github/agents/transformation-portal-specialist.md`,
  `docs/architecture/agent_governance.md`,
  `docs/guides/CUSTOM_AGENT_GUIDE.md`, and
  `docs/reference/AGENT_QUICK_REFERENCE.md`.
- Historical agent/RAG notes under `.github/agents/_archive/` or
  `.github/agents/rag_system/_archive/` are not live instructions.
- Use the narrowest live profile: Architect for governance/contract/CI/security,
  Steward for managed browser/frontdoor work, Specialist for backend/Lux
  Depth/archive/ingest/machine-mode execution.
- Blocking mypy authority is
  [docs/ci/TYPE_CHECKING_POLICY.md](docs/ci/TYPE_CHECKING_POLICY.md) plus
  `.github/workflows/build.yml`. Add paths only after
  `mypy --config-file=mypy.ini <path>` passes in the CI-pinned environment.

## Validation Ladders

- Quick local closeout:
  `make ci-quick`, `make test-fast`, `make pre-commit`, and
  `make check-worktree`.
- Full local CI when risk is broader:
  `make ci` or `make ci-full`.
- Orchestrator/API contracts:
  `make test-orchestrator-contract`,
  `make test-orchestrator-http-contract`, and
  `make validate-orchestrator-http` when a backend is running.
- Frontdoor/browser contracts:
  `make test-portal-contract`, `make test-frontdoor-contract`,
  `make test-accessibility-browser`, `make validate-portal-css-layer-parity`,
  `make validate-frontdoor-browser`, `make validate-portal-browser`, and
  `make validate-frontdoor-deployment-gate` for shared deployment posture.
  Before Playwright browser gates, install its managed Chromium once with
  `cd web/secure-landing && npm run test:browser:install`. The CDP smoke/parity
  validators require a valid `TP_PORTAL_BROWSER_BINARY` or Google Chrome. CSS
  layer parity additionally requires the Chrome for Testing product pinned in
  `tests/fixtures/portal-css/layer-parity-contract.json`; point
  `TP_PORTAL_BROWSER_BINARY` at that managed executable.
  For direct deployments, set `TP_FRONTDOOR_GATE_DEPLOYMENT_TARGET` and
  `TP_FRONTDOOR_GATE_DEPLOYMENT_URL`; the Vercel legacy alias is supported only
  for Vercel checks.
  Keep selectors and managed-auth observability stable. Build profiles are
  browser-local and actor-scoped: keep unsaved-draft discard explicit; migrate
  legacy profiles automatically only in standalone `direct_debug`, while
  managed actors must use the two-step legacy import. For pre-composite managed
  drafts, keep Build blocked and persistence paused until explicit claim or
  discard succeeds; failed recovery must preserve the legacy snapshot.
- Archive and pipeline gates:
  `make test-archive-gate-contract`, `make audit-pipeline-readiness`,
  `make validate-portal-lux-materials-live`, and
  `make validate-portal-fastvlm-captioning-live`.
- Service-backed lanes:
  `make test-orchestrator-postgres-contract`,
  `make test-orchestrator-postgres-app-contract`,
  `make test-worker-redis-contract`,
  `make test-frontdoor-redis-contract`,
  `make test-artifact-s3-contract`, and
  `make test-paid-pilot-services-contract`. Start only the services required by
  the lane and report missing services as environment blockers. Managed-provider
  staging uses `TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env make run-managed-paid-pilot-gate`;
  pass `MANAGED_PAID_PILOT_GATE_ARGS=--preflight-only` for clean-env preflight.
- Governance/docs gates:
  `make validate-ci`,
  `make check-stale-docs`, `make check-doc-heading-links`,
  `make check-todo-governance`, `make check-ci-sync`,
  `make check-piptools-cache`, `make check-requirements-lock-contract`,
  `make check-dependency-pinning`,
  `make check-json-serialization`, `make check-yaml-governance`,
  `make check-python-headers`, and
  `python3 scripts/governance/check_docs_structure.py --all`. For direct
  source-contract checks use
  `./scripts/setup/run_repo_python.sh scripts/governance/check_script_topology.py`
  and
  `./scripts/setup/run_repo_python.sh scripts/validation/check_gitleaks_workflow_contract.py`.
  For scoped dependency drift checks, the dedicated
  `.github/workflows/dependency-pinning-check.yml` workflow runs
  `python scripts/validation/check_dependency_pinning.py`; it enforces exact
  requirement pins and audits normal `constraints.txt` pins against the
  governed environment. When TODO or workflow baselines change, refresh
  committed docs from live repo state:
  `python3 scripts/validation/scan_todo_inventory.py --write-snapshot` updates
  `docs/analysis/todo_scanner_snapshot.json`, and
  `docs/ci/WORKFLOW_MATRIX.md` must match live `.github/workflows/*.yml`
  inventory counts plus per-workflow line estimates. Treat workflow-matrix
  recommendations as proposals until matching workflow YAML changes land.
- AI advisory workflow contracts:
  `./.venv/bin/pytest tests/test_summary_workflow.py tests/test_ai_code_review_workflow.py tests/test_smart_issue_management_workflow.py -q`.
  Issue summarizer fallback diagnostics carrying
  `<!-- ai-summarizer-diagnostic -->` stay log-only; only marker-free
  successful summaries should post PR comments.
- Secure-install hash pilot:
  `make -C requirements compile-hash-pilot LOCK_PYTHON_VERSION=3.11` then
  `make -C requirements check-hash-pilot LOCK_PYTHON_VERSION=3.11`; use
  `HASH_PILOT_OUT_DIR=/tmp/tp-hash-pilot` to keep pilot artifacts disposable.
  Match CI with `python -m pip install --upgrade "pip==26.2.1"` and
  `python -m pip install "pip-tools==7.6.1" "click==8.4.2"`. Use the same
  toolchain for the decisive generic-lock freshness check:
  `make -C requirements check-generic LOCK_PYTHON_VERSION=3.11`. Validate
  contract changes with
  `./.venv/bin/pytest tests/test_requirements_makefile_hash_pilot.py tests/test_secure_install_pilot_workflow.py -q`.
  This pilot is advisory and does not replace the standard checked-in lock or
  install flows.
- Root metadata contract tests:
  `./.venv/bin/pytest tests/validation/test_cloudflare_worker_root_shim_contract.py tests/validation/test_git_blame_ignore_revs_contract.py tests/validation/test_gitattributes_contract.py tests/validation/test_pylint_config_contract.py -v`.
- Coverage/cold-zone:
  `make coverage-report`, `make coverage-diff`, `make coverage-package`,
  `python3 scripts/ci/check_per_package_coverage.py coverage.xml`,
  `python3 scripts/ci/check_per_package_branch_coverage.py coverage.xml`, and
  `python3 scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref origin/main`.
  For floor ratchets, first reproduce the CI core-tier snapshot:
  `./.venv/bin/pytest -v tests/ -ra -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark" --cov=src/transformation_portal --cov=src/tp --cov=lux_depth_v3 --cov=app --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html --cov-fail-under=30 --cov-config=pyproject.toml`.
  Keep floor changes below measured baselines, and do not assume live-service
  Postgres/Redis/S3 coverage from this core lane.
- ML sampled coverage evidence:
  `TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf_home_cov TRANSFORMERS_CACHE=/tmp/transformers_cache_cov OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv/bin/pytest -v tests/vlm tests/spatial_ai/segmentation -ra -m "ml and not slow and not integration and not benchmark" --cov=src/transformation_portal/vlm --cov=src/transformation_portal/spatial_ai/segmentation --cov-report=term --cov-report=xml:coverage-ml-sampled.xml`.
  Treat this as cold-zone ratchet evidence, not a blocking local gate.
- Segmentation content-digest contract:
  `./.venv/bin/pytest tests/spatial_ai/segmentation/test_content_digest.py -v`.
- Legacy segmentation adapter contracts:
  `./.venv/bin/pytest tests/segmentation/test_legacy_segmentation_contracts.py -v`.
  Requires `torch`; report missing optional ML deps as environment blockers.
- SAM2 CPU fallback benchmark:
  `TP_RUN_BENCHMARKS=1 TP_SAM2_BENCHMARK_DEVICE=cpu ./.venv/bin/pytest tests/spatial_ai/segmentation/test_sam2_backend_performance.py::TestSAM2AutoModePerformance::test_auto_mode_latency_512x512 -v -s`.
  Requires `checkpoints/sam2.1_hiera_large.pt`.

## Local Runtime Commands

- Environment setup:
  `make venv`, `make install-core`, `make repair-core-venv`, and
  `make check-environment`.
- Local stack:
  `make dev-write-env`, `make dev-start`, and `make dev-stop`.
  Logs go to `/tmp/tp-backend.log` and `/tmp/tp-frontdoor.log`.
- Backend only:
  `make run-backend-local` or `make run-backend-local-noreload`; both require
  `TP_API_KEY`.
- Frontdoor only:
  `make seed-frontdoor-user` once for the canonical local managed-frontdoor
  credential fixture, then `make run-frontdoor-local`; the frontdoor requires
  backend readiness, auth env, and a free `localhost:3000`.
- Managed backend services:
  `docker compose up -d postgres`, `docker compose up -d redis`, and
  `docker compose --profile paid-pilot up -d minio minio-create-bucket`.
  Pair Postgres with `make db-upgrade`.
- External orchestrator workers:
  use `TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED=0 make run-backend-local-noreload`
  on backend hosts, then run `make run-orchestrator-worker` with
  `TP_ORCHESTRATOR_STATE_BACKEND=postgres`, `TP_ORCHESTRATOR_QUEUE_BACKEND=redis`,
  `TP_DATABASE_URL`, and `TP_REDIS_URL` set.
- Container smoke:
  `docker compose run --rm tp-init`,
  `docker compose up --build transformation-portal-cpu`, and
  `docker build --target cpu -t transformation-portal:cpu-nonroot-test .`.
  Do not add Compose healthchecks for CPU/GPU unless intentionally overriding
  Dockerfile healthchecks.

## ML And Optional Runtimes

- `make install-ml` and `make install-ml-raw` are intentionally disabled until
  trusted umbrella/raw lock contracts exist.
- Supported ML installs:
  `make install-ml-core`, `make install-ml-sam2` on native macOS Apple
  Silicon, and `make install-ml-coreml` on macOS when the trusted lock exists.
- Target-owned lock lanes:
  `make compile-ml-darwin-arm64`, `make update-ml-darwin-arm64`, and
  `make check-ml-darwin-arm64`. Linux x86_64 and Darwin x86_64 ML lock lanes
  are retired fail-closed stubs.
- Runtime installers:
  `./scripts/setup/install_da3_runtime.sh`,
  `./scripts/setup/install_depth_pro_runtime.sh`,
  `./scripts/setup/install_raw_runtime.sh`, and
  `./scripts/setup/install_fastvlm_runtime.sh`.
- FastVLM captioning is optional and subprocess-only. Keep it under
  `.runtime/fastvlm/`; its output is advisory and never quality-gate evidence.

## Workflow Scripts Worth Knowing

- Production pipeline helpers:
  `./scripts/pipelines/run_montecito_apex_full.sh`,
  `./scripts/pipelines/run_montecito_apex_lean.sh`,
  `./scripts/pipelines/process_source_tiffs_apex.sh`,
  `./scripts/pipelines/run_800_picacho_efficientsam_validation.sh`, and
  `./scripts/pipelines/hdr_production_pipeline.sh`.
- Archive/fixity:
  `./scripts/pipelines/run_sealed_eval_72h.sh --archive-index <path> --archive-root <path>`
  and
  `./scripts/pipelines/run_fixity_cycle.sh --archive-index <path> --archive-root <path>`.
- Dependency and environment guards:
  `./scripts/validate_dependency_constraints.sh`,
  `./scripts/setup/ensure_node_version.sh`, and
  `./scripts/setup/run_repo_python.sh scripts/validation/check_unsafe_torch_load.py --fix-suggestions`.
  Unicode-control scans use
  `./scripts/setup/run_repo_python.sh scripts/validation/check_unicode_controls.py <paths>`;
  omit paths to scan staged Python, YAML, and Markdown files.
- Script topology:
  canonical implementations live under governed `scripts/analysis`,
  `scripts/ci`, `scripts/maintenance`, `scripts/setup`, `scripts/pipelines`,
  `scripts/validation`, `scripts/verification`, `scripts/utilities`, or
  `src/transformation_portal` paths. Public compatibility wrappers in
  `scripts/` must delegate to the canonical module, bootstrap the right import
  root, and propagate exit status. Validate with
  `./scripts/setup/run_repo_python.sh scripts/governance/check_script_topology.py`.
- Cloudflare Worker root shim:
  root `package.json` is only a Workers Builds deploy shim. Use
  `npm run worker:dry-run` or `npm run worker:deploy`; deploys must preserve
  dashboard-managed vars via `--keep-vars`, and scripts stay aligned with
  `cloudflare/transformationportal-worker`.
- Unified Luxury batch I/O benchmark:
  `make benchmark-unified-luxury-batch-io`; pass harness options through
  `UNIFIED_LUXURY_BATCH_IO_BENCHMARK_ARGS`. Synthetic fixtures are smoke
  evidence only; representative production TIFFs are required before changing
  defaults.
- Presence Security CLI:
  `.venv/bin/presence-security params`, `anchor`, and `watermark` are the
  current helpers for sessionized Presence Compiler parameters, SHA3 anchor
  payloads, and manifest/session watermarks.
- Frontdoor source bundle:
  from `web/secure-landing`, use `npm run build:portal`,
  `npm run check:utility-ownership`, and `npm run check:css-layer-parity`.

## Review And Triage Policy

- CI triage: identify current live failures first, then group by likely root
  cause with job/test/error evidence. Keep downstream summary gates separate
  from the upstream failing job.
- Code review stance: findings first, ordered by severity with file/line
  references. If there are no findings, say so and name residual test gaps.
- PR review threads: address only actionable threads selected by the user or
  clearly requested by the latest instruction. Use precise fixes, focused
  validation, and reply/resolve only after the fix is pushed.
- Security review/threat modeling: use when requested or when a change adds
  external downloads, model/runtime integrations, artifact export mechanisms,
  workflow write permissions, filesystem/network scripts, or dependency policy
  changes.
- Documentation work: when commands, CLI flags, workflows, validation scripts,
  dependency lanes, schemas, or public operator guidance change, update docs in
  the same pass. If a docs command exposes a real source failure, fix the source
  issue rather than weakening docs.
- Visual evidence: capture screenshots or rendered artifacts when browser UI,
  image outputs, or visual comparison behavior changes. Do not add visual proof
  churn for backend-only changes.

## Closeout Expectations

- Report exactly what was changed, what was proven green, what is still
  failing, and whether any failure is product logic, stale test logic, or
  environment/tooling.
- Do not say "all green" unless the canonical path for the touched surface
  passed.
- Before staging, check `git status --short --branch` and stage explicit paths.
- Before final handoff, run `git diff --check` on touched files and confirm no
  unexpected generated artifacts remain.
