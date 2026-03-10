# AGENTS.md

Quick reference for common workflows and commands in this repo.

## Common commands (Makefile)
- `make venv` create local `.venv` if missing.
- `make setup` install package in editable mode.
- `make install-core` install core runtime + dev tooling dependencies (with constraints if present).
- `make install-ml` install ML tier dependencies (with constraints if present).
- `make test-fast` run fast test subset.
- `make test-novideo` run tests excluding luxury video master grader tests (filters out `video_master_grader`).
- `make test-full` run full test suite (parallel if xdist installed).
- `make test-integration` run DA3/HuggingFace model-loading integration (`tests/test_da3_inference_integration.py`) with `TP_RUN_HF_MODEL_TESTS=1` (downloads models from HF Hub unless offline; typically requires `HF_TOKEN`).
- `make test-structure` run codebase structure validation.
- `make test-utils` run performance/error utility tests.
- `make test-orchestrator-contract` run portal orchestrator contract tests (`tests/test_app_orchestrator_runtime.py` and `tests/test_app_orchestrator_contract_http.py`).
- `make clean` remove Python caches and build/test artifacts.
- `make lint` run flake8 + pylint (non-blocking).
- `make ci` run local CI checks (lint + check-json-serialization + check-piptools-cache + test-fast + test-orchestrator-contract).
- `make ci-full` run comprehensive local CI (`./scripts/local_ci_check.sh`).
- `make ci-quick` run quick local CI (`./scripts/local_ci_check.sh --quick`).
- `make pre-commit` run pre-commit checks.
- `make install-hooks` install git pre-commit hook.
- `make quality-check` run lint + CI validation + doc structure checks.
- `make fix-quality` auto-fix quality issues (`scripts/auto_fix_quality.py --fix-all`).
- `make check-quality` dry-run quality auto-fix checks (`scripts/auto_fix_quality.py --dry-run`).
- `make validate-ci` validate GitHub Actions configs.
- `make check-json-serialization` fail when raw `json.dump`/`json.dumps` usage is detected outside approved modules.
- `make check-piptools-cache` fail if `requirements/.pip-tools-cache` is tracked in git.
- `make organize-docs` move markdown files into `docs/` (repo hygiene).
- `make check-docs` dry-run docs organization.
- `make lock` regenerate all requirements lockfiles.
- `make lock-prod` regenerate `requirements.lock.txt`.
- `make lock-ci` regenerate `requirements-ci.lock.txt`.
- `make lock-dev` regenerate `requirements-dev.lock.txt`.
- `make docs` build API docs with Sphinx.
- `make docs-clean` remove generated docs output.

## Workflow scripts (bash)
- `./scripts/pipelines/run_montecito_apex_full.sh` run Montecito Shores APEX batch with all deliverables (interactive prompt).
- `./scripts/pipelines/run_montecito_apex_lean.sh` run Montecito Shores APEX batch (lean outputs, faster).
- `./scripts/pipelines/process_source_tiffs_apex.sh` batch APEX V2 enhancement for `input_images/source_tiffs` with optional depth generation.
- `./scripts/pipelines/process_source_tiffs_individual.sh` per-image APEX V2 enhancement commands (manual execution).
- `./scripts/test_v2_integration.sh` validate end-to-end lux-depth-v3 + V2 stage integration (`--verbose`, `--clean` available).
- `./scripts/validate_dependency_constraints.sh` enforce dependency pinning rules used by repo policy (`--verbose` available).
- `./scripts/pipelines/run_fixity_cycle.sh` run archive hash-manifest scan + verification cycle for fixity evidence (`--archive-index` and `--archive-root` required).
- `./scripts/diagnostics/full_chain_determinism_trial.sh` run Phase 4C/4D/4E determinism checks (`--input-root` or `--capture-metadata`).
- `./scripts/setup/auto-organize-install.sh` install repository file-organization guardrails and pre-commit hook.
- `./scripts/setup/pre-commit-check.sh` run root-file placement validation manually (also used by the hook).

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
