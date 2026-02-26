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
- `make lint` run flake8 + pylint (non-blocking).
- `make ci` run local CI checks (lint + test-fast).
- `make ci-full` run comprehensive local CI (`./scripts/local_ci_check.sh`).
- `make ci-quick` run quick local CI (`./scripts/local_ci_check.sh --quick`).
- `make pre-commit` run pre-commit checks.
- `make install-hooks` install git pre-commit hook.
- `make quality-check` run lint + CI validation + doc structure checks.
- `make fix-quality` auto-fix quality issues (`scripts/auto_fix_quality.py --fix-all`).
- `make validate-ci` validate GitHub Actions configs.
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

## ComfyUI workflows (`workflows/`)
- `python -c "from transformation_portal.comfyui import WorkflowTemplates; WorkflowTemplates.save_all_templates('workflows/templates')"` generate ComfyUI template workflows.
- `python -c "from transformation_portal.comfyui import WorkflowExecutor; from transformation_portal.comfyui.workflow_builder import Workflow; wf = Workflow.load('workflows/examples/simple_enhancement.json'); print(WorkflowExecutor(verbose=True).execute(wf)['success'])"` execute the `simple_enhancement` workflow example.
