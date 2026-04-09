SHELL := /bin/sh

# Resolve a Python interpreter: prefer local venv, otherwise fall back to python3
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)

# Common subsets (fast tests avoid heavy/optional paths)
FAST_TESTS := \
	tests/test_material_response.py \
	tests/test_board_material_aerial_enhancer.py \
	tests/test_coastal_estate_render.py \
	tests/test_codebase_philosophy_auditor.py \
	tests/test_cognitive_material_response.py \
	tests/test_decision_decay_dashboard.py \
	tests/test_depth_tools.py \
	tests/test_evolutionary_checkpoint.py \
	tests/test_float_roundtrip.py \
	tests/test_golden_hour_courtyard_workflow.py

PHASE6_SMOKE_TESTS := \
	tests/test_streaming_stages.py \
	tests/test_streaming_async_pipeline.py::test_async_pipeline_with_fake_stages_processes_end_to_end \
	tests/test_pipeline_unified_smoke.py \
	tests/test_rendering_4k_pipeline_smoke.py \
	tests/test_lux_render_pipeline_smoke.py \
	tests/lux_depth_v3/test_orchestrator_smoke.py

.PHONY: help test-fast test-novideo test-full test-integration test-structure test-utils test-orchestrator-contract test-orchestrator-http-contract test-portal-contract test-frontdoor-contract run-frontdoor-local validate-orchestrator-http validate-portal-browser validate-frontdoor-browser audit-pipeline-readiness coverage-fast-scope venv setup clean \
        lint lint-parity ci ci-full pre-commit install-hooks quality-check fix-quality validate-ci organize-docs check-json-serialization check-piptools-cache \
        check-yaml-governance check-stale-docs lock lock-prod lock-ci lock-dev install-core install-ml install-ml-core install-ml-raw install-ml-sam2 install-ml-coreml docs docs-clean \
        check-test-markers check-ci-sync

help:
	@echo "Targets:"
	@echo "  setup              Install package in editable mode (pip install -e .)"
	@echo "  install-core       Install core dependencies with constraints"
	@echo "  install-ml         Disabled: no trusted umbrella ML lockfile contract"
	@echo "  install-ml-core    Install ML core layer only (cross-platform baseline)"
	@echo "  install-ml-raw     Disabled: no trusted checked-in RAW lockfile contract"
	@echo "  install-ml-sam2    Install ML SAM2 layer (optional segmentation)"
	@echo "  install-ml-coreml  Disabled unless a trusted CoreML lockfile is present"
	@echo "  test-fast          Run fast subset plus Phase 6 smoke coverage"
	@echo "  test-novideo       Run all tests excluding video suite via -k filter"
	@echo "  test-full          Run entire test suite (parallel if xdist present)"
	@echo "  test-orchestrator-contract  Run route-level portal orchestrator contract suite"
	@echo "  test-orchestrator-http-contract  Run HTTP-only orchestrator contract tests"
	@echo "  test-portal-contract  Run portal runtime/browser contract tests"
	@echo "  test-frontdoor-contract  Run managed frontdoor Node contract/build checks"
	@echo "  run-frontdoor-local  Start the canonical local managed frontdoor on localhost:3000"
	@echo "  test-integration   Run integration tests (requires HF_TOKEN)"
	@echo "  test-structure     Run codebase structure validation tests"
	@echo "  test-utils         Run tests for performance and error handling utilities"
	@echo "  coverage-fast-scope  Run branch coverage for audited core/config and streaming paths"
	@echo "  validate-orchestrator-http  Run the live orchestrator HTTP smoke audit"
	@echo "  validate-portal-browser  Run the live browser smoke audit with an isolated local backend"
	@echo "  validate-frontdoor-browser  Run the live browser smoke audit with isolated local backend/frontdoor runtimes"
	@echo "  audit-pipeline-readiness  Run the local four-pipeline readiness audit"
	@echo "  venv               Create local .venv if missing"
	@echo "  clean              Remove Python cache files and build artifacts"
	@echo ""
	@echo "Quality & CI:"
	@echo "  lint               Run advisory lint checks (requires 'make install-core')"
	@echo "  lint-parity        Run the GitHub lint job locally using Python 3.12 + requirements-lint.txt"
	@echo "  ci                 Run local CI checks (lint + hygiene + fast tests)"
	@echo "  ci-full            Run comprehensive CI simulation (all checks)"
	@echo "  pre-commit         Run pre-commit hooks manually with CI-aligned formatter versions"
	@echo "  install-hooks      Install git pre-commit hook"
	@echo "  quality-check      Run all quality checks (lint + structure + tests)"
	@echo "  check-json-serialization  Fail on raw json.dump/json.dumps outside approved modules"
	@echo "  check-yaml-governance  Fail on raw yaml.safe_load outside approved preset/exempt boundaries"
	@echo "  check-piptools-cache  Fail if requirements/.pip-tools-cache is tracked in git"
	@echo "  check-stale-docs   Detect changed-file references to deleted docs root paths"
	@echo "  check-test-markers Audit test marker coverage (ADR-044)"
	@echo "  check-ci-sync      Verify CI dependency files are in sync (no drift)"
	@echo "  fix-quality        Auto-fix common quality issues"
	@echo "  validate-ci        Validate GitHub Actions workflow configs"
	@echo "  organize-docs      Organize markdown files to docs/ subdirectories"
	@echo ""
	@echo "Dependency locking:"
	@echo "  lock               Regenerate all requirements lockfiles (prod/ci/dev)"
	@echo "  lock-prod          Regenerate requirements.lock.txt"
	@echo "  lock-ci            Regenerate requirements-ci.lock.txt"
	@echo "  lock-dev           Regenerate requirements-dev.lock.txt"
	@echo ""
	@echo "Documentation:"
	@echo "  docs               Build API documentation with Sphinx"
	@echo "  docs-clean         Clean generated documentation files"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
	fi

setup: venv
	@echo "Installing package in editable mode..."
	@"$(PY)" -m pip install -e .

install-core: venv
	@echo "Installing core dependencies with constraints..."
	@if [ -f requirements/constraints.txt ]; then \
		"$(PY)" -m pip install -e ".[dev]" -c requirements/constraints.txt; \
	else \
		echo "Warning: requirements/constraints.txt not found, installing without constraints"; \
		"$(PY)" -m pip install -e ".[dev]"; \
	fi

# ML Layer Install Targets
# These support fine-grained ML capability installation per the layered strategy.
# See requirements/README.md for layer documentation.

install-ml: venv
	@echo "Error: install-ml no longer has a trusted checked-in umbrella lockfile contract."
	@echo "Error: use target-specific bootstrap profiles instead, for example:"
	@echo "Error:   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu"
	@echo "Error:   ./scripts/bootstrap/install_ml_stack.sh --profile core-mps"
	@echo "Error:   ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda"
	@exit 1

install-ml-core: venv
	@echo "Installing ML core layer (cross-platform baseline)..."
	@ml_lock=""; \
	py_os="$$(\"$(PY)\" -c 'import platform; print(platform.system())')"; \
	py_arch="$$(\"$(PY)\" -c 'import platform; print(platform.machine())')"; \
	case "$$py_arch" in \
		aarch64) py_arch="arm64" ;; \
		amd64) py_arch="x86_64" ;; \
	esac; \
	if [ "$$py_os" = "Darwin" ] && [ "$$py_arch" = "x86_64" ] && [ -f requirements/ml-core-darwin-x86_64.txt ]; then \
		ml_lock="requirements/ml-core-darwin-x86_64.txt"; \
	elif [ "$$py_os" = "Darwin" ] && [ "$$py_arch" = "arm64" ] && [ -f requirements/ml-core-darwin-arm64.txt ]; then \
		ml_lock="requirements/ml-core-darwin-arm64.txt"; \
	elif [ "$$py_os" = "Linux" ] && [ -f requirements/ml-core-linux.txt ]; then \
		ml_lock="requirements/ml-core-linux.txt"; \
	fi; \
	if [ -n "$$ml_lock" ]; then \
		echo "Using $$ml_lock"; \
		"$(PY)" -m pip install -r "$$ml_lock" && \
		"$(PY)" -m pip install -e .; \
	else \
		echo "Error: platform-specific ML core lockfile not found. Run 'cd requirements && make compile' first."; \
		exit 1; \
	fi

install-ml-raw: venv
	@echo "Error: install-ml-raw no longer has a trusted checked-in lockfile contract."
	@echo "Error: use a trusted target-specific flow or regenerate a target-correct lockfile in the appropriate environment."
	@exit 1

install-ml-sam2: venv
	@echo "Installing ML SAM2 segmentation layer via bootstrap script..."
	@echo "SAM2 requires non-standard install semantics and is scripted-only."
	@./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,sam2
	@"$(PY)" -m pip install -e .

install-ml-coreml: venv
	@echo "Installing ML CoreML layer (macOS only)..."
	@if [ "$$(uname -s)" != "Darwin" ]; then \
		echo "Warning: CoreML layer is only available on macOS. Skipping."; \
	elif [ -f requirements/ml-coreml.txt ]; then \
		"$(PY)" -m pip install -r requirements/ml-coreml.txt && \
		"$(PY)" -m pip install -e .; \
	else \
		echo "Error: install-ml-coreml no longer has a trusted checked-in lockfile contract."; \
		echo "Error: regenerate a target-correct CoreML lockfile in the appropriate environment first."; \
		exit 1; \
	fi

test-fast:
	@"$(PY)" -m pytest -q $(FAST_TESTS) $(PHASE6_SMOKE_TESTS)

test-novideo:
	@"$(PY)" -m pytest -q -k 'not video_master_grader'

test-full:
	@if "$(PY)" -m pip list | grep -q pytest-xdist; then \
		"$(PY)" -m pytest -q -n auto tests; \
	else \
		"$(PY)" -m pytest -q tests; \
	fi

test-integration:
	@echo "Running integration tests (requires HF_TOKEN)..."
	@TP_RUN_HF_MODEL_TESTS=1 "$(PY)" -m pytest -v tests/test_da3_inference_integration.py

test-structure:
	@echo "Running codebase structure validation tests..."
	@"$(PY)" -m pytest -v tests/test_codebase_structure.py

test-utils:
	@echo "Running utility tests..."
	@"$(PY)" -m pytest -v tests/test_performance_utils.py tests/test_error_handling.py

test-orchestrator-contract:
	@echo "Running portal orchestrator contract suite..."
	@"$(PY)" -m pytest -q tests/test_app_orchestrator_runtime.py tests/test_app_orchestrator_contract_http.py tests/validation/test_portal_smoke_scripts.py

test-orchestrator-http-contract:
	@echo "Running HTTP-only orchestrator contract tests..."
	@"$(PY)" -m pytest -q tests/test_app_orchestrator_contract_http.py

test-portal-contract:
	@echo "Running portal runtime/browser contract tests..."
	@"$(PY)" -m pytest -q tests/test_app_orchestrator_runtime.py tests/validation/test_portal_smoke_scripts.py

test-frontdoor-contract:
	@echo "Running managed frontdoor contract checks..."
	@cd web/secure-landing && npm test
	@cd web/secure-landing && npm run build

run-frontdoor-local:
	@echo "Starting the canonical local managed frontdoor on localhost:3000..."
	@./scripts/setup/run_frontdoor_local.sh

validate-orchestrator-http:
	@echo "Running live orchestrator HTTP smoke validation..."
	@"$(PY)" scripts/validation/validate_orchestrator_http_smoke.py

validate-portal-browser:
	@echo "Running live portal browser smoke validation..."
	@TP_API_KEY="$${TP_API_KEY:-contract-secret}" "$(PY)" scripts/validation/validate_portal_browser_smoke.py --spawn-local-backend --api-key "$${TP_API_KEY:-contract-secret}"

validate-frontdoor-browser:
	@echo "Running live managed frontdoor browser smoke validation..."
	@"$(PY)" scripts/validation/validate_frontdoor_browser_smoke.py --spawn-local-backend --spawn-local-frontdoor

audit-pipeline-readiness:
	@echo "Running safe local four-pipeline readiness audit..."
	@"$(PY)" scripts/validation/audit_pipeline_readiness.py

coverage-fast-scope:
	@rm -f .coverage.fast-scope .coverage.fast-scope.*
	@COVERAGE_FILE=.coverage.fast-scope "$(PY)" -m pytest \
		--cov=src/transformation_portal/core/config \
		--cov=src/transformation_portal/streaming \
		--cov-branch \
		--cov-report=term-missing \
		tests/test_core_config_presets.py \
		tests/test_streaming_async_pipeline.py \
		tests/test_preset_health.py

clean:
	@echo "Cleaning Python cache files and build artifacts..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .hypothesis/ 2>/dev/null || true
	@echo "✓ Cleanup complete"

# --- Additional developer + CI helpers ---

lint:
	@echo "Running advisory lint via shared policy..."
	@PYTHON_BIN="$(PY)" ./scripts/lint_runner.sh advisory

lint-parity:
	@echo "Running CI-aligned lint parity..."
	@./scripts/setup/run_lint_tool.sh parity

ci: lint check-json-serialization check-yaml-governance check-piptools-cache check-requirements-lock-contract check-ci-sync test-fast test-orchestrator-contract test-frontdoor-contract
	@echo "✅ Local CI checks completed successfully."

# Comprehensive CI simulation
ci-full:
	@echo "Running comprehensive CI simulation..."
	@./scripts/local_ci_check.sh

# Quick CI check (fast mode)
ci-quick:
	@echo "Running quick CI checks..."
	@./scripts/local_ci_check.sh --quick

# Pre-commit checks
pre-commit:
	@echo "Running pre-commit checks..."
	@pre-commit run --all-files --show-diff-on-failure

# Install git hooks
install-hooks:
	@echo "Installing git pre-commit hook..."
	@pre-commit install -f
	@echo "✓ Pre-commit hook installed via pre-commit"

# Quality check (all validations)
quality-check: lint validate-ci
	@echo "Running root file placement check..."
	@./scripts/setup/pre-commit-check.sh --all
	@echo "✅ Quality checks completed."

# Auto-fix quality issues
fix-quality:
	@echo "Auto-fixing quality issues..."
	@"$(PY)" scripts/auto_fix_quality.py --fix-all

# Fix quality issues (dry-run)
check-quality:
	@echo "Checking for quality issues (dry-run)..."
	@"$(PY)" scripts/auto_fix_quality.py --dry-run

check-stale-docs:
	@"$(PY)" scripts/governance/check_stale_docs_paths.py

# Validate CI configuration
validate-ci:
	@echo "Validating GitHub Actions workflows..."
	@"$(PY)" scripts/validate_ci_config.py
	@echo "Validating dependency-update workflow contract..."
	@"$(PY)" scripts/validation/check_dependency_update_workflow.py
	@echo "Validating Dependabot config contract..."
	@"$(PY)" scripts/validation/check_dependabot_config.py

check-json-serialization:
	@echo "Checking JSON serialization guardrails..."
	@"$(PY)" scripts/validation/check_raw_json_usage.py

check-yaml-governance:
	@echo "Checking YAML governance boundary..."
	@"$(PY)" scripts/validation/check_yaml_governance_boundary.py

check-piptools-cache:
	@echo "Checking pip-tools cache guardrails..."
	@"$(PY)" scripts/validation/check_piptools_cache_tracked.py

check-requirements-lock-contract:
	@echo "Checking requirements lock contract..."
	@"$(PY)" scripts/validation/check_requirements_lock_contract.py

check-test-markers:
	@echo "Auditing test marker coverage (ADR-044)..."
	@"$(PY)" scripts/validation/check_test_markers.py --audit

check-ci-sync:
	@echo "Checking CI dependency file sync..."
	@"$(PY)" scripts/validation/check_ci_dep_sync.py

# Organize documentation
organize-docs:
	@echo "Organizing documentation files..."
	@./scripts/organize_docs.sh --apply

# Organize documentation (dry-run)
check-docs:
	@echo "Checking documentation organization..."
	@./scripts/organize_docs.sh --dry-run

# --- Dependency locking (pip-tools) ---

lock: lock-prod lock-ci lock-dev
	@echo "✓ Lockfiles updated (prod/ci/dev)"

lock-prod:
	@echo "Locking production requirements -> requirements.lock.txt"
	@pip-compile --generate-hashes \
		-o requirements.lock.txt \
		requirements.txt

lock-ci:
	@echo "Locking CI requirements -> requirements-ci.lock.txt"
	@pip-compile --generate-hashes \
		-o requirements-ci.lock.txt \
		requirements-ci.txt

lock-dev:
	@echo "Locking dev requirements -> requirements-dev.lock.txt"
	@pip-compile --generate-hashes \
		-o requirements-dev.lock.txt \
		requirements-dev.txt

# --- Documentation ---

docs:
	@echo "Building API documentation with Sphinx..."
	@"$(PY)" -m pip install -q sphinx sphinx-rtd-theme sphinx-autodoc-typehints
	@"$(PY)" -m sphinx -b html -W --keep-going docs/api docs/api/_build/html
	@echo "✓ Documentation built in docs/api/_build/html"

docs-clean:
	@echo "Cleaning generated documentation..."
	@rm -rf docs/api/_build docs/api/_templates docs/api/_static
	@echo "✓ Documentation cleaned"
