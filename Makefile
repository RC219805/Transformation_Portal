SHELL := /bin/sh

# Resolve Python interpreters at recipe runtime so targets that create or repair
# .venv immediately switch to the repo interpreter on subsequent lines.
BOOTSTRAP_PY = $$(./scripts/setup/resolve_python_311.sh)
PY = $$(./scripts/setup/resolve_python_311.sh)
PRE_COMMIT_BIN := $(shell if [ -x .venv/bin/pre-commit ]; then printf '%s' .venv/bin/pre-commit; elif [ -x .venv/Scripts/pre-commit.exe ]; then printf '%s' .venv/Scripts/pre-commit.exe; fi)

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

.PHONY: help test-fast test-novideo test-full test-integration test-structure test-utils test-orchestrator-contract test-orchestrator-http-contract test-portal-contract test-frontdoor-contract test-archive-gate-contract seed-frontdoor-user run-frontdoor-local validate-orchestrator-http validate-portal-lux-materials-live validate-portal-css-layer-parity validate-portal-browser validate-frontdoor-browser validate-frontdoor-deployment-gate audit-pipeline-readiness coverage-fast-scope coverage-report coverage-diff coverage-package venv repair-core-venv setup clean \
        lint lint-parity ci ci-full pre-commit install-hooks quality-check fix-quality validate-ci organize-docs check-json-serialization check-piptools-cache \
        check-python-headers check-yaml-governance check-stale-docs check-doc-heading-links lock lock-prod lock-ci lock-dev install-core install-ml install-ml-core install-ml-raw install-ml-sam2 install-ml-coreml docs docs-clean \
        check check-test-markers check-ci-sync check-todo-governance check-environment check-portal-asset-budgets validate-full validate-quick clean-frontdoor clean-all check-worktree \
        compile-ml-darwin-arm64 update-ml-darwin-arm64 check-ml-darwin-arm64 \
        compile-ml-linux-x86_64 update-ml-linux-x86_64 check-ml-linux-x86_64 \
        compile-ml-darwin-x86_64 update-ml-darwin-x86_64 check-ml-darwin-x86_64

help:
	@echo "Targets:"
	@echo "  setup              Install package in editable mode (pip install -e .)"
	@echo "  install-core       Install pinned core runtime + dev tooling dependencies into .venv"
	@echo "  repair-core-venv   Recreate .venv and reinstall the pinned core environment"
	@echo "  install-ml         Disabled: no trusted umbrella ML lockfile contract"
	@echo "  install-ml-core    Install ML core layer only (supported Apple Silicon baseline)"
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
	@echo "  test-archive-gate-contract  Run archive gate readiness + HTTP contract tests (Gates A, B, C)"
	@echo "  seed-frontdoor-user  Seed the canonical local managed-frontdoor credential fixture under /tmp"
	@echo "  run-frontdoor-local  Start the canonical local managed frontdoor on localhost:3000"
	@echo "  test-integration   Run integration tests (requires HF_TOKEN)"
	@echo "  test-structure     Run codebase structure validation tests"
	@echo "  test-utils         Run tests for performance and error handling utilities"
	@echo "  coverage-fast-scope  Run branch coverage for audited core/config and streaming paths"
	@echo "  coverage-report    Generate comprehensive coverage report (HTML, XML, terminal)"
	@echo "  coverage-diff      Check diff coverage against main branch (≥85% required)"
	@echo "  coverage-package   Generate package-level coverage baseline report for ratcheting"
	@echo "  validate-orchestrator-http  Run the live orchestrator HTTP smoke audit"
	@echo "  validate-portal-lux-materials-live  Run live Lux Materials V3 segmentation backend smoke"
	@echo "  validate-portal-css-layer-parity  Validate production portal CSS layer contracts and computed-style parity"
	@echo "  validate-portal-browser  Run the live browser smoke audit with an isolated local backend"
	@echo "  validate-frontdoor-browser  Run the live browser smoke audit with isolated local backend/frontdoor runtimes"
	@echo "  validate-frontdoor-deployment-gate  Run the manual shared-deployment frontdoor posture gate"
	@echo "  validate-full      Run the full validation suite (all checks + browser smokes)"
	@echo "  validate-quick     Run quick validation (skip browser smokes)"
	@echo "  audit-pipeline-readiness  Run the local four-pipeline readiness audit"
	@echo "  venv               Create or validate .venv with Python 3.11+; fail on unsupported or broken environments"
	@echo "  clean              Remove Python cache files and build artifacts"
	@echo "  clean-frontdoor    Remove frontdoor build artifacts (.next)"
	@echo "  clean-all          Remove all build artifacts (Python + Node)"
	@echo ""
	@echo "Quality & CI:"
	@echo "  lint               Run advisory lint checks (requires 'make install-core')"
	@echo "  lint-parity        Run the GitHub lint job locally using Python 3.12 + requirements-lint.txt"
	@echo "  ci                 Run local CI checks (lint + hygiene + fast tests)"
	@echo "  ci-full            Run comprehensive CI simulation (all checks)"
	@echo "  pre-commit         Run pre-commit hooks manually with CI-aligned formatter versions"
	@echo "  install-hooks      Install git pre-commit hook"
	@echo "  quality-check      Run all quality checks (lint + structure + tests)"
	@echo "  check-environment  Run pre-flight environment validation"
	@echo "  check             Verify generic layered requirements under requirements/"
	@echo "  check-worktree     Check if git worktree is clean"
	@echo "  check-json-serialization  Fail on raw json.dump/json.dumps outside approved modules"
	@echo "  check-python-headers  Fail on invalid encoding-cookie-like text in Python header lines 1-2"
	@echo "  check-yaml-governance  Fail on raw yaml.safe_load outside approved preset/exempt boundaries"
	@echo "  check-piptools-cache  Fail if requirements/.pip-tools-cache is tracked in git"
	@echo "  compile-ml-darwin-arm64  Compile target-owned Darwin arm64 ML lock via requirements/"
	@echo "  update-ml-darwin-arm64   Update target-owned Darwin arm64 ML lock via requirements/"
	@echo "  check-ml-darwin-arm64    Verify target-owned Darwin arm64 ML lock via requirements/"
	@echo "  compile-ml-linux-x86_64  Retired unsupported Linux ML lane (fails closed)"
	@echo "  update-ml-linux-x86_64   Retired unsupported Linux ML lane (fails closed)"
	@echo "  check-ml-linux-x86_64    Retired unsupported Linux ML lane (fails closed)"
	@echo "  compile-ml-darwin-x86_64 Retired unsupported Darwin x86_64 ML lane (fails closed)"
	@echo "  update-ml-darwin-x86_64  Retired unsupported Darwin x86_64 ML lane (fails closed)"
	@echo "  check-ml-darwin-x86_64   Retired unsupported Darwin x86_64 ML lane (fails closed)"
	@echo "  check-stale-docs   Detect changed-file references to deleted docs root paths"
	@echo "  check-doc-heading-links  Validate markdown links that target related doc headings"
	@echo "  check-test-markers Audit test marker coverage (ADR-044)"
	@echo "  check-ci-sync      Verify CI dependency files are in sync (no drift)"
	@echo "  check-todo-governance  Verify TODO governance compliance (tracking refs)"
	@echo "  check-portal-asset-budgets  Validate raw/gzipped portal asset size budgets"
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
	@repo_venv_py=""; \
	if [ -x .venv/bin/python ]; then \
		repo_venv_py=.venv/bin/python; \
	elif [ -x .venv/Scripts/python.exe ]; then \
		repo_venv_py=.venv/Scripts/python.exe; \
	fi; \
	if [ -n "$$repo_venv_py" ]; then \
		if "$$repo_venv_py" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then \
			echo ".venv already present"; \
		else \
			venv_version="$$("$$repo_venv_py" -V 2>&1 || echo 'Python version unavailable')"; \
			echo "Error: existing .venv is not using Python 3.11+ ($$venv_version)."; \
			echo "Error: run 'make repair-core-venv' to recreate the repo environment."; \
			exit 1; \
		fi; \
	elif [ -d .venv ]; then \
		echo "Error: .venv exists but is missing a usable Python interpreter."; \
		echo "Error: run 'make repair-core-venv' to recreate the repo environment."; \
		exit 1; \
	else \
		bootstrap_py="$(BOOTSTRAP_PY)"; \
		"$$bootstrap_py" -m venv .venv && echo "Created .venv with $$bootstrap_py"; \
	fi

setup: venv
	@echo "Installing package in editable mode..."
	@"$(PY)" -m pip install -e .

install-core: venv
	@echo "Installing pinned core dependencies into .venv..."
	@"$(PY)" -m pip install -r requirements/base.txt -r requirements/dev.txt -c requirements/constraints.txt
	@"$(PY)" -m pip install -e . --no-deps
	@"$(PY)" -m pip check

repair-core-venv:
	@echo "Recreating repo .venv with a Python 3.11+ interpreter..."
	@rm -rf .venv
	@bootstrap_py="$(BOOTSTRAP_PY)"; \
		"$$bootstrap_py" -m venv .venv
	@"$(PY)" -m pip install -r requirements/base.txt -r requirements/dev.txt -c requirements/constraints.txt
	@"$(PY)" -m pip install -e . --no-deps
	@"$(PY)" -m pip check
	@echo "Repo .venv repaired."
	@echo "Reminder: install Depth Anything 3 into .runtime/Depth-Anything-3/.venv-da3 with ./scripts/setup/install_da3_runtime.sh"

# ML Layer Install Targets
# These support fine-grained ML capability installation per the layered strategy.
# See requirements/README.md for layer documentation.

install-ml: venv
	@echo "Error: install-ml no longer has a trusted checked-in umbrella lockfile contract."
	@echo "Error: use target-specific bootstrap profiles instead, for example:"
	@echo "Error:   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu"
	@echo "Error:   ./scripts/bootstrap/install_ml_stack.sh --profile core-mps"
	@exit 1

install-ml-core: venv
	@echo "Installing ML core layer (supported Apple Silicon baseline)..."
	@ml_lock=""; \
	py_os="$$("$(PY)" -c 'import platform; print(platform.system())')"; \
	py_arch="$$("$(PY)" -c 'import platform; print(platform.machine())')"; \
	case "$$py_arch" in \
		aarch64) py_arch="arm64" ;; \
		amd64) py_arch="x86_64" ;; \
	esac; \
	if [ "$$py_os" = "Darwin" ] && [ "$$py_arch" = "arm64" ] && [ -f requirements/ml-core-darwin-arm64.txt ]; then \
		ml_lock="requirements/ml-core-darwin-arm64.txt"; \
	fi; \
	if [ -n "$$ml_lock" ]; then \
		echo "Using $$ml_lock"; \
		"$(PY)" -m pip install -r "$$ml_lock" && \
		"$(PY)" -m pip install -e .; \
	else \
		echo "Error: no supported checked-in ML core lockfile for $$py_os/$$py_arch."; \
		echo "Error: Linux and macOS Intel ML lockfiles were retired from installable requirements."; \
		exit 1; \
	fi

install-ml-raw: venv
	@echo "Error: install-ml-raw no longer has a trusted checked-in lockfile contract."
	@echo "Error: use a trusted target-specific flow or regenerate a target-correct lockfile in the appropriate environment."
	@exit 1

install-ml-sam2: venv
	@echo "Installing ML SAM2 segmentation layer via bootstrap script..."
	@echo "SAM2 requires non-standard install semantics and is scripted-only."
	@py_os="$$("$(PY)" -c 'import platform; print(platform.system())')"; \
	py_arch="$$("$(PY)" -c 'import platform; print(platform.machine())')"; \
	case "$$py_arch" in \
		aarch64) py_arch="arm64" ;; \
		amd64) py_arch="x86_64" ;; \
	esac; \
	profile="core-cpu,sam2"; \
	if [ "$$py_os" = "Darwin" ] && [ "$$py_arch" = "arm64" ]; then \
		profile="core-mps,sam2"; \
	fi; \
	echo "Using ML SAM2 profile $$profile"; \
	./scripts/bootstrap/install_ml_stack.sh --profile "$$profile"
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
	@./scripts/setup/ensure_node_version.sh
	@cd web/secure-landing && npm test
	@cd web/secure-landing && npm run build

test-archive-gate-contract:
	@echo "Running archive gate readiness + HTTP contract tests (Gates A, B, C)..."
	@"$(PY)" -m pytest -v -k "archive_gate" tests/test_app_orchestrator_runtime.py tests/test_app_orchestrator_contract_http.py

check-portal-asset-budgets:
	@echo "Validating portal asset size budgets..."
	@"$(PY)" ./scripts/validation/check_portal_asset_budgets.py

seed-frontdoor-user:
	@echo "Seeding canonical local managed frontdoor credential fixture..."
	@cd web/secure-landing && node ./scripts/guard-runtime.mjs
	@cd web/secure-landing && node ./scripts/seed-frontdoor-user.mjs \
		--output "$${TP_FRONTDOOR_USERS_FILE:-/tmp/tp-frontdoor-users.json}" \
		--username "$${TP_FRONTDOOR_USERNAME:-smoke-admin}" \
		--password "$${TP_FRONTDOOR_PASSWORD:-correct horse battery staple}" \
		--access-email "$${TP_FRONTDOOR_ACCESS_EMAIL:-$${TP_FRONTDOOR_USERNAME:-smoke-admin}@local.invalid}" \
		--role "$${TP_FRONTDOOR_ROLE:-admin}"

run-frontdoor-local:
	@echo "Starting the canonical local managed frontdoor on localhost:3000..."
	@./scripts/setup/run_frontdoor_local.sh

validate-orchestrator-http:
	@echo "Running live orchestrator HTTP smoke validation..."
	@"$(PY)" scripts/validation/validate_orchestrator_http_smoke.py

validate-portal-lux-materials-live:
	@echo "Running live Lux Materials V3 segmentation backend smoke validation..."
	@TP_API_KEY="$${TP_API_KEY:-contract-secret}" "$(PY)" scripts/validation/validate_portal_lux_materials_live.py --api-key "$${TP_API_KEY:-contract-secret}"

validate-portal-css-layer-parity:
	@echo "Validating production portal CSS layer parity..."
	@./scripts/setup/ensure_node_version.sh
	@cd web/secure-landing && node ./scripts/build-portal-bundle.mjs --check-css
	@cd web/secure-landing && npm run check:css-layer-parity
	@"$(PY)" scripts/validation/validate_portal_css_layer_parity.py

validate-portal-browser:
	@echo "Running live portal browser smoke validation..."
	@TP_API_KEY="$${TP_API_KEY:-contract-secret}" TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT="$${TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT:-100}" "$(PY)" scripts/validation/validate_portal_browser_smoke.py --spawn-local-backend --api-key "$${TP_API_KEY:-contract-secret}"

validate-frontdoor-browser:
	@echo "Running live managed frontdoor browser smoke validation..."
	@"$(PY)" scripts/validation/validate_frontdoor_browser_smoke.py --spawn-local-backend --spawn-local-frontdoor

validate-frontdoor-deployment-gate:
	@echo "Running shared-deployment frontdoor posture gate..."
	@set -eu; \
	set -- "$(PY)" scripts/validation/check_frontdoor_deployment_gate.py \
		--environment "$${TP_FRONTDOOR_GATE_ENVIRONMENT:-}" \
		--frontdoor-url "$${TP_FRONTDOOR_GATE_FRONTDOOR_URL:-}" \
		--cf-access-team-domain "$${TP_FRONTDOOR_GATE_CF_ACCESS_TEAM_DOMAIN:-}" \
		--vercel-deployment-url "$${TP_FRONTDOOR_GATE_VERCEL_DEPLOYMENT_URL:-}"; \
	if [ -n "$${TP_FRONTDOOR_GATE_FASTAPI_PUBLIC_URL:-}" ]; then \
		set -- "$$@" --fastapi-public-url "$${TP_FRONTDOOR_GATE_FASTAPI_PUBLIC_URL}"; \
	fi; \
	if [ "$${TP_FRONTDOOR_GATE_CONFIRM_FASTAPI_NON_PUBLIC:-}" = "1" ]; then \
		set -- "$$@" --confirm-fastapi-non-public; \
	fi; \
	if [ -n "$${TP_FRONTDOOR_GATE_TIMEOUT_SECONDS:-}" ]; then \
		set -- "$$@" --timeout-seconds "$${TP_FRONTDOOR_GATE_TIMEOUT_SECONDS}"; \
	fi; \
	if [ -n "$${TP_FRONTDOOR_GATE_USER_AGENT:-}" ]; then \
		set -- "$$@" --user-agent "$${TP_FRONTDOOR_GATE_USER_AGENT}"; \
	fi; \
	"$$@"

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

# Coverage reporting targets (Phase 0 coverage infrastructure)
coverage-report:
	@echo "Running comprehensive coverage report..."
	@rm -f .coverage .coverage.* coverage.xml
	@"$(PY)" -m pytest tests/ \
		-m "not ml and not slow and not benchmark and not stress" \
		--cov=src/transformation_portal \
		--cov-branch \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-config=pyproject.toml \
		-q
	@echo ""
	@echo "✅ Coverage report generated:"
	@echo "  - Terminal: above"
	@echo "  - HTML:     htmlcov/index.html"
	@echo "  - XML:      coverage.xml"

coverage-diff:
	@echo "Running diff coverage comparison against main branch..."
	@if ! "$(PY)" -m pip show diff-cover >/dev/null 2>&1; then \
		echo "Error: diff-cover not installed. Run 'make install-core' or 'pip install diff-cover'"; \
		exit 1; \
	fi
	@if [ ! -f coverage.xml ]; then \
		echo "Error: coverage.xml not found. Run 'make coverage-report' first."; \
		exit 1; \
	fi
	@echo "Fetching origin/main for comparison..."
	@if ! git fetch origin main --quiet 2>/dev/null; then \
		echo "Warning: Could not fetch origin/main. Using local refs if available."; \
	fi
	@if ! git rev-parse --verify origin/main >/dev/null 2>&1; then \
		echo "Error: origin/main branch not available. Run 'git fetch origin main' first."; \
		exit 1; \
	fi
	@echo "Comparing against origin/main..."
	@"$(PY)" -m diff_cover.diff_cover_tool coverage.xml --compare-branch=origin/main --fail-under=85
	@echo "✅ Diff coverage check passed (≥85%)"

coverage-package:
	@echo "Generating package-level coverage baseline report..."
	@rm -f .coverage .coverage.* coverage.xml
	@"$(PY)" -m pytest tests/ \
		-m "not ml and not slow and not benchmark and not stress" \
		--cov=src/transformation_portal \
		--cov-branch \
		--cov-report=xml:coverage.xml \
		--cov-config=pyproject.toml \
		-q
	@echo ""
	@echo "=== Package-Level Coverage Baseline ==="
	@"$(PY)" -m coverage report --include="src/transformation_portal/*" --skip-covered 2>/dev/null || \
		"$(PY)" -m coverage report --include="src/transformation_portal/*" 2>/dev/null || \
		echo "  (no coverage data)"
	@echo ""
	@echo "--- Priority Package Coverage ---"
	@for pkg in events storage runtime lux_depth_v3 hardening; do \
		result=$$("$(PY)" -m coverage report --include="src/transformation_portal/$$pkg/*" 2>/dev/null | tail -1 | awk '{print $$NF}' || true); \
		if [ -z "$$result" ] || [ "$$result" = "TOTAL" ]; then result="0%"; fi; \
		printf "  %-20s %s\n" "$$pkg/:" "$$result"; \
	done
	@echo ""
	@echo "--- Root Module Coverage ---"
	@app_result=$$("$(PY)" -m coverage report --include="app.py" 2>/dev/null | tail -1 | awk '{print $$NF}' || true); \
	if [ -z "$$app_result" ] || [ "$$app_result" = "TOTAL" ]; then app_result="0%"; fi; \
	printf "  %-20s %s\n" "app.py:" "$$app_result"
	@echo ""
	@echo "✅ Package baseline report complete. Use this as a ratchet reference."

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

ci: lint check-json-serialization check-python-headers check-yaml-governance check-piptools-cache check-requirements-lock-contract check-ci-sync check-portal-asset-budgets test-fast test-orchestrator-contract test-frontdoor-contract
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
	@test -n "$(PRE_COMMIT_BIN)" || { echo "pre-commit is not installed in .venv; run 'make install-core' first"; exit 1; }
	@"$(PRE_COMMIT_BIN)" run --all-files --show-diff-on-failure

# Install git hooks
install-hooks:
	@echo "Installing git pre-commit hook..."
	@test -n "$(PRE_COMMIT_BIN)" || { echo "pre-commit is not installed in .venv; run 'make install-core' first"; exit 1; }
	@"$(PRE_COMMIT_BIN)" install -f
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

check-doc-heading-links:
	@"$(PY)" scripts/validation/check_doc_heading_links.py

# Validate CI configuration
validate-ci:
	@echo "Validating GitHub Actions workflows..."
	@"$(PY)" scripts/validate_ci_config.py
	@echo "Validating workflow concurrency contract..."
	@"$(PY)" scripts/validation/check_workflow_concurrency_contract.py
	@echo "Validating gitleaks workflow contract..."
	@"$(PY)" scripts/validation/check_gitleaks_workflow_contract.py
	@echo "Validating dependency-update workflow contract..."
	@"$(PY)" scripts/validation/check_dependency_update_workflow.py
	@echo "Validating Dependabot config contract..."
	@"$(PY)" scripts/validation/check_dependabot_config.py

check-json-serialization:
	@echo "Checking JSON serialization guardrails..."
	@"$(PY)" scripts/validation/check_raw_json_usage.py

check-python-headers:
	@echo "Checking Python header declarations..."
	@"$(PY)" scripts/validation/check_python_headers.py

check-yaml-governance:
	@echo "Checking YAML governance boundary..."
	@"$(PY)" scripts/validation/check_yaml_governance_boundary.py

check-piptools-cache:
	@echo "Checking pip-tools cache guardrails..."
	@"$(PY)" scripts/validation/check_piptools_cache_tracked.py

check-requirements-lock-contract:
	@echo "Checking requirements lock contract..."
	@"$(PY)" scripts/validation/check_requirements_lock_contract.py

check:
	@echo "Checking generic layered requirements in requirements/..."
	@$(MAKE) -C requirements check LOCK_PYTHON_VERSION=3.11

check-test-markers:
	@echo "Auditing test marker coverage (ADR-044)..."
	@"$(PY)" scripts/validation/check_test_markers.py --audit

check-ci-sync:
	@echo "Checking CI dependency file sync..."
	@"$(PY)" scripts/validation/check_ci_dep_sync.py

check-todo-governance:
	@echo "Checking TODO governance compliance..."
	@"$(PY)" scripts/validation/scan_todo_inventory.py --check-governance

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

compile-ml-darwin-arm64:
	@$(MAKE) -C requirements compile-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11

update-ml-darwin-arm64:
	@$(MAKE) -C requirements update-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11

check-ml-darwin-arm64:
	@$(MAKE) -C requirements check-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11

compile-ml-linux-x86_64:
	@$(MAKE) -C requirements compile-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11

update-ml-linux-x86_64:
	@$(MAKE) -C requirements update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11

check-ml-linux-x86_64:
	@$(MAKE) -C requirements check-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11

compile-ml-darwin-x86_64:
	@$(MAKE) -C requirements compile-ml-darwin-x86_64 LOCK_PYTHON_VERSION=3.11

update-ml-darwin-x86_64:
	@$(MAKE) -C requirements update-ml-darwin-x86_64 LOCK_PYTHON_VERSION=3.11

check-ml-darwin-x86_64:
	@$(MAKE) -C requirements check-ml-darwin-x86_64 LOCK_PYTHON_VERSION=3.11

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

# --- Environment and Validation ---

check-environment:
	@echo "Running pre-flight environment validation..."
	@"$(PY)" scripts/validation/check_local_environment.py

check-worktree:
	@echo "Checking if git worktree is clean..."
	@./scripts/validation/check_worktree_clean.sh

validate-full:
	@echo "Running full validation suite..."
	@./scripts/validation/run_full_validation_suite.sh

validate-quick:
	@echo "Running quick validation (skip browser smokes)..."
	@./scripts/validation/run_full_validation_suite.sh --quick

# --- Cleanup ---

clean-frontdoor:
	@echo "Cleaning frontdoor build artifacts..."
	@rm -rf web/secure-landing/.next web/secure-landing/.next-build-verify web/secure-landing/.next-smoke-* web/secure-landing/.next-codex-* 2>/dev/null || true
	@echo "✓ Frontdoor cleanup complete"
	@echo "Note: node_modules preserved. Run 'rm -rf web/secure-landing/node_modules' to remove."

clean-all: clean clean-frontdoor
	@echo "✓ Full cleanup complete"
