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

.PHONY: help test-fast test-novideo test-full test-integration test-structure test-utils test-orchestrator-contract venv setup clean \
        lint ci ci-full pre-commit install-hooks quality-check fix-quality validate-ci organize-docs check-json-serialization check-piptools-cache \
        lock lock-prod lock-ci lock-dev install-core install-ml docs docs-clean

help:
	@echo "Targets:"
	@echo "  setup              Install package in editable mode (pip install -e .)"
	@echo "  install-core       Install core dependencies with constraints"
	@echo "  install-ml         Install ML tier dependencies with constraints"
	@echo "  test-fast          Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo       Run all tests excluding video suite via -k filter"
	@echo "  test-full          Run entire test suite (parallel if xdist present)"
	@echo "  test-orchestrator-contract  Run route-level portal orchestrator contract suite"
	@echo "  test-integration   Run integration tests (requires HF_TOKEN)"
	@echo "  test-structure     Run codebase structure validation tests"
	@echo "  test-utils         Run tests for performance and error handling utilities"
	@echo "  venv               Create local .venv if missing"
	@echo "  clean              Remove Python cache files and build artifacts"
	@echo ""
	@echo "Quality & CI:"
	@echo "  lint               Run linting (flake8 + pylint)"
	@echo "  ci                 Run local CI checks (lint + test-fast)"
	@echo "  ci-full            Run comprehensive CI simulation (all checks)"
	@echo "  pre-commit         Run pre-commit checks manually"
	@echo "  install-hooks      Install git pre-commit hook"
	@echo "  quality-check      Run all quality checks (lint + structure + tests)"
	@echo "  check-json-serialization  Fail on raw json.dump/json.dumps outside approved modules"
	@echo "  check-piptools-cache  Fail if requirements/.pip-tools-cache is tracked in git"
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

install-ml: venv
	@echo "Installing ML tier dependencies with constraints..."
	@if [ -f requirements/constraints.txt ]; then \
		"$(PY)" -m pip install -e ".[ml]" -c requirements/constraints.txt; \
	else \
		echo "Warning: requirements/constraints.txt not found, installing without constraints"; \
		"$(PY)" -m pip install -e ".[ml]"; \
	fi

test-fast:
	@"$(PY)" -m pytest -q $(FAST_TESTS)

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
	@"$(PY)" -m pytest -q tests/test_app_orchestrator_runtime.py tests/test_app_orchestrator_contract_http.py

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
	@echo "Installing package for linting..."
	@$(PY) -m pip install -q -e . || echo "Warning: Package installation failed"
	@echo "Running advisory lint via shared policy..."
	@PYTHON_BIN="$(PY)" ./scripts/lint_runner.sh advisory

ci: lint check-json-serialization check-piptools-cache test-fast test-orchestrator-contract
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
	@pre-commit run --all-files

# Install git hooks
install-hooks:
	@echo "Installing git pre-commit hook..."
	@pre-commit install
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

# Validate CI configuration
validate-ci:
	@echo "Validating GitHub Actions workflows..."
	@"$(PY)" scripts/validate_ci_config.py

check-json-serialization:
	@echo "Checking JSON serialization guardrails..."
	@"$(PY)" scripts/validation/check_raw_json_usage.py

check-piptools-cache:
	@echo "Checking pip-tools cache guardrails..."
	@"$(PY)" scripts/validation/check_piptools_cache_tracked.py

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
