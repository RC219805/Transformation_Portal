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

.PHONY: help test-fast test-novideo test-full test-structure test-utils venv setup clean \
        lint ci ci-full pre-commit install-hooks quality-check fix-quality validate-ci organize-docs \
        lock lock-prod lock-ci lock-dev verify-security

help:
	@echo "Targets:"
	@echo "  setup              Install package in editable mode (pip install -e .)"
	@echo "  test-fast          Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo       Run all tests excluding video suite via -k filter"
	@echo "  test-full          Run entire test suite (parallel if xdist present)"
	@echo "  test-structure     Run codebase structure validation tests"
	@echo "  test-utils         Run tests for performance and error handling utilities"
	@echo "  venv               Create local .venv if missing"
	@echo "  clean              Remove Python cache files and build artifacts"
	@echo ""
	@echo "Security:"
	@echo "  verify-security    Verify no vulnerable basicsr imports (CVE-2024-27763)"
	@echo ""
	@echo "Quality & CI:"
	@echo "  lint               Run linting (flake8 + pylint)"
	@echo "  ci                 Run local CI checks (lint + test-fast)"
	@echo "  ci-full            Run comprehensive CI simulation (all checks)"
	@echo "  pre-commit         Run pre-commit checks manually"
	@echo "  install-hooks      Install git pre-commit hook"
	@echo "  quality-check      Run all quality checks (lint + structure + tests)"
	@echo "  fix-quality        Auto-fix common quality issues"
	@echo "  validate-ci        Validate GitHub Actions workflow configs"
	@echo "  organize-docs      Organize markdown files to docs/ subdirectories"
	@echo ""
	@echo "Dependency locking:"
	@echo "  lock               Regenerate all requirements lockfiles (prod/ci/dev)"
	@echo "  lock-prod          Regenerate requirements.lock.txt"
	@echo "  lock-ci            Regenerate requirements-ci.lock.txt"
	@echo "  lock-dev           Regenerate requirements-dev.lock.txt"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
	fi

setup: venv
	@echo "Installing package in editable mode..."
	@"$(PY)" -m pip install -c requirements/constraints.txt -e .

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

test-structure:
	@echo "Running codebase structure validation tests..."
	@"$(PY)" -m pytest -v tests/test_codebase_structure.py

test-utils:
	@echo "Running utility tests..."
	@"$(PY)" -m pytest -v tests/test_performance_utils.py tests/test_error_handling.py

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
	@$(PY) -m pip install -q -c requirements/constraints.txt -e . || echo "Warning: Package installation failed"
	@echo "Running flake8 critical checks..."
	@$(PY) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=deprecated,scripts,examples || true
	@echo "Running pylint (non-blocking)..."
	@$(PY) -m pylint $(shell git ls-files '*.py' | grep -v -e '/deprecated/' -e 'src/transformation_portal/' -e 'src/luxury_tiff_batch_processor/' -e 'scripts/' -e 'examples/' || echo '') || true

ci: lint test-fast
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
	@./scripts/pre_commit_hook.sh

# Install git hooks
install-hooks:
	@echo "Installing git pre-commit hook..."
	@cp scripts/pre_commit_hook.sh .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Pre-commit hook installed at .git/hooks/pre-commit"

# Quality check (all validations)
quality-check: lint validate-ci
	@echo "Running documentation structure check..."
	@MD_COUNT=$$(find . -maxdepth 1 -name "*.md" -type f | wc -l | tr -d ' '); \
	if [ $$MD_COUNT -gt 10 ]; then \
		echo "⚠ Too many markdown files in root: $$MD_COUNT (max: 10)"; \
		echo "💡 Run 'make organize-docs' to fix"; \
	else \
		echo "✓ Markdown file count OK ($$MD_COUNT/10)"; \
	fi
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

# Organize documentation
organize-docs:
	@echo "Organizing documentation files..."
	@./scripts/organize_docs.sh

# Organize documentation (dry-run)
check-docs:
	@echo "Checking documentation organization..."
	@./scripts/organize_docs.sh --dry-run

# Verify no vulnerable basicsr imports (CVE-2024-27763)
verify-security:
	@echo "Verifying security: basicsr CVE-2024-27763 mitigation..."
	@"$(PY)" scripts/utilities/verify_no_basicsr_imports.py --check-pkg

# --- Dependency locking (pip-tools) ---

lock: lock-prod lock-ci lock-dev
	@echo "✓ Lockfiles updated (prod/ci/dev)"

lock-prod:
	@echo "Locking production requirements -> requirements.lock.txt"
	@pip-compile --generate-hashes \
		-c requirements/constraints.txt \
		-o requirements.lock.txt \
		requirements.txt

lock-ci:
	@echo "Locking CI requirements -> requirements-ci.lock.txt"
	@pip-compile --generate-hashes \
		-c requirements/constraints.txt \
		-o requirements-ci.lock.txt \
		requirements-ci.txt

lock-dev:
	@echo "Locking dev requirements -> requirements-dev.lock.txt"
	@pip-compile --generate-hashes \
		-c requirements/constraints.txt \
		-o requirements-dev.lock.txt \
		requirements-dev.txt
