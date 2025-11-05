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

.PHONY: help test-fast test-novideo test-full test-structure test-utils venv setup clean

help:
	@echo "Targets:"
	@echo "  setup           Install package in editable mode (pip install -e .)"
	@echo "  test-fast       Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo    Run all tests excluding video suite via -k filter"
	@echo "  test-full       Run entire test suite (parallel if xdist present)"
	@echo "  test-structure  Run codebase structure validation tests"
	@echo "  test-utils      Run tests for performance and error handling utilities"
	@echo "  venv            Create local .venv if missing"
	@echo "  clean           Remove Python cache files and build artifacts"
	@echo "  lint            Run linting (flake8 + pylint)"
	@echo "  ci              Run local CI checks (lint + test-fast)"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
	fi

setup: venv
	@echo "Installing package in editable mode..."
	@"$(PY)" -m pip install -e .

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
	@echo "Running flake8 critical checks..."
	@$(PY) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=deprecated,scripts,examples || true
	@echo "Running pylint (non-blocking)..."
	@$(PY) -m pylint $(shell git ls-files '*.py' | grep -v -e '/deprecated/' -e 'src/transformation_portal/' -e 'scripts/' -e 'examples/' || echo '') || true

ci: lint test-fast
	@echo "✅ Local CI checks completed successfully."
