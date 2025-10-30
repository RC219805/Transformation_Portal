SHELL := /bin/sh

# Resolve a Python interpreter: prefer local venv, otherwise fall back to python3
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)

# Common subsets (fast tests avoid heavy/optional paths)
FAST_TESTS := \
	tests/test_material_response.py \
	tests/test_adjustments.py \
	tests/test_geometry.py \
	tests/test_io.py \
	tests/test_pipeline.py \
	tests/test_presets.py

.PHONY: help test-fast test-novideo test-full venv

help:
	@echo "Targets:"
	@echo "  test-fast     Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo  Run all tests excluding video suite via -k filter"
	@echo "  test-full     Run entire test suite (parallel if xdist present)"
	@echo "  venv          Create local .venv if missing"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
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

# --- Additional developer + CI helpers ---

lint:
	@echo "Running flake8 critical checks..."
	@$(PY) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo "Running pylint (non-blocking)..."
	@$(PY) -m pylint $(shell git ls-files '*.py' | grep -v -e '/deprecated/' -e 'src/transformation_portal/' -e 'scripts/' || echo '') || true

ci: lint test-fast
	@echo "✅ Local CI checks completed successfully."
