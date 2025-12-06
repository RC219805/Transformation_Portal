# Makefile for lux_depth_v2

.PHONY: help test test-fast test-unit test-integration test-cov clean docs docs-serve install install-dev lint format

help:
	@echo "Lux Depth V2 - Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-fast      - Run fast tests only (skip slow/gpu)"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build HTML documentation"
	@echo "  make docs-serve     - Serve documentation locally"
	@echo "  make docs-clean     - Clean documentation build"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code with black"
	@echo "  make format-check   - Check formatting without changes"
	@echo ""
	@echo "Installation:"
	@echo "  make install        - Install package"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make clean          - Clean build artifacts"

# Testing
test:
	pytest tests/

test-fast:
	pytest tests/ -m "not slow and not gpu"

test-unit:
	pytest tests/ -m "unit or not integration"

test-integration:
	pytest tests/ -m "integration"

test-cov:
	pytest tests/ --cov=lux_depth_v2 --cov-report=html --cov-report=term-missing

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	rm -rf docs/_build

# Code quality
lint:
	@echo "Running flake8..."
	flake8 lux_depth_v2 --max-line-length=127 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy lux_depth_v2 --ignore-missing-imports

format:
	black lux_depth_v2 tests examples --line-length=127

format-check:
	black lux_depth_v2 tests examples --line-length=127 --check

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Cleanup
clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage
