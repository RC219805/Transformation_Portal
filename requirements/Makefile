# Makefile for compiling layered requirements with pip-tools
# This ensures consistent dependency resolution across all layers
#
# Security: Uses constraints.txt to exclude packages with known vulnerabilities
# - See constraints.txt for details on excluded packages (e.g., basicsr CVE-2024-27763)
#
# IMPORTANT: Requirements MUST be compiled with Python 3.10 (minimum supported version)
# - See COMPILATION_NOTES.md for detailed explanation
# - Run 'make check-python' before compiling to verify correct Python version

SHELL := /bin/bash
PIP_COMPILE := pip-compile -q --resolver=backtracking -c constraints.txt

# Python version check
PYTHON_VERSION := $(shell python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_PYTHON := 3.10

# Default target: compile all requirements
.PHONY: all compile compile-all update help clean check check-python

help:
	@echo "Targets:"
	@echo "  check-python Check if using correct Python version (3.10 required)"
	@echo "  compile      Compile all pinned requirements from .in files"
	@echo "  compile-all  Same as compile (for compatibility)"
	@echo "  update       Update all dependencies to latest versions"
	@echo "  check        Verify that .txt files are up-to-date with .in files"
	@echo "  clean        Remove all compiled .txt files"
	@echo ""
	@echo "Individual compilation targets:"
	@echo "  all.txt      Compile all.txt (global resolution)"
	@echo "  base.txt     Compile base.txt (constrained by all.txt)"
	@echo "  ml.txt       Compile ml.txt (constrained by all.txt)"
	@echo "  dev.txt      Compile dev.txt (constrained by all.txt)"
	@echo "  ci.txt       Compile ci.txt (constrained by all.txt)"
	@echo ""
	@echo "⚠️  IMPORTANT: Requirements MUST be compiled with Python $(REQUIRED_PYTHON)"
	@echo "   Run 'make check-python' before compiling to verify."
	@echo "   See COMPILATION_NOTES.md for details."

# Check Python version before compilation
check-python:
	@echo "Current Python version: $(PYTHON_VERSION)"
	@echo "Required Python version: $(REQUIRED_PYTHON)"
	@if [ "$(PYTHON_VERSION)" != "$(REQUIRED_PYTHON)" ]; then \
		echo ""; \
		echo "❌ ERROR: Wrong Python version!"; \
		echo "   Current: $(PYTHON_VERSION)"; \
		echo "   Required: $(REQUIRED_PYTHON)"; \
		echo ""; \
		echo "Requirements must be compiled with Python $(REQUIRED_PYTHON) to ensure"; \
		echo "compatibility with all supported Python versions (3.10+)."; \
		echo ""; \
		echo "To fix this:"; \
		echo "  1. Install Python $(REQUIRED_PYTHON): pyenv install $(REQUIRED_PYTHON).15"; \
		echo "  2. Activate it: pyenv local $(REQUIRED_PYTHON).15"; \
		echo "  3. Verify: python --version"; \
		echo "  4. Then run: make compile"; \
		echo ""; \
		echo "See COMPILATION_NOTES.md for details."; \
		exit 1; \
	else \
		echo "✓ Python version is correct"; \
	fi

# Compile all requirements in correct order (with Python version check)
compile: check-python all.txt base.txt ml.txt dev.txt ci.txt
	@echo "✓ All requirements compiled successfully with Python $(PYTHON_VERSION)"

compile-all: compile

# Compile all.txt first for global resolution
all.txt: all.in base.in ml.in dev.in ci.in
	@echo "Compiling all.txt (global resolution)..."
	@$(PIP_COMPILE) all.in -o all.txt

# Compile individual files constrained by all.txt
base.txt: base.in all.txt
	@echo "Compiling base.txt..."
	@$(PIP_COMPILE) -c all.txt base.in -o base.txt

ml.txt: ml.in all.txt
	@echo "Compiling ml.txt..."
	@$(PIP_COMPILE) -c all.txt ml.in -o ml.txt

dev.txt: dev.in all.txt
	@echo "Compiling dev.txt..."
	@$(PIP_COMPILE) -c all.txt dev.in -o dev.txt

ci.txt: ci.in all.txt
	@echo "Compiling ci.txt..."
	@$(PIP_COMPILE) -c all.txt ci.in -o ci.txt

# Update all dependencies to latest allowed versions
update:
	@echo "Updating all dependencies to latest versions..."
	@$(PIP_COMPILE) --upgrade all.in -o all.txt
	@$(PIP_COMPILE) --upgrade -c all.txt base.in -o base.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ml.in -o ml.txt
	@$(PIP_COMPILE) --upgrade -c all.txt dev.in -o dev.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ci.in -o ci.txt
	@echo "✓ All requirements updated successfully"

# Check if .txt files are up-to-date
check:
	@echo "Checking if requirements are up-to-date..."
	@$(PIP_COMPILE) --dry-run --quiet all.in -o all.txt 2>&1 | grep -q "No changes" || \
		(echo "❌ all.txt is out of date. Run 'make compile' to update." && exit 1)
	@$(PIP_COMPILE) --dry-run --quiet -c all.txt base.in -o base.txt 2>&1 | grep -q "No changes" || \
		(echo "❌ base.txt is out of date. Run 'make compile' to update." && exit 1)
	@$(PIP_COMPILE) --dry-run --quiet -c all.txt ml.in -o ml.txt 2>&1 | grep -q "No changes" || \
		(echo "❌ ml.txt is out of date. Run 'make compile' to update." && exit 1)
	@$(PIP_COMPILE) --dry-run --quiet -c all.txt dev.in -o dev.txt 2>&1 | grep -q "No changes" || \
		(echo "❌ dev.txt is out of date. Run 'make compile' to update." && exit 1)
	@$(PIP_COMPILE) --dry-run --quiet -c all.txt ci.in -o ci.txt 2>&1 | grep -q "No changes" || \
		(echo "❌ ci.txt is out of date. Run 'make compile' to update." && exit 1)
	@echo "✓ All requirements are up-to-date"

# Clean compiled files
clean:
	@echo "Removing compiled .txt files..."
	@rm -f all.txt base.txt ml.txt dev.txt ci.txt
	@echo "✓ Cleanup complete"
