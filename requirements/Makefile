# Makefile for compiling layered requirements with pip-tools
# This ensures consistent dependency resolution across all layers

SHELL := /bin/bash
PIP_COMPILE := pip-compile -q --resolver=backtracking

# Default target: compile all requirements
.PHONY: all compile compile-all update help clean check

help:
	@echo "Targets:"
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

# Compile all requirements in correct order
compile: all.txt base.txt ml.txt dev.txt ci.txt
	@echo "✓ All requirements compiled successfully"

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
