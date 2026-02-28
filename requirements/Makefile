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
	@echo "  tools-archive.txt Compile tools-archive.txt (constrained by all.txt)"

# Compile all requirements in correct order
compile: all.txt base.txt ml.txt dev.txt ci.txt tools-archive.txt
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

tools-archive.txt: tools-archive.in all.txt
	@echo "Compiling tools-archive.txt..."
	@$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt

# Update all dependencies to latest allowed versions
update:
	@echo "Updating all dependencies to latest versions..."
	@$(PIP_COMPILE) --upgrade all.in -o all.txt
	@$(PIP_COMPILE) --upgrade -c all.txt base.in -o base.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ml.in -o ml.txt
	@$(PIP_COMPILE) --upgrade -c all.txt dev.in -o dev.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ci.in -o ci.txt
	@$(PIP_COMPILE) --upgrade -c all.txt tools-archive.in -o tools-archive.txt
	@echo "✓ All requirements updated successfully"

# Check if .txt files are up-to-date
check:
	@echo "Checking if requirements are up-to-date..."
	@set -euo pipefail; \
	repo_dir="$$(pwd)"; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp all.in base.in ml.in dev.in ci.in tools-archive.in "$$tmp_dir"/; \
	cp all.txt base.txt ml.txt dev.txt ci.txt tools-archive.txt "$$tmp_dir"/; \
	cd "$$tmp_dir"; \
	$(PIP_COMPILE) all.in -o all.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt base.in -o base.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ml.in -o ml.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt dev.in -o dev.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ci.in -o ci.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt >/dev/null; \
	normalize_req() { \
		awk 'BEGIN{start=0} { if (!start) { if ($$0 ~ /^#/ || $$0 ~ /^[[:space:]]*$$/) next; start=1 } print }' "$$1"; \
	}; \
	for f in all.txt base.txt ml.txt dev.txt ci.txt tools-archive.txt; do \
		if ! diff -u <(normalize_req "$$tmp_dir/$$f") <(normalize_req "$$repo_dir/$$f") >/dev/null; then \
			echo "❌ $$f is out of date. Run 'make compile' to update."; \
			exit 1; \
		fi; \
	done; \
	echo "✓ All requirements are up-to-date"

# Clean compiled files
clean:
	@echo "Removing compiled .txt files..."
	@rm -f all.txt base.txt ml.txt dev.txt ci.txt tools-archive.txt
	@echo "✓ Cleanup complete"
