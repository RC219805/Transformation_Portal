# Makefile for compiling layered requirements with pip-tools
# This ensures consistent dependency resolution across all layers
#
# LAYERED ML DEPENDENCY STRATEGY (ADR-032)
# ========================================
# ML dependencies are split into capability-specific layers for:
# - Platform-safe compilation (CPU-only baseline, platform-specific extras)
# - Deterministic installs (each layer has explicit contracts)
# - Capability-gated promotion (optional features can be added incrementally)
#
# PLATFORM MATRIX
# ---------------
# Axis       Values                    Detection
# OS         Darwin / Linux            platform_system
# ISA        arm64 / x86_64            platform_machine
# Accel      cpu / mps / cuda          explicit profile (NEVER inferred)
#
# PLATFORM-SPECIFIC LOCKFILES (pip-compile cannot resolve multi-platform .in)
# ---------------------------------------------------------------------------
# Platform          Lockfile                 Torch Version
# darwin-*          ml-core-darwin.txt       2.2.2
# linux-*           ml-core-linux.txt        2.2.2
#
# Checked-In ML Lockfiles (compiled via this Makefile):
#   ml-core-darwin.txt  macOS ML baseline (torch 2.2.2)
#   ml-core-linux.txt   Linux ML baseline (torch 2.2.2)
#
# Non-core optional ML layers are not part of the checked-in lockfile contract.
# Host-generated outputs for ml-core.txt, ml-cpu.txt, ml-mps.txt, ml-cuda.txt,
# ml-raw.txt, ml-coreml.txt, ml-research.txt, and ml.txt must not be checked in.
#
# Scripted-Only Layers (NOT compiled here - use bootstrap script):
#   ml-sam2          SAM2 segmentation - requires non-standard install semantics
#                    Install via: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,sam2
#
# See requirements/README.md for full documentation.

SHELL := /bin/bash
LOCK_PYTHON_VERSION ?= 3.11
PIP_TOOLS_CACHE_DIR ?= $(CURDIR)/.pip-tools-cache
PIP_COMPILE_BASE := pip-compile -q --resolver=backtracking
PIP_COMPILE_PY_FLAG := $(shell pip-compile --help 2>/dev/null | grep -q -- '--python-version' && echo "--python-version=$(LOCK_PYTHON_VERSION)" || true)
PIP_COMPILE := mkdir -p "$(PIP_TOOLS_CACHE_DIR)" && PIP_TOOLS_CACHE_DIR="$(PIP_TOOLS_CACHE_DIR)" $(PIP_COMPILE_BASE) $(PIP_COMPILE_PY_FLAG)

# PyTorch CPU index for CI/cross-platform lockfiles
PYTORCH_CPU_INDEX := --extra-index-url https://download.pytorch.org/whl/cpu

# Default target: compile all requirements
.PHONY: all compile compile-all compile-ml-layers update help clean check \
	ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt

help:
	@echo "Targets:"
	@echo "  compile           Compile all pinned requirements from .in files"
	@echo "  compile-all       Same as compile (for compatibility)"
	@echo "  compile-ml-layers Compile only ML layer lockfiles"
	@echo "  compile-accel     Compile acceleration layer lockfiles (ml-cpu, ml-mps, ml-cuda)"
	@echo "  update            Update all dependencies to latest versions"
	@echo "  check             Verify that .txt files are up-to-date with .in files"
	@echo "  clean             Remove all compiled .txt files"
	@echo ""
	@echo "Individual compilation targets:"
	@echo "  all.txt              Compile all.txt (checked-in contract aggregate)"
	@echo "  base.txt             Compile base.txt (constrained by all.txt)"
	@echo ""
	@echo "Platform-specific ML Core targets:"
	@echo "  ml-core-darwin.txt   macOS ML baseline (torch 2.2.2)"
	@echo "  ml-core-linux.txt    Linux ML baseline (torch 2.2.2)"
	@echo ""
	@echo "Checked-in ML lockfiles:"
	@echo "  ml-core-darwin.txt   macOS ML baseline (torch 2.2.2)"
	@echo "  ml-core-linux.txt    Linux ML baseline (torch 2.2.2)"
	@echo ""
	@echo "Non-core optional ML layers are not part of the checked-in lock contract:"
	@echo "  ml-core.txt, ml-cpu.txt, ml-mps.txt, ml-cuda.txt,"
	@echo "  ml-raw.txt, ml-coreml.txt, ml-research.txt, ml.txt"
	@echo ""
	@echo "Scripted-only layers (NOT compiled here):"
	@echo "  ml-sam2              SAM2 segmentation - use bootstrap script"
	@echo ""
	@echo "Other targets:"
	@echo "  dev.txt              Compile dev.txt (constrained by all.txt)"
	@echo "  ci.txt               Compile ci.txt (constrained by all.txt)"
	@echo "  security.txt         Compile security.txt (CI security scanning tools)"
	@echo "  tools-archive.txt    Compile tools-archive.txt (constrained by all.txt)"

# Compile all requirements in correct order
# Note: SAM2 is scripted-only and NOT compiled here - use bootstrap script
compile: all.txt base.txt ml-core-darwin.txt ml-core-linux.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ All requirements compiled successfully"

compile-all: compile

# Compile only ML layers (useful for ML-focused updates)
# Note: SAM2 is scripted-only and NOT compiled here
compile-ml-layers: all.txt base.txt ml-core-darwin.txt ml-core-linux.txt
	@echo "✓ ML layer requirements compiled successfully"

# Compile only acceleration layers (useful for profile updates)
compile-accel: all.txt base.txt ml-core-darwin.txt ml-core-linux.txt
	@echo "✓ Acceleration layer requirements compiled successfully"

# Compile all.txt first for global resolution
# Note: all.txt intentionally excludes optional ML layers from the checked-in contract.
all.txt: all.in base.in dev.in ci.in
	@echo "Compiling all.txt (checked-in contract aggregate)..."
	@$(PIP_COMPILE) all.in -o all.txt

# Compile individual files constrained by all.txt
base.txt: base.in all.txt
	@echo "Compiling base.txt..."
	@$(PIP_COMPILE) -c all.txt base.in -o base.txt

# ML Layer lockfiles - checked-in contract is limited to platform-specific core.
# Platform-specific core lockfiles are constrained by base.txt so host-resolved
# ML pins in all.txt cannot leak into darwin/linux target outputs.
# NOTE: Non-core optional ML layers are not compiled here.
# NOTE: SAM2 is scripted-only and NOT compiled here - use bootstrap script

# Platform-specific ml-core lockfiles (pip-compile cannot resolve multi-platform .in)
ml-core-darwin.txt: ml-core-darwin.in base.txt
	@echo "Compiling ml-core-darwin.txt (macOS, torch 2.2.2)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin.in -o ml-core-darwin.txt

ml-core-linux.txt: ml-core-linux.in base.txt
	@echo "Compiling ml-core-linux.txt (Linux, torch 2.2.2)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-linux.in -o ml-core-linux.txt

ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt:
	@echo "ERROR: $@ is not part of the checked-in lockfile contract."
	@echo "ERROR: Use target-specific bootstrap/install flows instead of checking in host-generated optional ML layer locks."
	@exit 1

dev.txt: dev.in all.txt
	@echo "Compiling dev.txt..."
	@$(PIP_COMPILE) -c all.txt dev.in -o dev.txt

ci.txt: ci.in all.txt
	@echo "Compiling ci.txt..."
	@$(PIP_COMPILE) -c all.txt ci.in -o ci.txt

# Security scanning tools (standalone layer for CI security job)
# NOT constrained by all.txt to ensure latest security tool versions
security.txt: security.in
	@echo "Compiling security.txt..."
	@$(PIP_COMPILE) security.in -o security.txt

tools-archive.txt: tools-archive.in all.txt
	@echo "Compiling tools-archive.txt..."
	@$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt

# Update all dependencies to latest allowed versions
# Note: SAM2 is scripted-only and NOT updated here
update:
	@echo "Updating all dependencies to latest versions..."
	@$(PIP_COMPILE) --upgrade all.in -o all.txt
	@$(PIP_COMPILE) --upgrade -c all.txt base.in -o base.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin.in -o ml-core-darwin.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c base.txt ml-core-linux.in -o ml-core-linux.txt
	@$(PIP_COMPILE) --upgrade -c all.txt dev.in -o dev.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ci.in -o ci.txt
	@$(PIP_COMPILE) --upgrade security.in -o security.txt
	@$(PIP_COMPILE) --upgrade -c all.txt tools-archive.in -o tools-archive.txt
	@echo "✓ All requirements updated successfully"

# Check if .txt files are up-to-date
# Note: SAM2 is scripted-only and NOT checked here
check:
	@echo "Checking if requirements are up-to-date..."
	@set -euo pipefail; \
	repo_dir="$$(pwd)"; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp all.in base.in ml.in ml-core.in ml-core-darwin.in ml-core-linux.in ml-cpu.in ml-mps.in ml-cuda.in ml-raw.in ml-coreml.in ml-research.in dev.in ci.in security.in tools-archive.in "$$tmp_dir"/; \
	for f in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			cp "$$repo_dir/$$f" "$$tmp_dir"/; \
		else \
			echo "ERROR: Required lockfile missing: $$f"; \
			exit 1; \
		fi; \
	done; \
	for f in ml-core-darwin.txt ml-core-linux.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			cp "$$repo_dir/$$f" "$$tmp_dir"/; \
		else \
			echo "ERROR: Required lockfile missing: $$f"; \
			exit 1; \
		fi; \
	done; \
	cd "$$tmp_dir"; \
	$(PIP_COMPILE) all.in -o all.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt base.in -o base.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin.in -o ml-core-darwin.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-linux.in -o ml-core-linux.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt dev.in -o dev.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ci.in -o ci.txt >/dev/null; \
	$(PIP_COMPILE) security.in -o security.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt >/dev/null; \
	normalize_req() { \
		awk 'BEGIN{start=0} { if (!start) { if ($$0 ~ /^#/ || $$0 ~ /^[[:space:]]*$$/) next; start=1 } print }' "$$1"; \
	}; \
	for f in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
		if ! diff -u <(normalize_req "$$tmp_dir/$$f") <(normalize_req "$$repo_dir/$$f") >/dev/null; then \
			echo "❌ $$f is out of date. Run 'make compile' to update."; \
			exit 1; \
		fi; \
	done; \
	for f in ml-core-darwin.txt ml-core-linux.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			if ! diff -u <(normalize_req "$$tmp_dir/$$f") <(normalize_req "$$repo_dir/$$f") >/dev/null; then \
				echo "❌ $$f is out of date. Run 'make compile' to update."; \
				exit 1; \
			fi; \
		fi; \
	done; \
	echo "✓ All requirements are up-to-date"

# Clean compiled files
# Note: ml-sam2.txt removed if present (historical artifact)
clean:
	@echo "Removing compiled .txt files..."
	@rm -f all.txt base.txt ml-core-darwin.txt ml-core-linux.txt ml-sam2.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ Cleanup complete"
