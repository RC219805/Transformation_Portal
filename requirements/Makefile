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
# Platform          Lockfile                 Baseline
# darwin-x86_64     ml-core-darwin-x86_64.txt  torch 2.2.2 + numpy<2 + transformers<5
# darwin-arm64      ml-core-darwin-arm64.txt   Apple Silicon baseline + CoreML tooling
# linux-*           ml-core-linux.txt          torch 2.2.2 + transformers<5
#
# Checked-In ML Lockfiles (compiled via this Makefile):
#   ml-core-darwin-x86_64.txt  macOS Intel conservative ML baseline
#   ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline
#   ml-core-linux.txt          Linux ML baseline (torch 2.2.2 + transformers<5)
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
HASH_PILOT_OUT_DIR ?= $(CURDIR)/.hash-pilot
PIP_COMPILE_BASE := pip-compile -q --resolver=backtracking
PIP_COMPILE_PY_FLAG := $(shell pip-compile --help 2>/dev/null | grep -q -- '--python-version' && echo "--python-version=$(LOCK_PYTHON_VERSION)" || true)
PIP_COMPILE := mkdir -p "$(PIP_TOOLS_CACHE_DIR)" && PIP_TOOLS_CACHE_DIR="$(PIP_TOOLS_CACHE_DIR)" $(PIP_COMPILE_BASE) $(PIP_COMPILE_PY_FLAG)
PIP_COMPILE_HASHED := mkdir -p "$(PIP_TOOLS_CACHE_DIR)" && PIP_TOOLS_CACHE_DIR="$(PIP_TOOLS_CACHE_DIR)" $(PIP_COMPILE_BASE) $(PIP_COMPILE_PY_FLAG) --generate-hashes

# PyTorch CPU index for CI/cross-platform lockfiles
PYTORCH_CPU_INDEX := --extra-index-url https://download.pytorch.org/whl/cpu

# Default target: compile all requirements
.PHONY: all compile compile-all compile-ml-layers compile-hash-pilot check-hash-pilot update help clean check \
	ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt

help:
	@echo "Targets:"
	@echo "  compile           Compile all pinned requirements from .in files"
	@echo "  compile-all       Same as compile (for compatibility)"
	@echo "  compile-ml-layers Compile only ML layer lockfiles"
	@echo "  compile-accel     Compile acceleration layer lockfiles (ml-cpu, ml-mps, ml-cuda)"
	@echo "  compile-hash-pilot Generate hash-enforced pilot lockfiles into HASH_PILOT_OUT_DIR"
	@echo "  update            Update all dependencies to latest versions"
	@echo "  check             Verify that .txt files are up-to-date with .in files"
	@echo "  check-hash-pilot  Validate the generated pilot lockfiles with --require-hashes"
	@echo "  clean             Remove all compiled .txt files"
	@echo ""
	@echo "Individual compilation targets:"
	@echo "  all.txt              Compile all.txt (checked-in contract aggregate)"
	@echo "  base.txt             Compile base.txt (constrained by all.txt)"
	@echo ""
	@echo "Platform-specific ML Core targets:"
	@echo "  ml-core-darwin-x86_64.txt  macOS Intel conservative ML baseline"
	@echo "  ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline"
	@echo "  ml-core-linux.txt          Linux ML baseline (torch 2.2.2 + transformers<5)"
	@echo ""
	@echo "Checked-in ML lockfiles:"
	@echo "  ml-core-darwin-x86_64.txt  macOS Intel conservative ML baseline"
	@echo "  ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline"
	@echo "  ml-core-linux.txt          Linux ML baseline (torch 2.2.2 + transformers<5)"
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
compile: all.txt base.txt ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ All requirements compiled successfully"

compile-all: compile

# Compile only ML layers (useful for ML-focused updates)
# Note: SAM2 is scripted-only and NOT compiled here
compile-ml-layers: all.txt base.txt ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt
	@echo "✓ ML layer requirements compiled successfully"

# Compile only acceleration layers (useful for profile updates)
compile-accel: all.txt base.txt ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt
	@echo "✓ Acceleration layer requirements compiled successfully"

# Hash pilot lane for secure-install evaluation.
# This intentionally covers only the non-ML checked-in contract so we can
# evaluate hash enforcement without pulling platform-specific ML lockfiles or
# changing the canonical repository install surface.
compile-hash-pilot:
	@echo "Compiling secure-install hash pilot into $(HASH_PILOT_OUT_DIR)..."
	@mkdir -p "$(HASH_PILOT_OUT_DIR)"
	@$(PIP_COMPILE_HASHED) all.in -o "$(HASH_PILOT_OUT_DIR)/all.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" base.in -o "$(HASH_PILOT_OUT_DIR)/base.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" dev.in -o "$(HASH_PILOT_OUT_DIR)/dev.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" ci.in -o "$(HASH_PILOT_OUT_DIR)/ci.txt"
	@$(PIP_COMPILE_HASHED) security.in -o "$(HASH_PILOT_OUT_DIR)/security.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" tools-archive.in -o "$(HASH_PILOT_OUT_DIR)/tools-archive.txt"
	@echo "✓ Secure-install hash pilot compiled successfully"

check-hash-pilot:
	@echo "Checking secure-install hash pilot in $(HASH_PILOT_OUT_DIR)..."
	@set -euo pipefail; \
	for lock in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
		lock_path="$(HASH_PILOT_OUT_DIR)/$$lock"; \
		if [[ ! -f "$$lock_path" ]]; then \
			echo "ERROR: Missing pilot lockfile $$lock_path"; \
			echo "ERROR: Run 'make compile-hash-pilot' first."; \
			exit 1; \
		fi; \
		python -m pip install --dry-run --require-hashes -r "$$lock_path" >/dev/null; \
	done; \
	echo "✓ Secure-install hash pilot passed"

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
# Platform-specific core lockfiles resolve against the base layer contract.
# Linux keeps the exact base.txt constraint. Darwin uses base.in so the
# conservative macOS ML lock can carry numpy<2 without inheriting the
# host-resolved base.txt NumPy exact pin.
# NOTE: Non-core optional ML layers are not compiled here.
# NOTE: SAM2 is scripted-only and NOT compiled here - use bootstrap script

# Platform-specific ml-core lockfiles (pip-compile cannot resolve multi-platform .in)
ml-core-darwin-x86_64.txt: ml-core-darwin-x86_64.in base.in
	@echo "Compiling ml-core-darwin-x86_64.txt (macOS Intel conservative baseline)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.in ml-core-darwin-x86_64.in -o ml-core-darwin-x86_64.txt

ml-core-darwin-arm64.txt: ml-core-darwin-arm64.in base.txt
	@echo "Compiling ml-core-darwin-arm64.txt (macOS Apple Silicon baseline)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt

ml-core-linux.txt: ml-core-linux.in base.txt
	@echo "Compiling ml-core-linux.txt (Linux baseline, transformers<5)..."
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
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c base.in ml-core-darwin-x86_64.in -o ml-core-darwin-x86_64.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt
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
	cp all.in base.in ml.in ml-core.in ml-core-darwin-x86_64.in ml-core-darwin-arm64.in ml-core-linux.in ml-cpu.in ml-mps.in ml-cuda.in ml-raw.in ml-coreml.in ml-research.in dev.in ci.in security.in tools-archive.in "$$tmp_dir"/; \
	for f in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			cp "$$repo_dir/$$f" "$$tmp_dir"/; \
		else \
			echo "ERROR: Required lockfile missing: $$f"; \
			exit 1; \
		fi; \
	done; \
	for f in ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt; do \
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
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.in ml-core-darwin-x86_64.in -o ml-core-darwin-x86_64.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt >/dev/null; \
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
	for f in ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt; do \
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
	@rm -f all.txt base.txt ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt ml-sam2.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ Cleanup complete"
