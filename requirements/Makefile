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
# Standard Lockfile Layers (compiled via this Makefile):
#   ml-core.txt      Cross-platform ML baseline (torch, diffusers, transformers, etc.)
#   ml-cpu.txt       CPU baseline acceleration layer (thin, includes ml-core)
#   ml-mps.txt       Apple Silicon MPS acceleration layer (darwin-arm64-mps)
#   ml-cuda.txt      NVIDIA CUDA acceleration layer (linux-x86_64-cuda)
#   ml-raw.txt       RAW camera file ingest (rawpy) - may have platform limitations
#   ml-coreml.txt    Apple CoreML acceleration - macOS only
#   ml-research.txt  Research/experimental extras - reserved for future use
#   ml.txt           Standard umbrella layer (ml-core + ml-raw)
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
.PHONY: all compile compile-all compile-ml-layers update help clean check

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
	@echo "  all.txt           Compile all.txt (global resolution)"
	@echo "  base.txt          Compile base.txt (constrained by all.txt)"
	@echo ""
	@echo "ML Layer targets (CPU-only PyTorch index):"
	@echo "  ml-core.txt       Cross-platform ML baseline"
	@echo "  ml-cpu.txt        CPU baseline acceleration layer"
	@echo "  ml-mps.txt        Apple Silicon MPS acceleration layer"
	@echo "  ml-cuda.txt       NVIDIA CUDA acceleration layer"
	@echo "  ml-raw.txt        RAW ingest capability layer"
	@echo "  ml-coreml.txt     Apple CoreML acceleration layer"
	@echo "  ml-research.txt   Research/experimental extras layer"
	@echo "  ml.txt            Standard umbrella layer (ml-core + ml-raw)"
	@echo ""
	@echo "Scripted-only layers (NOT compiled here):"
	@echo "  ml-sam2           SAM2 segmentation - use bootstrap script"
	@echo ""
	@echo "Other targets:"
	@echo "  dev.txt           Compile dev.txt (constrained by all.txt)"
	@echo "  ci.txt            Compile ci.txt (constrained by all.txt)"
	@echo "  tools-archive.txt Compile tools-archive.txt (constrained by all.txt)"

# Compile all requirements in correct order
# Note: SAM2 is scripted-only and NOT compiled here - use bootstrap script
compile: all.txt base.txt ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt dev.txt ci.txt tools-archive.txt
	@echo "✓ All requirements compiled successfully"

compile-all: compile

# Compile only ML layers (useful for ML-focused updates)
# Note: SAM2 is scripted-only and NOT compiled here
compile-ml-layers: all.txt ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt
	@echo "✓ ML layer requirements compiled successfully"

# Compile only acceleration layers (useful for profile updates)
compile-accel: all.txt ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt
	@echo "✓ Acceleration layer requirements compiled successfully"

# Compile all.txt first for global resolution
# Note: ml-sam2.in included for global resolution but NOT compiled as lockfile
# Note: ml-cpu.in, ml-mps.in, ml-cuda.in included for acceleration layers
all.txt: all.in base.in ml.in ml-core.in ml-cpu.in ml-mps.in ml-cuda.in ml-raw.in ml-sam2.in ml-coreml.in ml-research.in dev.in ci.in
	@echo "Compiling all.txt (global resolution)..."
	@$(PIP_COMPILE) all.in -o all.txt

# Compile individual files constrained by all.txt
base.txt: base.in all.txt
	@echo "Compiling base.txt..."
	@$(PIP_COMPILE) -c all.txt base.in -o base.txt

# ML Layer lockfiles - compiled with CPU-only PyTorch index for deterministic CI
# Each layer is compiled independently to allow fine-grained capability installs
# NOTE: SAM2 is scripted-only and NOT compiled here - use bootstrap script

ml-core.txt: ml-core.in all.txt
	@echo "Compiling ml-core.txt (CPU-only PyTorch)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-core.in -o ml-core.txt

# Acceleration layers (platform matrix profiles)
ml-cpu.txt: ml-cpu.in ml-core.in all.txt
	@echo "Compiling ml-cpu.txt (CPU baseline)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-cpu.in -o ml-cpu.txt

ml-mps.txt: ml-mps.in ml-core.in all.txt
	@echo "Compiling ml-mps.txt (Apple Silicon MPS)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-mps.in -o ml-mps.txt

ml-cuda.txt: ml-cuda.in ml-core.in all.txt
	@echo "Compiling ml-cuda.txt (NVIDIA CUDA)..."
	@# NOTE: ml-cuda.txt is compiled with CPU index for consistency,
	@# but contains CUDA packages with platform markers.
	@# Actual CUDA wheel installation uses PyTorch CUDA index at install time.
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-cuda.in -o ml-cuda.txt

ml-raw.txt: ml-raw.in all.txt
	@echo "Compiling ml-raw.txt..."
	@$(PIP_COMPILE) -c all.txt ml-raw.in -o ml-raw.txt

ml-coreml.txt: ml-coreml.in all.txt
	@echo "Compiling ml-coreml.txt..."
	@$(PIP_COMPILE) -c all.txt ml-coreml.in -o ml-coreml.txt

ml-research.txt: ml-research.in all.txt
	@echo "Compiling ml-research.txt..."
	@$(PIP_COMPILE) -c all.txt ml-research.in -o ml-research.txt

# Standard umbrella lockfile - includes ml-core + ml-raw
# Does NOT include: sam2 (scripted-only), coreml (platform-specific), research (experimental)
ml.txt: ml.in ml-core.in ml-raw.in all.txt
	@echo "Compiling ml.txt (standard umbrella, CPU-only PyTorch)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml.in -o ml.txt

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
# Note: SAM2 is scripted-only and NOT updated here
update:
	@echo "Updating all dependencies to latest versions..."
	@$(PIP_COMPILE) --upgrade all.in -o all.txt
	@$(PIP_COMPILE) --upgrade -c all.txt base.in -o base.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c all.txt ml-core.in -o ml-core.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c all.txt ml-cpu.in -o ml-cpu.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c all.txt ml-mps.in -o ml-mps.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c all.txt ml-cuda.in -o ml-cuda.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ml-raw.in -o ml-raw.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ml-coreml.in -o ml-coreml.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ml-research.in -o ml-research.txt
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c all.txt ml.in -o ml.txt
	@$(PIP_COMPILE) --upgrade -c all.txt dev.in -o dev.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ci.in -o ci.txt
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
	cp all.in base.in ml.in ml-core.in ml-cpu.in ml-mps.in ml-cuda.in ml-raw.in ml-coreml.in ml-research.in dev.in ci.in tools-archive.in "$$tmp_dir"/; \
	for f in all.txt base.txt ml.txt dev.txt ci.txt tools-archive.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			cp "$$repo_dir/$$f" "$$tmp_dir"/; \
		else \
			echo "ERROR: Required lockfile missing: $$f"; \
			exit 1; \
		fi; \
	done; \
	for f in ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt; do \
		if [ -f "$$repo_dir/$$f" ]; then \
			cp "$$repo_dir/$$f" "$$tmp_dir"/; \
		else \
			echo "INFO: Optional ML layer lockfile not found: $$f (will be compiled by CI)"; \
		fi; \
	done; \
	cd "$$tmp_dir"; \
	$(PIP_COMPILE) all.in -o all.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt base.in -o base.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-core.in -o ml-core.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-cpu.in -o ml-cpu.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-mps.in -o ml-mps.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml-cuda.in -o ml-cuda.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ml-raw.in -o ml-raw.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ml-coreml.in -o ml-coreml.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ml-research.in -o ml-research.txt >/dev/null; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c all.txt ml.in -o ml.txt >/dev/null; \
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
	for f in ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt; do \
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
	@rm -f all.txt base.txt ml.txt ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-sam2.txt ml-coreml.txt ml-research.txt dev.txt ci.txt tools-archive.txt
	@echo "✓ Cleanup complete"
