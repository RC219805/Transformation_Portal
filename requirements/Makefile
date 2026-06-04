# Makefile for compiling layered requirements with pip-tools
# This ensures consistent dependency resolution across all layers
#
# LAYERED ML DEPENDENCY STRATEGY (ADR-032)
# ========================================
# ML dependencies are split into capability-specific layers for:
# - Target-safe compilation (generic locks plus explicit target-owned ML lanes)
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
# TARGET-OWNED ML LOCKFILES (pip-compile cannot resolve multi-platform .in)
# -------------------------------------------------------------------------
# Target            Lockfile                    Baseline
# darwin-arm64      ml-core-darwin-arm64.txt   torch 2.8.0 + torchvision 0.23.0 + CoreML tooling
# darwin-x86_64     retired                    unsupported; no installable checked-in ML lock
# linux-x86_64      retired                    unsupported; no installable checked-in ML lock
#
# Target-owned ML lockfiles:
#   ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline
#
# Non-core optional ML layers are not part of the checked-in lockfile contract.
# Host-generated outputs for ml-core.txt, ml-cpu.txt, ml-mps.txt, ml-cuda.txt,
# ml-raw.txt, ml-coreml.txt, ml-research.txt, and ml.txt must not be checked in.
#
# Scripted-Only Layers (NOT compiled here - use bootstrap script):
#   ml-sam2          SAM2 segmentation - requires non-standard install semantics
#                    Install via: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,sam2
#
# See requirements/README.md and requirements/lock_ownership.yml for full documentation.

SHELL := /bin/bash
LOCK_PYTHON_VERSION ?= 3.11
PIP_TOOLS_CACHE_DIR ?= $(CURDIR)/.pip-tools-cache
HASH_PILOT_OUT_DIR ?= $(CURDIR)/.hash-pilot
PIP_COMPILE_BIN := $(if $(wildcard $(CURDIR)/../.venv/bin/pip-compile),$(CURDIR)/../.venv/bin/pip-compile,pip-compile)
PIP_COMPILE_BASE := $(PIP_COMPILE_BIN) -q --resolver=backtracking
PIP_COMPILE_PY_FLAG := $(shell $(PIP_COMPILE_BIN) --help 2>/dev/null | grep -q -- '--python-version' && echo "--python-version=$(LOCK_PYTHON_VERSION)" || true)
PIP_COMPILE := mkdir -p "$(PIP_TOOLS_CACHE_DIR)" && PIP_TOOLS_CACHE_DIR="$(PIP_TOOLS_CACHE_DIR)" $(PIP_COMPILE_BASE) $(PIP_COMPILE_PY_FLAG)
PIP_COMPILE_HASHED := mkdir -p "$(PIP_TOOLS_CACHE_DIR)" && PIP_TOOLS_CACHE_DIR="$(PIP_TOOLS_CACHE_DIR)" $(PIP_COMPILE_BASE) $(PIP_COMPILE_PY_FLAG) --generate-hashes
RESTORE_MARKER_PINS := python3 ../scripts/utilities/restore_requirement_marker_pins.py

# PyTorch CPU index for CI/cross-platform lockfiles
PYTORCH_CPU_INDEX := --extra-index-url https://download.pytorch.org/whl/cpu

# Default target: compile generic checked-in requirements
.PHONY: all compile compile-all compile-generic compile-ml-layers compile-accel compile-hash-pilot check-hash-pilot \
	update update-generic help clean check check-generic \
	compile-ml-darwin-arm64 update-ml-darwin-arm64 check-ml-darwin-arm64 \
	compile-ml-linux-x86_64 update-ml-linux-x86_64 check-ml-linux-x86_64 \
	compile-ml-darwin-x86_64 update-ml-darwin-x86_64 check-ml-darwin-x86_64 \
	require-pip-compile require-darwin-arm64-authority refuse-darwin-x86_64-authority refuse-linux-x86_64-authority \
	ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt

help:
	@echo "Targets:"
	@echo "  compile           Compile generic checked-in lockfiles only"
	@echo "  compile-all       Same as compile (for compatibility)"
	@echo "  compile-generic   Compile the generic checked-in lockfiles only"
	@echo "  compile-ml-layers Refuse broad ML regeneration; use explicit target-owned commands"
	@echo "  compile-accel     Refuse broad ML regeneration; use explicit target-owned commands"
	@echo "  compile-hash-pilot Generate hash-enforced pilot lockfiles into HASH_PILOT_OUT_DIR"
	@echo "  update            Update generic checked-in lockfiles only"
	@echo "  update-generic    Update generic checked-in lockfiles only"
	@echo "  check             Verify generic checked-in lockfiles only"
	@echo "  check-generic     Verify generic checked-in lockfiles only"
	@echo "  check-hash-pilot  Validate the generated pilot lockfiles with --require-hashes"
	@echo "  clean             Remove all compiled .txt files"
	@echo ""
	@echo "Individual compilation targets:"
	@echo "  all.txt              Compile all.txt (checked-in contract aggregate)"
	@echo "  base.txt             Compile base.txt (constrained by all.txt)"
	@echo ""
	@echo "Target-owned ML commands:"
	@echo "  compile-ml-darwin-arm64    Compile the Darwin arm64 ML lock on native Darwin arm64 only"
	@echo "  update-ml-darwin-arm64     Update the Darwin arm64 ML lock on native Darwin arm64 only"
	@echo "  check-ml-darwin-arm64      Verify the Darwin arm64 ML lock on native Darwin arm64 only"
	@echo "  compile-ml-linux-x86_64    Retired unsupported lane - always fails closed"
	@echo "  update-ml-linux-x86_64     Retired unsupported lane - always fails closed"
	@echo "  check-ml-linux-x86_64      Retired unsupported lane - always fails closed"
	@echo "  compile-ml-darwin-x86_64   Retired unsupported lane - always fails closed"
	@echo "  update-ml-darwin-x86_64    Retired unsupported lane - always fails closed"
	@echo "  check-ml-darwin-x86_64     Retired unsupported lane - always fails closed"
	@echo ""
	@echo "Target-owned ML lockfiles:"
	@echo "  ml-core-darwin-arm64.txt   macOS Apple Silicon ML baseline"
	@echo "  macOS Intel/Linux          retired unsupported lanes; no checked-in installable ML lock"
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

# Compile generic requirements in correct order
# Note: target-owned ML lockfiles must be regenerated via explicit target commands.
compile: compile-generic

compile-generic: all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ Generic requirements compiled successfully"

compile-all: compile

compile-ml-layers:
	@echo "ERROR: target-owned ML locks require explicit authoritative-lane commands."
	@echo "ERROR: Use 'make compile-ml-darwin-arm64' on native Darwin arm64."
	@exit 1

compile-accel:
	@echo "ERROR: target-owned ML locks require explicit authoritative-lane commands."
	@echo "ERROR: Acceleration/bootstrap flows must start from a target-owned ML core command, not compile-accel."
	@exit 1

require-pip-compile:
	@if ! command -v "$(PIP_COMPILE_BIN)" >/dev/null 2>&1; then \
		echo "ERROR: pip-compile is required for requirements/Makefile targets."; \
		echo "ERROR: Install pip-tools==7.5.2 in the active environment before running compile/update/check commands."; \
		exit 1; \
	fi

# Hash pilot lane for secure-install evaluation.
# This intentionally covers only the non-ML checked-in contract so we can
# evaluate hash enforcement without pulling platform-specific ML lockfiles or
# changing the canonical repository install surface.
compile-hash-pilot: require-pip-compile
	@echo "Compiling secure-install hash pilot into $(HASH_PILOT_OUT_DIR)..."
	@mkdir -p "$(HASH_PILOT_OUT_DIR)"
	@$(PIP_COMPILE_HASHED) all.in -o "$(HASH_PILOT_OUT_DIR)/all.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" base.in -o "$(HASH_PILOT_OUT_DIR)/base.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" dev.in -o "$(HASH_PILOT_OUT_DIR)/dev.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" ci.in -o "$(HASH_PILOT_OUT_DIR)/ci.txt"
	@$(PIP_COMPILE_HASHED) security.in -o "$(HASH_PILOT_OUT_DIR)/security.txt"
	@$(PIP_COMPILE_HASHED) -c "$(HASH_PILOT_OUT_DIR)/all.txt" tools-archive.in -o "$(HASH_PILOT_OUT_DIR)/tools-archive.txt"
	@echo "✓ Secure-install hash pilot compiled successfully"

check-hash-pilot: require-pip-compile
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
all.txt: require-pip-compile all.in base.in dev.in ci.in
	@echo "Compiling all.txt (checked-in contract aggregate)..."
	@$(PIP_COMPILE) all.in -o all.txt

# Compile individual files constrained by all.txt
base.txt: require-pip-compile base.in all.txt
	@echo "Compiling base.txt..."
	@$(PIP_COMPILE) -c all.txt base.in -o base.txt

# ML layer lockfiles - checked-in contract is limited to target-owned core locks.
# Target-owned core lockfiles resolve against the base layer contract.
# Linux keeps the exact base.txt constraint. Darwin uses base.in so the
# conservative macOS ML lock can carry numpy<2 without inheriting the
# host-resolved base.txt NumPy exact pin.
# NOTE: Non-core optional ML layers are not compiled here.
# NOTE: SAM2 is scripted-only and NOT compiled here - use bootstrap script

# Target-owned ml-core lockfiles (pip-compile cannot resolve multi-platform .in)
require-darwin-arm64-authority:
	@os="$$(uname -s)"; arch="$$(uname -m)"; \
	if [ "$$arch" = "aarch64" ]; then arch="arm64"; fi; \
	if [ "$$os" != "Darwin" ] || [ "$$arch" != "arm64" ]; then \
		echo "ERROR: Darwin arm64 ML lock generation is authoritative only on native Darwin arm64."; \
		echo "ERROR: Run 'make compile-ml-darwin-arm64' or related Darwin arm64 commands on an Apple Silicon macOS host."; \
		exit 1; \
	fi

refuse-darwin-x86_64-authority:
	@echo "ERROR: macOS Intel ML lockfiles are retired unsupported manifests."
	@echo "ERROR: Do not regenerate or reinstall retired vulnerable torch baselines."
	@exit 1

refuse-linux-x86_64-authority:
	@echo "ERROR: Linux ML lockfiles are retired unsupported manifests."
	@echo "ERROR: Do not regenerate or reinstall retired vulnerable torch baselines."
	@exit 1

ml-core-darwin-x86_64.txt: refuse-darwin-x86_64-authority

ml-core-darwin-arm64.txt: require-darwin-arm64-authority require-pip-compile ml-core-darwin-arm64.in base.txt
	@echo "Compiling ml-core-darwin-arm64.txt (macOS Apple Silicon baseline)..."
	@$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt

ml-core-linux.txt: refuse-linux-x86_64-authority

compile-ml-darwin-arm64: ml-core-darwin-arm64.txt
	@echo "✓ Darwin arm64 ML lock compiled successfully"

update-ml-darwin-arm64: require-darwin-arm64-authority require-pip-compile
	@echo "Updating ml-core-darwin-arm64.txt on authoritative Darwin arm64 lane..."
	@$(PIP_COMPILE) --upgrade $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt
	@echo "✓ Darwin arm64 ML lock updated successfully"

check-ml-darwin-arm64: require-darwin-arm64-authority require-pip-compile
	@echo "Checking Darwin arm64 ML lock..."
	@set -euo pipefail; \
	repo_dir="$$(pwd)"; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp base.txt ml-core-darwin-arm64.in "$$tmp_dir"/; \
	if [ -f "$$repo_dir/ml-core-darwin-arm64.txt" ]; then \
		cp "$$repo_dir/ml-core-darwin-arm64.txt" "$$tmp_dir"/; \
	else \
		echo "ERROR: Required lockfile missing: ml-core-darwin-arm64.txt"; \
		exit 1; \
	fi; \
	cd "$$tmp_dir"; \
	$(PIP_COMPILE) $(PYTORCH_CPU_INDEX) -c base.txt ml-core-darwin-arm64.in -o ml-core-darwin-arm64.txt >/dev/null; \
	normalize_req() { \
		awk 'BEGIN{start=0} { if (!start) { if ($$0 ~ /^#/ || $$0 ~ /^[[:space:]]*$$/) next; start=1 } print }' "$$1"; \
	}; \
	if ! diff -u <(normalize_req "$$tmp_dir/ml-core-darwin-arm64.txt") <(normalize_req "$$repo_dir/ml-core-darwin-arm64.txt") >/dev/null; then \
		echo "❌ ml-core-darwin-arm64.txt is out of date. Run 'make compile-ml-darwin-arm64' on native Darwin arm64."; \
		exit 1; \
	fi; \
	echo "✓ Darwin arm64 ML lock is up-to-date"

compile-ml-linux-x86_64 update-ml-linux-x86_64 check-ml-linux-x86_64: refuse-linux-x86_64-authority

compile-ml-darwin-x86_64 update-ml-darwin-x86_64 check-ml-darwin-x86_64: refuse-darwin-x86_64-authority

ml-core.txt ml-cpu.txt ml-mps.txt ml-cuda.txt ml-raw.txt ml-coreml.txt ml-research.txt ml.txt:
	@echo "ERROR: $@ is not part of the checked-in lockfile contract."
	@echo "ERROR: Use target-specific bootstrap/install flows instead of checking in host-generated optional ML layer locks."
	@exit 1

dev.txt: require-pip-compile dev.in all.txt
	@echo "Compiling dev.txt..."
	@$(PIP_COMPILE) -c all.txt dev.in -o dev.txt

ci.txt: require-pip-compile ci.in all.txt
	@echo "Compiling ci.txt..."
	@$(PIP_COMPILE) -c all.txt ci.in -o ci.txt

# Security scanning tools (standalone layer for CI security job)
# NOT constrained by all.txt to ensure latest security tool versions
security.txt: require-pip-compile security.in
	@echo "Compiling security.txt..."
	@$(PIP_COMPILE) security.in -o security.txt

tools-archive.txt: require-pip-compile tools-archive.in all.txt
	@echo "Compiling tools-archive.txt..."
	@$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt

# Update generic dependencies to latest allowed versions
# Note: target-owned ML locks must be updated via explicit target commands.
update: update-generic

update-generic: require-pip-compile
	@echo "Updating generic checked-in lockfiles to latest allowed versions..."
	@set -euo pipefail; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp all.txt "$$tmp_dir/all.txt"; \
	cp base.txt "$$tmp_dir/base.txt"; \
	$(PIP_COMPILE) --upgrade all.in -o all.txt; \
	$(RESTORE_MARKER_PINS) --previous "$$tmp_dir/all.txt" --lockfile all.txt; \
	$(PIP_COMPILE) --upgrade -c all.txt base.in -o base.txt; \
	$(RESTORE_MARKER_PINS) --previous "$$tmp_dir/base.txt" --lockfile base.txt
	@$(PIP_COMPILE) --upgrade -c all.txt dev.in -o dev.txt
	@$(PIP_COMPILE) --upgrade -c all.txt ci.in -o ci.txt
	@$(PIP_COMPILE) --upgrade security.in -o security.txt
	@$(PIP_COMPILE) --upgrade -c all.txt tools-archive.in -o tools-archive.txt
	@echo "✓ Generic requirements updated successfully"

# Check if .txt files are up-to-date
# Note: target-owned ML locks must be checked via explicit target commands.
check: check-generic

check-generic: require-pip-compile
	@echo "Checking if generic requirements are up-to-date..."
	@set -euo pipefail; \
	repo_dir="$$(pwd)"; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp all.in base.in dev.in ci.in security.in tools-archive.in "$$tmp_dir"/; \
	for f in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
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
	$(PIP_COMPILE) -c all.txt dev.in -o dev.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt ci.in -o ci.txt >/dev/null; \
	$(PIP_COMPILE) security.in -o security.txt >/dev/null; \
	$(PIP_COMPILE) -c all.txt tools-archive.in -o tools-archive.txt >/dev/null; \
	normalize_req() { \
		: "The lock contract separately enforces the Linux-only keyring chain and platform-marked OpenCV runtime pins, so host-side diffs ignore those packages and pip-compile provenance comments."; \
		file_name="$$(basename "$$1")"; \
		awk '!/^[[:space:]]*#/ && !/^[[:space:]]*$$/ { print }' "$$1" | \
		while IFS= read -r line; do \
			case "$$file_name:$$line" in \
				all.txt:jeepney==*|all.txt:secretstorage==*|all.txt:opencv-python==*|all.txt:opencv-python-headless==*|base.txt:opencv-python==*|base.txt:opencv-python-headless==*|ci.txt:cffi==*|ci.txt:cryptography==*|ci.txt:jeepney==*|ci.txt:pycparser==*|ci.txt:secretstorage==*) \
					continue; \
					;; \
			esac; \
			printf '%s\n' "$$line"; \
		done; \
	}; \
	for f in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \
		if ! diff -u <(normalize_req "$$tmp_dir/$$f") <(normalize_req "$$repo_dir/$$f") >/dev/null; then \
			echo "❌ $$f is out of date. Run 'make compile' to update."; \
			exit 1; \
		fi; \
	done; \
	echo "✓ Generic requirements are up-to-date"

# Clean compiled files
# Note: ml-sam2.txt removed if present (historical artifact)
clean:
	@echo "Removing compiled .txt files..."
	@rm -f all.txt base.txt ml-core-darwin-x86_64.txt ml-core-darwin-arm64.txt ml-core-linux.txt ml-sam2.txt dev.txt ci.txt security.txt tools-archive.txt
	@echo "✓ Cleanup complete"
