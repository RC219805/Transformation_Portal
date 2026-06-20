#!/usr/bin/env bash
# scripts/validate_dependency_constraints.sh
#
# Validates dependency constraints in requirements/*.in files.
# Enforces ADR-032: Dependency Pinning Strategy
#
# Exit codes:
#   0 - All validations pass
#   1 - Blocking violations found (unpinned, banned, security)
#   2 - Non-blocking warnings (suggested improvements)
#
# Usage:
#   ./scripts/validate_dependency_constraints.sh [--verbose]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="$("${REPO_ROOT}/scripts/setup/resolve_python_311.sh")"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0
FILES_CHECKED=0

# Verbose mode
VERBOSE=0
if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=1
fi

BANNED_REGISTRY="scripts/security/banned_dependencies.json"
BANNED_PACKAGE_ENTRIES=()

load_banned_packages() {
    if [[ ! -f "$BANNED_REGISTRY" ]]; then
        echo -e "${RED}❌ Missing banned registry: $BANNED_REGISTRY${NC}"
        exit 1
    fi

    local parsed_registry
    local parse_error
    parsed_registry="$(mktemp)"
    parse_error="$(mktemp)"

    if ! BANNED_REGISTRY="$BANNED_REGISTRY" "${PYTHON_BIN}" << 'PY' >"$parsed_registry" 2>"$parse_error"
import json
import os
from pathlib import Path

registry = Path(os.environ["BANNED_REGISTRY"])
data = json.loads(registry.read_text(encoding="utf-8"))
for entry in data.get("packages", []):
    name = str(entry.get("name", "")).strip().lower()
    reason = str(entry.get("reason", "")).strip()
    migration = str(entry.get("migration", "")).strip()
    if not name or not reason:
        continue
    print(f"{name}|{reason}|{migration}")
PY
    then
        echo -e "${RED}❌ Failed to parse banned registry: $BANNED_REGISTRY${NC}"
        cat "$parse_error" >&2
        rm -f "$parsed_registry" "$parse_error"
        exit 1
    fi

    rm -f "$parse_error"

    while IFS='|' read -r package reason migration; do
        [[ -z "$package" ]] && continue
        BANNED_PACKAGE_ENTRIES+=("${package}|${reason}|${migration}")
    done < "$parsed_registry"
    rm -f "$parsed_registry"

    if [[ ${#BANNED_PACKAGE_ENTRIES[@]} -eq 0 ]]; then
        echo -e "${RED}❌ No banned package entries loaded from $BANNED_REGISTRY${NC}"
        exit 1
    fi
}

# Function: Check if package is banned
is_banned_package() {
    local package="$1"
    local key
    key=$(printf "%s" "$package" | tr '[:upper:]' '[:lower:]')
    local entry=""
    local name=""
    local reason=""
    local migration=""

    for entry in "${BANNED_PACKAGE_ENTRIES[@]}"; do
        IFS='|' read -r name reason migration <<< "$entry"
        if [[ "$name" == "$key" ]]; then
            [[ -z "$migration" ]] && migration="Use approved alternatives and repository-native implementations."
            echo "${reason}|${migration}"
            return 0
        fi
    done
    return 1
}

# Function: Get security minimum version
get_security_minimum() {
    local package="$1"
    case "$package" in
        "sentence-transformers")
            echo "3.1.0|CVE-73169 (arbitrary code execution)"
            return 0
            ;;
        "Pillow")
            echo "10.3.0|CVE-2024-28219 and multiple 9.x CVEs"
            return 0
            ;;
        "starlette")
            echo "1.3.1|CVE-2026-48710 / PYSEC-2026-161 plus 2026 StaticFiles/HTTPEndpoint/form parsing fixes"
            return 0
            ;;
    esac
    return 1
}

# Function: Check if package has approved exception
get_approved_exception() {
    local package="$1"
    case "$package" in
        "mypy") echo "dev.in|Type checker: benefits from latest rules"; return 0 ;;
        "black") echo "dev.in|Formatter: deterministic, auto-updates OK"; return 0 ;;
        "flake8") echo "dev.in|Linter: new rules are improvements"; return 0 ;;
        "pylint") echo "dev.in|Linter: CLI stable across minor versions"; return 0 ;;
        "types-PyYAML") echo "dev.in|Type stubs: must track PyYAML version"; return 0 ;;
        "torch") echo "ml-core.in|Supported ML floor; target-owned locks and operator indexes control upper resolution"; return 0 ;;
        "torchvision") echo "ml-core.in|Supported ML floor paired with torch; target-owned locks control upper resolution"; return 0 ;;
        "PyYAML") echo "ml.in|Config parser: strong backward compatibility"; return 0 ;;
        "colour-science") echo "ml.in|Color math library: stable API"; return 0 ;;
        "coremltools") echo "ml.in|Apple ML tools: platform-specific updates"; return 0 ;;
        "psutil") echo "ml.in|System utilities: OS compatibility layer"; return 0 ;;
        "memory-profiler") echo "ml.in|Dev/profiling tool in optional deps"; return 0 ;;
        "pypdf") echo "ci.in|PDF utilities: backward-compatible 6.x security floor"; return 0 ;;
    esac
    return 1
}

# Production files (require range pins or strict pins)
PRODUCTION_FILES=("base.in" "ml.in")
TARGET_OWNED_ML_INPUTS=("ml-core-darwin-arm64.in")

echo -e "${BLUE}${BOLD}🔍 Validating dependency constraints...${NC}\n"
load_banned_packages

# Function: Extract package name from dependency line
extract_package_name() {
    local line="$1"
    # Remove version constraints and extras, handle pip-compile markers
    echo "$line" | sed -E 's/[>=<~!]=.*$//' | sed 's/\[.*\]$//' | tr -d ' '
}

is_target_owned_ml_input() {
    local basename="$1"
    local target_input=""
    for target_input in "${TARGET_OWNED_ML_INPUTS[@]}"; do
        if [[ "$basename" == "$target_input" ]]; then
            return 0
        fi
    done
    return 1
}

normalize_host_arch() {
    local arch="$1"
    case "$arch" in
        "aarch64")
            echo "arm64"
            ;;
        "amd64")
            echo "x86_64"
            ;;
        *)
            echo "$arch"
            ;;
    esac
}

current_host_system() {
    uname -s 2>/dev/null || echo ""
}

current_host_arch() {
    local arch
    arch="$(uname -m 2>/dev/null || echo "")"
    normalize_host_arch "$arch"
}

validate_target_owned_ml_freshness() {
    local basename="$1"
    local txt_file="$2"
    local host_os=""
    local host_arch=""
    local check_target=""
    local compile_fix=""
    local check_output=""

    case "$basename" in
        "ml-core-darwin-arm64.in")
            host_os="$(current_host_system)"
            host_arch="$(current_host_arch)"
            if [[ "$host_os" != "Darwin" || "$host_arch" != "arm64" ]]; then
                return 0
            fi
            check_target="check-ml-darwin-arm64"
            compile_fix="Run 'make -C requirements compile-ml-darwin-arm64' on native Darwin arm64."
            ;;
        *)
            return 0
            ;;
    esac

    if [[ $VERBOSE -eq 1 ]]; then
        if check_output=$(make -C requirements "$check_target" LOCK_PYTHON_VERSION=3.11 2>&1); then
            return 0
        fi
    else
        if make -C requirements "$check_target" LOCK_PYTHON_VERSION=3.11 >/dev/null 2>&1; then
            return 0
        fi
    fi

    echo -e "${YELLOW}⚠️  $basename: Target-owned ML lock freshness check failed${NC}"
    echo -e "   ${BOLD}WARNING:${NC} $(basename "$txt_file") could not be validated by $check_target on this authoritative lane"
    if [[ $VERBOSE -eq 1 && -n "$check_output" ]]; then
        echo -e "   ${BOLD}Details:${NC}"
        while IFS= read -r output_line; do
            echo "     $output_line"
        done <<< "$check_output"
    fi
    echo -e "   ${BOLD}Fix:${NC} $compile_fix\n"
    return 1
}

# Function: Extract version from constraint
extract_version() {
    local constraint="$1"
    # Extract version number from patterns like >=X.Y.Z or ==X.Y.Z
    echo "$constraint" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1
}

# Function: Compare semantic versions (returns 0 if v1 >= v2, 1 otherwise)
version_gte() {
    local v1="$1"
    local v2="$2"

    # Use Python for accurate semantic version comparison
    "${PYTHON_BIN}" << EOF
from packaging.version import Version
import sys
try:
    result = Version("$v1") >= Version("$v2")
    sys.exit(0 if result else 1)
except Exception as e:
    sys.exit(1)
EOF
}

# Function: Validate single .in file
validate_in_file() {
    local in_file="$1"
    local basename
    basename=$(basename "$in_file")
    local file_errors=0
    local file_warnings=0

    FILES_CHECKED=$((FILES_CHECKED + 1))

    [[ $VERBOSE -eq 1 ]] && echo -e "${BLUE}Checking $basename...${NC}"

    # Check if file is a production file
    local is_production=0
    for prod_file in "${PRODUCTION_FILES[@]}"; do
        if [[ "$basename" == "$prod_file" ]]; then
            is_production=1
            break
        fi
    done

    local line_num=0
    while IFS= read -r line; do
        line_num=$((line_num + 1))

        # Skip comments, empty lines, and -r includes
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^-r ]] && continue

        # Extract package name
        local package
        package=$(extract_package_name "$line")
        [[ -z "$package" ]] && continue

        # Check for banned packages
        if banned_info=$(is_banned_package "$package"); then
            IFS='|' read -r reason migration <<< "$banned_info"
            echo -e "${RED}❌ $basename:$line_num: $line${NC}"
            echo -e "   ${BOLD}ERROR:${NC} Package '$package' is BANNED ($reason)"
            echo -e "   ${BOLD}Migration:${NC} $migration\n"
            file_errors=$((file_errors + 1))
            continue
        fi

        # Validate constraint style
        if echo "$line" | grep -qE '^[a-zA-Z0-9_-]+[[:space:]]*$'; then
            # Unpinned dependency
            echo -e "${RED}❌ $basename:$line_num: $line${NC}"
            echo -e "   ${BOLD}ERROR:${NC} Unpinned dependency (no version constraint)"
            echo -e "   ${BOLD}Fix:${NC} Add version constraint:"
            echo -e "     - Production deps: Use range pin (>=X.Y,<Z)"
            echo -e "     - Dev tools: Use lower-bound (>=X.Y) if CLI is stable\n"
            file_errors=$((file_errors + 1))

        elif echo "$line" | grep -qE '^[a-zA-Z0-9_-]+>=[0-9.]+$'; then
            # Lower-bound-only constraint
            local pkg
            pkg=$(echo "$line" | sed -E 's/>=.*//')

            # Check if this is an approved exception
            if exception_info=$(get_approved_exception "$pkg"); then
                IFS='|' read -r allowed_file reason <<< "$exception_info"
                if [[ "$basename" != "$allowed_file" ]]; then
                    echo -e "${RED}❌ $basename:$line_num: $line${NC}"
                    echo -e "   ${BOLD}ERROR:${NC} Lower-bound-only constraint in wrong file"
                    echo -e "   ${BOLD}Approved for:${NC} $allowed_file ($reason)"
                    echo -e "   ${BOLD}Fix:${NC} Either move to $allowed_file or add upper bound\n"
                    file_errors=$((file_errors + 1))
                fi
            elif [[ $is_production -eq 1 ]]; then
                # Production file without approved exception
                echo -e "${RED}❌ $basename:$line_num: $line${NC}"
                echo -e "   ${BOLD}ERROR:${NC} Lower-bound-only constraint in production file"
                echo -e "   ${BOLD}Fix:${NC} Add upper bound for determinism (>=X.Y,<Z)"
                echo -e "   ${BOLD}Exception:${NC} Add to ADR-032 Section 4 if strong rationale exists\n"
                file_errors=$((file_errors + 1))
            else
                # Dev/CI file without approved exception
                echo -e "${YELLOW}⚠️  $basename:$line_num: $line${NC}"
                echo -e "   ${BOLD}WARNING:${NC} Lower-bound-only constraint without approved exception"
                echo -e "   ${BOLD}Consider:${NC} Adding upper bound for safety or documenting exception in ADR-032\n"
                file_warnings=$((file_warnings + 1))
            fi

        elif echo "$line" | grep -qE '^[a-zA-Z0-9_-]+==[0-9.]+'; then
            # Strict pin - should have inline comment
            if ! echo "$line" | grep -q '#'; then
                echo -e "${YELLOW}⚠️  $basename:$line_num: $line${NC}"
                echo -e "   ${BOLD}WARNING:${NC} Strict pin without inline comment"
                echo -e "   ${BOLD}Best practice:${NC} Add comment explaining rationale (e.g., '# Deterministic builds')\n"
                file_warnings=$((file_warnings + 1))
            fi
        fi

        # Check security minimums
        if security_info=$(get_security_minimum "$package"); then
            IFS='|' read -r min_version reason <<< "$security_info"
            local current_version=""
            local current_spec=""

            # Enforce security minimums for both lower bounds and strict pins.
            if echo "$line" | grep -qE '==[0-9.]+'; then
                current_spec="=="
                current_version=$(echo "$line" | grep -oE '==[0-9.]+' | head -1 | sed 's/==//') || current_version=""
            elif echo "$line" | grep -qE '>=[0-9.]+'; then
                current_spec=">="
                current_version=$(echo "$line" | grep -oE '>=[0-9.]+' | head -1 | sed 's/>=//') || current_version=""
            fi

            if [[ -n "$current_version" ]] && ! version_gte "$current_version" "$min_version"; then
                echo -e "${RED}❌ $basename:$line_num: $line${NC}"
                echo -e "   ${BOLD}ERROR:${NC} Security minimum not met (need >=$min_version for $reason)"
                echo -e "   ${BOLD}Current:${NC} ${current_spec}${current_version}"
                echo -e "   ${BOLD}Fix:${NC} Update constraint to >=$min_version,<... or pin to ==$min_version (or a later patched version)\n"
                file_errors=$((file_errors + 1))
            fi
        fi

    done < "$in_file"

    # Check corresponding .txt file freshness
    local txt_file="${in_file%.in}.txt"
    if [[ -f "$txt_file" ]]; then
        if [[ "$txt_file" -ot "$in_file" ]]; then
            if is_target_owned_ml_input "$basename"; then
                if ! validate_target_owned_ml_freshness "$basename" "$txt_file"; then
                    file_warnings=$((file_warnings + 1))
                fi
            else
                echo -e "${YELLOW}⚠️  $basename: Compiled .txt file is stale${NC}"
                echo -e "   ${BOLD}WARNING:${NC} $(basename "$txt_file") is older than $basename"
                echo -e "   ${BOLD}Fix:${NC} Run 'make -C requirements compile' to regenerate\n"
                file_warnings=$((file_warnings + 1))
            fi
        fi

        # Check for pip-compile header (ensures it wasn't manually edited)
        if ! head -5 "$txt_file" | grep -q "autogenerated by pip-compile"; then
            echo -e "${YELLOW}⚠️  $(basename "$txt_file"): Missing pip-compile header${NC}"
            echo -e "   ${BOLD}WARNING:${NC} File may have been manually edited"
            echo -e "   ${BOLD}Fix:${NC} Regenerate with 'make -C requirements compile'\n"
            file_warnings=$((file_warnings + 1))
        fi
    fi

    ERRORS=$((ERRORS + file_errors))
    WARNINGS=$((WARNINGS + file_warnings))

    if [[ $file_errors -eq 0 && $file_warnings -eq 0 ]]; then
        echo -e "${GREEN}✅ $basename: All constraints valid${NC}"
    fi
}

# Function: Validate pyproject dependency security minimums
validate_pyproject_security_minimums() {
    local pyproject_file="pyproject.toml"
    local file_errors=0
    local file_warnings=0

    if [[ ! -f "$pyproject_file" ]]; then
        return
    fi

    FILES_CHECKED=$((FILES_CHECKED + 1))
    [[ $VERBOSE -eq 1 ]] && echo -e "${BLUE}Checking pyproject.toml dependencies...${NC}"

    while IFS='|' read -r package spec raw_dep; do
        [[ -z "$package" ]] && continue

        if security_info=$(get_security_minimum "$package"); then
            IFS='|' read -r min_version reason <<< "$security_info"
            local current_version=""
            local current_spec=""

            if echo "$spec" | grep -qE '==[0-9.]+'; then
                current_spec="=="
                current_version=$(echo "$spec" | grep -oE '==[0-9.]+' | head -1 | sed 's/==//') || current_version=""
            elif echo "$spec" | grep -qE '>=[0-9.]+'; then
                current_spec=">="
                current_version=$(echo "$spec" | grep -oE '>=[0-9.]+' | head -1 | sed 's/>=//') || current_version=""
            fi

            if [[ -n "$current_version" ]] && ! version_gte "$current_version" "$min_version"; then
                echo -e "${RED}❌ pyproject.toml: $raw_dep${NC}"
                echo -e "   ${BOLD}ERROR:${NC} Security minimum not met for '$package' (need >=$min_version for $reason)"
                echo -e "   ${BOLD}Current:${NC} ${current_spec}${current_version}"
                echo -e "   ${BOLD}Fix:${NC} Update constraint to >=$min_version,<... or pin to ==$min_version (or a later patched version)\n"
                file_errors=$((file_errors + 1))
            elif [[ -z "$current_version" ]]; then
                echo -e "${YELLOW}⚠️  pyproject.toml: $raw_dep${NC}"
                echo -e "   ${BOLD}WARNING:${NC} Could not parse version lower bound for security-managed package '$package'"
                echo -e "   ${BOLD}Fix:${NC} Use an explicit >=X.Y (or ==X.Y) constraint in project.dependencies\n"
                file_warnings=$((file_warnings + 1))
            fi
        fi
    done < <(
        "${PYTHON_BIN}" << 'PY'
import re
import tomllib
from pathlib import Path

path = Path("pyproject.toml")
if not path.exists():
    raise SystemExit(0)

data = tomllib.loads(path.read_text(encoding="utf-8"))
deps = data.get("project", {}).get("dependencies", [])

for dep in deps:
    dep = dep.split(";", 1)[0].strip()
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)(\[[^\]]+\])?\s*(.*)$", dep)
    if not match:
        continue
    name = match.group(1)
    spec = (match.group(3) or "").replace(" ", "")
    print(f"{name}|{spec}|{dep}")
PY
    )

    ERRORS=$((ERRORS + file_errors))
    WARNINGS=$((WARNINGS + file_warnings))

    if [[ $file_errors -eq 0 && $file_warnings -eq 0 ]]; then
        echo -e "${GREEN}✅ pyproject.toml: Security minimums valid${NC}"
    fi
}

# Main validation loop
shopt -s nullglob
IN_FILES=(requirements/*.in)

if [[ ${#IN_FILES[@]} -eq 0 ]]; then
    echo -e "${RED}❌ No .in files found in requirements/ directory${NC}"
    exit 1
fi

for in_file in "${IN_FILES[@]}"; do
    # Skip all.in (it just includes other files)
    [[ "$(basename "$in_file")" == "all.in" ]] && continue

    validate_in_file "$in_file"
    echo # Blank line between files
done

validate_pyproject_security_minimums
echo

# Summary
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✅ All $FILES_CHECKED dependency files validated successfully!${NC}"
    echo -e "${BLUE}Run 'make -C requirements compile' for generic locks; use explicit target-owned ML commands for target-owned ML inputs.${NC}"
    exit 0
elif [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}${BOLD}❌ Validation failed: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo -e "${BLUE}Fix errors in .in files, then run 'make -C requirements compile' for generic locks.${NC}"
    exit 1
else
    echo -e "${YELLOW}${BOLD}⚠️  Validation passed with warnings: $WARNINGS warning(s)${NC}"
    echo -e "${BLUE}Consider addressing warnings for best practices.${NC}"
    exit 2
fi
