#!/bin/bash
# Reference bash + jq examples for parsing tp.meta.machine.v1 JSON output
# See docs/api/MACHINE_MODE_CONTRACT.md for full contract specification

set -euo pipefail

# ==============================================================================
# Example 1: Extract Command with Exit Code Routing
# ==============================================================================

extract_with_routing() {
    local input_path="$1"

    if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract "$input_path"); then
        exit_code=0
    else
        exit_code=$?
    fi

    # Validate schema
    schema=$(echo "$result" | jq -r '.schema')
    if [[ "$schema" != "tp.meta.machine.v1" ]]; then
        echo "ERROR: Unsupported schema: $schema" >&2
        return 99
    fi

    # Route by exit code
    if [[ $exit_code -eq 0 ]]; then
        input_path=$(echo "$result" | jq -r '.data.input_path')
        output_path=$(echo "$result" | jq -r '.data.output_path')
        elapsed=$(echo "$result" | jq -r '.data.elapsed_seconds')
        echo "✅ Extracted: $input_path → $output_path (${elapsed}s)"
        return 0
    else
        error_type=$(echo "$result" | jq -r '.data.error.type')
        error_name=$(echo "$result" | jq -r '.data.error.exit_code.name')
        echo "❌ Extract failed: $error_type ($error_name)" >&2
        return $exit_code
    fi
}

# ==============================================================================
# Example 2: Validate Command with Typed Error Handling
# ==============================================================================

validate_with_error_handling() {
    local sidecar_path="$1"

    if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json validate "$sidecar_path"); then
        exit_code=0
    else
        exit_code=$?
    fi

    success=$(echo "$result" | jq -r '.success')
    sidecar=$(echo "$result" | jq -r '.data.sidecar_path')

    if [[ "$success" == "true" ]]; then
        echo "✅ Validation passed: $sidecar"
        return 0
    else
        dominant_type=$(echo "$result" | jq -r '.data.dominant_error.type')
        dominant_name=$(echo "$result" | jq -r '.data.dominant_error.exit_code.name')
        error_count=$(echo "$result" | jq '.data.errors | length')

        echo "❌ Validation failed: $sidecar" >&2
        echo "   Dominant error: $dominant_type ($dominant_name)" >&2
        echo "   Total errors: $error_count" >&2

        # Detailed error breakdown
        echo "   Error breakdown:" >&2
        echo "$result" | jq -r '.data.errors[] | "     - \(.type): \(.message)"' >&2

        return $exit_code
    fi
}

# ==============================================================================
# Example 3: Batch Extract with Summary Statistics
# ==============================================================================

batch_extract_with_summary() {
    local input_root="$1"
    local output_dir="$2"

    if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract-batch "$input_root" --output "$output_dir"); then
        exit_code=0
    else
        exit_code=$?
    fi

    total=$(echo "$result" | jq '.data.summary_counts.total')
    success=$(echo "$result" | jq '.data.summary_counts.success')
    failure=$(echo "$result" | jq '.data.summary_counts.failure')

    echo "📦 Batch result: $success/$total succeeded, $failure failed"

    if [[ $exit_code -ne 0 ]]; then
        echo "Exit code breakdown:" >&2
        echo "$result" | jq -r '.data.summary_counts.by_exit_code | to_entries[] | "  \(.key): \(.value)"' >&2

        # Show first few failures
        echo "First 3 failures:" >&2
        echo "$result" | jq -r '.data.items[] | select(.success == false) | "  - \(.path): \(.error.type)"' | head -3 >&2
    fi

    return $exit_code
}

# ==============================================================================
# Example 4: Check System Readiness
# ==============================================================================

check_system_readiness() {
    if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json check-system); then
        exit_code=0
    else
        exit_code=$?
    fi

    all_ok=$(echo "$result" | jq -r '.data.all_required_ok')

    if [[ "$all_ok" == "true" ]]; then
        echo "✅ System check passed"

        # Show available tools
        echo "Tool versions:"
        echo "$result" | jq -r '
            .data |
            to_entries |
            map(select(.key | endswith("_version"))) |
            .[] |
            "  \(.key | sub("_version$"; "")): \(.value)"
        '
        return 0
    else
        echo "❌ System check failed" >&2
        echo "$result" | jq -r '.data.errors[]' >&2
        return $exit_code
    fi
}

# ==============================================================================
# Example 5: CI-Friendly Exit Code Routing
# ==============================================================================

ci_safe_validate() {
    local sidecar_path="$1"

    if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json validate "$sidecar_path"); then
        exit_code=0
    else
        exit_code=$?
    fi

    case $exit_code in
        0)
            echo "✅ Validation passed"
            return 0
            ;;
        1)
            echo "❌ BLOCKER: Schema validation failed" >&2
            echo "$result" | jq -r '.data.errors[] | select(.type == "SchemaValidationFailure") | .message' >&2
            return 1
            ;;
        2)
            echo "❌ QUALITY GATE: 8-bit conversion detected" >&2
            return 2
            ;;
        3)
            echo "❌ QUALITY GATE: Gamma correction detected" >&2
            return 3
            ;;
        4)
            echo "⚠️  WARNING: Schema drift detected (unknown fields)" >&2
            echo "$result" | jq -r '.data.errors[] | select(.type == "SchemaDriftFailure") | .message' >&2
            return 4
            ;;
        5)
            echo "❌ ERROR: Other failure" >&2
            dominant=$(echo "$result" | jq -r '.data.dominant_error.message')
            echo "   $dominant" >&2
            return 5
            ;;
        *)
            echo "❌ UNEXPECTED: Unknown exit code $exit_code" >&2
            return $exit_code
            ;;
    esac
}

# ==============================================================================
# Example 6: Compact Status Check (One-liner Style)
# ==============================================================================

compact_status_check() {
    local sidecar_path="$1"

    .venv/bin/python scripts/test_metadata_extraction.py --json validate "$sidecar_path" | \
        jq -r 'if .success then "✅ \(.data.sidecar_path)" else "❌ \(.data.sidecar_path): \(.data.dominant_error.type)" end'
}

# ==============================================================================
# Example 7: Extract Multiple Files with Error Collection
# ==============================================================================

extract_multiple_with_error_collection() {
    local output_dir="$1"
    shift
    local files=("$@")

    local errors=()
    local successes=0

    for file in "${files[@]}"; do
        if result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract "$file" --output "$output_dir"); then
            exit_code=0
        else
            exit_code=$?
        fi

        if [[ $exit_code -eq 0 ]]; then
            ((successes++))
        else
            error_type=$(echo "$result" | jq -r '.data.error.type')
            errors+=("$file: $error_type")
        fi
    done

    echo "Results: $successes/${#files[@]} succeeded"

    if [[ ${#errors[@]} -gt 0 ]]; then
        echo "Errors:" >&2
        printf '  - %s\n' "${errors[@]}" >&2
        return 1
    fi

    return 0
}

# ==============================================================================
# Main: Example Usage
# ==============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Machine JSON Parser Examples"
    echo "See docs/api/MACHINE_MODE_CONTRACT.md for full documentation"
    echo ""
    echo "Available functions:"
    echo "  extract_with_routing <input_path>"
    echo "  validate_with_error_handling <sidecar_path>"
    echo "  batch_extract_with_summary <input_root> <output_dir>"
    echo "  check_system_readiness"
    echo "  ci_safe_validate <sidecar_path>"
    echo "  compact_status_check <sidecar_path>"
    echo "  extract_multiple_with_error_collection <output_dir> <file1> [file2...]"
    echo ""
    echo "Example:"
    echo "  source tools/parse_machine_json_examples.sh"
    echo "  check_system_readiness"
fi
