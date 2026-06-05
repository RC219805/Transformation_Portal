#!/usr/bin/env python3
"""APEX Policy Validation Script.

Validates APEX policy files for:
- Schema conformance
- Internal consistency
- Alignment with code (DEFAULT_BUCKETS)

Usage:
    python scripts/apex_validate_policy.py --policy-dir docs/apex/policy/
    python scripts/apex_validate_policy.py --check consistency
    python scripts/apex_validate_policy.py --check schema

Exit codes:
    0: All validations passed
    1: Validation errors found
    2: Configuration or usage error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def validate_schema_version(policy: Dict[str, Any], filename: str, expected_version: str | None = None) -> List[str]:
    """Validate schema_version field.

    Args:
        policy: Policy data to validate
        filename: Name of policy file for error messages
        expected_version: If provided, enforce exact match against this version
    """
    errors = []

    if "schema_version" not in policy:
        errors.append(f"{filename}: Missing required field 'schema_version'")
        return errors

    version = policy["schema_version"]
    if not isinstance(version, str):
        errors.append(f"{filename}: schema_version must be string, got {type(version).__name__}")
        return errors

    # Validate semver format
    parts = version.split(".")
    if len(parts) != 3:
        errors.append(f"{filename}: schema_version must be semver (X.Y.Z), got '{version}'")
    else:
        for i, part in enumerate(parts):
            if not part.isdigit():
                errors.append(f"{filename}: schema_version part {i} must be numeric, got '{part}'")

    # Enforce expected version if provided
    if expected_version is not None and version != expected_version:
        errors.append(f"{filename}: schema_version '{version}' does not match expected '{expected_version}'")

    return errors


def validate_performance_budgets(policy_dir: Path, expected_version: str | None = None) -> List[str]:
    """Validate performance_budgets.yaml."""
    errors = []

    budgets_path = policy_dir / "performance_budgets.yaml"
    if not budgets_path.exists():
        return [f"Missing required file: {budgets_path}"]

    budgets = load_yaml_file(budgets_path)

    # Validate schema version
    errors.extend(validate_schema_version(budgets, "performance_budgets.yaml", expected_version))

    # Validate required top-level fields
    required_fields = ["schema_version", "effective_date", "review_date", "policy_owner", "budgets"]
    for field in required_fields:
        if field not in budgets:
            errors.append(f"performance_budgets.yaml: Missing required field '{field}'")

    if "budgets" not in budgets:
        return errors  # Can't validate budgets if field is missing

    # Validate each budget entry
    valid_workflow_versions = {"v1", "v2"}
    valid_stability_tiers = {"stable", "canary", "experimental"}
    valid_enforcement_modes = {"shadow", "enforce", "disabled"}

    for i, budget in enumerate(budgets["budgets"]):
        prefix = f"performance_budgets.yaml: budget[{i}]"

        # Required fields
        if "workflow_version" not in budget:
            errors.append(f"{prefix}: Missing 'workflow_version'")
        elif budget["workflow_version"] not in valid_workflow_versions:
            errors.append(
                f"{prefix}: Invalid workflow_version '{budget['workflow_version']}' (must be {valid_workflow_versions})"
            )

        if "bucket_name" not in budget:
            errors.append(f"{prefix}: Missing 'bucket_name'")

        if "stability_tier" not in budget:
            errors.append(f"{prefix}: Missing 'stability_tier'")
        elif budget["stability_tier"] not in valid_stability_tiers:
            errors.append(f"{prefix}: Invalid stability_tier '{budget['stability_tier']}' (must be {valid_stability_tiers})")

        # Validate thresholds
        if "thresholds" not in budget:
            errors.append(f"{prefix}: Missing 'thresholds'")
        else:
            thresholds = budget["thresholds"]
            required_threshold_fields = ["p50_sec", "p95_sec", "max_regression_pct"]
            for field in required_threshold_fields:
                if field not in thresholds:
                    errors.append(f"{prefix}.thresholds: Missing '{field}'")
                elif not isinstance(thresholds[field], (int, float)):
                    errors.append(f"{prefix}.thresholds.{field}: Must be numeric, got {type(thresholds[field]).__name__}")
                elif thresholds[field] < 0:
                    errors.append(f"{prefix}.thresholds.{field}: Must be non-negative, got {thresholds[field]}")

        # Validate enforcement
        if "enforcement" not in budget:
            errors.append(f"{prefix}: Missing 'enforcement'")
        else:
            enforcement = budget["enforcement"]
            if "mode" not in enforcement:
                errors.append(f"{prefix}.enforcement: Missing 'mode'")
            elif enforcement["mode"] not in valid_enforcement_modes:
                errors.append(
                    f"{prefix}.enforcement.mode: Invalid mode '{enforcement['mode']}' (must be {valid_enforcement_modes})"
                )

            if "effective_from" not in enforcement:
                errors.append(f"{prefix}.enforcement: Missing 'effective_from'")

    # Validate tier_policies if present
    if "tier_policies" in budgets:
        for tier in valid_stability_tiers:
            if tier in budgets["tier_policies"]:
                tier_policy = budgets["tier_policies"][tier]
                required_tier_fields = ["description", "required_baseline_days", "min_sample_size", "max_change_frequency"]
                for field in required_tier_fields:
                    if field not in tier_policy:
                        errors.append(f"performance_budgets.yaml: tier_policies.{tier}: Missing '{field}'")

    return errors


def validate_enforcement_policy(policy_dir: Path, expected_version: str | None = None) -> List[str]:
    """Validate enforcement_policy.yaml."""
    errors = []

    policy_path = policy_dir / "enforcement_policy.yaml"
    if not policy_path.exists():
        return [f"Missing required file: {policy_path}"]

    policy = load_yaml_file(policy_path)

    # Validate schema version
    errors.extend(validate_schema_version(policy, "enforcement_policy.yaml", expected_version))

    # Validate required top-level fields
    required_fields = ["schema_version", "evidence_gates", "statistical_methods", "sample_size_requirements"]
    for field in required_fields:
        if field not in policy:
            errors.append(f"enforcement_policy.yaml: Missing required field '{field}'")

    # Validate evidence gates
    if "evidence_gates" in policy:
        for mode in ["shadow_mode", "enforce_mode"]:
            if mode in policy["evidence_gates"]:
                gate = policy["evidence_gates"][mode]
                if "min_sample_size" not in gate:
                    errors.append(f"enforcement_policy.yaml: evidence_gates.{mode}: Missing 'min_sample_size'")
                elif not isinstance(gate["min_sample_size"], int) or gate["min_sample_size"] < 1:
                    errors.append(f"enforcement_policy.yaml: evidence_gates.{mode}.min_sample_size: Must be positive integer")

                if "allow_synthetic_data" not in gate:
                    errors.append(f"enforcement_policy.yaml: evidence_gates.{mode}: Missing 'allow_synthetic_data'")

    # Validate sample size requirements
    if "sample_size_requirements" in policy:
        reqs = policy["sample_size_requirements"]
        for percentile in ["p50", "p95", "p99"]:
            if percentile in reqs:
                if not isinstance(reqs[percentile], int) or reqs[percentile] < 1:
                    errors.append(f"enforcement_policy.yaml: sample_size_requirements.{percentile}: Must be positive integer")

    return errors


def validate_governance_rules(policy_dir: Path, expected_version: str | None = None) -> List[str]:
    """Validate governance_rules.yaml."""
    errors = []

    rules_path = policy_dir / "governance_rules.yaml"
    if not rules_path.exists():
        return [f"Missing required file: {rules_path}"]

    rules = load_yaml_file(rules_path)

    # Validate schema version
    errors.extend(validate_schema_version(rules, "governance_rules.yaml", expected_version))

    # Validate required top-level fields
    required_fields = ["schema_version", "waivers", "budget_changes", "incidents"]
    for field in required_fields:
        if field not in rules:
            errors.append(f"governance_rules.yaml: Missing required field '{field}'")

    # Validate waivers section
    if "waivers" in rules:
        waivers = rules["waivers"]
        required_waiver_fields = ["allowed_scopes", "required_fields", "labels", "expiry"]
        for field in required_waiver_fields:
            if field not in waivers:
                errors.append(f"governance_rules.yaml: waivers: Missing '{field}'")

        if "expiry" in waivers:
            expiry = waivers["expiry"]
            if "default_days" in expiry and "max_days" in expiry:
                if expiry["default_days"] > expiry["max_days"]:
                    errors.append(
                        f"governance_rules.yaml: waivers.expiry: default_days ({expiry['default_days']}) > max_days ({expiry['max_days']})"
                    )

    return errors


def validate_workload_suites(policy_dir: Path, expected_version: str | None = None) -> List[str]:
    """Validate workload_suites.yaml."""
    errors = []

    suites_path = policy_dir / "workload_suites.yaml"
    if not suites_path.exists():
        return [f"Missing required file: {suites_path}"]

    suites = load_yaml_file(suites_path)

    # Validate schema version
    errors.extend(validate_schema_version(suites, "workload_suites.yaml", expected_version))

    # Validate required top-level fields
    required_fields = ["schema_version", "golden_suite", "canary_suite", "fuzz_suite"]
    for field in required_fields:
        if field not in suites:
            errors.append(f"workload_suites.yaml: Missing required field '{field}'")

    # Validate golden suite
    if "golden_suite" in suites:
        golden = suites["golden_suite"]
        required_golden_fields = ["description", "fixture_dir", "images", "change_policy", "usage"]
        for field in required_golden_fields:
            if field not in golden:
                errors.append(f"workload_suites.yaml: golden_suite: Missing '{field}'")

        if "change_policy" in golden:
            policy = golden["change_policy"]
            if "requires_adr" not in policy:
                errors.append(f"workload_suites.yaml: golden_suite.change_policy: Missing 'requires_adr'")

    return errors


def check_consistency_with_code(policy_dir: Path) -> List[str]:
    """Check consistency between policy files and code."""
    errors = []

    # Try to import DEFAULT_BUCKETS
    try:
        # Add src to path if needed
        import sys

        repo_root = Path(__file__).parent.parent
        src_path = repo_root / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from transformation_portal.metrics.performance_capsule import DEFAULT_BUCKETS

        # Load budgets from policy file
        budgets_path = policy_dir / "performance_budgets.yaml"
        if not budgets_path.exists():
            return errors  # Already reported by validate_performance_budgets

        budgets_data = load_yaml_file(budgets_path)
        policy_buckets = {b["bucket_name"] for b in budgets_data.get("budgets", [])}
        code_buckets = {b.name for b in DEFAULT_BUCKETS}

        # Check for missing buckets
        missing_in_policy = code_buckets - policy_buckets
        if missing_in_policy:
            errors.append(f"Consistency: Buckets in DEFAULT_BUCKETS but not in policy: {missing_in_policy}")

        extra_in_policy = policy_buckets - code_buckets
        if extra_in_policy:
            # This is OK - policy can define budgets for buckets not yet in code
            pass

    except ImportError as e:
        errors.append(f"Consistency check skipped: Could not import DEFAULT_BUCKETS: {e}")

    return errors


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate APEX policy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("docs/apex/policy"),
        help="Directory containing policy files (default: docs/apex/policy)",
    )
    parser.add_argument(
        "--check",
        choices=["all", "schema", "consistency"],
        default="all",
        help="What to check (default: all)",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="1.0.0",
        help="Expected schema version (default: 1.0.0)",
    )

    args = parser.parse_args()

    if not args.policy_dir.exists():
        print(f"ERROR: Policy directory does not exist: {args.policy_dir}", file=sys.stderr)
        return 2

    errors: List[str] = []

    # Run requested checks
    if args.check in ["all", "schema"]:
        print("Validating policy file schemas...")
        errors.extend(validate_performance_budgets(args.policy_dir, args.schema_version))
        errors.extend(validate_enforcement_policy(args.policy_dir, args.schema_version))
        errors.extend(validate_governance_rules(args.policy_dir, args.schema_version))
        errors.extend(validate_workload_suites(args.policy_dir, args.schema_version))

    if args.check in ["all", "consistency"]:
        print("Checking consistency with code...")
        errors.extend(check_consistency_with_code(args.policy_dir))

    # Report results
    if errors:
        print(f"\n❌ Validation failed with {len(errors)} error(s):\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    else:
        print("\n✅ All validation checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
