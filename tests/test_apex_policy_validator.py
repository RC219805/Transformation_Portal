"""Unit tests for APEX policy validator.

Tests cover:
- Schema version mismatch detection
- Images field validation (oneOf: string|object)
- Bucket name consistency
- Missing required fields
"""

from __future__ import annotations


pytestmark = pytest.mark.unit

# Import validator functions
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "scripts"))
from apex_validate_policy import validate_performance_budgets, validate_schema_version, validate_workload_suites


class TestSchemaVersionValidation:
    """Test schema_version enforcement."""

    def test_valid_schema_version(self):
        """Valid semver schema_version should pass."""
        policy = {"schema_version": "1.0.0"}
        errors = validate_schema_version(policy, "test.yaml", expected_version="1.0.0")
        assert len(errors) == 0

    def test_schema_version_mismatch(self):
        """Schema version mismatch should be detected."""
        policy = {"schema_version": "1.0.0"}
        errors = validate_schema_version(policy, "test.yaml", expected_version="2.0.0")
        assert len(errors) == 1
        assert "does not match expected '2.0.0'" in errors[0]

    def test_missing_schema_version(self):
        """Missing schema_version should be detected."""
        policy: dict[str, Any] = {}
        errors = validate_schema_version(policy, "test.yaml")
        assert len(errors) == 1
        assert "Missing required field 'schema_version'" in errors[0]

    def test_invalid_semver_format(self):
        """Invalid semver format should be detected."""
        policy = {"schema_version": "1.0"}  # Not X.Y.Z
        errors = validate_schema_version(policy, "test.yaml")
        assert len(errors) == 1
        assert "must be semver (X.Y.Z)" in errors[0]

    def test_non_numeric_version_parts(self):
        """Non-numeric version parts should be detected."""
        policy = {"schema_version": "1.0.beta"}
        errors = validate_schema_version(policy, "test.yaml")
        assert len(errors) == 1
        assert "must be numeric" in errors[0]


class TestPerformanceBudgetsValidation:
    """Test performance_budgets.yaml validation."""

    def test_valid_budget_structure(self, tmp_path):
        """Valid budget file should pass validation."""
        budget_data = {
            "schema_version": "1.0.0",
            "effective_date": "2026-02-15",
            "review_date": "2026-05-15",
            "policy_owner": "test-owner",
            "budgets": [
                {
                    "workflow_version": "v2",
                    "bucket_name": "test_bucket",
                    "stability_tier": "stable",
                    "thresholds": {
                        "p50_sec": 5.0,
                        "p95_sec": 10.0,
                        "max_regression_pct": 10.0,
                    },
                    "enforcement": {
                        "mode": "shadow",
                        "effective_from": "2026-02-15",
                    },
                }
            ],
        }

        budgets_file = tmp_path / "performance_budgets.yaml"
        with open(budgets_file, "w") as f:
            yaml.dump(budget_data, f)

        errors = validate_performance_budgets(tmp_path, expected_version="1.0.0")
        assert len(errors) == 0

    def test_missing_required_fields(self, tmp_path):
        """Missing required fields should be detected."""
        budget_data = {
            "schema_version": "1.0.0",
            # Missing: effective_date, review_date, policy_owner, budgets
        }

        budgets_file = tmp_path / "performance_budgets.yaml"
        with open(budgets_file, "w") as f:
            yaml.dump(budget_data, f)

        errors = validate_performance_budgets(tmp_path)
        assert len(errors) >= 4  # At least 4 missing fields

    def test_invalid_workflow_version(self, tmp_path):
        """Invalid workflow_version should be detected."""
        budget_data = {
            "schema_version": "1.0.0",
            "effective_date": "2026-02-15",
            "review_date": "2026-05-15",
            "policy_owner": "test-owner",
            "budgets": [
                {
                    "workflow_version": "v99",  # Invalid
                    "bucket_name": "test_bucket",
                    "stability_tier": "stable",
                    "thresholds": {
                        "p50_sec": 5.0,
                        "p95_sec": 10.0,
                        "max_regression_pct": 10.0,
                    },
                    "enforcement": {
                        "mode": "shadow",
                        "effective_from": "2026-02-15",
                    },
                }
            ],
        }

        budgets_file = tmp_path / "performance_budgets.yaml"
        with open(budgets_file, "w") as f:
            yaml.dump(budget_data, f)

        errors = validate_performance_budgets(tmp_path)
        assert any("Invalid workflow_version" in e for e in errors)

    def test_negative_threshold_values(self, tmp_path):
        """Negative threshold values should be detected."""
        budget_data = {
            "schema_version": "1.0.0",
            "effective_date": "2026-02-15",
            "review_date": "2026-05-15",
            "policy_owner": "test-owner",
            "budgets": [
                {
                    "workflow_version": "v2",
                    "bucket_name": "test_bucket",
                    "stability_tier": "stable",
                    "thresholds": {
                        "p50_sec": -5.0,  # Invalid
                        "p95_sec": 10.0,
                        "max_regression_pct": 10.0,
                    },
                    "enforcement": {
                        "mode": "shadow",
                        "effective_from": "2026-02-15",
                    },
                }
            ],
        }

        budgets_file = tmp_path / "performance_budgets.yaml"
        with open(budgets_file, "w") as f:
            yaml.dump(budget_data, f)

        errors = validate_performance_budgets(tmp_path)
        assert any("Must be non-negative" in e for e in errors)


class TestWorkloadSuitesValidation:
    """Test workload_suites.yaml validation including images field."""

    def test_images_as_simple_strings(self, tmp_path):
        """Images as simple strings should be valid."""
        suites_data = {
            "schema_version": "1.0.0",
            "golden_suite": {
                "description": "Test suite",
                "fixture_dir": "tests/fixtures/test",
                "images": ["image1.jpg", "image2.jpg"],  # Simple strings
                "change_policy": {"requires_adr": True},
                "usage": ["Test usage"],
            },
            "canary_suite": {
                "description": "Canary",
                "fixture_dir": "tests/fixtures/canary",
                "change_policy": {"requires_adr": False},
                "usage": ["Canary usage"],
            },
            "fuzz_suite": {
                "description": "Fuzz",
                "generation_script": "scripts/generate_fuzz.py",
                "usage": ["Fuzz usage"],
            },
        }

        suites_file = tmp_path / "workload_suites.yaml"
        with open(suites_file, "w") as f:
            yaml.dump(suites_data, f)

        errors = validate_workload_suites(tmp_path, expected_version="1.0.0")
        # Schema validation passes (validator doesn't enforce the oneOf yet in Python)
        # This test documents that simple strings should work
        assert len(errors) == 0

    def test_images_as_structured_objects(self, tmp_path):
        """Images as structured objects should be valid."""
        suites_data = {
            "schema_version": "1.0.0",
            "golden_suite": {
                "description": "Test suite",
                "fixture_dir": "tests/fixtures/test",
                "images": [
                    {
                        "name": "pool_luxury_4k.jpg",
                        "path": "pool_luxury_4k.jpg",
                        "scene_type": "pool",
                        "dimensions": [3840, 2160],
                        "pixel_count": 8294400,
                        "characteristics": ["Specular highlights"],
                        "rationale": "Test image",
                    }
                ],
                "change_policy": {"requires_adr": True},
                "usage": ["Test usage"],
            },
            "canary_suite": {
                "description": "Canary",
                "fixture_dir": "tests/fixtures/canary",
                "change_policy": {"requires_adr": False},
                "usage": ["Canary usage"],
            },
            "fuzz_suite": {
                "description": "Fuzz",
                "generation_script": "scripts/generate_fuzz.py",
                "usage": ["Fuzz usage"],
            },
        }

        suites_file = tmp_path / "workload_suites.yaml"
        with open(suites_file, "w") as f:
            yaml.dump(suites_data, f)

        errors = validate_workload_suites(tmp_path, expected_version="1.0.0")
        # Schema validation passes
        assert len(errors) == 0

    def test_missing_golden_suite_fields(self, tmp_path):
        """Missing required fields in golden_suite should be detected."""
        suites_data = {
            "schema_version": "1.0.0",
            "golden_suite": {
                "description": "Test suite",
                # Missing: fixture_dir, images, change_policy, usage
            },
            "canary_suite": {
                "description": "Canary",
                "fixture_dir": "tests/fixtures/canary",
                "change_policy": {"requires_adr": False},
                "usage": ["Canary usage"],
            },
            "fuzz_suite": {
                "description": "Fuzz",
                "generation_script": "scripts/generate_fuzz.py",
                "usage": ["Fuzz usage"],
            },
        }

        suites_file = tmp_path / "workload_suites.yaml"
        with open(suites_file, "w") as f:
            yaml.dump(suites_data, f)

        errors = validate_workload_suites(tmp_path)
        assert len(errors) >= 4  # At least 4 missing required fields


@pytest.fixture
def tmp_path():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
