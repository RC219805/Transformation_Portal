"""Security regression tests for pipeline path safety.

This module validates that the DAG editor API correctly rejects
path traversal attacks and unsafe pipeline names.
"""

from __future__ import annotations

import pytest


@pytest.mark.security
class TestPipelineNameValidation:
    """Tests for _validate_pipeline_name function."""

    @pytest.fixture(autouse=True)
    def _setup_fastapi(self):
        """Skip tests if FastAPI is not available."""
        pytest.importorskip("fastapi")

    @pytest.mark.parametrize(
        "bad_name",
        [
            "",  # Empty
            ".",  # Current directory
            "..",  # Parent directory
            "../evil",  # Path traversal
            "a/b",  # Forward slash
            "a\\b",  # Backslash
            "a..b",  # Double dot in middle
            "a b",  # Space
            "a$b",  # Special character
            "🔥",  # Unicode emoji
            "a\x00b",  # Null byte
            "verylong" * 20,  # Exceeds 64 char limit
            ".hidden",  # Dot prefix
            "file.json",  # Dot in name (not allowed in strict pattern)
            "name with spaces",  # Multiple spaces
            "<script>",  # XSS attempt
        ],
    )
    def test_pipeline_name_rejected(self, bad_name: str) -> None:
        """Invalid pipeline names must be rejected."""
        from fastapi import HTTPException

        from transformation_portal.dashboard.dag_editor_api import (
            _validate_pipeline_name,
        )

        with pytest.raises(HTTPException) as exc_info:
            _validate_pipeline_name(bad_name)
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize(
        "valid_name",
        [
            "valid_name-123",
            "my-pipeline",
            "test_pipeline_v2",
            "UPPERCASE",
            "mixedCase123",
            "a",  # Single char
            "a" * 64,  # Max length
            "pipeline-2024-03-15",
        ],
    )
    def test_pipeline_name_valid(self, valid_name: str) -> None:
        """Valid pipeline names must be accepted."""
        from transformation_portal.dashboard.dag_editor_api import (
            _validate_pipeline_name,
        )

        result = _validate_pipeline_name(valid_name)
        assert result == valid_name

    def test_safe_path_construction(self, tmp_path) -> None:
        """_get_safe_pipeline_path constructs paths safely."""
        from transformation_portal.dashboard import dag_editor_api

        # Temporarily set pipelines dir
        original_dir = dag_editor_api._pipelines_dir
        dag_editor_api._pipelines_dir = tmp_path

        try:
            path = dag_editor_api._get_safe_pipeline_path("valid-name")
            assert path == tmp_path / "valid-name.json"
            # Path should not escape the directory
            assert str(tmp_path) in str(path)
        finally:
            dag_editor_api._pipelines_dir = original_dir

    def test_path_traversal_prevented(self, tmp_path) -> None:
        """Path traversal attempts must be blocked."""
        from fastapi import HTTPException

        from transformation_portal.dashboard import dag_editor_api

        original_dir = dag_editor_api._pipelines_dir
        dag_editor_api._pipelines_dir = tmp_path

        try:
            with pytest.raises(HTTPException) as exc_info:
                dag_editor_api._get_safe_pipeline_path("../../../etc/passwd")
            assert exc_info.value.status_code == 400
        finally:
            dag_editor_api._pipelines_dir = original_dir
