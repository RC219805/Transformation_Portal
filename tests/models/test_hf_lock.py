"""Tests for HF model lock record and file resolution utilities."""

from __future__ import annotations

import pytest

from transformation_portal.models.hf_lock import (
    HFModelLockError,
    HFModelLockRecord,
    HFRequiredFile,
)


class TestHFRequiredFile:
    """Tests for HFRequiredFile dataclass."""

    def test_basic_construction(self) -> None:
        """HFRequiredFile should store path and optional verification fields."""
        file = HFRequiredFile(path="config.json")
        assert file.path == "config.json"
        assert file.sha256 is None
        assert file.filesize_bytes is None

    def test_with_verification(self) -> None:
        """HFRequiredFile should store sha256 and filesize when provided."""
        file = HFRequiredFile(
            path="model.safetensors",
            sha256="abc123" * 10 + "abcd",
            filesize_bytes=1024,
        )
        assert file.path == "model.safetensors"
        assert file.sha256 is not None
        assert file.filesize_bytes == 1024


class TestHFModelLockRecord:
    """Tests for HFModelLockRecord dataclass and from_mapping."""

    def test_from_mapping_minimal(self) -> None:
        """HFModelLockRecord should parse minimal payload."""
        payload = {
            "repo_id": "llava-hf/llava-v1.6-mistral-7b-hf",
            "revision": "abc123" * 6 + "abcd",
        }
        record = HFModelLockRecord.from_mapping(payload)
        assert record.repo_id == "llava-hf/llava-v1.6-mistral-7b-hf"
        assert record.revision == "abc123" * 6 + "abcd"
        assert record.repo_type == "model"
        assert record.provider == "huggingface"
        assert record.required_files == []

    def test_from_mapping_full(self) -> None:
        """HFModelLockRecord should parse full payload with all fields."""
        payload = {
            "repo_id": "llava-hf/llava-v1.6-mistral-7b-hf",
            "revision": "abc123" * 6 + "abcd",
            "repo_type": "model",
            "provider": "huggingface",
            "license": "Apache-2.0",
            "owner": "evals/vision_language",
            "tier": "quality_validation_primary",
            "required_files": [
                "config.json",
                {"path": "model.safetensors", "sha256": "def456" * 10 + "defg"},
            ],
        }
        record = HFModelLockRecord.from_mapping(payload)
        assert record.license == "Apache-2.0"
        assert record.owner == "evals/vision_language"
        assert record.tier == "quality_validation_primary"
        assert len(record.required_files) == 2
        assert record.required_files[0].path == "config.json"
        assert record.required_files[1].sha256 is not None

    def test_from_mapping_missing_repo_id(self) -> None:
        """HFModelLockRecord should raise for missing repo_id."""
        with pytest.raises(HFModelLockError, match="repo_id"):
            HFModelLockRecord.from_mapping({"revision": "abc123"})

    def test_from_mapping_missing_revision(self) -> None:
        """HFModelLockRecord should raise for missing revision."""
        with pytest.raises(HFModelLockError, match="revision"):
            HFModelLockRecord.from_mapping({"repo_id": "org/repo"})


class TestRecordImmutability:
    """Tests for HFModelLockRecord immutability."""

    def test_record_is_frozen(self) -> None:
        """HFModelLockRecord should be frozen (immutable)."""
        record = HFModelLockRecord(
            repo_id="org/repo",
            revision="abc123" * 6 + "abcd",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            record.repo_id = "other/repo"  # type: ignore

    def test_required_file_is_frozen(self) -> None:
        """HFRequiredFile should be frozen (immutable)."""
        file = HFRequiredFile(path="config.json")
        with pytest.raises(Exception):  # FrozenInstanceError
            file.path = "other.json"  # type: ignore
