"""Tests for Phase 4 exception hierarchy."""

from __future__ import annotations

import pytest

from tp.phase4.exceptions import (
    Phase4ConfigError,
    Phase4Error,
    Phase4ExtractionError,
    Phase4InputError,
    Phase4IntegrityError,
    Phase4MerkleError,
    Phase4MetadataHashError,
    Phase4ProvenanceHashError,
    Phase4SchemaError,
)


class TestPhase4ExceptionHierarchy:
    """Tests for the Phase 4 exception class hierarchy."""

    def test_base_exception(self) -> None:
        """Phase4Error is the base exception."""
        exc = Phase4Error("test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "test error"

    def test_input_error_hierarchy(self) -> None:
        """Phase4InputError inherits correctly."""
        exc = Phase4InputError("input error")
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, ValueError)
        assert str(exc) == "input error"

    def test_schema_error_hierarchy(self) -> None:
        """Phase4SchemaError inherits correctly."""
        exc = Phase4SchemaError("schema error")
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, ValueError)

    def test_integrity_error_hierarchy(self) -> None:
        """Phase4IntegrityError inherits correctly."""
        exc = Phase4IntegrityError("integrity error")
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, ValueError)

    def test_metadata_hash_error_hierarchy(self) -> None:
        """Phase4MetadataHashError inherits from IntegrityError."""
        exc = Phase4MetadataHashError("metadata hash mismatch")
        assert isinstance(exc, Phase4IntegrityError)
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, ValueError)

    def test_provenance_hash_error_hierarchy(self) -> None:
        """Phase4ProvenanceHashError inherits from IntegrityError."""
        exc = Phase4ProvenanceHashError("provenance hash mismatch")
        assert isinstance(exc, Phase4IntegrityError)
        assert isinstance(exc, Phase4Error)

    def test_merkle_error_hierarchy(self) -> None:
        """Phase4MerkleError inherits from IntegrityError."""
        exc = Phase4MerkleError("merkle mismatch")
        assert isinstance(exc, Phase4IntegrityError)
        assert isinstance(exc, Phase4Error)

    def test_config_error_hierarchy(self) -> None:
        """Phase4ConfigError inherits correctly."""
        exc = Phase4ConfigError("config error")
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, ValueError)

    def test_extraction_error_hierarchy(self) -> None:
        """Phase4ExtractionError inherits correctly."""
        exc = Phase4ExtractionError("extraction failed")
        assert isinstance(exc, Phase4Error)
        assert isinstance(exc, RuntimeError)


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_all_phase4_errors(self) -> None:
        """Can catch all Phase 4 errors with base class."""
        exceptions_to_test = [
            Phase4InputError("test"),
            Phase4SchemaError("test"),
            Phase4IntegrityError("test"),
            Phase4MetadataHashError("test"),
            Phase4ProvenanceHashError("test"),
            Phase4MerkleError("test"),
            Phase4ConfigError("test"),
            Phase4ExtractionError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except Phase4Error:
                pass  # Should be caught
            else:
                pytest.fail(f"{type(exc).__name__} was not caught by Phase4Error")

    def test_catch_integrity_errors(self) -> None:
        """Can catch all integrity errors with base class."""
        integrity_exceptions = [
            Phase4MetadataHashError("test"),
            Phase4ProvenanceHashError("test"),
            Phase4MerkleError("test"),
        ]

        for exc in integrity_exceptions:
            try:
                raise exc
            except Phase4IntegrityError:
                pass  # Should be caught
            else:
                pytest.fail(f"{type(exc).__name__} was not caught by Phase4IntegrityError")

    def test_catch_value_errors(self) -> None:
        """Most Phase 4 errors can be caught as ValueError for compatibility."""
        value_error_exceptions = [
            Phase4InputError("test"),
            Phase4SchemaError("test"),
            Phase4IntegrityError("test"),
            Phase4ConfigError("test"),
        ]

        for exc in value_error_exceptions:
            try:
                raise exc
            except ValueError:
                pass  # Should be caught
            else:
                pytest.fail(f"{type(exc).__name__} was not caught by ValueError")

    def test_catch_runtime_error(self) -> None:
        """ExtractionError can be caught as RuntimeError."""
        try:
            raise Phase4ExtractionError("extraction failed")
        except RuntimeError:
            pass  # Should be caught
        except Exception:
            pytest.fail("Phase4ExtractionError was not caught by RuntimeError")
