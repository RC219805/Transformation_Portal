"""Tests for Phase 4 exception hierarchy."""

from __future__ import annotations

import pytest

from tp.phase4.canonicalize_capture_metadata import ConfigValidationError as CaptureConfigValidationError
from tp.phase4.canonicalize_capture_metadata import ExtractionFailure as CaptureExtractionFailure
from tp.phase4.canonicalize_capture_metadata import SchemaValidationError as CaptureSchemaValidationError
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
from tp.phase4.hash_capture_metadata import MetadataManifestInputError as HashMetadataManifestInputError
from tp.phase4.hash_capture_metadata import MetadataSchemaValidationError as HashMetadataSchemaValidationError
from tp.phase4.provenance_capture import ProvenanceInputError as CaptureProvenanceInputError
from tp.phase4.provenance_capture import ProvenanceSchemaValidationError as CaptureProvenanceSchemaValidationError
from tp.phase4.verify_phase4_chain import Phase4AlignmentError as VerifyAlignmentError
from tp.phase4.verify_phase4_chain import Phase4SchemaValidationError as VerifySchemaValidationError

pytestmark = pytest.mark.unit


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
        caught = False
        try:
            raise Phase4ExtractionError("extraction failed")
        except RuntimeError:
            caught = True

        if not caught:
            pytest.fail("Phase4ExtractionError was not caught by RuntimeError")


class TestLegacyExceptionBindings:
    """Tests that legacy module exception names bind to the unified hierarchy."""

    def test_legacy_module_exceptions_inherit_from_phase4_error(self) -> None:
        legacy_exception_types = [
            CaptureConfigValidationError,
            CaptureExtractionFailure,
            CaptureSchemaValidationError,
            HashMetadataManifestInputError,
            HashMetadataSchemaValidationError,
            CaptureProvenanceInputError,
            CaptureProvenanceSchemaValidationError,
            VerifyAlignmentError,
            VerifySchemaValidationError,
        ]

        for exception_type in legacy_exception_types:
            assert issubclass(exception_type, Phase4Error)

    def test_legacy_module_exceptions_preserve_compatibility_bases(self) -> None:
        assert issubclass(CaptureConfigValidationError, ValueError)
        assert issubclass(CaptureExtractionFailure, RuntimeError)
        assert issubclass(CaptureSchemaValidationError, RuntimeError)
        assert issubclass(HashMetadataManifestInputError, ValueError)
        assert issubclass(CaptureProvenanceInputError, ValueError)
        assert issubclass(VerifyAlignmentError, ValueError)

    def test_legacy_module_exceptions_can_be_caught_as_phase4_error(self) -> None:
        legacy_exceptions = [
            CaptureConfigValidationError("config error"),
            CaptureExtractionFailure("extraction error"),
            HashMetadataManifestInputError("manifest error"),
            CaptureProvenanceInputError("provenance error"),
            VerifyAlignmentError("alignment error"),
            VerifySchemaValidationError("schema error"),
        ]

        for exc in legacy_exceptions:
            try:
                raise exc
            except Phase4Error:
                pass
            else:
                pytest.fail(f"{type(exc).__name__} was not caught by Phase4Error")
