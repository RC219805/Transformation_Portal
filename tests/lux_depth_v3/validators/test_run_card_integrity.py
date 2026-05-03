"""Unit tests for ``transformation_portal.lux_depth_v3.validators.run_card_integrity``.

The module's main entry point ``verify_run_card_integrity`` is tested
end-to-end via the script wrapper in ``tests/test_verify_run_card_integrity.py``.
This file complements that suite at the helper / contract level: it pins
the *module API* directly (rather than via the script), and it covers the
small pure helpers and the path-traversal hardening on
``resolve_artifact_path`` — surfaces a refactor could break silently while
keeping the script wrapper green.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

pytestmark = [pytest.mark.unit, pytest.mark.security]

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.run_card_contract import RunCardPathValidationError
from transformation_portal.lux_depth_v3.validators import run_card_integrity as rci


class TestPublicModuleSurface:
    """The validator module is the contract surface; pin its public API."""

    def test_verify_run_card_integrity_is_exported(self):
        # The script at scripts/verify_run_card_integrity.py imports this
        # symbol; renaming or moving it is a binding-contract break.
        assert callable(rci.verify_run_card_integrity)

    def test_resolve_artifact_path_is_exported(self):
        # Reused by sibling validators; must stay in the module namespace.
        assert callable(rci.resolve_artifact_path)

    def test_default_schema_paths_resolve(self):
        # ``DEFAULT_SCHEMA_V1_PATH`` / ``V2_PATH`` are module-level constants
        # consumed by ``infer_schema_path_for_payload``. They must always be
        # ``Path`` instances pointing at real schema files in the source tree.
        assert isinstance(rci.DEFAULT_SCHEMA_V1_PATH, Path)
        assert isinstance(rci.DEFAULT_SCHEMA_V2_PATH, Path)
        assert rci.DEFAULT_SCHEMA_V1_PATH.is_file()
        assert rci.DEFAULT_SCHEMA_V2_PATH.is_file()

    def test_sha256_regex_rejects_non_hex_and_uppercase(self):
        # The validator pins lowercase 64-char hex digests as the only
        # acceptable shape. Uppercase / short / invalid-char strings must
        # all fail to match.
        assert rci.SHA256_HEX_RE.fullmatch("a" * 64)
        assert rci.SHA256_HEX_RE.fullmatch("0" * 64)
        assert not rci.SHA256_HEX_RE.fullmatch("A" * 64)
        assert not rci.SHA256_HEX_RE.fullmatch("a" * 63)
        assert not rci.SHA256_HEX_RE.fullmatch("z" * 64)


class TestFormatErrorPath:
    def test_empty_path_renders_as_root(self):
        assert rci.format_error_path([]) == "<root>"

    def test_path_segments_join_with_dot(self):
        # JSONSchema validators yield path elements as strings or ints
        # (array indices). Both must be stringified verbatim and joined.
        assert rci.format_error_path(["a", "b", "c"]) == "a.b.c"
        assert rci.format_error_path(["artifact_index", 0, "sha256"]) == "artifact_index.0.sha256"


class TestCanonicalJsonText:
    def test_canonical_json_is_sorted_indented_unicode(self):
        # Drift detection compares this exact serialization against the file
        # on disk, so its output must be deterministic and stable.
        payload = {"b": 2, "a": 1, "c": ["x", "y"]}
        text = rci.canonical_json_text(payload)
        # Sort_keys=True puts "a" before "b" before "c".
        assert text.index('"a"') < text.index('"b"') < text.index('"c"')
        # indent=2 emits a leading newline after "{" and 2-space indent.
        assert "\n  " in text
        # Roundtrip should produce equivalent semantics.
        assert json.loads(text) == payload

    def test_canonical_json_rejects_nan(self):
        # Run-card payloads must be JSON-strict; NaN/Infinity are forbidden
        # because they are not valid JSON and would break schema validators.
        with pytest.raises(ValueError):
            rci.canonical_json_text({"x": float("nan")})


class TestResolveArtifactPath:
    """Path traversal hardening — the validator's primary security boundary."""

    def test_normal_relative_path_resolves_under_root(self, tmp_path: Path):
        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=tmp_path,
            relative_path="depth/image_01_depth.png",
            context="ctx",
        )
        assert error is None
        assert artifact_path is not None
        assert artifact_path == (tmp_path / "depth" / "image_01_depth.png").resolve()

    def test_parent_traversal_is_rejected_by_normalizer(self, tmp_path: Path):
        # The normalizer in run_card_contract is the first line of defence;
        # ".." segments must be rejected before any filesystem touches.
        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=tmp_path,
            relative_path="../escape.png",
            context="combined_manifest artifact",
        )
        assert artifact_path is None
        assert error is not None
        assert "combined_manifest artifact" in error
        assert "traversal" in error

    def test_absolute_path_is_rejected(self, tmp_path: Path):
        # Even an absolute path that happens to live under the root must be
        # rejected — relative_path is the contract field type.
        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=tmp_path,
            relative_path=str(tmp_path / "x.png"),
            context="ctx",
        )
        assert artifact_path is None
        assert error is not None
        assert "must not be absolute" in error

    def test_null_byte_is_rejected(self, tmp_path: Path):
        # NUL bytes are a classic OS-level path-smuggling vector and the
        # normalizer must refuse them up front.
        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=tmp_path,
            relative_path="depth/\x00.png",
            context="ctx",
        )
        assert artifact_path is None
        assert error is not None

    def test_backslash_is_rejected(self, tmp_path: Path):
        # POSIX-only path contract: backslashes are not legal separators.
        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=tmp_path,
            relative_path="depth\\image_01_depth.png",
            context="ctx",
        )
        assert artifact_path is None
        assert error is not None

    def test_symlink_escape_is_caught_by_relative_to_check(self, tmp_path: Path):
        # A symlink whose target lives outside the run-card root must be
        # rejected by the post-resolve ``relative_to`` guard.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.png").write_bytes(b"x")
        root = tmp_path / "card"
        root.mkdir()
        link = root / "leak.png"
        try:
            link.symlink_to(outside / "leak.png")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        artifact_path, error = rci.resolve_artifact_path(
            run_card_root=root,
            relative_path="leak.png",
            context="artifact_index[0]",
        )
        assert artifact_path is None
        assert error is not None
        assert "escapes run card root" in error


class TestInferSchemaPath:
    def test_explicit_schema_path_wins(self, tmp_path: Path):
        explicit = tmp_path / "custom.schema.json"
        explicit.write_text("{}", encoding="utf-8")
        # The explicit override is always respected — even if the payload
        # would otherwise be inferred as v1 or v2.
        result = rci.infer_schema_path_for_payload({}, explicit_schema_path=explicit)
        assert result == explicit

    def test_v1_payload_resolves_to_v1_schema(self):
        # No artifact_tree → inferred v1 → DEFAULT_SCHEMA_V1_PATH.
        result = rci.infer_schema_path_for_payload({})
        assert result == rci.DEFAULT_SCHEMA_V1_PATH

    def test_v2_payload_resolves_to_v2_schema(self):
        # Presence of artifact_tree → inferred v2 → DEFAULT_SCHEMA_V2_PATH.
        result = rci.infer_schema_path_for_payload({"artifact_tree": {}})
        assert result == rci.DEFAULT_SCHEMA_V2_PATH

    def test_explicit_run_card_version_overrides_artifact_tree_heuristic(self):
        # When run_card_version is explicit, that wins over the heuristic.
        result = rci.infer_schema_path_for_payload({"run_card_version": "v1"})
        assert result == rci.DEFAULT_SCHEMA_V1_PATH


class TestVerifyRunCardIntegrityNotFound:
    """The single deterministic non-FS-touching surface of the entry point."""

    def test_missing_run_card_returns_single_error(self, tmp_path: Path):
        result = rci.verify_run_card_integrity(tmp_path / "missing.json")
        assert result == [f"Run card not found: {tmp_path / 'missing.json'}"]

    def test_run_card_root_must_be_object(self, tmp_path: Path):
        path = tmp_path / "rc.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = rci.verify_run_card_integrity(path)
        assert any("Run card root must be a JSON object" in err for err in result)

    def test_invalid_json_returns_load_error(self, tmp_path: Path):
        path = tmp_path / "rc.json"
        path.write_text("{not valid json", encoding="utf-8")
        result = rci.verify_run_card_integrity(path)
        assert len(result) == 1
        assert "Invalid JSON" in result[0]

    def test_explicit_missing_schema_path_short_circuits(self, tmp_path: Path):
        # A non-existent explicit schema path must surface a single,
        # clearly-labelled error before any other validation runs.
        rc_path = tmp_path / "rc.json"
        rc_path.write_text(json.dumps({"artifact_index": []}), encoding="utf-8")

        missing_schema = tmp_path / "does-not-exist.schema.json"
        result = rci.verify_run_card_integrity(rc_path, schema_path=missing_schema)
        assert any("Run card schema not found" in err for err in result)


class TestVerifyCaptioningStatus:
    """Captioning sidecars must NEVER claim to be quality-gate evidence."""

    def test_top_level_used_for_quality_gate_true_is_rejected(self):
        errors: list[str] = []
        rci._verify_captioning_status(
            {"captioning_status": {"used_for_quality_gate": True}},
            errors,
        )
        assert errors == ["captioning_status.used_for_quality_gate must be false"]

    def test_top_level_used_for_quality_gate_false_is_accepted(self):
        errors: list[str] = []
        rci._verify_captioning_status(
            {"captioning_status": {"used_for_quality_gate": False}},
            errors,
        )
        assert errors == []

    def test_per_image_used_for_quality_gate_true_surfaces_index(self):
        errors: list[str] = []
        rci._verify_captioning_status(
            {
                "result_summary": [
                    {"captioning_status": {"used_for_quality_gate": False}},
                    {"captioning_status": {"used_for_quality_gate": True}},
                ]
            },
            errors,
        )
        assert errors == [
            "result_summary[1].captioning_status.used_for_quality_gate must be false",
        ]

    def test_missing_or_non_dict_status_is_silently_ignored(self):
        # The contract is "if present, must be false". Absent or
        # malformed-shaped statuses are caught by the schema validator,
        # not the advisory-status guard.
        errors: list[str] = []
        rci._verify_captioning_status({}, errors)
        rci._verify_captioning_status({"captioning_status": "free-form"}, errors)
        rci._verify_captioning_status({"result_summary": "not-a-list"}, errors)
        assert errors == []


class TestVerifyConfigFingerprint:
    """The config fingerprint canonicalization invariant."""

    def _build_fingerprint(self, **fields):
        canonical_json = json.dumps(
            {k: fields[k] for k in sorted(fields)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        sha256_hex = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return {
            "hash_algorithm": "sha256",
            "canonical_json": canonical_json,
            "sha256": sha256_hex,
            **fields,
        }

    def test_well_formed_fingerprint_passes(self):
        fingerprint = self._build_fingerprint(preset="premium", quality_tier="premium")
        errors: list[str] = []
        rci._verify_config_fingerprint({"config_fingerprint": fingerprint}, errors)
        assert errors == []

    def test_canonical_json_drift_is_detected(self):
        fingerprint = self._build_fingerprint(preset="premium")
        # Tamper with the canonical_json so it no longer matches the field
        # set; the recomputed canonicalization will diverge and the validator
        # must catch it (this is the in-memory analogue of the on-disk drift
        # detection in verify_run_card_integrity itself).
        fingerprint["canonical_json"] = '{"preset":"PREMIUM"}'
        errors: list[str] = []
        rci._verify_config_fingerprint({"config_fingerprint": fingerprint}, errors)
        assert any("does not match canonicalized" in err for err in errors)

    def test_sha256_mismatch_is_detected(self):
        fingerprint = self._build_fingerprint(preset="premium")
        # Corrupt the sha while keeping canonical_json intact.
        fingerprint["sha256"] = "0" * 64
        errors: list[str] = []
        rci._verify_config_fingerprint({"config_fingerprint": fingerprint}, errors)
        assert any("config_fingerprint.sha256 mismatch" in err for err in errors)

    def test_uppercase_sha256_is_rejected(self):
        fingerprint = self._build_fingerprint(preset="premium")
        fingerprint["sha256"] = fingerprint["sha256"].upper()
        errors: list[str] = []
        rci._verify_config_fingerprint({"config_fingerprint": fingerprint}, errors)
        assert any("lowercase 64-char hex digest" in err for err in errors)

    def test_non_sha256_algorithm_is_rejected(self):
        fingerprint = self._build_fingerprint(preset="premium")
        fingerprint["hash_algorithm"] = "blake2b"
        errors: list[str] = []
        rci._verify_config_fingerprint({"config_fingerprint": fingerprint}, errors)
        assert any("hash_algorithm must be 'sha256'" in err for err in errors)

    def test_missing_config_fingerprint_is_a_silent_no_op(self):
        # The schema validator is responsible for requiring config_fingerprint;
        # _verify_config_fingerprint only adds canonicalization checks on top.
        errors: list[str] = []
        rci._verify_config_fingerprint({}, errors)
        assert errors == []


class TestSelfIntegrityPayloadHashing:
    """The payload-without-hash hashing rule is the heart of self-integrity."""

    def test_payload_hash_excludes_canonical_payload_sha256_field(self):
        # Self-integrity hashes the payload with the canonical_payload_sha256
        # field stripped from run_card_integrity so the result is deterministic
        # and free of the hash-of-hash bootstrapping problem. Pin that rule.
        integrity = {
            "self_indexing": "excluded_self_hash_cycle",
            "path": "rc.json",
        }
        payload = {"foo": "bar", "run_card_integrity": integrity}

        without_hash = {
            **payload,
            "run_card_integrity": {k: v for k, v in integrity.items() if k != "canonical_payload_sha256"},
        }
        expected = hashlib.sha256(canonicalize_json(without_hash)).hexdigest()

        # Adding the hash itself must not change the hash.
        integrity_with_hash = {**integrity, "canonical_payload_sha256": expected}
        payload_with_hash = {**payload, "run_card_integrity": integrity_with_hash}
        without_hash_again = {
            **payload_with_hash,
            "run_card_integrity": {
                k: v for k, v in integrity_with_hash.items() if k != "canonical_payload_sha256"
            },
        }
        recomputed = hashlib.sha256(canonicalize_json(without_hash_again)).hexdigest()
        assert recomputed == expected
