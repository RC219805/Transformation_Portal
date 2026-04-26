"""Tests for offline APEX model-family characterization."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from transformation_portal.evals.apex_model_family import (
    DuplicateFamilyError,
    ObservationBindingError,
    build_apex_model_family_characterization_report,
    canonical_json_bytes,
    collect_family_specs,
    parse_family_spec,
    parse_matrix_spec,
)
from transformation_portal.evals.apex_model_family_schema import (
    ALLOWED_DEPTH_BACKENDS,
    ALLOWED_QUALITY_TIERS,
    ALLOWED_SEGMENTATION_BACKENDS,
    REPORT_SCHEMA_VERSION,
    FamilySpec,
    canonical_family_name,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID_SUMMARY = FIXTURES / "redacted_summary_valid.json"
INVALID_EXIF_SUMMARY = FIXTURES / "redacted_summary_with_exif_should_reject.json"
RAW_RUN_CARD = FIXTURES / "run_card_raw_should_reject.json"
RAW_BATCH = FIXTURES / "batch_raw_should_reject.json"

DA3_ESAM_SPEC = "depth_backend=da3,segmentation_backend=efficientsam,quality_tier=apex,pbr_enabled=false,v2_enabled=false"
DA3_SAM2_SPEC = "depth_backend=da3,segmentation_backend=sam2,quality_tier=apex,pbr_enabled=false,v2_enabled=false"
DEPTHPRO_SAM2_SPEC = "depth_backend=depth_pro,segmentation_backend=sam2,quality_tier=apex,pbr_enabled=false,v2_enabled=false"


def _load_tool_module(script_name: str, module_name: str):
    script_path = REPO_ROOT / "tools" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(tmp_path: Path, **kwargs):
    return build_apex_model_family_characterization_report(output_path=tmp_path / "report.json", **kwargs)


def test_canonical_name_round_trips_for_all_tier_values() -> None:
    for depth in sorted(ALLOWED_DEPTH_BACKENDS):
        for seg in sorted(ALLOWED_SEGMENTATION_BACKENDS):
            for tier in sorted(ALLOWED_QUALITY_TIERS):
                for pbr in (False, True):
                    for v2 in (False, True):
                        spec = FamilySpec(
                            depth_backend=depth,
                            segmentation_backend=seg,
                            quality_tier=tier,
                            pbr_enabled=pbr,
                            v2_enabled=v2,
                        )
                        name = canonical_family_name(spec)
                        parsed = parse_family_spec(
                            {
                                "candidate_family": name,
                                "depth_backend": depth,
                                "segmentation_backend": seg,
                                "quality_tier": tier,
                                "pbr_enabled": pbr,
                                "v2_enabled": v2,
                            }
                        )
                        assert parsed["candidate_family"] == name


def test_quality_tier_naming_default_materials_version_and_suffix_order() -> None:
    spec = parse_family_spec(
        "depth_backend=depth_pro,segmentation_backend=sam2,quality_tier=premium,pbr_enabled=true,v2_enabled=true"
    )

    assert spec["materials_version"] == 3
    assert spec["candidate_family"] == "materials_v3_depthpro_sam2_premium_pbr_v2"


def test_name_mismatch_becomes_spec_validation_status(tmp_path: Path) -> None:
    report = _build(
        tmp_path,
        family_specs=[
            "candidate_family=materials_v3_depthpro_sam2_v2_pbr,depth_backend=depth_pro,segmentation_backend=sam2,quality_tier=apex,pbr_enabled=true,v2_enabled=true"
        ],
    )

    row = report["families"][0]
    assert row["spec_validation"]["status"] == "name_mismatch"
    assert row["spec_validation"]["expected_candidate_family"] == "materials_v3_depthpro_sam2_apex_pbr_v2"
    assert row["comparison_blockers"] == ["spec_invalid", "observation_missing", "license_blocked"]


def test_matrix_expands_and_sorts_candidate_families(tmp_path: Path) -> None:
    specs = parse_matrix_spec(
        "depth_backend=da3,depth_pro;segmentation_backend=sam2,efficientsam;quality_tier=apex;pbr_enabled=false;v2_enabled=false"
    )
    report = _build(tmp_path, family_specs=specs)

    assert [row["family_spec"]["candidate_family"] for row in report["families"]] == [
        "materials_v3_da3_efficientsam_apex",
        "materials_v3_da3_sam2_apex",
        "materials_v3_depthpro_efficientsam_apex",
        "materials_v3_depthpro_sam2_apex",
    ]


def test_family_file_import_rejects_observations(tmp_path: Path) -> None:
    family_file = tmp_path / "matrix.json"
    family_file.write_text(
        json.dumps(
            {
                "schema_version": "apex_family_matrix.v1",
                "default_governance": {},
                "families": [],
                "observations": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported top-level"):
        collect_family_specs(family_files=[family_file])


def test_default_family_file_shape() -> None:
    specs, governance = collect_family_specs(family_files=[REPO_ROOT / "config" / "apex_family_matrix.json"])

    assert governance == {"non_commercial_ok": False, "accept_depth_pro_license": False}
    assert len(specs) == 4
    assert {spec["candidate_family"] for spec in specs} == {
        "materials_v3_da3_efficientsam_apex",
        "materials_v3_da3_sam2_apex",
        "materials_v3_depthpro_efficientsam_apex",
        "materials_v3_depthpro_sam2_apex",
    }


def test_observation_binding_duplicate_and_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(ObservationBindingError, match="unknown candidate_family"):
        _build(
            tmp_path,
            family_specs=[DA3_ESAM_SPEC],
            observations=["candidate_family=unknown_family,source=mock_v1,status=mocked"],
        )

    with pytest.raises(ObservationBindingError, match="Duplicate observation"):
        _build(
            tmp_path,
            family_specs=[DA3_ESAM_SPEC],
            observations=[
                "candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked",
                "candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked",
            ],
        )


def test_depth_pro_license_blocked_and_research_only(tmp_path: Path) -> None:
    blocked = _build(tmp_path, family_specs=[DEPTHPRO_SAM2_SPEC])
    assert blocked["families"][0]["governance"]["status"] == "license_blocked"
    assert blocked["families"][0]["comparable"] is False

    research = build_apex_model_family_characterization_report(
        family_specs=[DEPTHPRO_SAM2_SPEC],
        output_path=tmp_path / "research.json",
        non_commercial_ok=True,
        accept_depth_pro_license=True,
    )
    assert research["families"][0]["governance"]["status"] == "research_only"


def test_not_run_fallback_used_is_null(tmp_path: Path) -> None:
    report = _build(tmp_path, family_specs=[DA3_ESAM_SPEC])

    assert report["families"][0]["observation"]["status"] == "not_run"
    assert report["families"][0]["observation"]["fallback_used"] is None


def test_mock_observation_records_comparable_summary(tmp_path: Path) -> None:
    report = _build(
        tmp_path,
        family_specs=[DA3_ESAM_SPEC],
        observations=[
            "candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked,fallback_used=false,runtime_ms=1234,promotion_verdict=eligible,metric_contract=apex_metrics.v1,mask_evidence_status=ok"
        ],
    )

    row = report["families"][0]
    assert row["comparable"] is True
    assert row["comparison_blockers"] == []
    assert row["observation"]["fallback_used"] is False
    assert row["observation"]["runtime_ms"] == pytest.approx(1234.0)
    assert row["observation"]["promotion_verdict"] == "eligible"
    assert row["observation"]["metric_contract"] == "apex_metrics.v1"
    assert row["observation"]["mask_evidence_status"] == "ok"


def test_redacted_summary_required_keys_and_unknown_keys_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "summary.json"
    missing.write_text(json.dumps({"schema_version": "apex_redacted_summary.v1"}), encoding="utf-8")
    unknown = tmp_path / "unknown.json"
    payload = json.loads(VALID_SUMMARY.read_text(encoding="utf-8"))
    payload["unexpected"] = "value"
    unknown.write_text(json.dumps(payload), encoding="utf-8")

    for path in (missing, unknown):
        with pytest.raises(ValueError):
            _build(
                tmp_path,
                family_specs=[DA3_ESAM_SPEC],
                redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={path}"],
            )


def test_redacted_summary_rejects_path_like_values_and_raw_artifacts(tmp_path: Path) -> None:
    for value in (
        "/Users/example/output/file.tif",
        "/tmp/private/report.json",
        "/home/user/data.json",
        "./local/file.json",
        "../secrets.json",
    ):
        path_like = tmp_path / f"path_like_{hashlib.sha256(value.encode('utf-8')).hexdigest()}.json"
        payload = json.loads(VALID_SUMMARY.read_text(encoding="utf-8"))
        payload["promotion_verdict"] = value
        path_like.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="path-like"):
            _build(
                tmp_path,
                family_specs=[DA3_ESAM_SPEC],
                redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={path_like}"],
            )

    for raw_path in (RAW_RUN_CARD, RAW_BATCH):
        with pytest.raises(ValueError, match="raw artifact"):
            _build(
                tmp_path,
                family_specs=[DA3_ESAM_SPEC],
                redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={raw_path}"],
            )


def test_redacted_summary_with_exif_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported field"):
        _build(
            tmp_path,
            family_specs=[DA3_ESAM_SPEC],
            redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={INVALID_EXIF_SUMMARY}"],
        )


def test_redacted_summary_observed_local_and_digest_includes_summary_hash(tmp_path: Path) -> None:
    first = _build(
        tmp_path,
        family_specs=[DA3_ESAM_SPEC],
        redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={VALID_SUMMARY}"],
    )
    changed = tmp_path / "changed_summary.json"
    payload = json.loads(VALID_SUMMARY.read_text(encoding="utf-8"))
    payload["runtime_ms"] = 9999.0
    changed.write_text(json.dumps(payload), encoding="utf-8")
    second = build_apex_model_family_characterization_report(
        family_specs=[DA3_ESAM_SPEC],
        redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={changed}"],
        output_path=tmp_path / "second.json",
    )

    row = first["families"][0]
    assert row["observation"]["status"] == "observed_local"
    assert row["observation"]["source"] == "redacted_summary_v1"
    assert row["observation"]["evidence_ref"]["summary_schema_version"] == "apex_redacted_summary.v1"
    assert "summary_path" not in row["observation"]["evidence_ref"]
    assert first["input_digest"]["sha256"] != second["input_digest"]["sha256"]


def test_non_finite_runtime_is_rejected_for_mock_and_redacted_summary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        _build(
            tmp_path,
            family_specs=[DA3_ESAM_SPEC],
            observations=["candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked,runtime_ms=nan"],
        )

    non_finite = tmp_path / "non_finite_summary.json"
    payload = json.loads(VALID_SUMMARY.read_text(encoding="utf-8"))
    non_finite.write_text(
        json.dumps({**payload, "runtime_ms": float("inf")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        _build(
            tmp_path,
            family_specs=[DA3_ESAM_SPEC],
            redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={non_finite}"],
        )


def test_allow_observation_invalid_uses_sanitized_error_code(tmp_path: Path) -> None:
    report = build_apex_model_family_characterization_report(
        family_specs=[DA3_ESAM_SPEC],
        redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={RAW_RUN_CARD}"],
        output_path=tmp_path / "report.json",
        allow_observation_invalid=True,
    )

    observation = report["families"][0]["observation"]
    assert observation["status"] == "evidence_missing"
    assert observation["error_code"] == "raw_artifact_rejected"
    serialized = canonical_json_bytes(report).decode("utf-8")
    assert str(RAW_RUN_CARD) not in serialized
    assert "run_card_raw_should_reject.json" not in serialized


def test_input_digest_stable_under_key_reordering_and_excludes_now(tmp_path: Path) -> None:
    spec_a = {
        "depth_backend": "da3",
        "segmentation_backend": "efficientsam",
        "quality_tier": "apex",
        "pbr_enabled": False,
        "v2_enabled": False,
    }
    spec_b = {
        "v2_enabled": False,
        "pbr_enabled": False,
        "quality_tier": "apex",
        "segmentation_backend": "efficientsam",
        "depth_backend": "da3",
    }
    first = build_apex_model_family_characterization_report(
        family_specs=[spec_a],
        output_path=tmp_path / "first.json",
        created_at="2026-04-25T00:00:00Z",
    )
    second = build_apex_model_family_characterization_report(
        family_specs=[spec_b],
        output_path=tmp_path / "second.json",
        created_at="2026-04-26T00:00:00Z",
    )

    assert first["input_digest"]["sha256"] == second["input_digest"]["sha256"]


def test_comparison_groups_require_two_members_and_closed_reasons(tmp_path: Path) -> None:
    report = _build(
        tmp_path,
        family_specs=[DA3_ESAM_SPEC, DA3_SAM2_SPEC],
        observations=["candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked"],
    )

    groups = report["comparison_groups"]
    assert groups
    for group in groups:
        assert set(group["blocking_reasons"]) <= {"only_one_member", "member_not_comparable"}
    segmentation_groups = [group for group in groups if group["axis"] == "segmentation_axis"]
    assert any(len(group["members"]) >= 2 and not group["comparable"] for group in segmentation_groups)


def test_comparison_groups_keep_materials_version_fixed(tmp_path: Path) -> None:
    report = _build(
        tmp_path,
        family_specs=[
            {
                "depth_backend": "da3",
                "segmentation_backend": "efficientsam",
                "materials_version": 3,
                "quality_tier": "apex",
                "pbr_enabled": False,
                "v2_enabled": False,
            },
            {
                "depth_backend": "da3",
                "segmentation_backend": "efficientsam",
                "materials_version": 4,
                "quality_tier": "apex",
                "pbr_enabled": False,
                "v2_enabled": False,
            },
        ],
        observations=[
            "candidate_family=materials_v3_da3_efficientsam_apex,source=mock_v1,status=mocked",
            "candidate_family=materials_v4_da3_efficientsam_apex,source=mock_v1,status=mocked",
        ],
    )

    for group in report["comparison_groups"]:
        assert set(group["members"]) != {
            "materials_v3_da3_efficientsam_apex",
            "materials_v4_da3_efficientsam_apex",
        }


def test_self_check_reconciliation_counts(tmp_path: Path) -> None:
    report = _build(tmp_path, family_specs=[DA3_ESAM_SPEC, DEPTHPRO_SAM2_SPEC])

    assert report["self_check"] == {"status": "ok", "failures": []}
    summary = report["summary"]
    assert summary["family_count"] == 2
    assert summary["spec_valid_count"] + summary["spec_invalid_count"] == summary["family_count"]
    assert summary["not_run_count"] == 2
    assert summary["license_blocked_count"] == 1


def test_markdown_output_byte_stable_with_fixed_now(tmp_path: Path) -> None:
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    kwargs = {
        "family_specs": [DA3_SAM2_SPEC, DA3_ESAM_SPEC],
        "output_format": "markdown",
        "created_at": "2026-04-25T00:00:00Z",
    }
    build_apex_model_family_characterization_report(output_path=first_path, **kwargs)
    build_apex_model_family_characterization_report(output_path=second_path, **kwargs)

    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == hashlib.sha256(second_path.read_bytes()).hexdigest()
    assert first_path.read_text(encoding="utf-8").endswith("\n")


def test_now_format_rejects_offsets_and_fractional_seconds(tmp_path: Path) -> None:
    for bad_now in ("2026-04-25T00:00:00+00:00", "2026-04-25T00:00:00.000Z"):
        with pytest.raises(ValueError, match="UTC format"):
            build_apex_model_family_characterization_report(
                family_specs=[DA3_ESAM_SPEC],
                output_path=tmp_path / "report.json",
                created_at=bad_now,
            )


def test_cli_exit_codes_and_invalid_allow_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = _load_tool_module("characterize_apex_model_families.py", "characterize_apex_model_families_exit_unit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "characterize_apex_model_families.py",
            "--family",
            "candidate_family=bad_name,depth_backend=da3,segmentation_backend=efficientsam,quality_tier=apex",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert module.main() == 2
    assert "spec validation failed" in capsys.readouterr().err

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "characterize_apex_model_families.py",
            "--family",
            DA3_ESAM_SPEC,
            "--redacted-summary",
            f"candidate_family=materials_v3_da3_efficientsam_apex,path={RAW_RUN_CARD}",
            "--output",
            str(tmp_path / "report2.json"),
        ],
    )
    assert module.main() == 3

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "characterize_apex_model_families.py",
            "--family",
            DA3_ESAM_SPEC,
            "--redacted-summary",
            "candidate_family=missing_family,path=missing.json",
            "--output",
            str(tmp_path / "report3.json"),
            "--allow-observation-invalid",
            "on",
        ],
    )
    assert module.main() == 4


def test_exit_code_5_overrides_allow_invalid_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool_module("characterize_apex_model_families.py", "characterize_apex_model_families_self_check_unit")

    def fake_builder(**_kwargs):
        return {
            "families": [],
            "self_check": {"status": "failed", "failures": [{"invariant": "forced"}]},
        }

    monkeypatch.setattr(module, "build_apex_model_family_characterization_report", fake_builder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "characterize_apex_model_families.py",
            "--family",
            DA3_ESAM_SPEC,
            "--output",
            str(tmp_path / "report.json"),
            "--allow-spec-invalid",
            "on",
            "--allow-observation-invalid",
            "on",
        ],
    )

    assert module.main() == 5


def test_cli_family_file_command_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool_module("characterize_apex_model_families.py", "characterize_apex_model_families_file_unit")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "characterize_apex_model_families.py",
            "--family-file",
            str(REPO_ROOT / "config" / "apex_family_matrix.json"),
            "--non-commercial-ok",
            "off",
            "--accept-depth-pro-license",
            "off",
            "--output",
            str(output),
            "--now",
            "2026-04-25T00:00:00Z",
        ],
    )

    assert module.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["summary"]["family_count"] == 4


def test_static_ast_import_guard() -> None:
    targets = [
        REPO_ROOT / "src/transformation_portal/evals/apex_model_family.py",
        REPO_ROOT / "src/transformation_portal/evals/apex_model_family_schema.py",
        REPO_ROOT / "src/transformation_portal/evals/apex_redacted_summary_schema.py",
        REPO_ROOT / "tools/characterize_apex_model_families.py",
    ]
    allowed_project_imports = {
        "transformation_portal.evals.apex_model_family",
        "transformation_portal.evals.apex_model_family_schema",
        "transformation_portal.evals.apex_redacted_summary_schema",
        "transformation_portal.ingest.canonical_json",
    }
    forbidden_prefixes = (
        "transformation_portal.depth",
        "transformation_portal.segmentation",
        "transformation_portal.candidates",
        "transformation_portal.runners",
        "torch",
        "tensorflow",
        "onnxruntime",
        "transformers",
        "diffusers",
        "requests",
        "httpx",
        "urllib",
        "urllib3",
        "socket",
        "ssl",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "boto3",
        "google.cloud",
        "subprocess",
    )
    violations = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("transformation_portal.") and name not in allowed_project_imports:
                    violations.append(f"{path.name}: disallowed project import {name}")
                if name.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}: forbidden import {name}")
    assert not violations


def test_filesystem_guard_reads_only_declared_inputs_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = []
    original_open = Path.open

    def recording_open(self, *args, **kwargs):
        opened.append((Path(self), args[0] if args else kwargs.get("mode", "r")))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    output = tmp_path / "report.json"
    build_apex_model_family_characterization_report(
        family_specs=[DA3_ESAM_SPEC],
        redacted_summaries=[f"candidate_family=materials_v3_da3_efficientsam_apex,path={VALID_SUMMARY}"],
        output_path=output,
    )

    read_paths = {path for path, mode in opened if "r" in str(mode)}
    write_paths = {path for path, mode in opened if "w" in str(mode)}
    assert read_paths == {VALID_SUMMARY}
    assert write_paths == {output}


def test_canonical_json_bytes_is_key_order_stable() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == canonical_json_bytes({"a": 2, "b": 1})
