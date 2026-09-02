import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_lock_ownership.py"
SPEC = importlib.util.spec_from_file_location("check_lock_ownership", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
ownership = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ownership)


def _entry(*, target_id: str, status: str, allowed_contexts: list[str]) -> dict[str, object]:
    return {
        "target_id": target_id,
        "python_version": "3.11",
        "status": status,
        "allowed_contexts": allowed_contexts,
    }


def manifest_fixture() -> dict[str, dict[str, object]]:
    return {
        "all.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "base.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "dev.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "ci.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "security.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "tools-archive.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "da3-runtime-darwin-arm64.txt": _entry(
            target_id="da3-runtime-darwin-arm64",
            status="active",
            allowed_contexts=["local-darwin-arm64"],
        ),
        "ml-core-darwin-arm64.txt": _entry(
            target_id="darwin-arm64",
            status="active",
            allowed_contexts=["local-darwin-arm64"],
        ),
    }


def test_manifest_must_cover_every_governed_lock_exactly_once() -> None:
    manifest = manifest_fixture()
    manifest.pop("base.txt")
    manifest["unexpected.txt"] = _entry(target_id="unexpected", status="active", allowed_contexts=["ubuntu-x64-generic"])

    errors = ownership.validate_manifest_contract(manifest)

    assert "requirements/lock_ownership.yml must declare governed lock 'base.txt'" in errors
    assert "requirements/lock_ownership.yml declares unexpected lock 'unexpected.txt'" in errors


def test_load_lock_ownership_parses_manifest_without_pyyaml(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lock_ownership.yml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "locks:",
                "  all.txt:",
                "    target_id: generic",
                '    python_version: "3.11"',
                "    status: active",
                "    allowed_contexts:",
                "      - ubuntu-x64-generic",
                "  ml-core-darwin-arm64.txt:",
                "    target_id: darwin-arm64",
                '    python_version: "3.11"',
                "    status: active",
                "    allowed_contexts:",
                "      - local-darwin-arm64",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = ownership.load_lock_ownership(manifest_path)

    assert manifest == {
        "all.txt": _entry(target_id="generic", status="active", allowed_contexts=["ubuntu-x64-generic"]),
        "ml-core-darwin-arm64.txt": _entry(
            target_id="darwin-arm64",
            status="active",
            allowed_contexts=["local-darwin-arm64"],
        ),
    }


def test_load_lock_ownership_rejects_unsupported_indentation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lock_ownership.yml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "locks:",
                " all.txt:",
                "    target_id: generic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="two-space indentation"):
        ownership.load_lock_ownership(manifest_path)


def test_load_lock_ownership_rejects_duplicate_lock_entry(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lock_ownership.yml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "locks:",
                "  all.txt:",
                "    target_id: generic",
                '    python_version: "3.11"',
                "    status: active",
                "    allowed_contexts:",
                "      - ubuntu-x64-generic",
                "  all.txt:",
                "    target_id: generic",
                '    python_version: "3.11"',
                "    status: active",
                "    allowed_contexts:",
                "      - ubuntu-x64-generic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate lock entry 'all.txt'"):
        ownership.load_lock_ownership(manifest_path)


def test_load_lock_ownership_rejects_duplicate_field_key(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lock_ownership.yml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "locks:",
                "  all.txt:",
                "    target_id: generic",
                "    target_id: generic",
                '    python_version: "3.11"',
                "    status: active",
                "    allowed_contexts:",
                "      - ubuntu-x64-generic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate field 'target_id' for lock entry 'all.txt'"):
        ownership.load_lock_ownership(manifest_path)


def test_malformed_manifest_entry_reports_contract_errors_without_keyerror() -> None:
    manifest = manifest_fixture()
    manifest["ml-core-darwin-arm64.txt"] = {
        "target_id": "darwin-arm64",
        "python_version": "3.11",
        # Missing status and malformed allowed_contexts would previously trigger KeyError.
        "allowed_contexts": "local-darwin-arm64",
    }

    errors = ownership.validate_changed_files_against_context(
        manifest,
        changed_files=["requirements/ml-core-darwin-arm64.txt"],
        contexts=["local-darwin-arm64"],
    )

    assert errors == [
        "requirements/lock_ownership.yml entry 'ml-core-darwin-arm64.txt' must declare status as one of ['active', 'frozen']",
        "requirements/lock_ownership.yml entry 'ml-core-darwin-arm64.txt' must declare allowed_contexts as a list of strings",
    ]


def test_ubuntu_generic_context_accepts_generic_locks_only() -> None:
    manifest = manifest_fixture()

    assert (
        ownership.validate_changed_files_against_context(
            manifest,
            changed_files=["requirements/base.txt"],
            contexts=["ubuntu-x64-generic"],
        )
        == []
    )

    errors = ownership.validate_changed_files_against_context(
        manifest,
        changed_files=["requirements/ml-core-darwin-arm64.txt"],
        contexts=["ubuntu-x64-generic"],
    )

    assert errors == [
        "requirements/ml-core-darwin-arm64.txt is owned by contexts ['local-darwin-arm64']; current contexts "
        "['ubuntu-x64-generic'] are not authoritative"
    ]


def test_darwin_arm64_context_accepts_only_darwin_arm64_lock() -> None:
    manifest = manifest_fixture()

    assert (
        ownership.validate_changed_files_against_context(
            manifest,
            changed_files=["requirements/ml-core-darwin-arm64.txt"],
            contexts=["local-darwin-arm64"],
        )
        == []
    )

    errors = ownership.validate_changed_files_against_context(
        manifest,
        changed_files=["requirements/base.txt"],
        contexts=["local-darwin-arm64"],
    )

    assert errors == [
        "requirements/base.txt is owned by contexts ['ubuntu-x64-generic']; current contexts "
        "['local-darwin-arm64'] are not authoritative"
    ]
