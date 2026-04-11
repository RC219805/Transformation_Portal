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
        "ml-core-linux.txt": _entry(target_id="linux-x86_64", status="active", allowed_contexts=["ubuntu-x64-linux"]),
        "ml-core-darwin-arm64.txt": _entry(
            target_id="darwin-arm64",
            status="active",
            allowed_contexts=["local-darwin-arm64"],
        ),
        "ml-core-darwin-x86_64.txt": _entry(target_id="darwin-x86_64", status="frozen", allowed_contexts=[]),
    }


def test_manifest_must_cover_every_governed_lock_exactly_once() -> None:
    manifest = manifest_fixture()
    manifest.pop("base.txt")
    manifest["unexpected.txt"] = _entry(target_id="unexpected", status="active", allowed_contexts=["ubuntu-x64-generic"])

    errors = ownership.validate_manifest_contract(manifest)

    assert "requirements/lock_ownership.yml must declare governed lock 'base.txt'" in errors
    assert "requirements/lock_ownership.yml declares unexpected lock 'unexpected.txt'" in errors


def test_ubuntu_generic_linux_lane_accepts_only_generic_and_linux_locks() -> None:
    manifest = manifest_fixture()

    assert (
        ownership.validate_changed_files_against_context(
            manifest,
            changed_files=["requirements/base.txt", "requirements/ml-core-linux.txt"],
            contexts=["ubuntu-x64-generic", "ubuntu-x64-linux"],
        )
        == []
    )

    errors = ownership.validate_changed_files_against_context(
        manifest,
        changed_files=["requirements/ml-core-darwin-arm64.txt"],
        contexts=["ubuntu-x64-generic", "ubuntu-x64-linux"],
    )

    assert errors == [
        "requirements/ml-core-darwin-arm64.txt is owned by contexts ['local-darwin-arm64']; current contexts "
        "['ubuntu-x64-generic', 'ubuntu-x64-linux'] are not authoritative"
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
        changed_files=["requirements/ml-core-linux.txt"],
        contexts=["local-darwin-arm64"],
    )

    assert errors == [
        "requirements/ml-core-linux.txt is owned by contexts ['ubuntu-x64-linux']; current contexts "
        "['local-darwin-arm64'] are not authoritative"
    ]


def test_frozen_darwin_x86_64_lock_is_always_rejected() -> None:
    manifest = manifest_fixture()

    errors = ownership.validate_changed_files_against_context(
        manifest,
        changed_files=["requirements/ml-core-darwin-x86_64.txt"],
        contexts=["local-darwin-arm64"],
    )

    assert errors == [
        "requirements/ml-core-darwin-x86_64.txt is frozen for target 'darwin-x86_64'; off-lane regeneration is not permitted"
    ]
