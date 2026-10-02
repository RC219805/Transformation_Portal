"""Require complete commit-bound dependency evidence before any graph POST."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import dependency_snapshot as collector

pytestmark = pytest.mark.unit
SCANNED = "2026-10-02T12:00:00Z"


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    for path in collector.LOCK_FILES:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("urllib3==2.8.0\npypdf==6.19.0\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "requirements", "config"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test: lock inventory",
        ],
        cwd=tmp_path,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    return tmp_path, {"sha": sha, "ref": "refs/heads/main", "run_id": "123", "repository": "example/repository"}


def payload(repository: tuple[Path, dict[str, str]]) -> bytes:
    root, context = repository
    return collector.canonical_bytes(collector.build_snapshot(root, context, SCANNED))


def configure_context(monkeypatch: pytest.MonkeyPatch, repository: tuple[Path, dict[str, str]]) -> None:
    root, context = repository
    monkeypatch.setattr(collector, "REPO_ROOT", root)
    for name, value in context.items():
        monkeypatch.setenv(f"GITHUB_{name.upper()}", value)


def test_real_governed_inventory_is_complete_and_has_current_security_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    context = {"sha": sha, "ref": "refs/heads/main", "run_id": "1", "repository": "RC219805/Transformation_Portal"}
    snapshot = collector.build_snapshot(root, context, SCANNED)
    assert len(snapshot["manifests"]) == 10
    for path, manifest in snapshot["manifests"].items():
        assert manifest["resolved"], path
        assert all(set(package) == {"package_url"} for package in manifest["resolved"].values())
        assert "pkg:pypi/urllib3@2.7.0" not in manifest["resolved"]
    assert "pkg:pypi/urllib3@2.8.0" in snapshot["manifests"]["requirements/all.txt"]["resolved"]
    assert "pkg:pypi/pypdf@6.19.0" in snapshot["manifests"]["requirements/all.txt"]["resolved"]


def test_serialized_snapshot_round_trips_all_required_manifests(repository) -> None:
    root, context = repository
    data = payload(repository)
    snapshot = collector.validate_payload(data, root, context, hashlib.sha256(data).hexdigest())
    assert set(snapshot["manifests"]) == set(collector.LOCK_FILES)
    assert snapshot["detector"]["name"] == "Component Detection"
    assert snapshot["job"]["correlator"] == "submit-pypi"
    assert context["sha"] in snapshot["detector"]["url"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snapshot: snapshot.update(manifests={}),
        lambda snapshot: snapshot["manifests"].pop("requirements/base.txt"),
        lambda snapshot: snapshot["manifests"].update({"requirements/constraints.txt": {}}),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"].update(resolved={}),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"]["resolved"].pop("pkg:pypi/urllib3@2.8.0"),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"]["resolved"].update(
            {"pkg:pypi/urllib3@2.7.0": {"package_url": "pkg:pypi/urllib3@2.7.0"}}
        ),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"]["file"].update(source_location="base.txt"),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"]["metadata"].update(lock_sha256="0" * 64),
        lambda snapshot: snapshot["manifests"]["requirements/base.txt"]["resolved"]["pkg:pypi/pypdf@6.19.0"].update(
            scope="runtime"
        ),
        lambda snapshot: snapshot.update(sha="0" * 40),
        lambda snapshot: snapshot.update(ref="refs/heads/develop"),
        lambda snapshot: snapshot["job"].update(correlator="other"),
        lambda snapshot: snapshot["job"].update(id="124"),
        lambda snapshot: snapshot["detector"].update(name="Other Detector"),
        lambda snapshot: snapshot["detector"].update(version="0.0.1"),
        lambda snapshot: snapshot["detector"].update(url="https://example.invalid"),
    ],
)
def test_rejects_empty_incomplete_or_mislabeled_serialized_snapshot(repository, mutation) -> None:
    root, context = repository
    snapshot = json.loads(payload(repository))
    mutation(snapshot)
    with pytest.raises(collector.EvidenceError):
        collector.validate_payload(collector.canonical_bytes(snapshot), root, context)


def test_checkout_and_lock_bytes_are_bound_to_source_commit(repository) -> None:
    root, context = repository
    with pytest.raises(collector.EvidenceError, match="checkout HEAD"):
        collector.build_snapshot(root, {**context, "sha": "0" * 40}, SCANNED)
    (root / collector.LOCK_FILES[0]).write_text("urllib3==2.7.0\n", encoding="utf-8")
    with pytest.raises(collector.EvidenceError, match="working bytes differ"):
        collector.build_snapshot(root, context, SCANNED)


def test_rejects_payload_digest_and_duplicate_json_keys(repository) -> None:
    root, context = repository
    data = payload(repository)
    with pytest.raises(collector.EvidenceError, match="SHA256"):
        collector.validate_payload(data, root, context, "0" * 64)
    with pytest.raises(collector.EvidenceError, match="duplicate JSON key"):
        collector.validate_payload(b'{"version":0,"version":0}', root, context)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# comments only\n",
        "urllib3>=2.8.0",
        "urllib3===2.8.0",
        "urllib3==nonsense",
        "urllib3==2.8.0garbage",
        "urllib3 @ https://example.invalid/wheel",
        "-r other.txt",
        "--unknown-option true",
        "urllib3==2.8.0 trailing-token",
        "urllib3==2.8.0 --hash=md5:abc",
        "urllib3==2.8.0 ; unknown_marker == 'Linux'",
        "urllib3==2.8.0 ; platform_system == 'Linux' or unknown == 'x'",
        "urllib3==2.8.0 \\",
        "urllib3==2.8.0\nurllib3==2.7.0",
    ],
)
def test_unknown_or_empty_lock_syntax_fails_closed(text: str) -> None:
    with pytest.raises(collector.EvidenceError):
        collector.parse_pins(text.encode(), "fixture.txt")


def test_exact_pin_parser_handles_extras_hash_wraps_and_retains_marked_inventory() -> None:
    text = f"""# target inventory
--extra-index-url https://download.pytorch.org/whl/cpu
Foo_Bar[one,
 two]==1.2.3
Hashed_Package==1.0 \\
 --hash=sha256:{'a' * 64}
Darwin_Package==2.0 ; sys_platform == 'darwin'
"""
    assert collector.parse_pins(text.encode(), "fixture.txt") == {
        "pkg:pypi/foo-bar@1.2.3",
        "pkg:pypi/hashed-package@1.0",
        "pkg:pypi/darwin-package@2.0",
    }


@pytest.mark.parametrize(
    "version", ["1.2.3", "2.0rc1", "2.0a2", "2.0b3", "1.2.post4", "1.2.dev3", "1!1.2+cpu", "1.2.post1.dev2"]
)
def test_generated_canonical_pep440_version_forms_are_supported(version: str) -> None:
    purls = collector.parse_pins(f"example=={version}".encode(), "fixture.txt")
    escaped_version = version.replace("!", "%21").replace("+", "%2B")
    assert purls == {f"pkg:pypi/example@{escaped_version}"}


def test_bounded_reads_reject_symlinks_and_oversized_files_before_reading(tmp_path, monkeypatch) -> None:
    regular = tmp_path / "regular"
    regular.write_bytes(b"inventory")
    link = tmp_path / "link"
    link.symlink_to(regular)
    large = tmp_path / "large"
    large.write_bytes(b"x" * (collector.MAX_FILE_BYTES + 1))
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("invalid file was read"))
    for path in (link, large, tmp_path):
        with pytest.raises(collector.EvidenceError, match="bounded regular file"):
            collector.read_bounded(path)


@pytest.mark.parametrize(
    "attempts",
    [
        ["failure:", "success:true"],
        ["success:", "skipped:"],
        ["failure:true", "skipped:"],
        ["skipped:", "skipped:"],
        ["cancelled:", "skipped:"],
    ],
)
def test_guard_rejects_any_invalid_attempt_even_if_another_succeeds(attempts) -> None:
    with pytest.raises(collector.EvidenceError):
        collector.validate_attempts(attempts)


@pytest.mark.parametrize(
    "attempts", [["success:true", "skipped:"], ["failure:true", "success:true"], ["failure:true", "failure:true"]]
)
def test_valid_evidence_preserves_success_retry_and_advisory_network_policy(attempts) -> None:
    collector.validate_attempts(attempts)


def test_invalid_submit_never_posts_or_exports_valid_evidence(repository, monkeypatch, tmp_path) -> None:
    configure_context(monkeypatch, repository)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(payload(repository))
    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(collector, "submit_payload", lambda *_: pytest.fail("invalid evidence reached POST"))
    assert collector.main(["submit", "--snapshot", str(snapshot_path), "--digest", "0" * 64]) == 1
    assert not output_path.exists()


def test_submit_posts_the_same_validated_bytes_and_exports_evidence_before_network_failure(
    repository, monkeypatch, tmp_path
) -> None:
    configure_context(monkeypatch, repository)
    data = payload(repository)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(data)
    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    def fail_post(posted: bytes, context: dict) -> None:
        assert posted == data
        assert output_path.read_text() == "evidence_valid=true\n"
        raise subprocess.CalledProcessError(1, "gh api")

    monkeypatch.setattr(collector, "submit_payload", fail_post)
    assert collector.main(["submit", "--snapshot", str(snapshot_path), "--digest", hashlib.sha256(data).hexdigest()]) == 1


@pytest.mark.parametrize(
    "receipt",
    [
        {"id": 7, "result": "SUCCESS"},
        {"id": 7, "result": "FAILED"},
        {"result": "SUCCESS"},
        {"id": True, "result": "SUCCESS"},
        {"id": 0, "result": "SUCCESS"},
        {"id": -1, "result": "SUCCESS"},
        [],
    ],
)
def test_network_success_requires_an_accepted_api_receipt(monkeypatch, receipt) -> None:
    data = b"exact validated payload"

    def run(command, **kwargs):
        assert command[-2:] == ["--input", "-"]
        assert kwargs["input"] == data
        assert kwargs["check"] is True
        return SimpleNamespace(stdout=json.dumps(receipt).encode())

    monkeypatch.setattr(collector.subprocess, "run", run)
    if receipt == {"id": 7, "result": "SUCCESS"}:
        collector.submit_payload(data, {"repository": "example/repository"})
    else:
        with pytest.raises(collector.EvidenceError):
            collector.submit_payload(data, {"repository": "example/repository"})
