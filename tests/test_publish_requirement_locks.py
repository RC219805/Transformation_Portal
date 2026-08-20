from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "utilities" / "publish_requirement_locks.py"
SPEC = importlib.util.spec_from_file_location("publish_requirement_locks", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def _write_lock_set(directory: Path, version: str) -> dict[str, bytes]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, bytes] = {}
    for name in publisher.GENERIC_LOCK_FILES:
        content = f"# generated candidate\nexample-{name.removesuffix('.txt')}=={version}\n".encode()
        (directory / name).write_bytes(content)
        written[name] = content
    return written


def test_publication_scope_is_the_exact_generic_lock_set() -> None:
    assert publisher.GENERIC_LOCK_FILES == (
        "all.txt",
        "base.txt",
        "dev.txt",
        "ci.txt",
        "security.txt",
        "tools-archive.txt",
    )


def test_publish_api_rejects_a_partial_generic_lock_set(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")

    with pytest.raises(publisher.LockPublicationError, match="exact governed generic lock set"):
        publisher.publish_generic_locks(
            staging,
            destination,
            lock_files=publisher.GENERIC_LOCK_FILES[:-1],
        )


def test_publish_generic_locks_successfully_replaces_complete_set(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    expected = _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")

    publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == expected
    assert not list(destination.glob(".generic-lock-publish-*"))


@pytest.mark.security
def test_preparation_command_runs_while_destination_lock_is_held(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    expected = _write_lock_set(tmp_path / "expected", "2.0")
    _write_lock_set(destination, "1.0")
    lock_path = publisher.destination_lock_path(destination)
    command = (
        sys.executable,
        "-c",
        (
            "import fcntl, os, sys\n"
            "from pathlib import Path\n"
            "handle = open(sys.argv[1], 'a+')\n"
            "try:\n"
            "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "except BlockingIOError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('destination lock was not held during preparation')\n"
            f"names = {publisher.GENERIC_LOCK_FILES!r}\n"
            f"contents = {expected!r}\n"
            f"staging = Path(os.environ[{publisher.PREPARATION_STAGING_ENV!r}])\n"
            "for name in names:\n"
            "    (staging / name).write_bytes(contents[name])\n"
        ),
        str(lock_path),
    )

    publisher.prepare_and_publish_generic_locks(destination, command)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == expected
    assert not list(destination.glob(f"{publisher.TRANSACTION_PREFIX}*"))


def test_clean_generic_locks_removes_complete_set_and_leaves_target_owned_lock(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    _write_lock_set(destination, "1.0")
    ml_lock = destination / "ml-core-darwin-arm64.txt"
    ml_lock.write_text("torch==2.13.0\n", encoding="utf-8")

    publisher.clean_generic_locks(destination)

    assert all(not (destination / name).exists() for name in publisher.GENERIC_LOCK_FILES)
    assert ml_lock.read_text(encoding="utf-8") == "torch==2.13.0\n"


def test_clean_generic_locks_fails_before_deletion_for_unsafe_target(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    original = _write_lock_set(destination, "1.0")
    unsafe_target = destination / "security.txt"
    unsafe_target.unlink()
    unsafe_target.mkdir()

    with pytest.raises(publisher.LockPublicationError, match="cleanup target is not a file"):
        publisher.clean_generic_locks(destination)

    for name, content in original.items():
        if name == "security.txt":
            assert unsafe_target.is_dir()
        else:
            assert (destination / name).read_bytes() == content


@pytest.mark.security
def test_clean_serializes_behind_inflight_publication(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")

    publication_started = threading.Event()
    release_publication = threading.Event()
    publication_finished = threading.Event()
    clean_finished = threading.Event()
    errors: list[BaseException] = []

    def blocking_replace(source: Path, target: Path) -> None:
        os.replace(source, target)
        if target.name == publisher.GENERIC_LOCK_FILES[0]:
            publication_started.set()
            if not release_publication.wait(timeout=5):
                raise TimeoutError("test did not release publication")

    def publish() -> None:
        try:
            publisher.publish_generic_locks(staging, destination, replace=blocking_replace)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            publication_finished.set()

    def clean() -> None:
        try:
            publisher.clean_generic_locks(destination)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            clean_finished.set()

    publish_thread = threading.Thread(target=publish)
    clean_thread = threading.Thread(target=clean)
    publish_thread.start()
    assert publication_started.wait(timeout=5)
    clean_thread.start()
    try:
        assert not clean_finished.wait(timeout=0.2)
        release_publication.set()
        assert publication_finished.wait(timeout=5)
        assert clean_finished.wait(timeout=5)
    finally:
        release_publication.set()
        publish_thread.join(timeout=5)
        clean_thread.join(timeout=5)

    assert errors == []
    assert not publish_thread.is_alive()
    assert not clean_thread.is_alive()
    assert all(not (destination / name).exists() for name in publisher.GENERIC_LOCK_FILES)


def test_cli_rejects_multiple_operation_modes(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--destination-dir",
            str(destination),
            "--staging-dir",
            str(staging),
            "--recover-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_committed_set_is_reported_as_committed_when_journal_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    expected = _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")

    def fail_cleanup(_transaction_dir: Path) -> None:
        raise OSError("injected committed cleanup failure")

    monkeypatch.setattr(publisher, "_remove_transaction", fail_cleanup)
    with pytest.raises(publisher.LockPublicationError, match="lock set was committed"):
        publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == expected
    recovery_dirs = list(destination.glob(f"{publisher.TRANSACTION_PREFIX}*"))
    assert len(recovery_dirs) == 1
    assert json.loads((recovery_dirs[0] / publisher.JOURNAL_NAME).read_text(encoding="utf-8"))["state"] == "committed"
    shutil.rmtree(recovery_dirs[0])


@pytest.mark.security
def test_injected_publication_failure_restores_entire_old_set_and_leaves_ml_untouched(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")
    ml_lock = destination / "ml-core-darwin-arm64.txt"
    ml_lock.write_text("torch==2.13.0\n", encoding="utf-8")

    def fail_during_ci(source: Path, target: Path) -> None:
        if target.name == "ci.txt":
            raise OSError("injected publication failure")
        os.replace(source, target)

    with pytest.raises(publisher.LockPublicationError, match="previous lock set restored"):
        publisher.publish_generic_locks(staging, destination, replace=fail_during_ci)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert ml_lock.read_text(encoding="utf-8") == "torch==2.13.0\n"
    assert not list(destination.glob(".generic-lock-publish-*"))


def test_restored_set_is_distinguished_from_rollback_failure_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")

    def fail_during_ci(source: Path, target: Path) -> None:
        if target.name == "ci.txt":
            raise OSError("injected publication failure")
        os.replace(source, target)

    def fail_cleanup(_transaction_dir: Path) -> None:
        raise OSError("injected restored cleanup failure")

    monkeypatch.setattr(publisher, "_remove_transaction", fail_cleanup)
    with pytest.raises(
        publisher.LockPublicationError,
        match="previous lock set was restored, but recovery journal cleanup failed",
    ):
        publisher.publish_generic_locks(staging, destination, replace=fail_during_ci)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    recovery_dirs = list(destination.glob(f"{publisher.TRANSACTION_PREFIX}*"))
    assert len(recovery_dirs) == 1
    assert json.loads((recovery_dirs[0] / publisher.JOURNAL_NAME).read_text(encoding="utf-8"))["state"] == "rolled_back"
    shutil.rmtree(recovery_dirs[0])


@pytest.mark.security
def test_injected_keyboard_interrupt_restores_entire_old_set(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")

    def interrupt_during_ci(source: Path, target: Path) -> None:
        if target.name == "ci.txt":
            raise KeyboardInterrupt
        os.replace(source, target)

    with pytest.raises(publisher.LockPublicationError, match="previous lock set restored"):
        publisher.publish_generic_locks(staging, destination, replace=interrupt_during_ci)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert not list(destination.glob(".generic-lock-publish-*"))


def test_missing_staged_lock_is_rejected_before_any_destination_changes(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")
    (staging / "security.txt").unlink()

    with pytest.raises(publisher.LockPublicationError, match="staged lockfile is missing"):
        publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original


def test_floating_requirement_is_rejected_even_when_file_has_an_exact_pin(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")
    (staging / "all.txt").write_text(
        "exact-package==2.0\nfloating-package>=3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(publisher.LockPublicationError, match="violates the exact-pin contract"):
        publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original


@pytest.mark.security
def test_sigterm_during_publication_restores_entire_old_set(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")

    def terminate_during_ci(source: Path, target: Path) -> None:
        if target.name == "ci.txt":
            os.kill(os.getpid(), signal.SIGTERM)
        os.replace(source, target)

    with pytest.raises(publisher.LockPublicationError, match="previous lock set restored"):
        publisher.publish_generic_locks(staging, destination, replace=terminate_during_ci)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert not list(destination.glob(f"{publisher.TRANSACTION_PREFIX}*"))


def test_broken_destination_symlink_is_rejected_before_publication(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")
    (destination / "all.txt").unlink()
    (destination / "all.txt").symlink_to(tmp_path / "missing-lock.txt")

    with pytest.raises(publisher.LockPublicationError, match="destination lockfile is a symlink"):
        publisher.publish_generic_locks(staging, destination)

    assert (destination / "all.txt").is_symlink()
    for name, content in original.items():
        if name != "all.txt":
            assert (destination / name).read_bytes() == content


@pytest.mark.security
def test_incomplete_rollback_retains_recovery_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")
    real_replace = os.replace

    def fail_during_ci(source: Path, target: Path) -> None:
        if target.name == "ci.txt":
            raise OSError("injected publication failure")
        real_replace(source, target)

    def fail_base_rollback(source: Path, target: Path) -> None:
        source_path = Path(source)
        if source_path.name.startswith(".restore-base.txt."):
            raise OSError("injected rollback failure")
        real_replace(source, target)

    monkeypatch.setattr(publisher.os, "replace", fail_base_rollback)
    with pytest.raises(publisher.LockPublicationError, match="recovery data retained at") as exc_info:
        publisher.publish_generic_locks(staging, destination, replace=fail_during_ci)

    recovery_dir = Path(str(exc_info.value).rsplit("recovery data retained at ", maxsplit=1)[1])
    try:
        assert recovery_dir.is_dir()
        assert (recovery_dir / "backups" / "base.txt").is_file()
    finally:
        shutil.rmtree(recovery_dir)


@pytest.mark.security
def test_next_writer_recovers_stale_publication_before_validating_new_candidates(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "3.0")
    original = _write_lock_set(destination, "1.0")

    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}stale-test"
    backups_dir = transaction_dir / "backups"
    backups_dir.mkdir(parents=True)
    for name, content in original.items():
        (backups_dir / name).write_bytes(content)

    touched = list(publisher.GENERIC_LOCK_FILES[:2])
    for name in touched:
        (destination / name).write_text(f"mixed-{name}==2.0\n", encoding="utf-8")
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )
    (staging / "security.txt").unlink()

    with pytest.raises(publisher.LockPublicationError, match="staged lockfile is missing"):
        publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert not transaction_dir.exists()


@pytest.mark.security
def test_recover_only_api_restores_stale_mixed_set(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    original = _write_lock_set(destination, "1.0")
    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}recover-only"
    backups_dir = transaction_dir / "backups"
    backups_dir.mkdir(parents=True)
    for name, content in original.items():
        (backups_dir / name).write_bytes(content)

    touched = list(publisher.GENERIC_LOCK_FILES[:3])
    for name in touched:
        (destination / name).write_text(f"mixed-{name}==2.0\n", encoding="utf-8")
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )

    publisher.recover_generic_locks(destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert not transaction_dir.exists()


@pytest.mark.security
def test_recovery_does_not_follow_precreated_restore_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    original = _write_lock_set(destination, "1.0")
    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}restore-symlink"
    backups_dir = transaction_dir / "backups"
    backups_dir.mkdir(parents=True)
    for name, content in original.items():
        (backups_dir / name).write_bytes(content)

    touched = [publisher.GENERIC_LOCK_FILES[0]]
    (destination / touched[0]).write_text("mixed-all==2.0\n", encoding="utf-8")
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )
    victim = tmp_path / "restore-victim.txt"
    victim.write_text("do not replace\n", encoding="utf-8")
    (transaction_dir / ".restore-all.txt").symlink_to(victim)

    publisher.recover_generic_locks(destination)

    assert victim.read_text(encoding="utf-8") == "do not replace\n"
    assert (destination / touched[0]).read_bytes() == original[touched[0]]
    assert not transaction_dir.exists()


@pytest.mark.security
def test_recovery_does_not_follow_precreated_journal_temp_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    original = _write_lock_set(destination, "1.0")
    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}journal-symlink"
    backups_dir = transaction_dir / "backups"
    backups_dir.mkdir(parents=True)
    for name, content in original.items():
        (backups_dir / name).write_bytes(content)

    touched = [publisher.GENERIC_LOCK_FILES[0]]
    (destination / touched[0]).write_text("mixed-all==2.0\n", encoding="utf-8")
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )
    victim = tmp_path / "journal-victim.txt"
    victim.write_text("do not replace\n", encoding="utf-8")
    (transaction_dir / f".{publisher.JOURNAL_NAME}.tmp").symlink_to(victim)

    publisher.recover_generic_locks(destination)

    assert victim.read_text(encoding="utf-8") == "do not replace\n"
    assert (destination / touched[0]).read_bytes() == original[touched[0]]
    assert not transaction_dir.exists()


@pytest.mark.security
def test_recovery_rejects_symlinked_backup_directory_before_changes(tmp_path: Path) -> None:
    destination = tmp_path / "requirements"
    original = _write_lock_set(destination, "1.0")
    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}backup-symlink"
    transaction_dir.mkdir()
    external_backups = tmp_path / "external-backups"
    _write_lock_set(external_backups, "0.9")
    (transaction_dir / "backups").symlink_to(external_backups, target_is_directory=True)

    touched = [publisher.GENERIC_LOCK_FILES[0]]
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(publisher.LockPublicationError, match="recovery data retained"):
        publisher.recover_generic_locks(destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert transaction_dir.is_dir()


@pytest.mark.security
def test_incomplete_committed_journal_is_rejected_and_retained(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    original = _write_lock_set(destination, "1.0")

    transaction_dir = destination / f"{publisher.TRANSACTION_PREFIX}invalid-committed"
    transaction_dir.mkdir()
    (transaction_dir / publisher.JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": publisher.JOURNAL_VERSION,
                "destination": str(destination.resolve()),
                "names": list(publisher.GENERIC_LOCK_FILES),
                "existing": list(publisher.GENERIC_LOCK_FILES),
                "touched": list(publisher.GENERIC_LOCK_FILES[:2]),
                "state": "committed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(publisher.LockPublicationError, match="journal is invalid; recovery data retained"):
        publisher.publish_generic_locks(staging, destination)

    assert {name: (destination / name).read_bytes() for name in publisher.GENERIC_LOCK_FILES} == original
    assert transaction_dir.is_dir()


@pytest.mark.security
def test_destination_lock_serializes_concurrent_publishers(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "requirements"
    _write_lock_set(staging, "2.0")
    _write_lock_set(destination, "1.0")
    lock_path = publisher.destination_lock_path(destination)
    holder_code = (
        "import fcntl, sys\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
        "print('locked', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdin is not None
    assert holder.stdout.readline().strip() == "locked"

    finished = threading.Event()
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish_generic_locks(staging, destination)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert not finished.wait(timeout=0.2)
        holder.stdin.write("release\n")
        holder.stdin.flush()
        assert holder.wait(timeout=5) == 0
        assert finished.wait(timeout=5)
        thread.join(timeout=1)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)

    assert errors == []
    assert not thread.is_alive()
