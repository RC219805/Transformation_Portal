#!/usr/bin/env python3
"""Publish the governed generic requirement locks as one recoverable set.

The dependency writer compiles every candidate into a staging directory first.
This helper validates the complete six-file generic set, serializes writers,
and records a durable rollback journal before replacing live files. Handled
failures restore the previous set immediately; a later writer repairs a stale
journal left by a process crash before attempting another publication.

The six public lock paths remain regular files, so their sequential renames do
not provide an atomic snapshot to readers that ignore the advisory writer lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from check_dependency_pinning import find_violations

GENERIC_LOCK_FILES = (
    "all.txt",
    "base.txt",
    "dev.txt",
    "ci.txt",
    "security.txt",
    "tools-archive.txt",
)
TRANSACTION_PREFIX = ".generic-lock-publish-"
JOURNAL_NAME = "journal.json"
JOURNAL_VERSION = 1
PREPARATION_STAGING_ENV = "TP_GENERIC_LOCK_STAGING_DIR"

Replace = Callable[[Path, Path], None]
Journal = dict[str, Any]


class LockPublicationError(RuntimeError):
    """Raised when a generic lock set cannot be safely published."""


class _TerminationRequested(BaseException):
    """Internal signal used to route SIGTERM through rollback handling."""


def _validate_candidate(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise LockPublicationError(f"staged lockfile is missing or not a regular file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LockPublicationError(f"staged lockfile is unreadable: {path} ({exc})") from exc

    if not text.strip():
        raise LockPublicationError(f"staged lockfile is empty: {path}")

    violations = find_violations([path])
    if violations:
        raise LockPublicationError("staged lockfile violates the exact-pin contract: " + "; ".join(violations))

    has_exact_pin = any("==" in line and not line.lstrip().startswith(("#", "--")) for line in text.splitlines())
    if not has_exact_pin:
        raise LockPublicationError(f"staged lockfile contains no exact dependency pins: {path}")


def destination_lock_path(destination_dir: Path) -> Path:
    """Return the stable system-temp lock path for ``destination_dir``."""
    destination_key = hashlib.sha256(os.fsencode(str(destination_dir.resolve()))).hexdigest()[:32]
    return Path(tempfile.gettempdir()) / f"tp-generic-lock-publish-{destination_key}.lock"


@contextmanager
def _exclusive_destination_lock(destination_dir: Path) -> Iterator[None]:
    """Serialize publishers for one resolved destination without repo artifacts."""
    lock_path = destination_lock_path(destination_dir)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise LockPublicationError(f"could not open publication lock {lock_path}: {exc}") from exc

    with os.fdopen(descriptor, "a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _rollback_on_sigterm() -> Iterator[None]:
    """Convert SIGTERM into a rollback-aware exception on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_termination(_signum: int, _frame: object) -> None:
        raise _TerminationRequested("SIGTERM received during generic lock publication")

    signal.signal(signal.SIGTERM, request_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_copy(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    _fsync_file(destination)


def _replace_from_exclusive_copy(source: Path, destination: Path, temporary_dir: Path) -> None:
    """Replace ``destination`` from an exclusively created durable copy."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".restore-{destination.name}.",
        dir=temporary_dir,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
            descriptor_open = False
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _write_journal(transaction_dir: Path, journal: Journal) -> None:
    journal_path = transaction_dir / JOURNAL_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{JOURNAL_NAME}.",
        suffix=".tmp",
        dir=transaction_dir,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor_open = False
            json.dump(journal, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, journal_path)
        _fsync_directory(transaction_dir)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _read_journal(transaction_dir: Path) -> Journal:
    journal_path = transaction_dir / JOURNAL_NAME
    if journal_path.is_symlink() or not journal_path.is_file():
        raise LockPublicationError(f"invalid generic lock recovery journal: {journal_path}")
    try:
        value = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockPublicationError(f"could not read generic lock recovery journal {journal_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockPublicationError(f"generic lock recovery journal is not an object: {journal_path}")
    return value


def _validated_journal(
    transaction_dir: Path,
    journal: Journal,
    destination_dir: Path,
    names: tuple[str, ...],
) -> tuple[set[str], list[str], str]:
    expected_destination = str(destination_dir.resolve())
    if journal.get("version") != JOURNAL_VERSION:
        raise LockPublicationError(f"unsupported recovery journal version at {transaction_dir}")
    if journal.get("destination") != expected_destination:
        raise LockPublicationError(f"recovery journal destination mismatch at {transaction_dir}")
    if journal.get("names") != list(names):
        raise LockPublicationError(f"recovery journal lock set mismatch at {transaction_dir}")

    state = journal.get("state")
    existing_value = journal.get("existing")
    touched_value = journal.get("touched")
    if state not in {"publishing", "rolled_back", "committed"}:
        raise LockPublicationError(f"invalid recovery journal state at {transaction_dir}")
    if not isinstance(existing_value, list) or any(name not in names for name in existing_value):
        raise LockPublicationError(f"invalid recovery journal existing set at {transaction_dir}")
    if len(existing_value) != len(set(existing_value)):
        raise LockPublicationError(f"duplicate recovery journal existing entries at {transaction_dir}")
    if not isinstance(touched_value, list) or touched_value != list(names[: len(touched_value)]):
        raise LockPublicationError(f"invalid recovery journal publication order at {transaction_dir}")
    if state == "committed" and touched_value != list(names):
        raise LockPublicationError(f"incomplete committed recovery journal at {transaction_dir}")

    return set(existing_value), list(touched_value), state


def _restore_transaction(
    transaction_dir: Path,
    destination_dir: Path,
    existing: set[str],
    touched: list[str],
) -> None:
    backups_dir = transaction_dir / "backups"
    if backups_dir.is_symlink() or not backups_dir.is_dir():
        raise LockPublicationError(f"recovery backup directory is missing or unsafe: {backups_dir}")

    errors: list[str] = []
    for name in reversed(touched):
        destination = destination_dir / name
        try:
            if name in existing:
                backup = backups_dir / name
                if backup.is_symlink() or not backup.is_file():
                    raise LockPublicationError(f"recovery backup is missing or unsafe: {backup}")
                _replace_from_exclusive_copy(backup, destination, transaction_dir)
            else:
                destination.unlink(missing_ok=True)
            _fsync_directory(destination_dir)
        except BaseException as exc:
            errors.append(f"{name}: {exc}")

    if errors:
        raise LockPublicationError("; ".join(errors))


def _remove_transaction(transaction_dir: Path) -> None:
    shutil.rmtree(transaction_dir)


def _recover_stale_transactions(destination_dir: Path, names: tuple[str, ...]) -> None:
    """Recover or clean transactions left by a previously terminated writer."""
    for transaction_dir in sorted(destination_dir.glob(f"{TRANSACTION_PREFIX}*")):
        if transaction_dir.is_symlink() or not transaction_dir.is_dir():
            raise LockPublicationError(f"unsafe generic lock recovery path: {transaction_dir}")

        journal_path = transaction_dir / JOURNAL_NAME
        if not journal_path.exists():
            # A journal is made durable before the first live replacement, so
            # an orphan without one contains staging data only.
            try:
                _remove_transaction(transaction_dir)
            except BaseException as exc:
                raise LockPublicationError(f"stale pre-publication data could not be cleaned at {transaction_dir}") from exc
            continue

        try:
            journal = _read_journal(transaction_dir)
            existing, touched, state = _validated_journal(
                transaction_dir,
                journal,
                destination_dir,
                names,
            )
        except BaseException as exc:
            raise LockPublicationError(
                f"stale generic lock journal is invalid; recovery data retained at {transaction_dir}"
            ) from exc

        if state == "publishing":
            try:
                _restore_transaction(transaction_dir, destination_dir, existing, touched)
            except BaseException as exc:
                raise LockPublicationError(
                    "stale generic lock publication could not be rolled back; " f"recovery data retained at {transaction_dir}"
                ) from exc
            journal["state"] = "rolled_back"
            try:
                _write_journal(transaction_dir, journal)
            except BaseException as exc:
                raise LockPublicationError(
                    "stale generic lock publication was rolled back, but the recovery state "
                    f"could not be persisted at {transaction_dir}"
                ) from exc
            try:
                _remove_transaction(transaction_dir)
            except BaseException as exc:
                raise LockPublicationError(
                    "stale generic lock publication was rolled back, but its recovery journal "
                    f"could not be cleaned at {transaction_dir}"
                ) from exc
        elif state == "committed":
            try:
                # A committed journal proves every replacement and the state
                # transition were made durable. It must never trigger rollback.
                _remove_transaction(transaction_dir)
            except BaseException as exc:
                raise LockPublicationError(
                    "generic lock set was already committed, but its stale journal "
                    f"could not be cleaned at {transaction_dir}"
                ) from exc
        else:
            try:
                _remove_transaction(transaction_dir)
            except BaseException as exc:
                raise LockPublicationError(
                    "previous generic lock set was already restored, but its stale journal "
                    f"could not be cleaned at {transaction_dir}"
                ) from exc


def _publish_locked(
    staging_dir: Path,
    destination_dir: Path,
    names: tuple[str, ...],
    replace: Replace,
) -> None:
    """Publish a validated candidate set while the destination lock is held."""
    transaction_dir = Path(tempfile.mkdtemp(prefix=TRANSACTION_PREFIX, dir=destination_dir))
    candidates_dir = transaction_dir / "candidates"
    backups_dir = transaction_dir / "backups"
    candidates_dir.mkdir()
    backups_dir.mkdir()

    existing: set[str] = set()
    touched: list[str] = []
    journal: Journal | None = None
    try:
        # Materialize every candidate and backup on the destination filesystem.
        # No live lockfile is replaced before the rollback journal is durable.
        for name in names:
            _durable_copy(staging_dir / name, candidates_dir / name)
            destination = destination_dir / name
            if destination.is_symlink():
                raise LockPublicationError(f"destination lockfile is a symlink: {destination}")
            if destination.exists():
                if not destination.is_file():
                    raise LockPublicationError(f"destination lockfile is not a regular file: {destination}")
                _durable_copy(destination, backups_dir / name)
                existing.add(name)
        _fsync_directory(candidates_dir)
        _fsync_directory(backups_dir)

        journal = {
            "version": JOURNAL_VERSION,
            "destination": str(destination_dir.resolve()),
            "names": list(names),
            "existing": [name for name in names if name in existing],
            "touched": [],
            "state": "publishing",
        }
        _write_journal(transaction_dir, journal)
        # Persist the transaction directory entry itself in the destination
        # directory before the first live lockfile rename.
        _fsync_directory(destination_dir)

        # Record each target before replacement. Recovery may restore a file
        # that was not yet changed, which is safe and closes the crash window.
        for name in names:
            touched.append(name)
            journal["touched"] = list(touched)
            _write_journal(transaction_dir, journal)
            replace(candidates_dir / name, destination_dir / name)
            _fsync_directory(destination_dir)

        journal["state"] = "committed"
        _write_journal(transaction_dir, journal)
    except BaseException as exc:
        if journal is None or not touched:
            try:
                _remove_transaction(transaction_dir)
            except BaseException as cleanup_exc:
                raise LockPublicationError(
                    "generic lock staging failed before publication and temporary data cleanup failed; "
                    f"data retained at {transaction_dir}"
                ) from cleanup_exc
            if isinstance(exc, LockPublicationError):
                raise
            raise LockPublicationError("generic lock staging failed before publication") from exc

        try:
            _restore_transaction(transaction_dir, destination_dir, existing, touched)
        except BaseException as rollback_exc:
            raise LockPublicationError(
                "generic lock publication failed and rollback was incomplete "
                f"({rollback_exc}); recovery data retained at {transaction_dir}"
            ) from exc

        journal["state"] = "rolled_back"
        try:
            _write_journal(transaction_dir, journal)
        except BaseException as journal_exc:
            raise LockPublicationError(
                "generic lock publication failed and the previous lock set was restored, "
                f"but the rollback state could not be persisted at {transaction_dir}"
            ) from journal_exc

        try:
            _remove_transaction(transaction_dir)
        except BaseException as cleanup_exc:
            raise LockPublicationError(
                "generic lock publication failed and the previous lock set was restored, "
                f"but recovery journal cleanup failed at {transaction_dir}"
            ) from cleanup_exc
        raise LockPublicationError("generic lock publication failed; previous lock set restored") from exc
    else:
        try:
            _remove_transaction(transaction_dir)
        except BaseException as cleanup_exc:
            raise LockPublicationError(
                "generic lock set was committed, but committed journal cleanup failed at " f"{transaction_dir}"
            ) from cleanup_exc


def publish_generic_locks(
    staging_dir: Path,
    destination_dir: Path,
    *,
    lock_files: Sequence[str] = GENERIC_LOCK_FILES,
    replace: Replace = os.replace,
) -> None:
    """Validate and publish the exact generic lock set under one writer lock.

    ``replace`` is injectable so tests can prove rollback behavior at any
    publication boundary. Recovery and rollback always use ``os.replace`` so
    an injected publication failure cannot disable restoration.
    """
    staging_dir = staging_dir.resolve()
    if not staging_dir.is_dir():
        raise LockPublicationError(f"staging directory does not exist: {staging_dir}")
    destination_dir, normalized_names = _validated_destination(destination_dir, lock_files)

    with _exclusive_destination_lock(destination_dir), _rollback_on_sigterm():
        _recover_stale_transactions(destination_dir, normalized_names)
        for name in normalized_names:
            _validate_candidate(staging_dir / name)
        _publish_locked(staging_dir, destination_dir, normalized_names, replace)


def _run_preparation_command(command: tuple[str, ...], destination_dir: Path, staging_dir: Path) -> None:
    environment = os.environ.copy()
    environment[PREPARATION_STAGING_ENV] = str(staging_dir)
    try:
        process = subprocess.Popen(command, cwd=destination_dir, env=environment)
    except OSError as exc:
        raise LockPublicationError(f"could not start generic lock preparation command: {exc}") from exc

    try:
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    if return_code != 0:
        raise LockPublicationError(f"generic lock preparation command failed with exit status {return_code}")


def prepare_and_publish_generic_locks(
    destination_dir: Path,
    prepare_command: Sequence[str],
    *,
    lock_files: Sequence[str] = GENERIC_LOCK_FILES,
) -> None:
    """Run preparation and publication under one destination writer lock."""
    command = tuple(prepare_command)
    if not command:
        raise LockPublicationError("generic lock preparation command must not be empty")

    destination_dir, normalized_names = _validated_destination(destination_dir, lock_files)
    with _exclusive_destination_lock(destination_dir), _rollback_on_sigterm():
        _recover_stale_transactions(destination_dir, normalized_names)
        with tempfile.TemporaryDirectory(prefix="tp-generic-lock-prepare-") as temporary_name:
            staging_dir = Path(temporary_name)
            _run_preparation_command(command, destination_dir, staging_dir)
            for name in normalized_names:
                _validate_candidate(staging_dir / name)
            _publish_locked(staging_dir, destination_dir, normalized_names, os.replace)


def _validated_destination(
    destination_dir: Path,
    lock_files: Sequence[str],
) -> tuple[Path, tuple[str, ...]]:
    destination_dir = destination_dir.resolve()
    if not destination_dir.is_dir():
        raise LockPublicationError(f"destination directory does not exist: {destination_dir}")

    normalized_names = tuple(lock_files)
    if normalized_names != GENERIC_LOCK_FILES:
        raise LockPublicationError("lockfile operation set must be the exact governed generic lock set")
    return destination_dir, normalized_names


def recover_generic_locks(
    destination_dir: Path,
    *,
    lock_files: Sequence[str] = GENERIC_LOCK_FILES,
) -> None:
    """Recover stale generic publication state under the destination lock."""
    destination_dir, normalized_names = _validated_destination(destination_dir, lock_files)
    with _exclusive_destination_lock(destination_dir), _rollback_on_sigterm():
        _recover_stale_transactions(destination_dir, normalized_names)


def clean_generic_locks(
    destination_dir: Path,
    *,
    lock_files: Sequence[str] = GENERIC_LOCK_FILES,
) -> None:
    """Remove the complete generic lock set while excluding target-owned locks."""
    destination_dir, normalized_names = _validated_destination(destination_dir, lock_files)
    with _exclusive_destination_lock(destination_dir):
        _recover_stale_transactions(destination_dir, normalized_names)

        # Validate every target before deleting any, so a directory or other
        # unexpected filesystem object fails closed without a partial clean.
        for name in normalized_names:
            path = destination_dir / name
            if path.exists() and not path.is_file() and not path.is_symlink():
                raise LockPublicationError(f"generic lock cleanup target is not a file: {path}")

        for name in normalized_names:
            (destination_dir / name).unlink(missing_ok=True)
        _fsync_directory(destination_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage governed generic requirements locks recoverably")
    parser.add_argument("--destination-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staging-dir", type=Path, help="publish a validated staged generic lock set")
    mode.add_argument(
        "--prepare-command",
        nargs=argparse.REMAINDER,
        help=(
            "run a command under the writer lock with "
            f"{PREPARATION_STAGING_ENV} set, then validate and publish its staged lock set"
        ),
    )
    mode.add_argument("--recover-only", action="store_true", help="recover stale publication state and exit")
    mode.add_argument("--clean-generic", action="store_true", help="remove the six governed generic locks")
    args = parser.parse_args()

    try:
        if args.recover_only:
            recover_generic_locks(args.destination_dir)
        elif args.clean_generic:
            clean_generic_locks(args.destination_dir)
        elif args.prepare_command is not None:
            prepare_and_publish_generic_locks(args.destination_dir, args.prepare_command)
        else:
            publish_generic_locks(args.staging_dir, args.destination_dir)
    except LockPublicationError as exc:
        parser.error(str(exc))

    if args.recover_only:
        print("generic requirements lock recovery completed successfully")
    elif args.clean_generic:
        print("generic requirements lock set removed successfully")
    else:
        print("generic requirements lock set published successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
