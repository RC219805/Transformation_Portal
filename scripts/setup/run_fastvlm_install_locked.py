#!/usr/bin/env python3
"""Run the complete FastVLM install transaction under an interprocess lock."""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

DEFAULT_LOCK_TIMEOUT_SECONDS = 1800.0
MAX_PORTABLE_LOCK_OFFSET = (1 << 31) - 1
LOCK_FD_ENV = "TP_FASTVLM_INSTALL_LOCK_FD"
LOCK_TOKEN_ENV = "TP_FASTVLM_INSTALL_LOCK_TOKEN"


class InstallLockError(RuntimeError):
    """Raised when the FastVLM transaction lock cannot be trusted or acquired."""


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _ensure_real_directory(path: Path, *, create: bool) -> Path:
    target = _lexical_absolute(path)
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    current = Path(existing.anchor)
    for part in existing.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise InstallLockError(f"FastVLM lock parent must contain only real directories: {current}")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise InstallLockError(f"FastVLM lock parent must contain only real directories: {current}")
    return target


def _open_lock(lock_path: Path) -> int:
    target = _lexical_absolute(lock_path)
    _ensure_real_directory(target.parent, create=True)
    try:
        existing = target.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)):
        raise InstallLockError(f"FastVLM install lock must be a regular file: {target}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise InstallLockError(f"FastVLM install lock could not be opened safely: {target}") from exc
    try:
        opened = os.fstat(descriptor)
        current = target.lstat()
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise InstallLockError(f"FastVLM install lock changed while being opened: {target}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _acquire_lock(descriptor: int, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as exc:
            if time.monotonic() >= deadline:
                raise InstallLockError(f"Timed out after {timeout_seconds:g}s waiting for the FastVLM install lock") from exc
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def assert_lock_held(lock_path: Path, descriptor: int, token: int) -> None:
    """Validate the inherited lock descriptor used by the installer body."""

    if token <= 0:
        raise InstallLockError("FastVLM installer inherited an invalid transaction lock token")
    target = _lexical_absolute(lock_path)
    try:
        opened = os.fstat(descriptor)
        current = target.lstat()
    except (OSError, ValueError) as exc:
        raise InstallLockError("FastVLM installer did not inherit its transaction lock") from exc
    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise InstallLockError("FastVLM installer inherited an invalid transaction lock")
    try:
        inherited_token = os.lseek(descriptor, 0, os.SEEK_CUR)
    except OSError as exc:
        raise InstallLockError("FastVLM installer could not validate its transaction lock token") from exc
    if inherited_token != token:
        raise InstallLockError("FastVLM installer inherited a mismatched transaction lock token")

    probe = _open_lock(target)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(probe, fcntl.LOCK_UN)
        raise InstallLockError("FastVLM installer does not hold its transaction lock")
    finally:
        os.close(probe)


def run_locked(lock_path: Path, command: Sequence[str], *, timeout_seconds: float) -> int:
    if not command:
        raise InstallLockError("FastVLM install lock runner requires a command")
    descriptor = _open_lock(lock_path)
    try:
        _acquire_lock(descriptor, timeout_seconds=timeout_seconds)
        token = secrets.randbelow(MAX_PORTABLE_LOCK_OFFSET) + 1
        os.lseek(descriptor, token, os.SEEK_SET)
        environment = os.environ.copy()
        environment[LOCK_FD_ENV] = str(descriptor)
        environment[LOCK_TOKEN_ENV] = str(token)
        completed = subprocess.run(
            list(command),
            check=False,
            env=environment,
            pass_fds=(descriptor,),
        )
        return completed.returncode
    finally:
        os.close(descriptor)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Acquire the lock and run a command.")
    run_parser.add_argument("--lock-file", required=True)
    run_parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_LOCK_TIMEOUT_SECONDS)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    assert_parser = subparsers.add_parser("assert-held", help="Validate an inherited lock descriptor.")
    assert_parser.add_argument("--lock-file", required=True)
    assert_parser.add_argument("--fd", required=True, type=int)
    assert_parser.add_argument("--token", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.action == "assert-held":
            assert_lock_held(Path(args.lock_file), int(args.fd), int(args.token))
            return 0
        command = list(args.command)
        if command and command[0] == "--":
            command.pop(0)
        if args.timeout_seconds <= 0:
            raise InstallLockError("FastVLM install lock timeout must be positive")
        return run_locked(Path(args.lock_file), command, timeout_seconds=float(args.timeout_seconds))
    except (InstallLockError, OSError) as exc:
        print(f"FastVLM install lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
