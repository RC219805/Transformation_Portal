"""Process ownership, secret redaction and progress parsing for job execution."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from contextlib import suppress
from types import ModuleType
from typing import Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)
CANCEL_GRACE_SECONDS = 5.0
PROGRESS_RE = re.compile(r"progress=(\d{1,3})%")

_LOG_REDACT_KV_KEYS = (
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "authorization",
    "auth_token",
    "auth-token",
    "token",
    "secret",
    "password",
    "passwd",
    "private_key",
    "private-key",
    "client_secret",
    "client-secret",
    "session_token",
    "session-token",
    "aws_secret_access_key",
    "aws-secret-access-key",
)

_LOG_REDACTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(authorization|proxy-authorization)\s*:\s*" r"(?:bearer|basic|digest|token|apikey)\s+\S+"),
        r"\1: <redacted>",
    ),
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
        "Bearer <redacted>",
    ),
    (
        re.compile(
            r"(?i)(^|[^A-Za-z0-9])(" + "|".join(re.escape(key) for key in _LOG_REDACT_KV_KEYS) + r")(\s*[:=]\s*)([^\s,;]+)"
        ),
        r"\1\2\3<redacted>",
    ),
)


def _redact_log_line(line: str) -> str:
    if not line:
        return line
    redacted = line
    for pattern, replacement in _LOG_REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _extract_progress_percent(line: str) -> Optional[int]:
    match = PROGRESS_RE.search(line)
    if not match:
        return None
    try:
        return max(0, min(100, int(match.group(1))))
    except ValueError:
        return None


def _sanitized_child_env() -> Dict[str, str]:
    child_env = os.environ.copy()
    sensitive_exact = {
        "TP_API_KEY",
        # Native consumers never receive the operational store/broker authority.
        "TP_DATABASE_URL",
        "TP_REDIS_URL",
        "DATABASE_URL",
        "REDIS_URL",
        "TP_TEST_POSTGRES_URL",
        "TP_DISPATCH_TEST_DATABASE_URL",
        "TP_PHOTOGRAPHY_TEST_DATABASE_URL",
        "TP_DISPATCH_TEST_REDIS_URL",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    sensitive_suffixes = (
        "_TOKEN",
        "_SECRET",
        "_PASSWORD",
        "_API_KEY",
        "_ACCESS_KEY",
        "_PRIVATE_KEY",
    )
    for key in list(child_env.keys()):
        upper = key.upper()
        if upper in sensitive_exact or upper.endswith(sensitive_suffixes):
            child_env.pop(key, None)
    return child_env


def _owned_process_group_id(proc: asyncio.subprocess.Process, *, os_module: ModuleType = os) -> Optional[int]:
    """Return only a group owned by this execution's isolated POSIX session.

    A successful ``start_new_session`` spawn pins its PID on the live process
    handle. That process-local authority survives leader exit until cleanup;
    it is never read from persisted job or broker state.
    """

    os = os_module
    if os.name == "nt":
        return None
    pid = proc.pid
    if type(pid) is not int or pid <= 0:
        return None
    pinned = getattr(proc, "_tp_owned_process_group_id", None)
    if pinned is not None:
        return pinned if type(pinned) is int and pinned == pid else None
    if proc.returncode is not None:
        return None
    try:
        if os.getpgid(pid) == pid and os.getsid(pid) == pid:
            return pid
    except OSError:
        pass
    return None


def _signal_process_tree(
    proc: asyncio.subprocess.Process,
    sig: int,
    *,
    os_module: ModuleType = os,
) -> bool:
    """Deliver *sig* to the subprocess's full process group on POSIX.

    Returns True when the signal was delivered via :func:`os.killpg`; False
    when the caller must fall back to the direct ``proc`` methods (e.g. the
    spawn did not create a new session, or we are on Windows).
    """

    os = os_module
    pgid = _owned_process_group_id(proc, os_module=os)
    if pgid is None:
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


async def _terminate_process(
    proc: asyncio.subprocess.Process,
    grace_seconds: float = CANCEL_GRACE_SECONDS,
    *,
    os_module: ModuleType = os,
) -> None:
    os = os_module
    owned_group = _owned_process_group_id(proc, os_module=os)
    if proc.returncode is not None and owned_group is None:
        return
    if owned_group is not None:
        # Retain a verified group for escalation after the leader is reaped.
        setattr(proc, "_tp_owned_process_group_id", owned_group)
    cleanup_completed = False
    deadline = asyncio.get_running_loop().time() + grace_seconds
    try:
        if not _signal_process_tree(proc, signal.SIGTERM, os_module=os):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
            except Exception:
                return
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            pass

        # A leader can exit before descendants finish, even when they close
        # stdout. Give the owned group the remaining TERM grace before KILL.
        while owned_group is not None:
            try:
                os.killpg(owned_group, 0)
            except ProcessLookupError:
                cleanup_completed = True
                return
            except OSError:
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))

        if owned_group is not None or proc.returncode is None:
            if not _signal_process_tree(proc, signal.SIGKILL, os_module=os):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    return
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            # A detached descendant can retain a pipe outside our group.
            # The runner owns bounded pipe draining/closure in its finally.
            LOGGER.warning("subprocess %s wait exceeded force-kill grace; runner must close remaining pipes", proc.pid)
        cleanup_completed = True
    finally:
        if cleanup_completed:
            with suppress(AttributeError):
                delattr(proc, "_tp_owned_process_group_id")
