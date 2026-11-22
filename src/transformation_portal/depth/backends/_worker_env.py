"""Shared subprocess-worker environment and timeout handling for depth backends.

The DA3 and Depth Pro backends both run their model in a dedicated Python
runtime. The parent's interpreter configuration (``PYTHONPATH``,
``PYTHONHOME``, ``VIRTUAL_ENV`` ...) must not leak into that runtime, and a
hung worker must not block a job forever.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_WORKER_SUBPROCESS_TIMEOUT_SECONDS = 900

_STRIPPED_INTERPRETER_KEYS = frozenset({"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"})


def build_isolated_worker_env(repo_src: Optional[Path]) -> dict[str, str]:
    """Return the parent environment without its Python interpreter configuration.

    Every ``PYTHON*`` variable plus ``VIRTUAL_ENV``/``__PYVENV_LAUNCHER__`` is
    dropped, bytecode writes, user site-packages and implicit cwd imports are
    disabled, and ``PYTHONPATH`` points only at the checkout's ``src`` tree
    when one exists.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON") and key.upper() not in _STRIPPED_INTERPRETER_KEYS
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    if repo_src is not None and repo_src.exists():
        env["PYTHONPATH"] = str(repo_src)
    return env


def resolve_worker_timeout_seconds(
    config: Any,
    *,
    config_attr: str,
    env_var: str,
    label: str,
    default: int = DEFAULT_WORKER_SUBPROCESS_TIMEOUT_SECONDS,
) -> int:
    """Resolve a positive worker timeout from config, then the environment, then the default."""
    candidate: Any = None
    if config is not None:
        candidate = getattr(config, config_attr, None)

    if candidate in (None, ""):
        env_candidate = os.environ.get(env_var)
        if env_candidate and env_candidate.strip():
            candidate = env_candidate.strip()

    if candidate in (None, ""):
        return default

    try:
        timeout_seconds = int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} subprocess timeout must be a positive integer number of seconds") from exc

    if timeout_seconds <= 0:
        raise ValueError(f"{label} subprocess timeout must be greater than zero")
    return timeout_seconds
