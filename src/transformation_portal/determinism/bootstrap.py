from __future__ import annotations

import os
import sys

THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _ensure_thread_env() -> None:
    for k, v in THREAD_ENV_VARS.items():
        os.environ.setdefault(k, v)


def _ensure_pythonhashseed_zero() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        # Re-exec to ensure deterministic hashing in this interpreter while
        # preserving module invocation flags (e.g. "-m ...").
        orig_argv = getattr(sys, "orig_argv", None)
        if orig_argv:
            os.execv(sys.executable, [sys.executable] + list(orig_argv[1:]))
        os.execv(sys.executable, [sys.executable] + sys.argv)


def bootstrap() -> None:
    # Must run before importing NumPy / BLAS-linked extensions.
    _ensure_thread_env()
    _ensure_pythonhashseed_zero()
