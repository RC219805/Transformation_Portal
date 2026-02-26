# verify_rag_install.py
#!/usr/bin/env python3
"""
Transformation Portal — RAG Environment Verifier

Checks:
  1) Python version and virtualenv status
  2) Torch with Metal (MPS) backend availability and a tiny MPS matmul
  3) FAISS availability and a minimal index/search roundtrip
  4) Sentence-Transformers availability (no model download)
  5) RAG modules importability from .github/agents/rag_system

Usage:
  python verify_rag_install.py [--repo /path/to/repo] [--verbose]

Exit codes:
  0 = all good
  1 = warnings only (non-fatal)
  2 = failure (at least one required check failed)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


def _seed_repo_root_for_imports() -> None:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".github" / "workflows").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_seed_repo_root_for_imports()

from scripts.lib.repo_root import RepoRootError, resolve_repo_root


# --- UI helpers ---
class UI:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def ok(msg: str) -> None:
        print(f"{UI.GREEN}✅ {msg}{UI.RESET}")

    @staticmethod
    def warn(msg: str) -> None:
        print(f"{UI.YELLOW}⚠️  {msg}{UI.RESET}")

    @staticmethod
    def err(msg: str) -> None:
        print(f"{UI.RED}❌ {msg}{UI.RESET}")

    @staticmethod
    def info(msg: str) -> None:
        print(f"{UI.DIM}{msg}{UI.RESET}")


def check_python(args) -> Tuple[bool, str]:
    ok = True
    msgs: List[str] = []
    ver = sys.version.split()[0]
    in_venv = (
        hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", "") != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    )
    msgs.append(f"Python: {ver}")
    msgs.append(f"Virtualenv active: {in_venv} (sys.prefix={sys.prefix})")

    # Expect 3.11.x
    major_minor = ".".join(ver.split(".")[:2])
    if major_minor != "3.11":
        ok = False
        msgs.append("Expected Python 3.11.x for best compatibility on macOS ARM.")
    if not in_venv:
        ok = False
        msgs.append("Not running inside a virtual environment.")
    return ok, "\n".join(msgs)


def check_torch(args) -> Tuple[bool, str]:
    try:
        import torch  # type: ignore

        msgs = [f"Torch: {torch.__version__}"]
        mps_built = getattr(torch.backends.mps, "is_built", lambda: False)()
        mps_avail = getattr(torch.backends.mps, "is_available", lambda: False)()
        msgs.append(f"MPS built: {mps_built}, available: {mps_avail}")
        ok = True

        # Tiny runtime test if available
        if mps_avail:
            try:
                a = torch.randn((2, 2), device="mps")
                b = torch.randn((2, 2), device="mps")
                _ = a @ b
                msgs.append("MPS runtime matmul: OK")
            except Exception as e:
                ok = False
                msgs.append(f"MPS runtime matmul FAILED: {e}")
        else:
            UI.warn("MPS not available. Torch will fall back to CPU.")

        return ok, "\n".join(msgs)
    except Exception as e:
        return False, f"Torch import failed: {e}"


def check_faiss(args) -> Tuple[bool, str]:
    try:
        import faiss  # type: ignore

        msgs = [f"FAISS: {getattr(faiss, '__version__', 'unknown')}"]
        # Minimal index roundtrip
        import numpy as np

        d = 4
        idx = faiss.IndexFlatL2(d)
        x = np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.9, 0.8, 0.7, 0.6],
            ],
            dtype="float32",
        )
        idx.add(x)
        _D, I = idx.search(x, 1)
        ok = I.shape == (2, 1)
        msgs.append(f"FAISS search: OK (neighbors={I.ravel().tolist()})" if ok else "FAISS search: FAILED")
        return ok, "\n".join(msgs)
    except Exception as e:
        return False, f"FAISS check failed: {e}"


def check_sentence_transformers(args) -> Tuple[bool, str]:
    try:
        import sentence_transformers  # type: ignore

        ver = getattr(sentence_transformers, "__version__", "unknown")
        return True, f"Sentence-Transformers: {ver} (import OK)"
    except Exception as e:
        return False, f"Sentence-Transformers import failed: {e}"


def ensure_repo_on_path(repo_root: Path) -> None:
    # Keep imports local to the checked-out repository and src layout.
    for path in (repo_root, repo_root / "src", repo_root / ".github" / "agents"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def check_rag_imports(args, repo_root: Path) -> Tuple[bool, str]:
    ensure_repo_on_path(repo_root)
    try:
        from rag_system.citation import CitationGenerator  # type: ignore
        from rag_system.indexer import RepositoryIndexer  # type: ignore
        from rag_system.retriever import HybridRetriever  # type: ignore

        return True, "RAG modules import OK (rag_system.* via .github/agents)"
    except Exception as e:
        return False, f"RAG import failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Repository root path override")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        repo_path = Path(args.repo).expanduser() if args.repo else None
        repo_root = resolve_repo_root(start=Path(__file__), repo=repo_path)
    except RepoRootError as exc:
        UI.err(str(exc))
        sys.exit(2)

    UI.info(f"Verifying environment for repo: {repo_root}")

    failures = 0
    warnings = 0

    # 1) Python / venv
    ok, msg = check_python(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 2) Torch + MPS
    ok, msg = check_torch(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 3) FAISS
    ok, msg = check_faiss(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 4) Sentence-Transformers
    ok, msg = check_sentence_transformers(args)
    if ok:
        UI.ok(msg)
    else:
        UI.warn(msg)
        warnings += 1

    # 5) RAG imports
    ok, msg = check_rag_imports(args, repo_root)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # Summary / exit code
    print()
    if failures == 0 and warnings == 0:
        UI.ok("System Ready — all checks passed.")
        sys.exit(0)
    if failures == 0 and warnings > 0:
        UI.warn(f"System Ready with warnings ({warnings}).")
        sys.exit(1)

    UI.err(f"System NOT ready — failures: {failures}, warnings: {warnings}.")
    sys.exit(2)


if __name__ == "__main__":
    main()
