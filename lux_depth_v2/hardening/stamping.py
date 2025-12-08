from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime import get_runtime_info
from .safe_io import to_jsonable


def stable_dumps(obj: Any) -> str:
    """
    Deterministic JSON serialization for config hashing and reproducibility.
    """
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def config_hash(config_obj: Any) -> str:
    return sha256_bytes(stable_dumps(config_obj).encode("utf-8"))


def stamp_report(
    report: Dict[str, Any],
    *,
    config: Optional[Any] = None,
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    include_input_hash: bool = False,
    profiler: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Add reproducibility + observability metadata without mutating the original report.
    """
    out = deepcopy(report)
    meta: Dict[str, Any] = out.get("meta", {})

    meta.update(
        {
            "run_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": get_runtime_info(),
        }
    )

    if config is not None:
        meta["config_hash"] = config_hash(config)

    if input_path is not None:
        meta["input_file"] = str(input_path)
        if include_input_hash:
            meta["input_sha256"] = sha256_file(input_path)

    if output_dir is not None:
        meta["output_dir"] = str(output_dir)

    if profiler is not None:
        out["profile_ms"] = dict(profiler)

    out["meta"] = meta
    return out


def write_run_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2))
