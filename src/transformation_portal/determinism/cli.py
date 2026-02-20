from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from . import cas as caslib
from .ingest import ingest_from_npy, ingest_phase2_xyz_d50_linear_fp32, probe_subnormals_preserved, seed_everything
from .ingest import sha256_file as sha256_file_bytes
from .jcs import sha256_hex_of_canonical_json
from .policy import load_policy
from .tensor import compute_artifact_id, load_tensor_bin, write_tensor_bin, write_tensor_npy
from .trace import get_or_create_trace_context
from .verify import verify_against_policy

app = typer.Typer(add_completion=False, help="Determinism harness (ADR-030 / SPEC-DH-001).")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_artifact_id(artifact_id: str) -> str:
    if not artifact_id.startswith("sha256:"):
        raise typer.BadParameter("artifact_id must be of the form sha256:<hex>")
    hex_ = artifact_id.split("sha256:", 1)[1]
    if len(hex_) != 64 or any(c not in "0123456789abcdef" for c in hex_.lower()):
        raise typer.BadParameter("artifact_id hash must be 64 lowercase hex chars")
    return hex_


def _load_tensor_from_cas(cas_root: Path, artifact_id: str):
    import numpy as np

    tensor_hash = _parse_artifact_id(artifact_id)
    artifact_dir = cas_root / "sha256" / tensor_hash
    meta_path = artifact_dir / "tensor_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing tensor_meta.json for {artifact_id} at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    bin_path = artifact_dir / "output_tensor.bin"
    if bin_path.exists():
        arr = load_tensor_bin(bin_path, shape=shape)  # dtype <f4
        # Normalize to native float32 for compute; values identical.
        arr = arr.astype(np.float32, copy=False)
        return arr
    npy_path = artifact_dir / "output_tensor.npy"
    if npy_path.exists():
        arr = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
        return arr
    raise FileNotFoundError(f"No tensor payload found for {artifact_id} in {artifact_dir}")


def _collect_environment(policy_source: str) -> Dict[str, Any]:
    env = {
        "created_at": _now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env_vars": {
            k: os.environ.get(k)
            for k in [
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "numpy": None,
        "policy_source": policy_source,
    }
    try:
        import numpy as np  # noqa: F401

        env["numpy"] = {"version": np.__version__}
    except Exception:
        env["numpy"] = None
    try:
        import rawpy  # type: ignore

        env["rawpy"] = {"version": getattr(rawpy, "__version__", "unknown")}
    except Exception:
        env["rawpy"] = None
    return env


@app.command("run")
def run(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, help="RAW file path or .npy tensor input"),
    contract: str = typer.Option("npy_tensor", "--contract", help="Ingest contract: npy_tensor | camera_native_linear"),
    tensor_role: str = typer.Option("xyz_d50_linear_fp32", "--tensor-role", help="Certified tensor role (lowercase)"),
    cas_root: Path = typer.Option(Path(".cas"), "--cas-root", help="CAS root directory"),
    policy_path: Path = typer.Option(Path("policy/adr030_v1.json"), "--policy", help="Executable ADR-030 policy file"),
    execution_id: Optional[str] = typer.Option(
        None, "--execution-id", help="Execution id (uuid4). Auto-generated if omitted."
    ),
    traceparent: Optional[str] = typer.Option(None, "--traceparent", help="W3C traceparent header (optional)"),
    wb_mode: str = typer.Option("camera", "--wb-mode", help="White balance mode for camera_native_linear: none|camera|auto"),
    demosaic: str = typer.Option("AHD", "--demosaic", help="Demosaic algorithm (rawpy enum name), e.g., AHD"),
    baseline_artifact: Optional[str] = typer.Option(
        None, "--baseline", help="Baseline artifact id for immediate verification (optional)"
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Fail (non-zero) if verification fails or environment violates policy"
    ),
):
    """Run ingest -> emit CAS artifact -> optionally verify against a baseline."""
    # Deterministic seeding for any local randomness in tooling.
    seed_everything(0)

    policy, policy_hash = load_policy(policy_path)

    # Basic policy sanity guard.
    if tensor_role != policy.certified_tensor_role:
        raise typer.BadParameter(
            f"tensor_role '{tensor_role}' does not match policy certified_tensor_role '{policy.certified_tensor_role}'"
        )

    # FTZ/DAZ probe (best-effort). Fail closed when policy demands it.
    subnormals_ok = probe_subnormals_preserved()
    if policy.ftz_daz_policy == "fail_closed" and strict and not subnormals_ok:
        raise RuntimeError("FTZ/DAZ appears enabled (subnormals flushed). Policy requires fail-closed.")

    tc = get_or_create_trace_context(traceparent)
    exec_id = execution_id or str(uuid.uuid4())

    contract_n = contract.strip().lower()
    if contract_n == "npy_tensor":
        tensor, fingerprint = ingest_from_npy(input_path)
    elif contract_n == "camera_native_linear":
        tensor, fingerprint = ingest_phase2_xyz_d50_linear_fp32(input_path, wb_mode=wb_mode, demosaic=demosaic)
    else:
        raise typer.BadParameter("Unsupported contract. Use npy_tensor or camera_native_linear.")

    raw_hash = f"sha256:{sha256_file_bytes(input_path)}"
    fingerprint_v2_hash = f"sha256:{sha256_hex_of_canonical_json(fingerprint)}"

    artifact_id = compute_artifact_id(tensor_role=tensor_role, tensor=tensor)
    tensor_hash = _parse_artifact_id(artifact_id)

    cas_paths = caslib.CasPaths(cas_root=cas_root)
    caslib.ensure_dir(cas_paths.sha256_root)

    artifact_dir = cas_paths.artifact_dir(tensor_hash)
    artifact_created = False

    if not artifact_dir.exists():
        # Stage and atomically commit artifact directory.
        stage = caslib.stage_dir(cas_root, prefix=".stage_artifact_")
        try:
            # Write payloads.
            write_tensor_bin(stage / "output_tensor.bin", tensor)
            write_tensor_npy(stage / "output_tensor.npy", tensor)

            tensor_meta = {
                "tensor_role": tensor_role,
                "dtype": "float32",
                "order": "C",
                "shape": list(tensor.shape),
                "artifact_id": artifact_id,
            }
            caslib.write_json(stage / "tensor_meta.json", tensor_meta)

            manifest = caslib.build_artifact_manifest(stage)
            caslib.write_json(stage / "artifact_manifest.json", manifest)

            # Ensure parent exists.
            caslib.ensure_dir(artifact_dir.parent)

            caslib.atomic_commit_dir(stage, artifact_dir)
            artifact_created = True
        except FileExistsError:
            # Concurrent writer or pre-existing artifact.
            pass
        finally:
            # If staging still exists, clean it up.
            if stage.exists() and not artifact_created:
                # best-effort cleanup
                import shutil

                shutil.rmtree(stage, ignore_errors=True)

    # Stage and commit run evidence (append-only).
    runs_dir = cas_paths.runs_dir(tensor_hash)
    caslib.ensure_dir(runs_dir)

    run_stage = caslib.stage_dir(cas_root, prefix=".stage_run_")
    run_dir = runs_dir / exec_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")

    environment = _collect_environment(policy_source=policy.policy_source)
    environment_hash = f"sha256:{sha256_hex_of_canonical_json(environment)}"

    ingest_record = {
        "created_at": _now_iso(),
        "execution_id": exec_id,
        "traceparent": tc.traceparent,
        "trace_id": tc.trace_id,
        "contract": contract_n,
        "tensor_role": tensor_role,
        "artifact_id": artifact_id,
        "raw_input": {
            "path": str(input_path),
            "sha256": raw_hash,
        },
        "fingerprint_v2_hash": fingerprint_v2_hash,
        "environment_hash": environment_hash,
        "policy": {
            "path": str(policy_path),
            "policy_hash": f"sha256:{policy_hash}",
            "verification_policy_version": policy.verification_policy_version,
        },
        "ftz_daz_probe": {
            "subnormals_preserved": subnormals_ok,
            "policy": policy.ftz_daz_policy,
        },
    }

    caslib.write_json(run_stage / "environment.json", environment)
    caslib.write_json(run_stage / "ingest.json", ingest_record)

    verification_record = None
    if baseline_artifact:
        baseline_tensor = _load_tensor_from_cas(cas_root, baseline_artifact)
        candidate_tensor = tensor  # already in memory
        vr = verify_against_policy(baseline_tensor, candidate_tensor, policy)
        verification_record = vr.to_dict(
            policy=policy,
            baseline_artifact=baseline_artifact,
            candidate_artifact=artifact_id,
        )
        caslib.write_json(run_stage / "verification.json", verification_record)

        if strict and vr.status != "pass":
            # Commit evidence anyway (forensics), then fail.
            pass

    certificate = {
        "created_at": _now_iso(),
        "execution_id": exec_id,
        "trace_id": tc.trace_id,
        "traceparent": tc.traceparent,
        "artifact_id": artifact_id,
        "tensor_role": tensor_role,
        "raw_sha256": raw_hash,
        "fingerprint_v2_hash": fingerprint_v2_hash,
        "environment_hash": environment_hash,
        "policy_hash": f"sha256:{policy_hash}",
        "verification": verification_record,
    }
    caslib.write_json(run_stage / "reproducibility_certificate.json", certificate)

    run_manifest = caslib.build_artifact_manifest(run_stage)
    caslib.write_json(run_stage / "run_manifest.json", run_manifest)

    # Atomic commit run evidence.
    try:
        caslib.atomic_commit_dir(run_stage, run_dir)
    except FileExistsError as e:
        raise RuntimeError(f"Run evidence target already exists: {run_dir}") from e

    result = {
        "status": "ok",
        "artifact_id": artifact_id,
        "artifact_dir": str(artifact_dir),
        "execution_id": exec_id,
        "trace_id": tc.trace_id,
        "traceparent": tc.traceparent,
        "raw_sha256": raw_hash,
        "fingerprint_v2_hash": fingerprint_v2_hash,
        "environment_hash": environment_hash,
        "policy_hash": f"sha256:{policy_hash}",
        "verification": verification_record,
    }
    typer.echo(json.dumps(result, indent=2))

    if strict and verification_record and verification_record.get("status") != "pass":
        raise typer.Exit(code=2)


@app.command("verify")
def verify(
    baseline_artifact: str = typer.Option(..., "--baseline", help="Baseline artifact id sha256:<hex>"),
    candidate_artifact: str = typer.Option(..., "--artifact", help="Candidate artifact id sha256:<hex>"),
    cas_root: Path = typer.Option(Path(".cas"), "--cas-root", help="CAS root directory"),
    policy_path: Path = typer.Option(Path("policy/adr030_v1.json"), "--policy", help="Executable ADR-030 policy file"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Exit non-zero on verification failure"),
):
    """Verify candidate artifact against baseline using ADR-030 gates."""
    policy, _policy_hash = load_policy(policy_path)

    baseline_tensor = _load_tensor_from_cas(cas_root, baseline_artifact)
    candidate_tensor = _load_tensor_from_cas(cas_root, candidate_artifact)

    vr = verify_against_policy(baseline_tensor, candidate_tensor, policy)
    record = vr.to_dict(policy=policy, baseline_artifact=baseline_artifact, candidate_artifact=candidate_artifact)
    typer.echo(json.dumps(record, indent=2))
    if strict and vr.status != "pass":
        raise typer.Exit(code=2)
