from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from transformation_portal.determinism import cas as caslib

# Backward-compatibility export for legacy tests (do not use directly)
from transformation_portal.determinism.artifact_utils import parse_artifact_id as _parse_artifact_id
from transformation_portal.determinism.hardware_fpstate import enforce_fpstate_and_probe
from transformation_portal.determinism.ingest import ingest_from_npy, seed_everything, sha256_file
from transformation_portal.determinism.jcs import sha256_hex_of_canonical_json
from transformation_portal.determinism.manifest import build_artifact_manifest, stable_manifest_json
from transformation_portal.determinism.policy import load_policy
from transformation_portal.determinism.tensor import compute_artifact_id, load_tensor_bin, write_tensor_bin, write_tensor_npy
from transformation_portal.determinism.verify import verify_against_policy

__all__ = ["_parse_artifact_id"]

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------
# Root Callback (Prevents Typer collapsing to single-command mode)
# ---------------------------------------------------------------------
@app.callback()
def main_callback() -> None:
    """Determinism CLI root."""
    pass


@dataclass(frozen=True)
class DeterminismSummary:
    execution_id: str
    contract: str
    input_path: str
    artifact_id: str
    tensor_hash: str
    raw_hash: str
    fingerprint_hash: str
    subnormals_preserved: bool
    artifact_created: bool


@dataclass(frozen=True)
class HarnessRunReport:
    """Full harness run report including environment fingerprint per SPEC-DH-001."""

    summary: DeterminismSummary
    environment: Dict[str, Any]  # Environment fingerprint (SPEC-DH-001 Section 5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self.summary),
            "environment": self.environment,
        }


# ---------------------------------------------------------------------
# RUN COMMAND
# ---------------------------------------------------------------------
@app.command("run")
def run(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, file_okay=True),
    contract: str = typer.Option("npy_tensor", "--contract"),
    tensor_role: str = typer.Option("xyz_d50_linear_fp32", "--tensor-role"),
    wb_mode: str = typer.Option("camera", "--wb-mode"),
    demosaic: str = typer.Option("AHD", "--demosaic"),
    print_hash: bool = typer.Option(False, "--print-hash"),
    json_out: bool = typer.Option(True, "--json/--no-json"),
    include_env: bool = typer.Option(False, "--include-env", help="Include environment fingerprint per SPEC-DH-001"),
    cas_root: Path = typer.Option(Path(".cas"), "--cas-root"),
):
    """
    Minimal deterministic ingest runner.

    - Deterministic seeding
    - FTZ/DAZ probe reporting
    - Artifact ID generation
    - Minimal deterministic manifest emission
    """

    seed_everything(0)

    contract_n = contract.strip().lower()
    if contract_n == "npy_tensor":
        tensor, fingerprint = ingest_from_npy(input_path)
    elif contract_n == "camera_native_linear":
        try:
            from transformation_portal.spatial_ai.ingest.phase2_camera_native_linear import ingest_phase2_xyz_d50_linear_fp32
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "camera_native_linear contract requires optional ML ingest dependencies "
                "(e.g., rawpy). Use `./scripts/setup/install_raw_runtime.sh` for the "
                "isolated RAW runtime, or see `requirements/README.md` for the governed "
                "RAW lock contract."
            ) from e
        tensor, fingerprint = ingest_phase2_xyz_d50_linear_fp32(input_path, wb_mode=wb_mode, demosaic=demosaic)
    else:
        raise typer.BadParameter("Unsupported contract. Use 'npy_tensor' or 'camera_native_linear'.")

    fpstate_report = enforce_fpstate_and_probe(require_subnormals=False)
    subnormals_ok = bool(fpstate_report.subnormals_preserved)

    # Intentionally random per-run identifier for tracing/log correlation only.
    # It is not part of artifact identity or CAS layout.
    execution_id = str(uuid.uuid4())

    raw_hash = f"sha256:{sha256_file(input_path)}"
    fingerprint_hash = f"sha256:{sha256_hex_of_canonical_json(fingerprint)}"

    try:
        artifact_id = compute_artifact_id(
            tensor_role=tensor_role,
            tensor=tensor,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--tensor-role") from e

    # artifact_id is expected to be "sha256:<hex>"
    if not artifact_id.startswith("sha256:"):
        raise ValueError("Unexpected artifact_id format")

    tensor_hash = artifact_id.split("sha256:", 1)[1]

    # -----------------------------------------------------------------
    # CAS Write (Canonical directory naming + legacy migration)
    # -----------------------------------------------------------------
    cas_paths = caslib.CasPaths(cas_root=cas_root)
    caslib.ensure_dir(cas_paths.sha256_root)

    artifact_dir = cas_paths.artifact_dir(tensor_hash)

    # Legacy directory format: sha256:<hex>
    legacy_dir = cas_paths.artifact_dir(f"sha256:{tensor_hash}")

    # If legacy exists but canonical does not, migrate it
    if legacy_dir.exists() and not artifact_dir.exists():
        try:
            legacy_dir.rename(artifact_dir)
        except OSError as e:
            typer.echo(
                f"Warning: legacy CAS migration failed ({legacy_dir} -> {artifact_dir}): {e}",
                err=True,
            )

    artifact_created = False

    if not artifact_dir.exists():
        stage = caslib.stage_dir(cas_root, prefix=".stage_artifact_")
        try:
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

            manifest = build_artifact_manifest(
                artifact_id=artifact_id,
                tensor_role=tensor_role,
                tensor_hash=tensor_hash,
                raw_hash=raw_hash,
                fingerprint_hash=fingerprint_hash,
                fpstate_enforced=fpstate_report.enforced,
                fpstate_backend=fpstate_report.backend,
                probe_version=fpstate_report.probe_version,
                probe_policy=fpstate_report.probe_policy,
                subnormals_preserved=subnormals_ok,
                fpstate_note=fpstate_report.note,
            )

            caslib.write_json(stage / "artifact_manifest.json", manifest)

            caslib.ensure_dir(artifact_dir.parent)
            caslib.atomic_commit_dir(stage, artifact_dir)
            artifact_created = True
        except FileExistsError:
            pass
        finally:
            if stage.exists() and not artifact_created:
                import shutil

                shutil.rmtree(stage, ignore_errors=True)

    summary = DeterminismSummary(
        execution_id=execution_id,
        contract=contract_n,
        input_path=str(input_path),
        artifact_id=artifact_id,
        tensor_hash=tensor_hash,
        raw_hash=raw_hash,
        fingerprint_hash=fingerprint_hash,
        subnormals_preserved=subnormals_ok,
        artifact_created=artifact_created,
    )

    if print_hash:
        typer.echo(tensor_hash)
        return

    if include_env:
        from transformation_portal.determinism.environment import environment_fingerprint_dict

        env_fp = environment_fingerprint_dict()
        report = HarnessRunReport(summary=summary, environment=env_fp)
        if json_out:
            typer.echo(stable_manifest_json(report.to_dict()))
        else:
            for k, v in report.to_dict().items():
                typer.echo(f"{k}: {v}")
    else:
        if json_out:
            typer.echo(stable_manifest_json(asdict(summary)))
        else:
            for k, v in asdict(summary).items():
                typer.echo(f"{k}: {v}")


def _load_tensor_from_cas(cas_root: Path, artifact_id: str):
    import numpy as np

    tensor_hash = _parse_artifact_id(artifact_id)
    artifact_dir = cas_root / "sha256" / tensor_hash

    npy_path = artifact_dir / "output_tensor.npy"
    if npy_path.exists():
        try:
            return np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load tensor npy payload for artifact {artifact_id} from {npy_path}") from e

    meta_path = artifact_dir / "tensor_meta.json"
    bin_path = artifact_dir / "output_tensor.bin"
    if meta_path.exists() and bin_path.exists():
        try:
            meta_text = meta_path.read_text(encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"Failed to read tensor metadata for artifact {artifact_id} from {meta_path}") from e

        try:
            meta = json.loads(meta_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid tensor metadata JSON for artifact {artifact_id} at {meta_path}") from e

        if "shape" not in meta:
            raise KeyError(f"Missing 'shape' in tensor metadata for artifact {artifact_id} at {meta_path}")

        shape = tuple(meta["shape"])
        try:
            return load_tensor_bin(bin_path, shape=shape).astype(np.float32, copy=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tensor binary payload for artifact {artifact_id} " f"from {bin_path} with shape {shape}"
            ) from e

    raise FileNotFoundError(f"No tensor payload found for {artifact_id} in {artifact_dir}")


@app.command("verify")
def verify(
    baseline_artifact: str = typer.Option(..., "--baseline", help="Baseline artifact id sha256:<hex>"),
    candidate_artifact: str = typer.Option(..., "--artifact", help="Candidate artifact id sha256:<hex>"),
    cas_root: Path = typer.Option(Path(".cas"), "--cas-root", help="CAS root directory"),
    policy_path: Path = typer.Option(Path("policy/adr030_v1.json"), "--policy", help="ADR-030 policy file"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Exit non-zero on verification failure"),
):
    """Verify candidate artifact against baseline using ADR-030 gates."""
    try:
        _parse_artifact_id(baseline_artifact)
        _parse_artifact_id(candidate_artifact)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    policy, _policy_hash = load_policy(policy_path)
    baseline_tensor = _load_tensor_from_cas(cas_root, baseline_artifact)
    candidate_tensor = _load_tensor_from_cas(cas_root, candidate_artifact)

    vr = verify_against_policy(baseline_tensor, candidate_tensor, policy)
    record = vr.to_dict(policy=policy, baseline_artifact=baseline_artifact, candidate_artifact=candidate_artifact)
    typer.echo(stable_manifest_json(record))

    if strict and vr.status != "pass":
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------
# INFO COMMAND (Environment Fingerprint)
# ---------------------------------------------------------------------
@app.command("info")
def info(
    json_out: bool = typer.Option(True, "--json/--no-json"),
):
    """
    Print environment fingerprint and harness engine version.

    Per SPEC-DH-001 Section 5, the harness must report OS, ISA, runtime version,
    and dependency lock IDs for cross-ISA audit and reproducibility.
    """
    from transformation_portal.determinism.environment import HARNESS_ENGINE_VERSION, environment_fingerprint_dict

    fingerprint = environment_fingerprint_dict()

    if json_out:
        typer.echo(stable_manifest_json(fingerprint))
    else:
        typer.echo(f"Harness Engine Version: {HARNESS_ENGINE_VERSION}")
        typer.echo(f"OS: {fingerprint['os_system']} {fingerprint['os_release']}")
        typer.echo(f"ISA: {fingerprint['os_machine']}")
        typer.echo(f"Python: {fingerprint['python_version']} ({fingerprint['python_implementation']})")
        typer.echo(f"NumPy: {fingerprint['numpy_version']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
