from __future__ import annotations

# Allow running as a script (pytest calls file directly)
if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import typer

from transformation_portal.determinism import cas as caslib

# Backward-compatibility export for legacy tests (do not use directly)
from transformation_portal.determinism.artifact_utils import parse_artifact_id as _parse_artifact_id
from transformation_portal.determinism.ingest import ingest_from_npy, probe_subnormals_preserved, seed_everything, sha256_file
from transformation_portal.determinism.jcs import sha256_hex_of_canonical_json
from transformation_portal.determinism.manifest import build_artifact_manifest
from transformation_portal.determinism.tensor import compute_artifact_id

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


def _stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


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


# ---------------------------------------------------------------------
# RUN COMMAND
# ---------------------------------------------------------------------
@app.command("run")
def run(
    input_path: Path = typer.Option(..., "--input", exists=True),
    contract: str = typer.Option("npy_tensor", "--contract"),
    print_hash: bool = typer.Option(False, "--print-hash"),
    json_out: bool = typer.Option(True, "--json/--no-json"),
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

    if contract != "npy_tensor":
        raise typer.BadParameter("Only 'npy_tensor' supported in minimal runner.")

    subnormals_ok = probe_subnormals_preserved()

    execution_id = str(uuid.uuid4())

    tensor, fingerprint = ingest_from_npy(input_path)

    raw_hash = f"sha256:{sha256_file(input_path)}"
    fingerprint_hash = f"sha256:{sha256_hex_of_canonical_json(fingerprint)}"

    artifact_id = compute_artifact_id(
        tensor_role="xyz_d50_linear_fp32",
        tensor=tensor,
    )

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
        legacy_dir.rename(artifact_dir)

    artifact_created = False

    if not artifact_dir.exists():
        stage = caslib.stage_dir(cas_root, prefix=".stage_artifact_")
        try:
            manifest = build_artifact_manifest(
                artifact_id=artifact_id,
                tensor_role="xyz_d50_linear_fp32",
                tensor_hash=tensor_hash,
                raw_hash=raw_hash,
                fingerprint_hash=fingerprint_hash,
            )

            caslib.write_json(stage / "artifact_manifest.json", manifest)

            caslib.ensure_dir(artifact_dir.parent)
            caslib.atomic_commit_dir(stage, artifact_dir)
            artifact_created = True
        except FileExistsError:
            pass

    summary = DeterminismSummary(
        execution_id=execution_id,
        contract=contract,
        input_path=str(input_path),
        artifact_id=artifact_id,
        tensor_hash=tensor_hash,
        raw_hash=raw_hash,
        fingerprint_hash=fingerprint_hash,
        subnormals_preserved=subnormals_ok,
        artifact_created=artifact_created,
    )

    if print_hash:
        print(tensor_hash)
        return

    if json_out:
        print(_stable_json(asdict(summary)))
    else:
        for k, v in asdict(summary).items():
            print(f"{k}: {v}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
