import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_determinism_cli_hash(tmp_path):
    # Arrange: write canonical deterministic tensor to temp dir
    repo_root = Path(__file__).parent.parent.parent
    test_input = tmp_path / "tensor.npy"
    np.save(test_input, np.zeros((4, 4, 3), dtype=np.float32), allow_pickle=False)

    # Act: run CLI and capture output
    cas_root = tmp_path / "cas"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_path}" if existing_path else src_path

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformation_portal.determinism.cli",
            "run",
            "--input",
            str(test_input),
            "--contract",
            "npy_tensor",
            "--cas-root",
            str(cas_root),
            "--print-hash",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
        env=env,
    )
    output = result.stdout.strip()

    # Remove 'sha256:' prefix if present
    if output.startswith("sha256:"):
        output = output[len("sha256:") :]

    # Assert: compare to baseline
    baseline_path = repo_root / "tests" / "cli" / "baselines" / "cli_hash.txt"
    with open(baseline_path) as f:
        baseline = f.read().strip()
    if baseline.startswith("sha256:"):
        baseline = baseline[len("sha256:") :]
    assert output == baseline, f"Hash drift: {output} != {baseline}"


def test_determinism_cli_writes_tensor_payloads(tmp_path):
    repo_root = Path(__file__).parent.parent.parent
    test_input = tmp_path / "tensor.npy"
    np.save(test_input, np.zeros((4, 4, 3), dtype=np.float32), allow_pickle=False)

    cas_root = tmp_path / "cas"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_path}" if existing_path else src_path

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformation_portal.determinism.cli",
            "run",
            "--input",
            str(test_input),
            "--contract",
            "npy_tensor",
            "--cas-root",
            str(cas_root),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
        env=env,
    )
    summary = json.loads(result.stdout)
    artifact_dir = cas_root / "sha256" / summary["tensor_hash"]

    assert (artifact_dir / "output_tensor.bin").exists()
    assert (artifact_dir / "output_tensor.npy").exists()
    assert (artifact_dir / "tensor_meta.json").exists()
    assert (artifact_dir / "artifact_manifest.json").exists()


def test_determinism_cli_verify_roundtrip(tmp_path):
    repo_root = Path(__file__).parent.parent.parent
    test_input = tmp_path / "tensor.npy"
    np.save(test_input, np.zeros((4, 4, 3), dtype=np.float32), allow_pickle=False)

    cas_root = tmp_path / "cas"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_path}" if existing_path else src_path

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformation_portal.determinism.cli",
            "run",
            "--input",
            str(test_input),
            "--contract",
            "npy_tensor",
            "--cas-root",
            str(cas_root),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
        env=env,
    )
    artifact_id = json.loads(run_result.stdout)["artifact_id"]

    verify_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformation_portal.determinism.cli",
            "verify",
            "--baseline",
            artifact_id,
            "--artifact",
            artifact_id,
            "--cas-root",
            str(cas_root),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
        env=env,
    )
    verification = json.loads(verify_result.stdout)
    assert verification["status"] == "pass"
