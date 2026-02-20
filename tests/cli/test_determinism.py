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
    cli_path = repo_root / "src" / "transformation_portal" / "determinism" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "run", "--input", str(test_input), "--contract", "npy_tensor", "--print-hash"],
        capture_output=True,
        text=True,
        check=True,
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
