"""Pipeline Foundation Model (PFM) data pipeline.

This module handles conversion of execution logs into canonical
sequences for foundation model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """Single execution step record."""

    node_id: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    diff: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    action: int | None = None
    timestamp: float = 0.0


@dataclass
class RunRecord:
    """Complete run record."""

    run_id: str
    pipeline_id: str
    steps: list[StepRecord] = field(default_factory=list)
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def build_sequence(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a run into a time-ordered sequence.

    Each step contains:
    - node_id: Which node executed
    - config: Node configuration (normalized)
    - metrics: Evaluation metrics (score, psnr, lpips, llava)
    - diff: Semantic diff summary (counts by type)
    - artifacts: Artifact hashes

    Args:
        run: Raw run dictionary from logs

    Returns:
        List of step dictionaries
    """
    seq = []

    for step in run.get("steps", []):
        seq.append(
            {
                "node_id": step.get("node_id", "unknown"),
                "config": _normalize_config(step.get("config", {})),
                "metrics": _normalize_metrics(step.get("metrics", {})),
                "diff": _normalize_diff(step.get("diff", {})),
                "artifacts": step.get("artifacts", {}),
                "action": step.get("action"),
            }
        )

    return seq


def _normalize_config(config: dict[str, Any]) -> dict[str, float]:
    """Normalize configuration values to [0, 1] range."""
    normalized = {}

    # Common config keys with their ranges
    ranges = {
        "threshold": (0.0, 1.0),
        "steps": (0, 1000),
        "iterations": (0, 1000),
        "roughness_bias": (-1.0, 1.0),
        "metalness": (0.0, 1.0),
        "exposure": (0.0, 2.0),
    }

    for key, value in config.items():
        if key in ranges and isinstance(value, (int, float)):
            lo, hi = ranges[key]
            normalized[key] = (float(value) - lo) / (hi - lo + 1e-8)
        elif isinstance(value, (int, float)):
            normalized[key] = float(value)

    return normalized


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize metrics to consistent ranges."""
    return {
        "score": float(metrics.get("score", 0.0)),
        "psnr": float(metrics.get("psnr", 0.0)) / 50.0,  # Typical range 0-50
        "lpips": 1.0 - float(metrics.get("lpips", 0.0)),  # Invert (lower is better)
        "llava": float(metrics.get("llava", 0.0)),
        "ssim": float(metrics.get("ssim", 0.0)),
    }


def _normalize_diff(diff: dict[str, Any]) -> dict[str, int]:
    """Extract diff type counts."""
    counts = {
        "geometry": 0,
        "texture": 0,
        "missing": 0,
        "artifact": 0,
        "semantic": 0,
    }

    for change in diff.get("changes", []):
        ctype = change.get("type", "").lower()
        if ctype in counts:
            counts[ctype] += 1

    return counts


class SequenceDataset:
    """Dataset of canonicalized run sequences for PFM training."""

    def __init__(self, runs: list[dict[str, Any]], seq_len: int = 16) -> None:
        self.sequences = [build_sequence(run) for run in runs]
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def sample(self, batch_size: int) -> list[list[dict[str, Any]]]:
        """Sample batch of sequences.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of sequences
        """
        import random

        batch = []
        for _ in range(batch_size):
            seq = random.choice(self.sequences)

            # Truncate or pad to seq_len
            if len(seq) >= self.seq_len:
                start = random.randint(0, len(seq) - self.seq_len)
                batch.append(seq[start : start + self.seq_len])
            else:
                # Pad with last step repeated
                padded = seq + [seq[-1]] * (self.seq_len - len(seq))
                batch.append(padded)

        return batch


def load_runs_from_experiment_db(db_path: str) -> list[dict[str, Any]]:
    """Load run records from experiment database.

    Args:
        db_path: Path to experiment database

    Returns:
        List of run dictionaries
    """
    import json
    from pathlib import Path

    runs = []
    db_file = Path(db_path)

    if db_file.exists():
        with open(db_file) as f:
            data = json.load(f)
            runs = data.get("runs", [])

    return runs


def load_runs_from_merkle_dag(merkle_path: str) -> list[dict[str, Any]]:
    """Load run records from Merkle DAG lineage.

    Reconstructs execution history from artifact lineage.

    Args:
        merkle_path: Path to Merkle DAG JSON

    Returns:
        List of reconstructed run dictionaries
    """
    import json
    from pathlib import Path

    runs = []
    dag_file = Path(merkle_path)

    if dag_file.exists():
        with open(dag_file) as f:
            dag = json.load(f)

        # Reconstruct runs from lineage
        for node_hash, node_data in dag.get("nodes", {}).items():
            metadata = node_data.get("metadata", {})
            if "run_id" in metadata:
                # This is a run node
                runs.append(
                    {
                        "run_id": metadata["run_id"],
                        "steps": metadata.get("steps", []),
                        "final_score": metadata.get("score", 0.0),
                    }
                )

    return runs
