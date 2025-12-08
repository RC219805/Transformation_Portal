"""Dataset registry for benchmark validation packs.

This module maps a dataset_id (e.g. validation_v1) to a predictable on-disk layout:

data/benchmark_datasets/<dataset_id>/
  input/
  metadata.json
  baselines/
    <baseline_id>/
    manifest.json

This file is safe to add even if the broader validation system is still being implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    root_dir: Path
    input_dir: Path
    metadata_file: Path
    baselines_dir: Path
    baselines: Dict[str, Path]


def _repo_root() -> Path:
    # .../lux_depth_v2/validation/benchmark/dataset_registry.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    repo = _repo_root()
    root = repo / "data" / "benchmark_datasets" / dataset_id
    input_dir = root / "input"
    metadata_file = root / "metadata.json"
    baselines_dir = root / "baselines"

    baselines = {
        "topaz_photo": baselines_dir / "topaz_photo",
        "topaz_gigapixel": baselines_dir / "topaz_gigapixel",
        "topaz_video": baselines_dir / "topaz_video",
        "adobe_sr": baselines_dir / "adobe_sr",
        "adobe_neutral": baselines_dir / "adobe_neutral",
    }
    return DatasetSpec(
        dataset_id=dataset_id,
        root_dir=root,
        input_dir=input_dir,
        metadata_file=metadata_file,
        baselines_dir=baselines_dir,
        baselines=baselines,
    )


def list_input_images(dataset_id: str) -> List[Path]:
    spec = get_dataset_spec(dataset_id)
    if not spec.input_dir.exists():
        return []
    imgs: List[Path] = []
    for ext in ("*.tif", "*.tiff"):
        imgs.extend(spec.input_dir.glob(ext))
    return sorted(imgs)


def baseline_dir(dataset_id: str, baseline_id: str) -> Path:
    spec = get_dataset_spec(dataset_id)
    if baseline_id not in spec.baselines:
        raise KeyError(f"Unknown baseline_id: {baseline_id}")
    return spec.baselines[baseline_id]
