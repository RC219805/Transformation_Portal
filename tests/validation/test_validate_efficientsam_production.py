"""Focused contracts for the maintained EfficientSAM validation entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/validation/validate_efficientsam_production.py"


def _load_script(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_efficientsam_production = _load_script("validate_efficientsam_production_under_test")


def test_import_does_not_configure_process_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test collection must not mutate the process-wide root logger."""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    _load_script("validate_efficientsam_production_logging_test")

    assert not calls


def test_direct_execution_logging_configuration_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    validate_efficientsam_production._configure_logging()

    assert calls == [
        (
            (),
            {
                "level": logging.INFO,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        )
    ]


def test_validation_executes_the_complete_prepared_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Python validator must use the evidence-authoritative batch API."""

    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module
    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle_module
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_images" / "800 Picacho"
    input_dir.mkdir(parents=True)
    image_paths = (input_dir / "sample.jpg", input_dir / "second.jpg")
    for image_path in image_paths:
        image_path.touch()
    captured: dict[str, object] = {}
    prepared_plan = object()

    def prepare(config: object, root: Path, files: list[Path]) -> SimpleNamespace:
        assert all(path.is_absolute() for path in files)
        captured["prepared_files"] = tuple(files)
        return SimpleNamespace(
            runtime_config=config,
            plan=prepared_plan,
            input_root=root.resolve(),
            input_files=tuple(path.resolve() for path in files),
        )

    class CapturingOrchestrator:
        def __init__(self, output_root: Path) -> None:
            self.output_root = output_root

        @classmethod
        def from_prepared(
            cls,
            prepared: SimpleNamespace,
            output_root: Path,
        ) -> "CapturingOrchestrator":
            captured["prepared"] = prepared
            captured["output_root"] = output_root
            return cls(output_root)

        def enhance_batch(self, input_root: Path, *, input_files: list[Path]) -> list[dict[str, object]]:
            captured["batch_input_root"] = input_root
            captured["batch_input_files"] = tuple(input_files)
            evidence_path = self.output_root / "manifests" / "execution_evidence_test.json"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text("{}", encoding="utf-8")
            results: list[dict[str, object]] = []
            for input_file in input_files:
                stem = input_file.stem
                mask_path = self.output_root / "segmentation" / f"{stem}_materials_v3_masks.npz"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(mask_path, stone=np.ones((2, 3), dtype=np.float32))
                manifest_path = self.output_root / "manifests" / f"{stem}_combined.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "environment": {
                                "execution_contract": {
                                    "execution_evidence_path": ("manifests/execution_evidence_test.json"),
                                }
                            },
                            "materials_v3": {
                                "enabled": True,
                                "segmentation_metadata": {
                                    "backend": "efficientsam",
                                    "mask_count": 1,
                                    "mask_artifact_path": str(mask_path),
                                    "mask_artifact_format": "npz",
                                    "mask_artifact_shape": [2, 3],
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                results.append(
                    {
                        "status": "ok",
                        "runtime_s": 0.25,
                        "manifest": str(manifest_path),
                        "segmentation_mask_path": str(mask_path),
                    }
                )
            return results

    monkeypatch.setattr(lifecycle_module, "prepare_lux_execution", prepare)
    monkeypatch.setattr(orchestrator_module, "EnhanceOrchestrator", CapturingOrchestrator)
    verification_calls: list[tuple[Path, Path, object]] = []

    def verify_evidence(path: Path, *, output_root: Path, plan: object) -> dict[str, object]:
        verification_calls.append((path, output_root, plan))
        artifacts = []
        for artifact_path in sorted((*output_root.glob("manifests/*_combined.json"), *output_root.glob("segmentation/*.npz"))):
            data = artifact_path.read_bytes()
            artifacts.append(
                {
                    "path": artifact_path.relative_to(output_root).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        return {"failed_artifacts": [], "produced_artifacts": [{"artifacts": artifacts}]}

    monkeypatch.setattr(evidence_module, "verify_execution_evidence_file", verify_evidence)
    monkeypatch.setattr(evidence_module, "require_required_artifacts", lambda payload: None)

    assert validate_efficientsam_production.run_validation() == 0
    assert captured["prepared_files"] == tuple(path.absolute() for path in image_paths)
    assert captured["batch_input_root"] == input_dir.resolve()
    assert captured["batch_input_files"] == tuple(path.resolve() for path in image_paths)
    assert len(verification_calls) == 1
    assert verification_calls[0][1:] == (captured["output_root"].resolve(), prepared_plan)


@pytest.mark.parametrize(
    "archive_kind",
    ("not-npz", "truncated-zip", "duplicate-members"),
)
def test_validation_rejects_corrupt_mask_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_kind: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    manifest_path = output_root / "manifests" / "sample_combined.json"
    mask_path = output_root / "segmentation" / "sample_materials_v3_masks.npz"
    evidence_path = output_root / "manifests" / "execution_evidence_test.json"
    manifest_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    if archive_kind == "duplicate-members":
        array_buffer = io.BytesIO()
        np.save(array_buffer, np.ones((2, 3), dtype=np.float32))
        archive_buffer = io.BytesIO()
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(archive_buffer, mode="w") as archive:
                archive.writestr("stone.npy", array_buffer.getvalue())
                archive.writestr("stone.npy", array_buffer.getvalue())
        mask_path.write_bytes(archive_buffer.getvalue())
    elif archive_kind == "truncated-zip":
        mask_path.write_bytes(b"PK\x03\x04")
    else:
        mask_path.write_bytes(b"not-an-npz")
    evidence_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "environment": {
                    "execution_contract": {
                        "execution_evidence_path": "manifests/execution_evidence_test.json",
                    }
                },
                "materials_v3": {
                    "enabled": True,
                    "segmentation_metadata": {
                        "backend": "efficientsam",
                        "mask_count": 1,
                        "mask_artifact_path": str(mask_path),
                        "mask_artifact_format": "npz",
                        "mask_artifact_shape": [2, 3],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def verify_evidence(*args, **kwargs):
        artifacts = []
        for artifact_path in (manifest_path, mask_path):
            data = artifact_path.read_bytes()
            artifacts.append(
                {
                    "path": artifact_path.relative_to(output_root).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        return {"produced_artifacts": [{"artifacts": artifacts}]}

    monkeypatch.setattr(evidence_module, "verify_execution_evidence_file", verify_evidence)
    monkeypatch.setattr(evidence_module, "require_required_artifacts", lambda payload: None)

    with pytest.raises(ValueError, match="not a readable safe NPZ"):
        validate_efficientsam_production._validate_efficientsam_evidence(
            {
                "manifest": str(manifest_path),
                "segmentation_mask_path": str(mask_path),
            },
            output_root=output_root,
            plan=object(),
        )


def test_validation_rejects_mask_replaced_after_evidence_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    manifest_path = output_root / "manifests" / "sample_combined.json"
    mask_path = output_root / "segmentation" / "sample_materials_v3_masks.npz"
    evidence_path = output_root / "manifests" / "execution_evidence_test.json"
    manifest_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    np.savez_compressed(mask_path, stone=np.ones((2, 3), dtype=np.float32))
    evidence_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "environment": {
                    "execution_contract": {
                        "execution_evidence_path": "manifests/execution_evidence_test.json",
                    }
                },
                "materials_v3": {
                    "enabled": True,
                    "segmentation_metadata": {
                        "backend": "efficientsam",
                        "mask_count": 1,
                        "mask_artifact_path": str(mask_path),
                        "mask_artifact_format": "npz",
                        "mask_artifact_shape": [2, 3],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_data = manifest_path.read_bytes()
    original_mask_data = mask_path.read_bytes()

    def verify_then_replace(*args, **kwargs):
        np.savez_compressed(mask_path, stone=np.zeros((2, 3), dtype=np.float32))
        return {
            "produced_artifacts": [
                {
                    "artifacts": [
                        {
                            "path": manifest_path.relative_to(output_root).as_posix(),
                            "sha256": hashlib.sha256(manifest_data).hexdigest(),
                            "size_bytes": len(manifest_data),
                        },
                        {
                            "path": mask_path.relative_to(output_root).as_posix(),
                            "sha256": hashlib.sha256(original_mask_data).hexdigest(),
                            "size_bytes": len(original_mask_data),
                        },
                    ]
                }
            ]
        }

    monkeypatch.setattr(evidence_module, "verify_execution_evidence_file", verify_then_replace)
    monkeypatch.setattr(evidence_module, "require_required_artifacts", lambda payload: None)

    with pytest.raises(ValueError, match="mask bytes do not match canonical execution evidence"):
        validate_efficientsam_production._validate_efficientsam_evidence(
            {
                "manifest": str(manifest_path),
                "segmentation_mask_path": str(mask_path),
            },
            output_root=output_root,
            plan=object(),
        )


def test_validation_rejects_ok_status_without_efficientsam_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``ok`` status alone must not make the production validator green."""

    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle_module
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_images" / "800 Picacho"
    input_dir.mkdir(parents=True)
    (input_dir / "sample.jpg").touch()

    def prepare(config: object, root: Path, files: list[Path]) -> SimpleNamespace:
        return SimpleNamespace(
            runtime_config=config,
            plan=object(),
            input_root=root.resolve(),
            input_files=tuple(path.resolve() for path in files),
        )

    class ArtifactFreeOrchestrator:
        @classmethod
        def from_prepared(
            cls,
            prepared: SimpleNamespace,
            output_root: Path,
        ) -> "ArtifactFreeOrchestrator":
            return cls()

        def enhance_batch(self, input_root: Path, *, input_files: list[Path]) -> list[dict[str, object]]:
            return [{"status": "ok", "runtime_s": 0.25}]

    monkeypatch.setattr(lifecycle_module, "prepare_lux_execution", prepare)
    monkeypatch.setattr(orchestrator_module, "EnhanceOrchestrator", ArtifactFreeOrchestrator)

    assert validate_efficientsam_production.run_validation() == 1
