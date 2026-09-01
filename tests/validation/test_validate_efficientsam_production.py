"""Focused contracts for the maintained EfficientSAM validation entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validation import validate_efficientsam_production
from transformation_portal.lux_depth_v3.input_manager import ImageInput

pytestmark = pytest.mark.unit


def test_validation_wraps_planned_paths_as_image_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Python validator must call the typed orchestrator API."""

    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle_module
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_images" / "800 Picacho"
    input_dir.mkdir(parents=True)
    image_path = input_dir / "sample.jpg"
    image_path.touch()
    captured: dict[str, object] = {}

    def prepare(config: object, root: Path, files: list[Path]) -> SimpleNamespace:
        captured["prepared_files"] = tuple(files)
        return SimpleNamespace(runtime_config=config, input_files=tuple(path.resolve() for path in files))

    class CapturingOrchestrator:
        @classmethod
        def from_prepared(
            cls,
            prepared: SimpleNamespace,
            output_root: Path,
        ) -> "CapturingOrchestrator":
            captured["prepared"] = prepared
            captured["output_root"] = output_root
            return cls()

        def enhance_image(self, image_input: ImageInput) -> dict[str, str]:
            captured["image_input"] = image_input
            return {"status": "ok"}

    monkeypatch.setattr(lifecycle_module, "prepare_lux_execution", prepare)
    monkeypatch.setattr(orchestrator_module, "EnhanceOrchestrator", CapturingOrchestrator)

    assert validate_efficientsam_production.run_validation() == 0
    assert captured["prepared_files"] == (Path("input_images/800 Picacho/sample.jpg"),)
    image_input = captured["image_input"]
    assert isinstance(image_input, ImageInput)
    assert image_input.path == image_path.resolve()
