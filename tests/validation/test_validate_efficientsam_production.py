"""Focused contracts for the maintained EfficientSAM validation entrypoint."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from transformation_portal.lux_depth_v3.input_manager import ImageInput

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
        assert all(path.is_absolute() for path in files)
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
    assert captured["prepared_files"] == (image_path.absolute(),)
    image_input = captured["image_input"]
    assert isinstance(image_input, ImageInput)
    assert image_input.path == image_path.resolve()
