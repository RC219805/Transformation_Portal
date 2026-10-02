"""Named-time propagation and validation for the limited SkyGAN executor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

NAMED_TIMES = [
    ("sunrise", 6.5),
    ("morning", 9.0),
    ("midday", 12.0),
    ("golden_hour", 17.0),
    ("sunset", 18.5),
    ("twilight", 19.5),
]


@pytest.fixture
def capture_executor(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    from transformation_portal.comfyui import executor as executor_module

    observed = {}

    class CapturePresets:
        def get_atmospheric_parameters(self, location, season):
            observed["atmosphere"] = (location, season)
            return SimpleNamespace()

        def get_sky_parameters(self, location, *, season, time_of_day=None):
            observed["sky"] = (location, season, time_of_day)
            return SimpleNamespace(cloud_coverage=None)

    def capture_render(**kwargs):
        observed["render"] = kwargs
        return kwargs["source_image"], SimpleNamespace(message="captured sky")

    monkeypatch.setattr(executor_module, "LocationPresets", CapturePresets)
    executor = executor_module.WorkflowExecutor(cache_models=False)
    executor._sky_blender = SimpleNamespace(smart_render=capture_render)
    return executor, np.zeros((2, 2, 3), dtype=np.uint8), observed


@pytest.mark.parametrize(("slot", "hour"), NAMED_TIMES)
def test_executor_passes_named_time_to_presets(capture_executor, slot: str, hour: float) -> None:
    import numpy as np

    from transformation_portal.comfyui.workflow_builder import Node, NodeType, Workflow

    executor, image, observed = capture_executor
    node = Node(
        "sky",
        NodeType.SKYGAN_SKY,
        parameters={"image": image, "location": "montecito", "season": "summer", "time_of_day": slot},
    )

    result = executor.execute(Workflow(nodes={"sky": node}))

    assert result["success"] is True
    assert observed["sky"] == ("montecito", "summer", hour)
    assert observed["atmosphere"] == ("montecito", "summer")
    assert result["node_outputs"]["sky"]["REPORT"] == "captured sky"
    np.testing.assert_array_equal(result["node_outputs"]["sky"]["IMAGE"], image)


def test_executor_defaults_to_golden_hour(capture_executor) -> None:
    from transformation_portal.comfyui.workflow_builder import Node, NodeType

    executor, image, observed = capture_executor
    executor._execute_skygan_node(Node("sky", NodeType.SKYGAN_SKY), {"image": image})

    assert observed["sky"] == ("montecito", "summer", 17.0)


@pytest.mark.parametrize("slot", ["not_a_real_slot", "", None])
def test_executor_rejects_invalid_time_before_preset_or_render_work(monkeypatch: pytest.MonkeyPatch, slot) -> None:
    from transformation_portal.comfyui import executor as executor_module
    from transformation_portal.comfyui.workflow_builder import Node, NodeType

    def forbidden_work(*args, **kwargs):
        pytest.fail("invalid time reached preset or render work")

    monkeypatch.setattr(executor_module, "LocationPresets", forbidden_work)
    executor = executor_module.WorkflowExecutor(cache_models=False)
    executor._sky_blender = SimpleNamespace(smart_render=forbidden_work)

    with pytest.raises(ValueError, match="Unknown time_of_day") as error:
        executor._execute_skygan_node(Node("sky", NodeType.SKYGAN_SKY), {"image": [[0]], "time_of_day": slot})

    assert str([name for name, _ in NAMED_TIMES]) in str(error.value)


def test_executor_preserves_missing_image_error_before_time_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.comfyui import executor as executor_module
    from transformation_portal.comfyui.workflow_builder import Node, NodeType

    def forbidden_presets():
        pytest.fail("missing image reached preset work")

    monkeypatch.setattr(executor_module, "LocationPresets", forbidden_presets)
    executor = executor_module.WorkflowExecutor(cache_models=False)

    with pytest.raises(ValueError, match="SkyGAN missing image"):
        executor._execute_skygan_node(Node("sky", NodeType.SKYGAN_SKY), {"time_of_day": "not_a_real_slot"})


def test_named_time_authority_imports_without_optional_runtimes() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "from transformation_portal.comfyui.skygan_time import resolve_skygan_time_of_day; "
                "import sys; "
                "assert resolve_skygan_time_of_day('golden_hour') == 17.0; "
                "assert not {'torch', 'numpy', 'cv2', 'transformation_portal.comfyui.custom_nodes', "
                "'transformation_portal.comfyui.executor'}.intersection(sys.modules)"
            ),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
