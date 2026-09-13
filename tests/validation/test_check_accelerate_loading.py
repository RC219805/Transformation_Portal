"""Regression protection for unpatched Accelerate checkpoint-loader reachability."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

# Validation scripts are CLI-owned files rather than an importable package.
# Match the explicit file loader used by the adjacent torch-load policy tests.
TOOL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "check_accelerate_loading.py"
SPEC = importlib.util.spec_from_file_location("check_accelerate_loading", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)
check_source = checker.check_source


@pytest.mark.parametrize(
    "source",
    [
        "from accelerate import load_checkpoint_and_dispatch as load\nload(model, path)",
        "from accelerate.utils.modeling import load_checkpoint_in_model",
        "import accelerate as a\na.load_checkpoint_and_dispatch(model, path)",
        "import accelerate.utils.modeling as m\nm.load_checkpoint_in_model(model, path)",
        "from accelerate import utils as u\nloader = u.load_checkpoint_in_model\nloader(model, path)",
        'import accelerate as a\ngetattr(a, "load_checkpoint_and_dispatch")(model, path)',
        'import importlib as i\na = i.import_module("accelerate")\na.load_checkpoint_and_dispatch(model, path)',
        'a = __import__("accelerate")\na.load_checkpoint_and_dispatch(model, path)',
        "from accelerate.utils import *",
    ],
)
def test_unsafe_checkpoint_api_references_fail(source: str) -> None:
    assert check_source(source)


def test_dispatch_offload_and_examples_remain_supported() -> None:
    source = """
from accelerate import dispatch_model, cpu_offload, init_empty_weights
from accelerate.utils import load_offloaded_weights
dispatch_model(model, device_map="auto")
cpu_offload(model)
# accelerate.load_checkpoint_and_dispatch is prohibited.
example = "from accelerate import load_checkpoint_in_model"
other.load_checkpoint_in_model(model, path)
"""
    assert check_source(source) == []


def test_unparseable_source_fails_closed() -> None:
    assert check_source("def incomplete(")


def test_self_referential_module_alias_fails_with_bounded_work() -> None:
    findings = check_source("import accelerate as a\na = a.foo\n")
    assert findings == [(1, "Accelerate alias resolution did not converge within its 16-pass bound")]
