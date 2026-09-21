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


@pytest.mark.parametrize(
    "source",
    [
        "import accelerate as a\na.load_checkpoint_and_dispatch(model, path)\nimport math as a",
        "def unsafe():\n import accelerate as a\n a.load_checkpoint_and_dispatch(model, path)\n"
        "def unrelated():\n import math as a\n return a.pi",
        "def unsafe():\n from accelerate import utils as u\n u.load_checkpoint_in_model(model, path)\n"
        "def unrelated():\n from pathlib import Path as u\n return u('.')",
        "import accelerate as a\nloader = a\nloader.load_checkpoint_and_dispatch(model, path)\n"
        "import importlib\nloader = importlib",
        "import importlib as i\na = i.import_module('accelerate')\n"
        "a.load_checkpoint_and_dispatch(model, path)\nimport math as i",
    ],
)
def test_reused_names_cannot_erase_unsafe_provenance(source: str) -> None:
    assert any("unsafe Accelerate checkpoint API" in reason for _, reason in check_source(source))


def test_reused_names_preserve_safe_dispatch_and_offload() -> None:
    source = """
def supported():
    import accelerate as a
    a.dispatch_model(model, device_map="auto")
    a.cpu_offload(model)

def unrelated():
    import math as a
    return a.pi
"""
    assert check_source(source) == []


def test_unparseable_source_fails_closed() -> None:
    assert check_source("def incomplete(")


def test_self_referential_module_alias_fails_with_bounded_work() -> None:
    findings = check_source("import accelerate as a\na = a.foo\n")
    assert findings == [(1, "Accelerate alias resolution did not converge within its 16-pass bound")]


def test_branching_alias_cycle_fails_before_identity_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checker, "MAX_IDENTITIES_PER_ALIAS", 8)
    findings = check_source("import accelerate as a\na = a.foo\na = a.bar\na = a.baz\n")
    assert findings == [(1, "Accelerate alias resolution exceeded its per-alias identity budget")]


def test_alias_fanout_fails_at_total_identity_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checker, "MAX_TOTAL_ALIAS_IDENTITIES", 3)
    findings = check_source("import accelerate as a\nb = a\nc = a\nd = a\n")
    assert findings == [(1, "Accelerate alias resolution exceeded its total identity budget")]


def test_identity_budget_is_checked_before_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checker, "MAX_IDENTITIES_PER_ALIAS", 1)
    identities = {"accelerate"}
    with pytest.raises(checker._AliasBudgetExceeded, match="per-alias identity budget"):
        checker._add_identity(identities, "accelerate.utils")
    assert identities == {"accelerate"}


def test_safe_self_assignment_converges() -> None:
    assert check_source("import accelerate as a\na = a\na.dispatch_model(model, device_map='auto')\n") == []
