"""Contract tests for the raw json.dump/json.dumps source guardrail."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_GATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "check_raw_json_usage.py"
_spec = importlib.util.spec_from_file_location("check_raw_json_usage", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
# Register before exec: the gate defines a dataclass, which resolves its module via sys.modules.
sys.modules.setdefault(_spec.name, _module)
_spec.loader.exec_module(_module)
check_raw_json_usage = _module.check_raw_json_usage


def _scan(tmp_path: Path, source: str, approved: str = "# none\n") -> list[tuple[int, str]]:
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.py").write_text(source, encoding="utf-8")
    approved_file = tmp_path / "approved.txt"
    approved_file.write_text(approved, encoding="utf-8")
    violations = check_raw_json_usage(roots=[root], approved_file=approved_file)
    return sorted((v.lineno, v.call_name) for v in violations)


def test_direct_json_dumps_is_flagged(tmp_path: Path) -> None:
    assert _scan(tmp_path, "import json\nprint(json.dumps({}))\n") == [(2, "dumps")]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("from json import dumps, dump\nprint(dumps({}))\ndump({}, None)\n", [(2, "dumps"), (3, "dump")]),
        ("from json import dumps as to_text\nprint(to_text({}))\n", [(2, "dumps")]),
        ("import json as _j\nprint(_j.dumps({}))\n", [(2, "dumps")]),
        ("def f():\n    return _j.dump({}, None)\n\nimport json as _j\n", [(2, "dump")]),
    ],
    ids=["from-import", "from-import-alias", "module-alias", "alias-bound-after-use"],
)
def test_aliased_json_imports_are_flagged(tmp_path: Path, source: str, expected: list[tuple[int, str]]) -> None:
    assert _scan(tmp_path, source) == expected


def test_unrelated_names_are_not_flagged(tmp_path: Path) -> None:
    source = (
        "import json\n"
        "from mypkg import dumps\n"
        "from json import loads\n"
        "import other as _j\n"
        "print(dumps({}), loads('{}'), _j.dumps({}), json.loads('{}'))\n"
    )
    assert _scan(tmp_path, source) == []


def test_approved_module_is_exempt_for_aliased_imports(tmp_path: Path) -> None:
    assert _scan(tmp_path, "from json import dumps\nprint(dumps({}))\n", approved="*/src/a.py\n") == []
