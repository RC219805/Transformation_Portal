"""Unit tests for ``scripts/ci/check_per_package_coverage.py``.

The script gates CI by enforcing per-package coverage floors against
``coverage.xml``. Three behaviours are non-obvious enough to warrant
dedicated tests:

1. **Filename normalization** — Cobertura emits filenames in absolute,
   relative, ``./``-prefixed, and Windows-backslash forms depending on
   how coverage was configured. Prefix matching must work for all of
   them, otherwise CI gets false ``matched 0 covered source files``
   failures whenever the runner happens to write absolute paths.
2. **Nested-prefix exclusion** — a parent floor (e.g. ``lux_depth_v3/``)
   must NOT double-count files that belong to a stricter nested floor
   (e.g. ``lux_depth_v3/validators/``). Without exclusion the high
   validator coverage masks regressions in the rest of the parent.
3. **src/-stripping fallback** — some coverage configurations emit
   filenames without the leading ``src/`` segment; the script must
   match against the stripped form too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ci" / "check_per_package_coverage.py"
    spec = importlib.util.spec_from_file_location("check_per_package_coverage", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module so the @dataclass decorator inside the
    # script can resolve the module via sys.modules.get(__module__).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cobertura_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ci" / "cobertura_xml.py"
    spec = importlib.util.spec_from_file_location("cobertura_xml", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


@pytest.fixture(scope="module")
def cobertura_module():
    return _load_cobertura_module()


def _write_coverage(
    path: Path,
    classes: list[tuple[str, list[int]]],
    sources: list[str] | None = None,
) -> None:
    """Build a tiny synthetic Cobertura XML.

    ``classes`` is a list of (filename, hits_per_line) tuples — e.g.
    ``[("src/tp/x.py", [1, 0, 1])]`` means file with 3 lines, 2 hit.

    ``sources`` is an optional list of absolute ``<source>`` paths that
    coverage.py emits when ``--cov`` is given source roots; class
    filenames are typically relative to one of those roots when this
    is set. Pass ``None`` for the simpler case where filenames are
    already repo-relative.
    """
    body = ['<?xml version="1.0" ?>', "<coverage>"]
    if sources:
        body.append("  <sources>")
        for source in sources:
            body.append(f"    <source>{source}</source>")
        body.append("  </sources>")
    body.extend(["  <packages>", "    <package>", "      <classes>"])
    for filename, hits in classes:
        body.append(f'        <class name="x" filename="{filename}">')
        body.append("          <lines>")
        for i, h in enumerate(hits, start=1):
            body.append(f'            <line number="{i}" hits="{h}"/>')
        body.append("          </lines>")
        body.append("        </class>")
    body.extend(["      </classes>", "    </package>", "  </packages>", "</coverage>"])
    path.write_text("\n".join(body), encoding="utf-8")


class TestNormalizeFilename:
    def test_relative_path_unchanged(self, cobertura_module):
        assert cobertura_module._normalize_filename("src/tp/x.py") == "src/tp/x.py"

    def test_dot_slash_prefix_stripped(self, cobertura_module):
        assert cobertura_module._normalize_filename("./src/tp/x.py") == "src/tp/x.py"

    def test_backslashes_to_forward_slashes(self, cobertura_module):
        assert cobertura_module._normalize_filename("src\\tp\\x.py") == "src/tp/x.py"

    def test_absolute_path_sliced_at_src(self, cobertura_module):
        # The bug review thread #4 caught: lstrip("./") on an absolute
        # path silently dropped only the leading "/" producing
        # "home/.../src/tp/x.py" which would not match "src/tp/" prefix.
        absolute = "/home/runner/work/Transformation_Portal/Transformation_Portal/src/tp/foo.py"
        assert cobertura_module._normalize_filename(absolute) == "src/tp/foo.py"

    def test_absolute_path_with_src_in_repo_name_uses_last_src(self, cobertura_module):
        # If the repo path itself contains "src" (e.g. the repo is named
        # "my-src-repo"), only the LAST /src/ should be the slice point —
        # otherwise we'd grab a fictional "src" segment from the repo
        # name and produce broken paths.
        absolute = "/work/my-src-repo/src/tp/foo.py"
        assert cobertura_module._normalize_filename(absolute) == "src/tp/foo.py"

    def test_absolute_path_without_src_segment_unchanged_after_norm(self, cobertura_module):
        # Defensive: if there's no /src/ in an absolute path, leave it
        # alone (callers will see a 0-match failure with a clear cause
        # rather than a silently-truncated filename).
        absolute = "/var/tmp/standalone/x.py"
        assert cobertura_module._normalize_filename(absolute) == absolute


class TestMatchesPrefix:
    def test_direct_match(self, script_module):
        assert script_module._matches_prefix("src/tp/x.py", "src/tp/")

    def test_no_match_outside_prefix(self, script_module):
        assert not script_module._matches_prefix("src/transformation_portal/x.py", "src/tp/")

    def test_src_stripped_fallback(self, script_module):
        # Coverage configured with source = ["src/tp"] would emit
        # "tp/x.py" instead of "src/tp/x.py"; the matcher must still
        # accept it under the configured "src/tp/" prefix.
        assert script_module._matches_prefix("tp/x.py", "src/tp/")


class TestAggregateBasic:
    def test_passing_floor(self, script_module, tmp_path: Path):
        coverage = tmp_path / "coverage.xml"
        _write_coverage(coverage, [("src/tp/x.py", [1, 1, 1, 0])])  # 3/4 = 75%
        floors = (script_module.PackageFloor("src/tp/", 60.0),)

        results = script_module.aggregate(coverage, floors)
        assert len(results) == 1
        assert results[0].covered == 3
        assert results[0].valid == 4
        assert results[0].percentage == 75.0
        assert results[0].passed is True

    def test_failing_floor(self, script_module, tmp_path: Path):
        coverage = tmp_path / "coverage.xml"
        _write_coverage(coverage, [("src/tp/x.py", [1, 0, 0, 0])])  # 1/4 = 25%
        floors = (script_module.PackageFloor("src/tp/", 60.0),)

        results = script_module.aggregate(coverage, floors)
        assert results[0].percentage == 25.0
        assert results[0].passed is False

    def test_zero_match_is_failure(self, script_module, tmp_path: Path):
        coverage = tmp_path / "coverage.xml"
        _write_coverage(coverage, [("src/transformation_portal/x.py", [1, 1])])
        floors = (script_module.PackageFloor("src/nonexistent/", 0.0),)

        results = script_module.aggregate(coverage, floors)
        # A prefix that matches zero source files must fail rather than
        # silently pass with a 0/0 ratio — protects against typos in
        # PACKAGE_FLOORS or packages that have moved.
        assert results[0].valid == 0
        assert results[0].passed is False


class TestAggregateAbsolutePaths:
    def test_absolute_filename_matches_relative_prefix(self, script_module, tmp_path: Path):
        # This is the regression for review thread #4: absolute Cobertura
        # filenames must match relative configured prefixes.
        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [("/home/runner/work/Transformation_Portal/Transformation_Portal/src/tp/x.py", [1, 1, 1, 1])],
        )
        floors = (script_module.PackageFloor("src/tp/", 60.0),)

        results = script_module.aggregate(coverage, floors)
        assert results[0].valid == 4
        assert results[0].percentage == 100.0
        assert results[0].passed is True


class TestAggregateSourceRelative:
    """Regression tests for coverage.py's <source>-relative class filenames.

    coverage.py emits a top-level ``<sources>`` block listing the absolute
    roots that each ``<class filename>`` is relative to. The first
    iteration of this script naively prefix-matched filenames like
    ``merkle.py`` against ``src/tp/`` and got 0 matches across the board,
    silently failing every per-package floor with ``matched 0 covered
    source files``. The resolver now joins each filename to a probed
    source root and uses the first one where the file actually exists.
    """

    def test_filename_relative_to_source_root_resolves(self, script_module, tmp_path: Path):
        # Simulate the coverage.py shape: <source> root + bare filename.
        # The class file must exist on disk for the resolver to pick it,
        # so write the actual files into tmp_path.
        src_tp = tmp_path / "src" / "tp"
        src_tp.mkdir(parents=True)
        (src_tp / "merkle.py").write_text("# real file\n", encoding="utf-8")

        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [("merkle.py", [1, 1])],
            sources=[str(src_tp)],
        )
        floors = (script_module.PackageFloor("src/tp/", 50.0),)

        results = script_module.aggregate(coverage, floors)
        # The resolver must reconstruct "src/tp/merkle.py" so the floor
        # matches; otherwise this test fails the same way CI did with
        # "0/0 lines, matched 0 covered source files".
        assert results[0].valid == 2
        assert results[0].percentage == 100.0
        assert results[0].passed is True

    def test_filename_attributed_to_correct_source_when_two_roots(self, script_module, tmp_path: Path):
        # Two source roots, two unambiguous files (one per root, with
        # distinguishing path components in the filename). The src-only
        # file must aggregate under src/tp/, not under
        # src/transformation_portal/, even though both roots are listed.
        src_tp = tmp_path / "src" / "tp"
        src_tp.mkdir(parents=True)
        (src_tp / "phase4").mkdir()
        (src_tp / "phase4" / "exceptions.py").write_text("# tp file\n", encoding="utf-8")

        src_tp_main = tmp_path / "src" / "transformation_portal"
        src_tp_main.mkdir(parents=True)
        (src_tp_main / "lux_depth_v3").mkdir()
        (src_tp_main / "lux_depth_v3" / "orchestrator.py").write_text("# main file\n", encoding="utf-8")

        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [
                ("phase4/exceptions.py", [1, 1, 0, 0]),  # src/tp/ file: 50%
                ("lux_depth_v3/orchestrator.py", [1, 1, 1, 1]),  # main file: 100%
            ],
            sources=[str(src_tp), str(src_tp_main)],
        )
        floors = (
            script_module.PackageFloor("src/tp/", 40.0),
            script_module.PackageFloor("src/transformation_portal/lux_depth_v3/", 80.0),
        )

        results = script_module.aggregate(coverage, floors)
        tp_result, ldv3_result = results

        # phase4/exceptions.py only — src/transformation_portal/phase4/
        # doesn't exist on disk so it must NOT be cross-attributed.
        assert tp_result.valid == 4
        assert tp_result.percentage == 50.0

        # lux_depth_v3/orchestrator.py only — src/tp/lux_depth_v3/
        # doesn't exist on disk so it must NOT be cross-attributed.
        assert ldv3_result.valid == 4
        assert ldv3_result.percentage == 100.0

    def test_missing_file_does_not_inflate_other_packages(self, script_module, tmp_path: Path):
        # If a class filename doesn't exist under any source root (e.g.
        # the file was deleted between pytest and the coverage check),
        # the resolver falls back to the bare normalized filename rather
        # than fabricating an attribution. The class still appears in
        # results, but only against a prefix that matches the bare form
        # — it must not be silently rolled into an unrelated package.
        src_tp = tmp_path / "src" / "tp"
        src_tp.mkdir(parents=True)

        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [("ghost.py", [1, 0])],
            sources=[str(src_tp)],
        )
        floors = (script_module.PackageFloor("src/tp/", 50.0),)

        results = script_module.aggregate(coverage, floors)
        # Bare "ghost.py" doesn't match "src/tp/" prefix → 0/0 → fail loud.
        assert results[0].valid == 0
        assert results[0].passed is False


class TestAggregateNestedExclusion:
    def test_parent_excludes_nested_floor(self, script_module, tmp_path: Path):
        # The regression for review threads #1 and #3: parent rollup must
        # exclude files belonging to a stricter nested floor, otherwise
        # high nested coverage can mask regressions in the parent.
        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [
                # 2 lines in "rest of lux_depth_v3", 1 hit → 50%
                ("src/transformation_portal/lux_depth_v3/orchestrator.py", [1, 0]),
                # 4 lines in validators, all hit → 100% (would inflate parent
                # to (1+4)/(2+4) = 83% if double-counted, masking the 50%
                # parent regression)
                (
                    "src/transformation_portal/lux_depth_v3/validators/run_card_validator.py",
                    [1, 1, 1, 1],
                ),
            ],
        )
        floors = (
            script_module.PackageFloor("src/transformation_portal/lux_depth_v3/validators/", 80.0),
            script_module.PackageFloor(
                "src/transformation_portal/lux_depth_v3/",
                70.0,  # would pass with double-count (83%), fail without (50%)
                exclude_prefixes=("src/transformation_portal/lux_depth_v3/validators/",),
            ),
        )

        results = script_module.aggregate(coverage, floors)
        validators_result, parent_result = results

        assert validators_result.valid == 4
        assert validators_result.passed is True

        assert parent_result.valid == 2  # validators excluded
        assert parent_result.percentage == 50.0
        assert parent_result.passed is False

    def test_parent_without_exclusion_double_counts(self, script_module, tmp_path: Path):
        # Sanity check: with NO exclude_prefixes the parent does include
        # the nested files. This guards against accidentally making
        # exclusion the only path through aggregate().
        coverage = tmp_path / "coverage.xml"
        _write_coverage(
            coverage,
            [
                ("src/transformation_portal/lux_depth_v3/orchestrator.py", [1, 0]),
                (
                    "src/transformation_portal/lux_depth_v3/validators/run_card_validator.py",
                    [1, 1, 1, 1],
                ),
            ],
        )
        floors = (script_module.PackageFloor("src/transformation_portal/lux_depth_v3/", 70.0),)

        results = script_module.aggregate(coverage, floors)
        # 5/6 covered → 83% — passes the 70% floor because the high
        # nested coverage inflates the rollup. This is the *bug* the
        # exclude_prefixes feature defends against in the real config.
        assert results[0].covered == 5
        assert results[0].valid == 6


class TestMain:
    def test_missing_coverage_xml_returns_2(self, script_module, tmp_path: Path):
        rc = script_module.main([str(tmp_path / "missing.xml")])
        assert rc == 2

    def test_failure_returns_1(self, script_module, tmp_path: Path, capsys, monkeypatch):
        coverage = tmp_path / "coverage.xml"
        _write_coverage(coverage, [("src/tp/x.py", [1, 0, 0, 0])])
        monkeypatch.setattr(
            script_module,
            "PACKAGE_FLOORS",
            (script_module.PackageFloor("src/tp/", 80.0),),
        )

        rc = script_module.main([str(coverage)])
        assert rc == 1

    def test_success_returns_0(self, script_module, tmp_path: Path, monkeypatch):
        coverage = tmp_path / "coverage.xml"
        _write_coverage(coverage, [("src/tp/x.py", [1, 1, 1, 1])])
        monkeypatch.setattr(
            script_module,
            "PACKAGE_FLOORS",
            (script_module.PackageFloor("src/tp/", 80.0),),
        )

        rc = script_module.main([str(coverage)])
        assert rc == 0


class TestDefaultFloors:
    def test_stable_cold_zone_line_ratchets_are_configured(self, script_module):
        prefixes = [floor.prefix for floor in script_module.PACKAGE_FLOORS]
        assert len(prefixes) == len(set(prefixes))

        floors = {floor.prefix: floor.floor for floor in script_module.PACKAGE_FLOORS}
        expected_floors = {
            "src/tp/": 75.0,
            "src/transformation_portal/lux_depth_v3/validators/": 80.0,
            "src/transformation_portal/lux_depth_v3/": 72.0,
            "src/transformation_portal/plugins/": 50.0,
            "src/transformation_portal/stage_graph/": 74.0,
            "src/transformation_portal/vlm/": 69.0,
            "src/transformation_portal/depth/": 57.0,
            "src/transformation_portal/streaming/": 53.0,
            "src/transformation_portal/spatial_ai/reconstruction/": 42.0,
            "app.py": 79.0,
            "src/transformation_portal/orchestrator/storage/": 64.0,
            "src/transformation_portal/orchestrator/queue/": 63.0,
            "src/transformation_portal/orchestrator/artifact_store/": 62.0,
            "src/transformation_portal/metrics/ledger.py": 95.0,
            "src/transformation_portal/comfyui/workflow_builder.py": 96.0,
            "src/transformation_portal/comfyui/workflow_templates.py": 97.0,
            "src/transformation_portal/hardening/": 95.0,
            "src/transformation_portal/storage/cas_store.py": 92.0,
            "src/transformation_portal/orchestrator/worker.py": 95.0,
            "src/transformation_portal/dashboard/node_state_store.py": 95.0,
            "src/transformation_portal/dashboard/execution_manager.py": 85.0,
            "src/transformation_portal/dashboard/time_travel.py": 90.0,
            "src/transformation_portal/dashboard/studio_inspector.py": 82.0,
            "src/transformation_portal/dashboard/node_api.py": 92.0,
            "src/transformation_portal/dashboard/experiment_api.py": 92.0,
            "src/transformation_portal/dashboard/server.py": 78.0,
            "src/transformation_portal/dashboard/gpu_api.py": 55.0,
            "src/transformation_portal/dashboard/dag_api.py": 88.0,
            "src/transformation_portal/dashboard/artifact_api.py": 80.0,
            "src/transformation_portal/dashboard/artifact_preview.py": 88.0,
            "src/transformation_portal/dashboard/optimization_api.py": 65.0,
            "src/transformation_portal/dashboard/rl_api.py": 80.0,
            "src/transformation_portal/dashboard/dag_editor_api.py": 76.0,
            "src/transformation_portal/dashboard/execution_api.py": 86.0,
            "src/transformation_portal/dashboard/": 80.0,
        }

        assert floors == expected_floors


class TestRenderTable:
    def test_pass_and_fail_rows_distinguishable(self, script_module):
        results = [
            script_module.PackageResult(prefix="src/a/", floor=50.0, covered=8, valid=10),
            script_module.PackageResult(prefix="src/b/", floor=80.0, covered=4, valid=10),
        ]
        text = script_module.render_table(results)
        # Both packages appear; one PASS, one FAIL.
        assert "src/a/" in text and "PASS" in text
        assert "src/b/" in text and "FAIL" in text

    def test_zero_valid_renders_n_a(self, script_module):
        results = [
            script_module.PackageResult(prefix="src/missing/", floor=50.0, covered=0, valid=0),
        ]
        text = script_module.render_table(results)
        assert "N/A" in text
