"""Exercise CI path classification and preserve independent enforcement checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _classify(files: list[dict[str, str]], *, event: str = "pull_request", changed_files: int | None = None) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the GitHub Actions classifier")
    script = _workflow("build.yml")["jobs"]["preflight"]["steps"][0]["with"]["script"]
    harness = """
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const outputs = {};
const core = {setOutput: (key, value) => { outputs[key] = value; }, info: () => {}};
const context = {
  eventName: input.event,
  payload: {pull_request: {number: 123, changed_files: input.changed_files}},
  repo: {owner: 'example', repo: 'portal'},
};
const github = {
  rest: {pulls: {listFiles: 'listFiles'}},
  paginate: async (method, args) => {
    if (context.eventName !== 'pull_request') throw new Error('Non-PR must not list files');
    if (method !== 'listFiles' || args.pull_number !== 123 || args.per_page !== 100) {
      throw new Error('Incorrect paginated PR inventory request');
    }
    return input.files;
  },
};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('core', 'context', 'github', input.script)(core, context, github)
  .then(() => process.stdout.write(JSON.stringify(outputs)))
  .catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps(
            {
                "script": script,
                "event": event,
                "files": files,
                "changed_files": len(files) if changed_files is None else changed_files,
            }
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "path",
    [
        "app.py",
        "src/tp/phase4/hash_capture_metadata.py",
        "tests/test_app_orchestrator_runtime.py",
        "migrations/versions/0001_state.py",
        "policy/ingest.yaml",
        "schemas/phase4/metadata.schema.json",
        "docs/schemas/run_card/run_card.v2.schema.json",
        "docs/archive/schemas/archive_manifest_v2.schema.json",
        "docs/compliance/schemas/risk_assessment_report.schema.json",
        ".github/agents/transformation-portal-architect.md",
        ".github/agents/rag_system/classifier.py",
        ".pre-commit-config.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "requirements/core.txt",
    ],
)
def test_runtime_and_governance_changes_require_full_ci(path: str) -> None:
    outputs = _classify([{"filename": path}])

    assert outputs["run_full"] == "true"
    assert outputs["run_frontdoor"] == "false"


@pytest.mark.parametrize(
    "path",
    [
        "web/secure-landing/package-lock.json",
        "web/shared/shared-ui-tokens.css",
        "cloudflare/transformationportal-worker/src/index.ts",
        "public/portal-assets/portal.js",
        "public/shared/tokens.css",
        "config/portal_asset_manifest.json",
        "config/portal_asset_budgets.json",
        "package.json",
        "package-lock.json",
        "wrangler.jsonc",
        ".github/workflows/build.yml",
        "portal.html",
        "scripts/validation/validate_frontdoor_browser_smoke.py",
        "scripts/validation/validate_portal_css_layer_parity.py",
        "tests/fixtures/portal-css/layer-parity-contract.json",
        "tests/fixtures/frontdoor_identity_v1.json",
        "docs/guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md",
    ],
)
def test_frontdoor_inputs_require_browser_contracts(path: str) -> None:
    outputs = _classify([{"filename": path}])

    assert outputs["run_full"] == "true"
    assert outputs["run_frontdoor"] == "true"


@pytest.mark.parametrize("path", ["README.md", "docs/guides/QUICKSTART.md", "CHANGELOG.md"])
def test_ordinary_documentation_keeps_lightweight_ci(path: str) -> None:
    outputs = _classify([{"filename": path}])

    assert outputs["run_full"] == "false"
    assert outputs["run_frontdoor"] == "false"


@pytest.mark.parametrize("previous_path,frontdoor", [("app.py", "false"), ("public/portal-assets/portal.js", "true")])
def test_renames_out_of_runtime_paths_retain_required_validation(previous_path: str, frontdoor: str) -> None:
    outputs = _classify([{"filename": "docs/archive/old-source.txt", "previous_filename": previous_path}])

    assert outputs["run_full"] == "true"
    assert outputs["run_frontdoor"] == frontdoor


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_non_pr_events_require_all_contracts_without_api_inventory(event: str) -> None:
    outputs = _classify([], event=event)

    assert outputs["run_full"] == "true"
    assert outputs["run_frontdoor"] == "true"


@pytest.mark.parametrize("returned_count,declared_count", [(0, 0), (1, 2), (2, 1), (3000, 3000), (3000, 3001)])
def test_empty_stale_or_capped_inventory_fails_closed(returned_count: int, declared_count: int) -> None:
    files = [{"filename": f"docs/note-{index}.md"} for index in range(returned_count)]

    outputs = _classify(files, changed_files=declared_count)

    assert outputs["run_full"] == "true"
    assert outputs["run_frontdoor"] == "true"
    assert "inventory" in outputs["reason"]


def test_enforcement_classifier_has_read_only_pr_inventory_permission() -> None:
    workflow = _workflow("enforcement.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["changes"]["permissions"] == {"contents": "read", "pull-requests": "read"}


@pytest.mark.parametrize(
    "job_id",
    ["action-pins", "no-tautological-tests", "banned-dependencies", "hf-revision-policy", "test-layer1", "artifact-boundary"],
)
def test_independent_enforcement_checks_do_not_wait_for_classifier(job_id: str) -> None:
    job = _workflow("enforcement.yml")["jobs"][job_id]

    assert "needs" not in job
    assert "if" not in job
    assert "needs.changes" not in json.dumps(job)


@pytest.mark.parametrize("job_id,output", [("test-layer2-ml", "ml_changed"), ("golden-regression", "core_changed")])
def test_change_gated_enforcement_checks_retain_classifier_dependency(job_id: str, output: str) -> None:
    job = _workflow("enforcement.yml")["jobs"][job_id]

    assert job["needs"] == ["changes"]
    assert job["if"] == f"needs.changes.outputs.{output} == 'true' || github.event_name == 'schedule'"


def test_documentation_cancels_only_superseded_runs_of_same_event_and_ref() -> None:
    concurrency = _workflow("docs.yml")["concurrency"]

    assert concurrency["group"] == "${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}"
    assert concurrency["cancel-in-progress"] is True


def _pip_cache_inputs(workflow_name: str, job_id: str) -> str:
    job = _workflow(workflow_name)["jobs"][job_id]
    for step in job["steps"]:
        if step.get("with", {}).get("cache") == "pip":
            return step["with"]["cache-dependency-path"]
        if step.get("name") == "Cache pip packages":
            return step["with"]["key"]
    pytest.fail(f"No pip cache found for {workflow_name}:{job_id}")


@pytest.mark.parametrize(
    "workflow_name,job_id",
    [
        ("build.yml", "test"),
        ("ci.yml", "test-core"),
        ("ci.yml", "test-ml"),
        ("ci-quality-firewall.yml", "validate-python-compatibility"),
        ("ci-quality-firewall.yml", "test-core"),
        ("ci-quality-firewall.yml", "test-ml"),
    ],
)
def test_ci_dependency_caches_follow_transitive_base_lock(workflow_name: str, job_id: str) -> None:
    cache_inputs = _pip_cache_inputs(workflow_name, job_id)

    for dependency_path in ("requirements-ci.txt", "requirements.txt", "requirements/base.txt"):
        assert dependency_path in cache_inputs


@pytest.mark.parametrize(
    "workflow_name,job_id", [("build.yml", "test"), ("ci.yml", "test-core"), ("ci-quality-firewall.yml", "test-core")]
)
def test_core_caches_follow_archive_dependency_lock(workflow_name: str, job_id: str) -> None:
    assert "requirements/tools-archive.txt" in _pip_cache_inputs(workflow_name, job_id)


@pytest.mark.parametrize(
    "workflow_name,job_id", [("build.yml", "test"), ("ci.yml", "test-ml"), ("ci-quality-firewall.yml", "test-ml")]
)
def test_ml_caches_follow_installer_and_extra_inputs(workflow_name: str, job_id: str) -> None:
    cache_inputs = _pip_cache_inputs(workflow_name, job_id)

    for dependency_path in ("requirements/ml-raw.in", "pyproject.toml", "scripts/ci/install_ml_test_dependencies.sh"):
        assert dependency_path in cache_inputs


def test_build_matrix_publishes_distinct_python_and_tier_caches() -> None:
    key = _pip_cache_inputs("build.yml", "test")

    assert "${{ matrix.python-version }}" in key
    assert "${{ matrix.test-type }}" in key
