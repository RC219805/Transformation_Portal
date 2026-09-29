"""Known photographic failures retain their contract and expose safe recovery."""

from __future__ import annotations

from copy import deepcopy

import pytest

from transformation_portal.orchestrator.execution_diagnostics import annotate_job_failure
from transformation_portal.orchestrator.execution_runtime import WorkerJob
from transformation_portal.orchestrator.execution_summary import refresh_job_run_summary

pytestmark = pytest.mark.unit

_ICC_FAILURE = "Stage preprocess failed: Unsupported ICC profile; provide an explicit input_color correction"
_AMBIGUOUS_FAILURE = "Stage preprocess failed: Ambiguous input color; provide input_color='srgb' or 'linear_srgb'"


def _failed_job(*, pipeline: str = "lux-depth", line: str = _ICC_FAILURE) -> WorkerJob:
    return WorkerJob(
        id="job_failed_color",
        created_at=1.0,
        state="failed",
        exit_code=1,
        request={"pipeline": pipeline},
        logs_tail=[line],
        error={
            "code": "RUNNER_EXIT_NONZERO",
            "message": "runner exited with code 1",
            "details": {"exit_code": 1},
            "retriable": False,
        },
    )


@pytest.mark.parametrize("pipeline", ["lux-depth", "lux-depth-v5", "lux-depth-v6"])
@pytest.mark.parametrize("prefix", ["", "RuntimeError: "])
@pytest.mark.parametrize(
    ("line", "reason", "advice"),
    [
        (_ICC_FAILURE, "unsupported_icc_profile", "Do not relabel Adobe RGB as sRGB"),
        (_AMBIGUOUS_FAILURE, "ambiguous_input_color", "only when it matches the source encoding"),
    ],
)
def test_historical_failure_projection_provides_known_recovery_without_changing_status(pipeline, prefix, line, reason, advice):
    job = _failed_job(pipeline=pipeline, line=prefix + line)
    job.finished_at = job.done_published_at = 2.0

    assert refresh_job_run_summary(job, None) == {}
    assert job.error["code"] == "RUNNER_EXIT_NONZERO"
    assert advice in job.error["message"]
    assert job.error["details"] == {"exit_code": 1, "stage": "preprocess", "reason": reason}
    assert job.error["retriable"] is False
    assert job.state == "failed"
    assert job.finished_at == job.done_published_at == 2.0
    assert job.exit_code == 1
    first_projection = deepcopy(job.error)
    refresh_job_run_summary(job, None)
    assert job.error == first_projection


@pytest.mark.parametrize(
    "line",
    [
        "Unexpected native runtime failure",
        _ICC_FAILURE + " token=secret /private/input.tif <script>alert(1)</script>",
        "filename='" + _ICC_FAILURE + "'",
        _ICC_FAILURE + "\n" + "x" * 1024,
    ],
)
def test_unknown_or_embedded_log_messages_do_not_change_failure(line):
    job = _failed_job(line=line)
    before = deepcopy(job.error)
    annotate_job_failure(job)
    assert job.error == before


def test_guidance_does_not_copy_tracebacks_or_other_log_content():
    job = _failed_job()
    job.logs_tail += ['File "/private/customer/input.tif", token=secret', "<script>alert(1)</script>", "x" * 1000000]
    annotate_job_failure(job)
    projected = str(job.error)
    assert "profile-aware editor" in projected
    assert all(value not in projected for value in ("/private/", "secret", "<script>", "xxxx"))


def test_diagnostic_scan_is_bounded_to_recent_log_tail():
    job = _failed_job()
    job.logs_tail += ["unrelated output"] * 200
    before = deepcopy(job.error)
    annotate_job_failure(job)
    assert job.error == before


@pytest.mark.parametrize("state", ["queued", "running", "succeeded", "partial", "canceled", "worker_lost"])
def test_nonfailed_jobs_preserve_error(state):
    job = _failed_job()
    job.state = state
    before = deepcopy(job.error)
    annotate_job_failure(job)
    assert job.error == before


@pytest.mark.parametrize("code", ["RUNNER_ERROR", "RUNNER_PARTIAL_FAILURE", "WORKER_LOST"])
def test_specific_failure_codes_preserve_their_error(code):
    job = _failed_job()
    job.error["code"] = code
    before = deepcopy(job.error)
    annotate_job_failure(job)
    assert job.error == before


def test_other_pipelines_do_not_interpret_photography_log_messages():
    job = _failed_job(pipeline="archive-gate-a")
    before = deepcopy(job.error)
    annotate_job_failure(job)
    assert job.error == before


def test_status_projection_exposes_historical_guidance_when_logs_are_omitted():
    import app

    job = app.Job(**vars(_failed_job()))
    payload = app._serialize_job(job, include_logs=False)
    assert "logs_tail" not in payload
    assert payload["error"]["details"]["reason"] == "unsupported_icc_profile"
    assert "profile-aware editor" in payload["error"]["message"]


@pytest.mark.parametrize(
    "profile, reason", [(b"unsupported profile", "unsupported_icc_profile"), (None, "ambiguous_input_color")]
)
def test_current_photography_errors_remain_classifiable(profile, reason):
    from transformation_portal.lux_depth_v4.photography import InputColorError, _resolve_color

    with pytest.raises(InputColorError) as failure:
        _resolve_color(input_color="auto", image_format="PNG", profile=profile, declared_color=None, exif_color=None)
    job = _failed_job(line="Stage preprocess failed: " + str(failure.value))
    annotate_job_failure(job)
    assert job.error["details"]["reason"] == reason
    if reason == "unsupported_icc_profile":
        assert "If Auto rejects its sRGB profile, select sRGB only for that converted file" in job.error["message"]


def test_previous_conversion_guidance_logs_remain_classifiable():
    previous = (
        "Stage preprocess failed: Unsupported ICC profile. Convert the photographs to sRGB with color management and "
        "upload the converted copies. An input color override only reinterprets existing pixels; it does not convert them."
    )
    job = _failed_job(line=previous)
    annotate_job_failure(job)
    assert job.error["details"]["reason"] == "unsupported_icc_profile"
    assert "select sRGB only for that converted file" in job.error["message"]
