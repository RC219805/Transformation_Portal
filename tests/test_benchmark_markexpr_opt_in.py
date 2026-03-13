from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import _benchmark_run_explicitly_requested, _markexpr_requests_marker


def _make_config(*, markexpr: str = "", args: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        option=SimpleNamespace(markexpr=markexpr),
        invocation_params=SimpleNamespace(args=args),
    )


def test_markexpr_requests_benchmark_for_positive_selector() -> None:
    assert _markexpr_requests_marker("benchmark and not slow", "benchmark") is True


def test_markexpr_does_not_request_benchmark_for_negative_selector() -> None:
    assert _markexpr_requests_marker("not benchmark", "benchmark") is False
    assert _markexpr_requests_marker("ml and not slow and not benchmark", "benchmark") is False


def test_benchmark_opt_in_detects_positive_markexpr(monkeypatch) -> None:
    monkeypatch.delenv("TP_RUN_BENCHMARKS", raising=False)
    config = _make_config(markexpr="benchmark or smoke")
    assert _benchmark_run_explicitly_requested(config) is True


def test_benchmark_opt_in_ignores_negative_markexpr(monkeypatch) -> None:
    monkeypatch.delenv("TP_RUN_BENCHMARKS", raising=False)
    config = _make_config(markexpr="not benchmark")
    assert _benchmark_run_explicitly_requested(config) is False


def test_benchmark_opt_in_detects_explicit_benchmark_path(monkeypatch) -> None:
    monkeypatch.delenv("TP_RUN_BENCHMARKS", raising=False)
    config = _make_config(args=("tests/benchmarks/test_lux_depth_v3_perf_smoke.py",))
    assert _benchmark_run_explicitly_requested(config) is True
