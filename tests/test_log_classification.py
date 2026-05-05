"""Unit tests for src/transformation_portal/core/observability/log_classification.py."""

from __future__ import annotations

import logging

import pytest

from transformation_portal.core.observability.log_classification import (
    install_healthcheck_log_filter,
    log_kind,
)


@pytest.fixture()
def isolated_logger() -> logging.Logger:
    logger = logging.getLogger("uvicorn.access.test_log_classification")
    # Reset any leftover filters between tests so order is deterministic.
    for f in list(logger.filters):
        logger.removeFilter(f)
    if hasattr(logger, "_tp_healthcheck_filter_installed"):
        delattr(logger, "_tp_healthcheck_filter_installed")
    return logger


def _emit(logger: logging.Logger, message: str) -> logging.LogRecord:
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, message, args=None, exc_info=None)
    return record


@pytest.mark.unit
def test_install_filter_drops_successful_healthcheck_lines(
    monkeypatch: pytest.MonkeyPatch, isolated_logger: logging.Logger
) -> None:
    monkeypatch.setenv("TP_LOG_HEALTHCHECKS", "0")
    installed = install_healthcheck_log_filter(logger_name=isolated_logger.name)
    assert installed is True

    record = _emit(isolated_logger, '127.0.0.1 - "GET /healthz HTTP/1.1" 200')
    keep = all(f.filter(record) for f in isolated_logger.filters)
    assert keep is False, "successful /healthz access lines should be suppressed"


@pytest.mark.unit
def test_install_filter_keeps_failed_healthcheck_lines(
    monkeypatch: pytest.MonkeyPatch, isolated_logger: logging.Logger
) -> None:
    monkeypatch.setenv("TP_LOG_HEALTHCHECKS", "0")
    install_healthcheck_log_filter(logger_name=isolated_logger.name)
    record = _emit(isolated_logger, '127.0.0.1 - "GET /healthz HTTP/1.1" 503')
    keep = all(f.filter(record) for f in isolated_logger.filters)
    assert keep is True, "/healthz error responses must still surface in logs"


@pytest.mark.unit
def test_install_filter_keeps_non_healthcheck_lines(monkeypatch: pytest.MonkeyPatch, isolated_logger: logging.Logger) -> None:
    monkeypatch.setenv("TP_LOG_HEALTHCHECKS", "0")
    install_healthcheck_log_filter(logger_name=isolated_logger.name)
    record = _emit(isolated_logger, '127.0.0.1 - "GET /v1/jobs HTTP/1.1" 200')
    keep = all(f.filter(record) for f in isolated_logger.filters)
    assert keep is True


@pytest.mark.unit
def test_install_filter_default_keeps_everything(monkeypatch: pytest.MonkeyPatch, isolated_logger: logging.Logger) -> None:
    monkeypatch.delenv("TP_LOG_HEALTHCHECKS", raising=False)
    installed = install_healthcheck_log_filter(logger_name=isolated_logger.name)
    assert installed is False
    assert isolated_logger.filters == []


@pytest.mark.unit
def test_log_kind_emits_structured_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test_log_classification.kind")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_kind(logger, "protected_api", "config_failure", path="/v1/jobs", status=503)
    assert any(
        "kind=protected_api" in record.message
        and "event=config_failure" in record.message
        and "path=/v1/jobs" in record.message
        and "status=503" in record.message
        for record in caplog.records
    )


@pytest.mark.unit
def test_log_kind_normalizes_unknown_kind(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test_log_classification.unknown")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_kind(logger, "made_up", "something")
    assert any("kind=audit" in record.message for record in caplog.records)
