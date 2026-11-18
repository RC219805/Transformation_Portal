"""Behavioral coverage for ``dashboard.rl_api``.

Covers the torch-free surface of the RL optimization router: the
action-listing and policy-config endpoints (backed by the pure
``transformation_portal.rl.action_space`` / ``policy_guard`` modules), the
job-start handshake, the unknown-job status path, and the FastAPI guard.
Background execution is tested with capture substitutes so the deprecated,
ignored model_path field cannot enable pre-trained model loading. Actual RL
training requires torch and is intentionally not asserted here.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import rl_api


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(rl_api.create_rl_api_router())
    return TestClient(app)


def test_list_actions(client: TestClient) -> None:
    body = client.get("/rl/actions").json()
    assert body["count"] > 0
    first = body["actions"][0]
    assert {"index", "node", "action_type", "params"} <= set(first)


def test_get_policy_config(client: TestClient) -> None:
    body = client.get("/rl/policy").json()
    assert {"safe_actions", "risky_actions", "blocked_actions"} <= set(body)
    assert isinstance(body["safe_actions"], list)


def test_status_unknown_job(client: TestClient) -> None:
    assert client.get("/rl/status/ghost").json() == {"error": "Job not found"}


def test_optimize_returns_job_handshake(client: TestClient) -> None:
    # The scheduled background task needs torch (absent in the core lane) and
    # will fail closed; we only assert the synchronous handshake here.
    resp = client.post("/rl/optimize", json={"max_iterations": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert isinstance(body["job_id"], str) and body["job_id"]


@pytest.mark.parametrize(
    "model_path_fields",
    [
        {},
        {"model_path": None},
        {"model_path": "/unsupported/pretrained.pt"},
        {"model_path": "/another/nonexistent/model.pt"},
    ],
    ids=["absent", "null", "first-path", "second-path"],
)
@pytest.mark.parametrize("max_iterations", [None, 3])
def test_optimize_uses_fresh_mock_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    model_path_fields: dict[str, str | None],
    max_iterations: int | None,
) -> None:
    actions = [object(), object()]
    env, model, trainer = object(), object(), object()
    environment_factory = Mock(return_value=env)
    model_factory = Mock(return_value=model)
    trainer_factory = Mock(return_value=trainer)
    expected_iterations = 50 if max_iterations is None else max_iterations
    config = SimpleNamespace(max_iterations=expected_iterations)
    config_factory = Mock(return_value=config)
    result_body = {"best_score": 0.8, "iterations": expected_iterations}
    result = SimpleNamespace(
        best_score=0.8,
        iterations=expected_iterations,
        to_dict=Mock(return_value=result_body),
    )
    train_calls: list[bool] = []

    def train(*args: object) -> SimpleNamespace:
        # Record whether training ran on the event-loop thread.
        try:
            asyncio.get_running_loop()
            train_calls.append(True)
        except RuntimeError:
            train_calls.append(False)
        assert args == (env, trainer, pipeline, config)
        return result

    # Replace lazy RL imports, and block torch even in an ML-enabled environment.
    modules = {
        "action_space": SimpleNamespace(enumerate_actions=lambda: actions),
        "env": SimpleNamespace(MockPipelineEnv=environment_factory),
        "model": SimpleNamespace(create_model=model_factory),
        "optimize_rl": SimpleNamespace(RLOptimizationConfig=config_factory, train_rl=train),
        "state_encoder": SimpleNamespace(get_state_dim=lambda: 13),
        "trainer": SimpleNamespace(RLTrainer=trainer_factory),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, f"transformation_portal.rl.{name}", module)
    monkeypatch.setitem(sys.modules, "torch", None)

    pipeline = {"nodes": [{"id": "source", "params": {"strength": 0.5}}]}
    payload = {"pipeline": pipeline, **model_path_fields}
    if max_iterations is not None:
        payload["max_iterations"] = max_iterations

    response = client.post("/rl/optimize", json=payload)
    assert response.status_code == 200
    handshake = response.json()
    assert set(handshake) == {"job_id", "status"}
    assert handshake["status"] == "started"
    assert isinstance(handshake["job_id"], str) and handshake["job_id"]
    environment_factory.assert_called_once_with(actions)
    model_factory.assert_called_once_with(13, len(actions))
    trainer_factory.assert_called_once_with(model, actions)
    config_factory.assert_called_once_with(max_iterations=expected_iterations, save_model=False)
    # Training ran exactly once and off the event loop (D10).
    assert train_calls == [False]
    assert client.get(f"/rl/status/{handshake['job_id']}").json() == {
        "status": "completed",
        "progress": 1.0,
        "best_score": 0.8,
        "iterations": expected_iterations,
        "result": result_body,
    }


@pytest.mark.parametrize("max_iterations", [0, -1, 10_001, "50", 2.5, True, None, [3]])
def test_optimize_rejects_invalid_max_iterations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, max_iterations: object
) -> None:
    train = Mock()
    monkeypatch.setitem(
        sys.modules,
        "transformation_portal.rl.optimize_rl",
        SimpleNamespace(RLOptimizationConfig=Mock(), train_rl=train),
    )
    response = client.post("/rl/optimize", json={"max_iterations": max_iterations})
    assert response.status_code == 422
    assert "max_iterations" in response.json()["detail"]
    train.assert_not_called()


@pytest.mark.ml
def test_optimize_real_training_writes_nothing_into_working_directory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)

    response = client.post("/rl/optimize", json={"max_iterations": 2})
    assert response.status_code == 200
    status = client.get(f"/rl/status/{response.json()['job_id']}").json()
    assert status["status"] == "completed", status
    assert list(tmp_path.iterdir()) == []


def test_optimize_openapi_documents_mock_model_compatibility(client: TestClient) -> None:
    description = client.get("/openapi.json").json()["paths"]["/rl/optimize"]["post"]["description"]
    assert "MockPipelineEnv" in description
    assert "pre-trained model loading is unsupported" in description
    assert "model_path: Deprecated, unsupported field; accepted and ignored" in description


def test_create_router_returns_none_without_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl_api, "FASTAPI_AVAILABLE", False)
    assert rl_api.create_rl_api_router() is None
