# test_prophetic_orchestrator.py

from prophetic_orchestrator import (
    CausalityEngine,
    PropheticOrchestrator,
    TemporalAntibody,
    WeakPoint,
)

def test_causality_engine_normalizes_inputs() -> None:
    engine = CausalityEngine()
    predicted_failure = {
        "weak_points": [
            {"component": "database", "failure_mode": "replication lag", "severity": "high"},
            "api:timeout",
            WeakPoint(component="cache", failure_mode="eviction storm", severity="low"),
        ]
    }

    result = engine.trace_failure_origins(predicted_failure)

    assert [wp.component for wp in result] == ["database", "api", "cache"]
    assert [wp.failure_mode for wp in result] == ["replication lag", "timeout", "eviction storm"]
    assert [wp.severity for wp in result] == ["high", None, "low"]

def test_prophetic_orchestrator_deploys_temporal_antibodies() -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {
        "weak_points": [
            {"component": "ingest", "failure_mode": "queue saturation", "severity": "critical"},
            {"component": "renderer", "failure_mode": "color drift", "severity": "medium"},
        ]
    }

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    assert all(isinstance(item, TemporalAntibody) for item in antibodies)
    assert orchestrator.deployed_antibodies == antibodies

    for deployed in orchestrator.deployed_antibodies:
        probability = orchestrator.probability_weaver.probability_of(deployed.target)
        assert probability == 0.9999

    for antibody in antibodies:
        assert antibody.target.component in antibody.countermeasure