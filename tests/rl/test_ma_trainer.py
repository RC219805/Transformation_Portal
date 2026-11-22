"""Regression tests for the decentralized multi-agent actor-critic training loop."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ml]

torch = pytest.importorskip("torch")


def _pipeline() -> dict:
    from transformation_portal.rl.ma_action_space import DEFAULT_AGENT_NODES

    return {"nodes": [{"id": n, "config": {}} for n in DEFAULT_AGENT_NODES]}


@pytest.mark.parametrize("iterations", [11, 20])
def test_optimize_multi_agent_survives_overlapping_replay_windows(iterations: int) -> None:
    """Re-training on transitions already used must not backward a freed graph (finding R2)."""
    from transformation_portal.rl.ma_action_space import DEFAULT_AGENT_NODES
    from transformation_portal.rl.ma_env import MockMultiAgentEnv
    from transformation_portal.rl.ma_optimize import create_agents, optimize_multi_agent

    torch.manual_seed(0)
    env = MockMultiAgentEnv(DEFAULT_AGENT_NODES)
    agents = create_agents()

    result = optimize_multi_agent(env, agents, _pipeline(), iterations=iterations)

    # One update per agent per iteration from the 10th collected transition on.
    expected_updates = len(agents) * (result.iterations - 9)
    assert result.iterations == iterations
    assert len(result.loss_history) == expected_updates
    assert all(torch.isfinite(torch.tensor(result.loss_history)))
    for agent in agents.values():
        assert all(len(item) == 5 for item in agent.replay)


def test_agent_trainer_steps_twice_on_the_same_batch_and_accepts_legacy_tuples() -> None:
    from transformation_portal.rl.ma_model import create_agent
    from transformation_portal.rl.ma_trainer import AgentTrainer

    torch.manual_seed(0)
    model = create_agent(4, 3)
    trainer = AgentTrainer(model, actions=["a", "b", "c"])
    s = torch.zeros(4).numpy()
    s2 = torch.ones(4).numpy()
    _, logp, v = model.act(torch.tensor(s).unsqueeze(0))
    batch = [(s, 1, 0.5, s2, False, logp, v), (s2, 2, -0.25, s, True)]

    first = trainer.step(batch)
    second = trainer.step(batch)

    assert torch.isfinite(torch.tensor([first, second])).all()
    assert first != second
