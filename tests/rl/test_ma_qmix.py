"""Regression tests for the QMIX/VDN centralized critic and its training loop."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ml]

torch = pytest.importorskip("torch")


def _global_dim() -> int:
    from transformation_portal.rl.ma_state import get_global_dim

    return get_global_dim()


@pytest.mark.parametrize("critic_type", ["qmix", "vdn", "hybrid"])
def test_central_critic_forward_mixes_agent_qs(critic_type: str) -> None:
    """The plain-class mixer must be callable from CentralCritic.forward (finding R1)."""
    from transformation_portal.rl.ma_qmix import create_critic

    torch.manual_seed(0)
    critic = create_critic(n_agents=3, state_dim=_global_dim(), critic_type=critic_type)
    agent_qs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
    global_state = torch.zeros(2, _global_dim())

    q_total = critic.forward(agent_qs, global_state)

    assert q_total.shape == (2, 1)
    assert torch.isfinite(q_total).all()
    assert torch.equal(q_total, critic(agent_qs, global_state))
    if critic_type == "vdn":
        assert torch.allclose(q_total, agent_qs.sum(dim=1, keepdim=True))


def test_mixing_network_output_is_differentiable() -> None:
    from transformation_portal.rl.ma_qmix import MixingNetwork

    torch.manual_seed(0)
    mixer = MixingNetwork(n_agents=2, state_dim=4, embed_dim=8)
    agent_qs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    out = mixer(agent_qs, torch.ones(1, 4))
    out.sum().backward()

    assert out.shape == (1, 1)
    assert agent_qs.grad is not None
    # QMIX monotonicity: dQ_total/dQ_i >= 0 for every agent.
    assert (agent_qs.grad >= 0).all()


@pytest.mark.parametrize(
    ("iterations", "batch_size", "expected_updates"),
    [
        (20, 32, 1),  # defaults: a run shorter than one batch still trains once
        (40, 32, 9),  # full batches from iteration 32 onwards
    ],
)
def test_optimize_with_central_critic_trains(iterations: int, batch_size: int, expected_updates: int) -> None:
    from transformation_portal.rl.ma_action_space import DEFAULT_AGENT_NODES
    from transformation_portal.rl.ma_env import MockMultiAgentEnv
    from transformation_portal.rl.ma_optimize import optimize_with_central_critic

    torch.manual_seed(0)
    pipeline = {"nodes": [{"id": n, "config": {}} for n in DEFAULT_AGENT_NODES]}
    env = MockMultiAgentEnv(DEFAULT_AGENT_NODES)

    result = optimize_with_central_critic(env, pipeline, iterations=iterations, batch_size=batch_size)

    if result.iterations == iterations:
        assert len(result.loss_history) == expected_updates
    else:  # episode ended early: at most one update per step after the threshold
        assert len(result.loss_history) <= expected_updates
    assert result.loss_history, "the centralized critic never trained"
    assert all(loss == loss and loss >= 0.0 for loss in result.loss_history)
