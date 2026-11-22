"""Regression tests for the MuZero latent-space MCTS (``rl.muzero_mcts``)."""

import math

import pytest

torch = pytest.importorskip("torch")

from transformation_portal.rl import muzero_mcts as mz  # noqa: E402
from transformation_portal.rl.muzero_model import MuZeroModel  # noqa: E402

pytestmark = [pytest.mark.ml, pytest.mark.unit, pytest.mark.regression]


def _child(prior: float, visits: int, reward: float, value_sum: float) -> mz.MuZeroNode:
    node = mz.MuZeroNode(prior=prior)
    node.visit_count = visits
    node.reward = reward
    node.value_sum = value_sum
    return node


def test_ucb_score_uses_reward_plus_discounted_value() -> None:
    cfg = mz.MCTSConfig()
    parent = mz.MuZeroNode(prior=1.0)
    parent.visit_count = 20
    good = _child(0.5, 10, reward=1.0, value_sum=0.0)  # Q = 1.0
    poor = _child(0.5, 10, reward=0.0, value_sum=5.0)  # Q = 0.99 * 0.5

    assert mz.ucb_score(parent, good, cfg) > mz.ucb_score(parent, poor, cfg)
    # Identical priors and visit counts: the gap is exactly the Q gap.
    gap = mz.ucb_score(parent, good, cfg) - mz.ucb_score(parent, poor, cfg)
    assert gap == pytest.approx(1.0 - cfg.discount * 0.5)
    _, picked = mz.select_child(_with_children(parent, {0: poor, 1: good}), cfg)
    assert picked is good


def _with_children(parent: mz.MuZeroNode, children: dict) -> mz.MuZeroNode:
    parent.children = children
    return parent


def test_get_action_probs_greedy_and_degenerate_inputs() -> None:
    assert mz.get_action_probs({0: 50, 1: 30}, temperature=0) == {0: 1.0, 1: 0.0}
    assert mz.get_action_probs({0: 5, 1: 5, 2: 1}, temperature=0) == {0: 0.5, 1: 0.5, 2: 0.0}
    tiny = mz.get_action_probs({0: 50, 1: 30}, temperature=0.001)
    assert tiny[0] == pytest.approx(1.0)
    assert math.isclose(sum(tiny.values()), 1.0)
    assert mz.get_action_probs({0: 0, 1: 0}) == {0: 0.5, 1: 0.5}
    assert mz.get_action_probs({}) == {}
    assert mz.get_action_probs({0: 30, 1: 10}, temperature=1.0) == pytest.approx({0: 0.75, 1: 0.25})
    with pytest.raises(ValueError):
        mz.get_action_probs({0: 1}, temperature=-1.0)


def test_run_mcts_searches_without_autograd_in_eval_mode() -> None:
    torch.manual_seed(0)
    model = MuZeroModel(obs_dim=6, action_dim=4, latent_dim=8)
    model.train()
    seen: list[tuple[bool, bool]] = []
    original = model.recurrent_inference

    def probe(state, action):
        seen.append((torch.is_grad_enabled(), model.training))
        return original(state, action)

    model.recurrent_inference = probe  # type: ignore[method-assign]
    action, visits = mz.run_mcts(model, torch.randn(1, 6), mz.MCTSConfig(num_simulations=5))

    assert seen and all(not grad and not training for grad, training in seen)
    assert model.training is True  # caller's mode restored
    assert sum(visits.values()) == 5
    assert action in visits
