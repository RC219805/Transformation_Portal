"""Tests for rl.mcts — PUCT selection, Dirichlet noise, and MCTSConfig."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from transformation_portal.rl.mcts import MCTSConfig, add_dirichlet_noise, puct_select
from transformation_portal.rl.mcts_node import MCTSNode

pytestmark = pytest.mark.unit


def _expanded_node(n_actions: int = 3, parent_N: int = 0) -> MCTSNode:
    """Return an expanded root node with n_actions children (all None/unvisited)."""
    node = MCTSNode(state="root")
    node.N = parent_N
    priors = {i: 1.0 / n_actions for i in range(n_actions)}
    node.P = priors
    node.children = {i: None for i in range(n_actions)}
    return node


def _node_with_visited_child(child_N: int, child_Q: float, child_prior: float = 0.5) -> MCTSNode:
    """Return an expanded root with one visited child."""
    root = MCTSNode(state="root")
    root.N = child_N
    root.P = {0: child_prior}

    child = MCTSNode(state="child", parent=root, action_from_parent=0)
    child.N = child_N
    child.W = child_Q * child_N
    child.Q = child_Q

    root.children = {0: child}
    return root


class TestMCTSConfig:
    def test_default_num_simulations(self):
        """Default num_simulations is 50."""
        assert MCTSConfig().num_simulations == 50

    def test_default_c_puct(self):
        """Default c_puct is 1.5."""
        assert MCTSConfig().c_puct == pytest.approx(1.5)

    def test_custom_values_stored(self):
        """Custom config values are stored correctly."""
        cfg = MCTSConfig(num_simulations=100, c_puct=2.0, temperature=0.5)
        assert cfg.num_simulations == 100
        assert cfg.c_puct == pytest.approx(2.0)
        assert cfg.temperature == pytest.approx(0.5)


class TestPuctSelect:
    def test_returns_none_when_no_children(self):
        """puct_select returns None for a leaf node."""
        node = MCTSNode(state="leaf")
        assert puct_select(node) is None

    def test_selects_unvisited_over_low_q_visited(self):
        """Unvisited child beats a visited child with Q=0 due to exploration bonus."""
        root = MCTSNode(state="root")
        root.N = 1
        root.P = {0: 0.5, 1: 0.5}
        # Action 1 is visited but has Q=0 — exploration bonus for action 0 dominates
        visited = MCTSNode(state="visited", parent=root, action_from_parent=1)
        visited.N = 5
        visited.Q = 0.0
        root.children = {0: None, 1: visited}
        # Unvisited score ≈ 1.5 * 0.5 * sqrt(2)/1 ≈ 1.06; visited ≈ 0 + 0.18 → unvisited wins
        assert puct_select(root) == 0

    def test_higher_prior_preferred_when_visits_equal(self):
        """Higher prior leads to higher PUCT score when visits are equal."""
        root = MCTSNode(state="root")
        root.N = 10
        root.P = {0: 0.1, 1: 0.9}

        c0 = MCTSNode(state="c0", parent=root, action_from_parent=0)
        c0.N = 5
        c0.Q = 0.5
        c1 = MCTSNode(state="c1", parent=root, action_from_parent=1)
        c1.N = 5
        c1.Q = 0.5
        root.children = {0: c0, 1: c1}

        # action 1 has prior 0.9 vs 0.1, same Q and N → action 1 wins
        assert puct_select(root) == 1

    def test_returns_integer_action(self):
        """puct_select returns an integer."""
        root = _expanded_node(n_actions=3, parent_N=5)
        result = puct_select(root)
        assert isinstance(result, int)

    def test_higher_c_puct_favours_exploration(self):
        """With high c_puct, a high-prior unvisited action beats a high-Q visited one."""
        root = MCTSNode(state="root")
        root.N = 1
        # action 0: visited, high Q; action 1: unvisited, high prior
        root.P = {0: 0.1, 1: 0.9}
        c0 = MCTSNode(state="c0")
        c0.N = 10
        c0.Q = 0.9
        root.children = {0: c0, 1: None}
        # With c_puct=10, the exploration bonus for unvisited child 1 dominates
        assert puct_select(root, c_puct=10.0) == 1


class TestAddDirichletNoise:
    def test_output_keys_match_input(self):
        """Output priors have the same action keys as input."""
        priors = {0: 0.5, 1: 0.3, 2: 0.2}
        noisy = add_dirichlet_noise(priors, alpha=0.3, weight=0.25)
        assert set(noisy.keys()) == set(priors.keys())

    def test_empty_priors_returned_unchanged(self):
        """Empty priors dict returns unchanged."""
        assert not add_dirichlet_noise({})

    def test_weight_zero_returns_unchanged_priors(self):
        """weight=0 → noisy = (1-0)*prior + 0*noise = original prior."""
        priors = {0: 0.6, 1: 0.4}
        noisy = add_dirichlet_noise(priors, alpha=0.3, weight=0.0)
        for k, v in priors.items():
            assert noisy[k] == pytest.approx(v, abs=1e-6)

    def test_noisy_priors_sum_to_approx_one(self):
        """Mixed priors still sum to approximately 1.0."""
        priors = {0: 0.5, 1: 0.3, 2: 0.2}
        # Run multiple times to reduce randomness sensitivity
        for _ in range(5):
            noisy = add_dirichlet_noise(priors, alpha=0.3, weight=0.25)
            total = sum(noisy.values())
            assert total == pytest.approx(1.0, abs=0.05)

    def test_output_values_are_floats(self):
        """All output values are floats."""
        priors = {0: 0.7, 1: 0.3}
        noisy = add_dirichlet_noise(priors)
        assert all(isinstance(v, float) for v in noisy.values())


class TestMCTSSearch:
    """Integration-level test using mocked world model and action_fn."""

    def _make_world_model(self):
        """Minimal world model mock: predict returns object with .next_state and .score."""
        from types import SimpleNamespace

        wm = MagicMock()
        wm.predict.return_value = SimpleNamespace(next_state="next_state", score=0.5)
        return wm

    def _uniform_action_fn(self, n_actions=3):
        """Returns uniform priors over n_actions."""

        def fn(state):
            actions = list(range(n_actions))
            priors = [1.0 / n_actions] * n_actions
            return actions, priors

        return fn

    def test_search_returns_integer(self):
        """MCTS.search() returns an int."""
        from transformation_portal.rl.mcts import MCTS

        mcts = MCTS(
            world_model=self._make_world_model(),
            action_fn=self._uniform_action_fn(3),
            config=MCTSConfig(num_simulations=5),
        )
        result = mcts.search("initial_state")
        assert isinstance(result, int)

    def test_search_returns_valid_action_index(self):
        """Action index is in [0, n_actions)."""
        from transformation_portal.rl.mcts import MCTS

        n = 4
        mcts = MCTS(
            world_model=self._make_world_model(),
            action_fn=self._uniform_action_fn(n),
            config=MCTSConfig(num_simulations=5),
        )
        result = mcts.search("initial_state")
        assert 0 <= result < n

    def test_search_with_no_actions_returns_zero(self):
        """When action_fn returns empty list, search returns 0."""
        from transformation_portal.rl.mcts import MCTS

        mcts = MCTS(
            world_model=self._make_world_model(),
            action_fn=lambda state: ([], []),
            config=MCTSConfig(num_simulations=5),
        )
        assert mcts.search("state") == 0

    def test_mcts_search_convenience_function(self):
        """mcts_search() returns a valid int."""
        from transformation_portal.rl.mcts import mcts_search

        result = mcts_search(
            root_state="s0",
            world_model=self._make_world_model(),
            action_fn=self._uniform_action_fn(3),
            num_simulations=5,
        )
        assert isinstance(result, int)
        assert result >= 0


class _ImprovingActionWorldModel:
    """Stub world model: the score is the state quality; only ``improving`` raises it."""

    def __init__(self, improving: int = 7, step: float = 0.4) -> None:
        self.improving = improving
        self.step = step

    def predict(self, state, action):
        from types import SimpleNamespace

        nxt = state + (self.step if action == self.improving else 0.0)
        return SimpleNamespace(next_state=nxt, score=nxt)


def _uniform_actions(n_actions: int):
    return lambda state: (list(range(n_actions)), [1.0 / n_actions] * n_actions)


class TestMCTSExploration:
    """Regression tests for R13: the search must explore and find improving actions."""

    def test_unvisited_child_uses_parent_value_as_first_play_urgency(self):
        """An unvisited sibling beats a visited child whose positive Q equals the parent's value."""
        root = MCTSNode(state="root")
        root.N = 3
        root.W = 1.8
        root.Q = 0.6
        root.P = {0: 0.5, 1: 0.5}
        visited = MCTSNode(state="c0", parent=root, action_from_parent=0)
        visited.N = 3
        visited.W = 1.8
        visited.Q = 0.6
        root.children = {0: visited, 1: None}
        # c_puct=0.5, sqrt(N+1)=2: visited 0.6 + 0.5*0.5*2/4 = 0.725;
        # unvisited scored at Q=0 -> 0.5 (loses), with first-play urgency 0.6 + 0.5 = 1.1 (wins).
        assert puct_select(root, c_puct=0.5) == 1

    def test_leaf_value_includes_transition_score(self):
        """The predicted score of the expanded transition is backed up with the rollout."""
        from types import SimpleNamespace

        from transformation_portal.rl.mcts import MCTS

        class WM:
            def predict(self, state, action):
                # Leaving the root scores 1.0; every later step scores 0.0.
                return SimpleNamespace(next_state="leaf", score=1.0 if state == "root" else 0.0)

        mcts = MCTS(WM(), _uniform_actions(2), MCTSConfig(num_simulations=1, rollout_depth=3, dirichlet_weight=0.0))
        root = MCTSNode(state="root")
        root.expand([0, 1], [0.5, 0.5])
        mcts._simulate(root)
        child = next(c for c in root.children.values() if c is not None)
        assert child.Q == pytest.approx(0.25)
        assert root.Q == pytest.approx(0.25)

    def test_search_finds_the_only_improving_action(self):
        """With uniform priors the search tries many root actions and returns the improving one."""
        from transformation_portal.rl.mcts import MCTS

        config = MCTSConfig(num_simulations=50, dirichlet_weight=0.0)
        mcts = MCTS(_ImprovingActionWorldModel(improving=7), _uniform_actions(39), config)
        assert mcts.search(0.5) == 7

    def test_search_finds_improving_action_with_root_noise(self):
        """Dirichlet root noise does not prevent the search from finding the improving action."""
        np = pytest.importorskip("numpy")
        from transformation_portal.rl.mcts import MCTS

        picks = []
        for seed in range(10):
            np.random.seed(seed)
            mcts = MCTS(_ImprovingActionWorldModel(improving=7), _uniform_actions(39), MCTSConfig(num_simulations=50))
            picks.append(mcts.search(0.5))
        assert picks.count(7) >= 8


class _ConstantWorldModel:
    def predict(self, state, action):
        from types import SimpleNamespace

        return SimpleNamespace(next_state=state, score=0.5)


class TestMCTSPipelineOptimizer:
    """Regression tests for R13: RL actions are converted before apply_fix."""

    def test_optimize_applies_rl_actions(self):
        """optimize() converts RLAction to FixSuggestion and returns a modified pipeline."""
        from transformation_portal.rl.action_space import RLAction
        from transformation_portal.rl.mcts import MCTSPipelineOptimizer

        actions = [
            RLAction(node="recon", action_type="increase_iterations", params={"steps": 4000}, index=0),
            RLAction(node="recon", action_type="increase_iterations", params={"steps": 8000}, index=1),
        ]
        pipeline = {"nodes": [{"id": "recon", "config": {"steps": 1000}}]}
        opt = MCTSPipelineOptimizer(
            _ConstantWorldModel(),
            run_fn=lambda p: p,
            eval_fn=lambda out: {"score": out["nodes"][0]["config"]["steps"] / 10000.0},
            action_list=actions,
            encode_fn=lambda p: 0.0,
            config=MCTSConfig(num_simulations=4, dirichlet_weight=0.0),
        )
        out = opt.optimize(pipeline, steps=1, validate_top_k=1)
        assert out["nodes"][0]["config"]["steps"] in (4000, 8000)
        assert pipeline["nodes"][0]["config"]["steps"] == 1000

    def test_optimize_skips_inapplicable_actions(self):
        """An action whose target node is absent stops that candidate without raising."""
        from transformation_portal.rl.action_space import RLAction
        from transformation_portal.rl.mcts import MCTSPipelineOptimizer

        pipeline = {"nodes": [{"id": "other", "config": {}}]}
        opt = MCTSPipelineOptimizer(
            _ConstantWorldModel(),
            run_fn=lambda p: p,
            eval_fn=lambda out: {"score": 0.1},
            action_list=[RLAction(node="recon", action_type="increase_iterations", params={"steps": 4000})],
            encode_fn=lambda p: 0.0,
            config=MCTSConfig(num_simulations=2, dirichlet_weight=0.0),
        )
        assert opt.optimize(pipeline, steps=2, validate_top_k=1) == pipeline
