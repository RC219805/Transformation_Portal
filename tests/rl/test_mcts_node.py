"""Tests for rl.mcts_node — MCTS tree node data structure."""

from __future__ import annotations

import math

import pytest

from transformation_portal.rl.mcts_node import MCTSNode

pytestmark = pytest.mark.unit


def _root(state: object = "root") -> MCTSNode:
    return MCTSNode(state=state)


def _expanded_root(n_actions: int = 3) -> MCTSNode:
    node = _root()
    actions = list(range(n_actions))
    priors = [1.0 / n_actions] * n_actions
    node.expand(actions, priors)
    return node


class TestMCTSNodeDefaults:
    def test_new_node_is_leaf(self):
        """Freshly created node has no children → is a leaf."""
        assert _root().is_leaf() is True

    def test_new_node_not_expanded(self):
        """Freshly created node has no priors → not expanded."""
        assert _root().is_expanded() is False

    def test_initial_stats_are_zero(self):
        """N, W, Q start at 0."""
        node = _root()
        assert node.N == 0
        assert node.W == 0.0
        assert node.Q == 0.0

    def test_depth_root_is_zero(self):
        """Root node depth is 0."""
        assert _root().depth() == 0

    def test_parent_of_root_is_none(self):
        """Root has no parent."""
        assert _root().parent is None

    def test_action_from_parent_of_root_is_none(self):
        """Root has no action from parent."""
        assert _root().action_from_parent is None


class TestExpand:
    def test_expand_sets_priors(self):
        """After expand, P contains the supplied priors."""
        node = _root()
        node.expand([0, 1, 2], [0.5, 0.3, 0.2])
        assert node.P[0] == pytest.approx(0.5)
        assert node.P[1] == pytest.approx(0.3)
        assert node.P[2] == pytest.approx(0.2)

    def test_expand_creates_none_children(self):
        """After expand, children exist but are all None (unvisited)."""
        node = _root()
        node.expand([0, 1], [0.6, 0.4])
        assert 0 in node.children
        assert 1 in node.children
        assert node.children[0] is None
        assert node.children[1] is None

    def test_is_expanded_after_expand(self):
        """is_expanded is True after expand."""
        node = _expanded_root()
        assert node.is_expanded() is True

    def test_is_not_leaf_after_expand(self):
        """is_leaf is False after expand (children dict is non-empty)."""
        node = _expanded_root()
        assert node.is_leaf() is False


class TestUpdate:
    def test_update_increments_n(self):
        """update increments N by 1 each call."""
        node = _root()
        node.update(0.5)
        assert node.N == 1
        node.update(0.5)
        assert node.N == 2

    def test_update_accumulates_w(self):
        """update accumulates values in W."""
        node = _root()
        node.update(0.4)
        node.update(0.6)
        assert node.W == pytest.approx(1.0)

    def test_q_equals_w_over_n(self):
        """Q is always W/N after updates."""
        node = _root()
        node.update(0.8)
        node.update(0.4)
        assert node.Q == pytest.approx(node.W / node.N)


class TestAddChild:
    def test_add_child_returns_mcts_node(self):
        """add_child creates and returns an MCTSNode."""
        root = _expanded_root()
        child = root.add_child(0, "child_state")
        assert isinstance(child, MCTSNode)

    def test_child_stored_in_children(self):
        """After add_child, children[action] is the new node."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        assert root.children[0] is child

    def test_child_has_correct_parent(self):
        """Child's parent reference points to root."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        assert child.parent is root

    def test_child_has_correct_action_from_parent(self):
        """Child's action_from_parent is the action used to create it."""
        root = _expanded_root()
        child = root.add_child(1, "s")
        assert child.action_from_parent == 1

    def test_child_depth_is_one(self):
        """Direct child of root has depth 1."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        assert child.depth() == 1


class TestGetChild:
    def test_get_child_returns_none_before_visit(self):
        """get_child returns None for unexpanded child slot."""
        root = _expanded_root()
        assert root.get_child(0) is None

    def test_get_child_returns_node_after_add(self):
        """get_child returns the node after add_child."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        assert root.get_child(0) is child

    def test_get_child_unknown_action_returns_none(self):
        """get_child returns None for action not in children dict."""
        root = _root()
        assert root.get_child(99) is None


class TestActionStats:
    def test_get_action_visits_all_zero_before_visits(self):
        """Unvisited children have visit count 0."""
        root = _expanded_root(n_actions=3)
        visits = root.get_action_visits()
        assert all(v == 0 for v in visits.values())

    def test_get_action_visits_after_update(self):
        """Visit counts reflect child N after updates."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        child.update(1.0)
        visits = root.get_action_visits()
        assert visits[0] == 1

    def test_get_action_values_unvisited_zero(self):
        """Unvisited children have Q=0.0."""
        root = _expanded_root()
        values = root.get_action_values()
        assert all(v == 0.0 for v in values.values())

    def test_get_action_values_after_update(self):
        """Q values reflect child state after updates."""
        root = _expanded_root()
        child = root.add_child(0, "s")
        child.update(0.8)
        values = root.get_action_values()
        assert values[0] == pytest.approx(0.8)


class TestBestAction:
    def test_best_action_none_when_no_children(self):
        """best_action returns None when no children dict."""
        assert _root().best_action() is None

    def test_best_action_by_visits(self):
        """best_action(by='visits') returns most-visited action."""
        root = _expanded_root(n_actions=3)
        c0 = root.add_child(0, "s0")
        c1 = root.add_child(1, "s1")
        c0.update(0.5)
        c0.update(0.5)
        c1.update(0.5)
        assert root.best_action(by="visits") == 0

    def test_best_action_by_value(self):
        """best_action(by='value') returns highest-Q action."""
        root = _expanded_root(n_actions=2)
        c0 = root.add_child(0, "s0")
        c1 = root.add_child(1, "s1")
        c0.update(0.2)
        c1.update(0.9)
        assert root.best_action(by="value") == 1


class TestGetPolicy:
    def test_get_policy_empty_when_no_children(self):
        """get_policy returns {} when no children."""
        assert _root().get_policy() == {}

    def test_get_policy_sums_to_one(self):
        """get_policy probabilities sum to ~1.0."""
        root = _expanded_root(n_actions=3)
        for i in range(3):
            c = root.add_child(i, f"s{i}")
            for _ in range(i + 1):
                c.update(0.5)
        policy = root.get_policy(temperature=1.0)
        assert sum(policy.values()) == pytest.approx(1.0, abs=1e-6)

    def test_get_policy_temperature_zero_is_greedy(self):
        """temperature=0 assigns 1.0 to the most-visited action."""
        root = _expanded_root(n_actions=3)
        for i in range(3):
            c = root.add_child(i, f"s{i}")
            for _ in range(i + 1):
                c.update(0.5)
        policy = root.get_policy(temperature=0)
        # action 2 has 3 visits — should get probability 1.0
        assert policy[2] == pytest.approx(1.0)
        assert policy[0] == pytest.approx(0.0)
        assert policy[1] == pytest.approx(0.0)


class TestDepthAndPath:
    def test_depth_increases_with_nesting(self):
        """Each level adds 1 to depth."""
        root = _expanded_root()
        child = root.add_child(0, "s1")
        child.expand([0], [1.0])
        grandchild = child.add_child(0, "s2")
        assert grandchild.depth() == 2

    def test_path_to_root_single_node(self):
        """Root's path_to_root is [root]."""
        root = _root()
        path = root.path_to_root()
        assert path == [root]

    def test_path_to_root_two_levels(self):
        """Grandchild's path is [grandchild, child, root]."""
        root = _expanded_root()
        child = root.add_child(0, "s1")
        child.expand([0], [1.0])
        grandchild = child.add_child(0, "s2")
        path = grandchild.path_to_root()
        assert path == [grandchild, child, root]

    def test_repr_does_not_raise(self):
        """__repr__ returns a string without error."""
        node = _root()
        assert isinstance(repr(node), str)
