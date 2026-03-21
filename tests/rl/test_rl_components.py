"""Tests for RL action space module."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestActionSpace:
    """Tests for single-agent action space."""

    def test_enumerate_actions(self):
        """Test action enumeration."""
        from transformation_portal.rl.action_space import enumerate_actions

        actions = enumerate_actions()

        assert len(actions) > 0
        assert all(hasattr(a, "node") for a in actions)
        assert all(hasattr(a, "action_type") for a in actions)
        assert all(hasattr(a, "params") for a in actions)

    def test_action_indices_unique(self):
        """Test that action indices are unique."""
        from transformation_portal.rl.action_space import enumerate_actions

        actions = enumerate_actions()
        indices = [a.index for a in actions]

        assert len(indices) == len(set(indices))

    def test_get_action_dim(self):
        """Test action dimension calculation."""
        from transformation_portal.rl.action_space import enumerate_actions, get_action_dim

        actions = enumerate_actions()
        dim = get_action_dim()

        assert dim == len(actions)
        assert dim > 0

    def test_action_to_fix(self):
        """Test converting action to fix suggestion."""
        from transformation_portal.rl.action_space import RLAction

        action = RLAction(
            node="sam2",
            action_type="increase_mask_coverage",
            params={"threshold": 0.3},
            index=0,
        )

        fix = action.to_fix_suggestion()

        assert fix["target_node"] == "sam2"
        assert fix["action"] == "increase_mask_coverage"
        assert fix["params"]["threshold"] == 0.3


class TestMultiAgentActionSpace:
    """Tests for multi-agent action space."""

    def test_enumerate_node_actions(self):
        """Test per-node action enumeration."""
        from transformation_portal.rl.ma_action_space import enumerate_node_actions

        sam2_actions = enumerate_node_actions("sam2")
        nvdiffrec_actions = enumerate_node_actions("nvdiffrec")

        assert len(sam2_actions) > 0
        assert len(nvdiffrec_actions) > 0

        # All SAM2 actions should have node_id "sam2"
        assert all(a.node_id == "sam2" for a in sam2_actions)

    def test_get_all_node_actions(self):
        """Test getting actions for all nodes."""
        from transformation_portal.rl.ma_action_space import get_all_node_actions

        all_actions = get_all_node_actions()

        assert "sam2" in all_actions
        assert "nvdiffrec" in all_actions
        assert "material_backend" in all_actions

    def test_node_action_to_fix(self):
        """Test converting node action to fix."""
        from transformation_portal.rl.ma_action_space import NodeAction

        action = NodeAction(
            node_id="nvdiffrec",
            action_type="increase_iterations",
            params={"steps": 400},
        )

        fix = action.to_fix()

        assert fix["target_node"] == "nvdiffrec"
        assert fix["params"]["steps"] == 400


class TestStateEncoder:
    """Tests for state encoding."""

    def test_encode_state(self):
        """Test full state encoding."""
        from transformation_portal.rl.state_encoder import encode_state, get_state_dim

        pipeline = {"nodes": [{"id": "sam2", "config": {"threshold": 0.3}}]}
        metrics = {"score": 0.7, "psnr": 30.0}
        diff = {"changes": [{"type": "texture", "severity": "low"}]}

        state = encode_state(pipeline, metrics, diff)

        assert state is not None
        assert len(state) == get_state_dim()

    def test_state_dim_consistency(self):
        """Test state dimension is consistent."""
        from transformation_portal.rl.state_encoder import get_state_dim

        dim = get_state_dim()

        assert dim > 0
        # Should be: 5 metrics + 15 diff + 80 node configs + 5 history = 105
        assert dim == 105


class TestMultiAgentState:
    """Tests for multi-agent state encoding."""

    def test_encode_global(self):
        """Test global state encoding."""
        from transformation_portal.rl.ma_state import encode_global, get_global_dim

        metrics = {"score": 0.8, "psnr": 35.0}
        diff = {"changes": []}

        global_state = encode_global(metrics, diff)

        assert len(global_state) == get_global_dim()

    def test_encode_local(self):
        """Test local state encoding."""
        from transformation_portal.rl.ma_state import encode_local, get_local_dim

        node_cfg = {"threshold": 0.4, "steps": 500}

        local_state = encode_local(node_cfg, "sam2")

        assert len(local_state) == get_local_dim()

    def test_encode_full_state(self):
        """Test combined state encoding."""
        from transformation_portal.rl.ma_state import encode_state, get_state_dim

        node_cfg = {"threshold": 0.3}
        metrics = {"score": 0.7}
        diff = {"changes": []}

        state = encode_state(node_cfg, metrics, diff, "sam2")

        assert len(state) == get_state_dim()


class TestReplayBuffer:
    """Tests for replay buffer."""

    def test_replay_buffer_add_sample(self):
        """Test adding and sampling from replay buffer."""
        from transformation_portal.rl.replay import ReplayBuffer, Transition

        buffer = ReplayBuffer(capacity=100)

        # Add transitions
        for i in range(20):
            buffer.add(
                Transition(
                    state=[float(i)],
                    action=i % 5,
                    reward=float(i) / 20,
                    next_state=[float(i + 1)],
                    done=False,
                )
            )

        assert len(buffer) == 20

        # Sample
        batch = buffer.sample(10)
        assert len(batch) == 10

    def test_replay_buffer_capacity(self):
        """Test buffer respects capacity."""
        from transformation_portal.rl.replay import ReplayBuffer, Transition

        buffer = ReplayBuffer(capacity=10)

        for i in range(20):
            buffer.add(
                Transition(
                    state=[float(i)],
                    action=0,
                    reward=0.0,
                    next_state=[float(i)],
                    done=False,
                )
            )

        assert len(buffer) == 10


class TestPolicyGuard:
    """Tests for policy guard."""

    def test_is_safe_action(self):
        """Test safe action detection."""
        from transformation_portal.rl.action_space import RLAction
        from transformation_portal.rl.policy_guard import is_safe

        safe_action = RLAction(
            node="sam2",
            action_type="increase_mask_coverage",
            params={},
        )

        assert is_safe(safe_action)

    def test_filter_actions(self):
        """Test action filtering."""
        from transformation_portal.rl.action_space import enumerate_actions
        from transformation_portal.rl.policy_guard import filter_actions

        all_actions = enumerate_actions()
        safe_actions = filter_actions(all_actions)

        assert len(safe_actions) <= len(all_actions)
        assert len(safe_actions) > 0


class TestMessageBus:
    """Tests for multi-agent message bus."""

    def test_publish_read(self):
        """Test publishing and reading messages."""
        from transformation_portal.rl.ma_comm import MessageBus

        bus = MessageBus()

        bus.publish("sam2", {"intent": "increase_coverage"})
        bus.publish("nvdiffrec", {"intent": "increase_iterations"})

        messages = bus.read_all()

        assert "sam2" in messages
        assert "nvdiffrec" in messages
        assert messages["sam2"]["intent"] == "increase_coverage"

    def test_read_others(self):
        """Test reading others' messages."""
        from transformation_portal.rl.ma_comm import MessageBus

        bus = MessageBus()

        bus.publish("sam2", {"intent": "a"})
        bus.publish("nvdiffrec", {"intent": "b"})

        others = bus.read_others("sam2")

        assert "sam2" not in others
        assert "nvdiffrec" in others

    def test_clear(self):
        """Test clearing messages."""
        from transformation_portal.rl.ma_comm import MessageBus

        bus = MessageBus()

        bus.publish("sam2", {"intent": "test"})
        bus.clear()

        messages = bus.read_all()
        assert len(messages) == 0
