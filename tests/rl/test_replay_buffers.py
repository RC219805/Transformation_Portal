"""Tests for rl.replay and rl.temporal_replay buffer implementations."""

from __future__ import annotations

import pytest

from transformation_portal.rl.replay import (
    BatchedTransitions,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    Transition,
)
from transformation_portal.rl.temporal_replay import (
    Episode,
    TemporalReplayBuffer,
    TemporalTransition,
)

pytestmark = pytest.mark.unit


def _transition(state=0, action=0, reward=1.0, next_state=1, done=False) -> Transition:
    return Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)


def _temporal_transition(reward=1.0, done=False) -> TemporalTransition:
    return TemporalTransition(
        states={"sam2": [0.0] * 8},
        actions={"sam2": 0},
        reward=reward,
        global_state=[0.0] * 20,
        done=done,
    )


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_add_increases_length(self):
        """Adding one transition increases buffer size to 1."""
        buf = ReplayBuffer(capacity=100)
        buf.add(_transition())
        assert len(buf) == 1

    def test_capacity_enforced(self):
        """Buffer never exceeds its capacity."""
        buf = ReplayBuffer(capacity=10)
        for i in range(25):
            buf.add(_transition(state=i))
        assert len(buf) == 10

    def test_sample_returns_correct_count(self):
        """sample(n) returns exactly n transitions when buffer is full enough."""
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(_transition(state=i))
        result = buf.sample(16)
        assert len(result) == 16

    def test_sample_capped_by_buffer_size(self):
        """sample(n) returns at most len(buffer) items."""
        buf = ReplayBuffer(capacity=100)
        for i in range(5):
            buf.add(_transition(state=i))
        result = buf.sample(100)
        assert len(result) == 5

    def test_is_ready_below_threshold_false(self):
        """is_ready returns False when below min_size."""
        buf = ReplayBuffer(capacity=100)
        for i in range(5):
            buf.add(_transition())
        assert buf.is_ready(10) is False

    def test_is_ready_at_threshold_true(self):
        """is_ready returns True when at min_size."""
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(_transition())
        assert buf.is_ready(10) is True

    def test_clear_empties_buffer(self):
        """clear() resets length to 0."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(5):
            buf.add(_transition())
        buf.clear()
        assert len(buf) == 0

    def test_transition_values_preserved(self):
        """Sampled transition retains its original field values."""
        buf = ReplayBuffer(capacity=100)
        t = _transition(state=42, action=3, reward=9.9, next_state=99, done=True)
        buf.add(t)
        samples = buf.sample(1)
        assert samples[0].state == 42
        assert samples[0].action == 3
        assert samples[0].reward == pytest.approx(9.9)
        assert samples[0].done is True

    def test_add_batch_increases_length(self):
        """add_batch adds all supplied transitions."""
        buf = ReplayBuffer(capacity=100)
        batch = [_transition(state=i) for i in range(5)]
        buf.add_batch(batch)
        assert len(buf) == 5

    def test_sample_batched_returns_none_when_insufficient(self):
        """sample_batched returns None when fewer items than batch_size."""
        buf = ReplayBuffer(capacity=100)
        buf.add(_transition())
        assert buf.sample_batched(32) is None

    def test_sample_batched_returns_batched_transitions(self):
        """sample_batched returns BatchedTransitions when sufficient."""
        buf = ReplayBuffer(capacity=100)
        for i in range(40):
            buf.add(_transition(state=i, action=i % 3))
        result = buf.sample_batched(32)
        assert result is not None
        assert isinstance(result, BatchedTransitions)
        assert len(result.states) == 32


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBuffer:
    def test_add_with_default_priority(self):
        """Adding without explicit priority succeeds."""
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add(_transition())
        assert len(buf) == 1

    def test_add_with_explicit_priority(self):
        """Adding with explicit priority succeeds."""
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add(_transition(), priority=2.0)
        assert len(buf) == 1

    def test_capacity_overflow_wraps(self):
        """Buffer wraps around at capacity."""
        buf = PrioritizedReplayBuffer(capacity=10)
        for i in range(15):
            buf.add(_transition(state=i))
        assert len(buf) == 10

    def test_sample_returns_three_tuple(self):
        """sample() returns (transitions, indices, weights)."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(_transition(state=i))
        transitions, indices, weights = buf.sample(16)
        assert len(transitions) == 16
        assert len(indices) == 16
        assert len(weights) == 16

    def test_sample_empty_returns_empty(self):
        """Sampling from empty buffer returns three empty lists."""
        buf = PrioritizedReplayBuffer(capacity=100)
        transitions, indices, weights = buf.sample(16)
        assert transitions == []
        assert indices == []
        assert weights == []

    def test_update_priorities_accepted(self):
        """update_priorities does not raise."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(_transition(state=i))
        _, indices, _ = buf.sample(5)
        buf.update_priorities(indices, [1.0] * len(indices))

    def test_weights_normalized(self):
        """Importance-sampling weights are in (0, 1]."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(_transition(state=i), priority=float(i + 1))
        _, _, weights = buf.sample(20)
        assert all(0.0 < w <= 1.0 for w in weights)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def _mock_value(self, v: float):
        """Return a simple object with .item() → v (mimics torch.Tensor)."""
        class _V:
            def item(self_):
                return v
        return _V()

    def test_add_increases_length(self):
        """Each add increases length by 1."""
        buf = RolloutBuffer()
        buf.add(state=0, action=1, reward=1.0, value=self._mock_value(0.5), log_prob=0.0, done=False)
        assert len(buf) == 1

    def test_clear_resets_all_lists(self):
        """clear() empties all stored lists."""
        buf = RolloutBuffer()
        buf.add(state=0, action=0, reward=1.0, value=self._mock_value(0.5), log_prob=0.0, done=False)
        buf.clear()
        assert len(buf) == 0
        assert buf.states == []
        assert buf.actions == []
        assert buf.rewards == []

    def test_compute_returns_length_matches_steps(self):
        """compute_returns returns lists of same length as stored steps."""
        buf = RolloutBuffer()
        for i in range(5):
            buf.add(state=i, action=0, reward=1.0, value=self._mock_value(0.5), log_prob=0.0, done=False)
        returns, advantages = buf.compute_returns(last_value=0.0)
        assert len(returns) == 5
        assert len(advantages) == 5

    def test_compute_returns_finite(self):
        """All computed returns and advantages are finite numbers."""
        import math
        buf = RolloutBuffer()
        for i in range(3):
            buf.add(state=i, action=0, reward=1.0, value=self._mock_value(0.5), log_prob=0.0, done=False)
        returns, advantages = buf.compute_returns(last_value=0.0)
        assert all(math.isfinite(r) for r in returns)
        assert all(math.isfinite(a) for a in advantages)

    def test_done_flag_resets_gae(self):
        """A done=True step zeroes out future GAE contributions."""
        buf = RolloutBuffer()
        # Step that terminates: delta = reward - value, gae = delta only
        buf.add(state=0, action=0, reward=1.0, value=self._mock_value(0.5), log_prob=0.0, done=True)
        # Step after terminal (only used for last_value bootstrap, not accumulated)
        buf.add(state=1, action=0, reward=0.5, value=self._mock_value(0.3), log_prob=0.0, done=False)
        returns, advantages = buf.compute_returns(last_value=0.0)
        # Terminal step advantage should equal reward - value exactly
        assert advantages[0] == pytest.approx(1.0 - 0.5)


# ---------------------------------------------------------------------------
# Transition NamedTuple
# ---------------------------------------------------------------------------

class TestTransition:
    def test_required_fields(self):
        """Transition has all required fields."""
        t = _transition()
        assert hasattr(t, "state")
        assert hasattr(t, "action")
        assert hasattr(t, "reward")
        assert hasattr(t, "next_state")
        assert hasattr(t, "done")

    def test_optional_fields_default_none(self):
        """log_prob and value default to None."""
        t = _transition()
        assert t.log_prob is None
        assert t.value is None


# ---------------------------------------------------------------------------
# TemporalTransition / Episode
# ---------------------------------------------------------------------------

class TestTemporalTransition:
    def test_default_done_false(self):
        """done defaults to False."""
        t = _temporal_transition()
        assert t.done is False

    def test_info_default_empty(self):
        """info defaults to empty dict."""
        t = _temporal_transition()
        assert t.info == {}


class TestEpisode:
    def test_add_increments_length(self):
        """Episode.add increments length by 1."""
        ep = Episode()
        ep.add(_temporal_transition(reward=1.0))
        assert len(ep) == 1

    def test_add_accumulates_reward(self):
        """Episode.add accumulates total_reward."""
        ep = Episode()
        ep.add(_temporal_transition(reward=2.0))
        ep.add(_temporal_transition(reward=3.0))
        assert ep.total_reward == pytest.approx(5.0)

    def test_get_sequence_returns_slice(self):
        """get_sequence(start, length) returns correct slice."""
        ep = Episode()
        for i in range(5):
            ep.add(_temporal_transition(reward=float(i)))
        seq = ep.get_sequence(1, 3)
        assert len(seq) == 3

    def test_get_sequence_clips_at_end(self):
        """get_sequence clips when start+length exceeds episode length."""
        ep = Episode()
        for _ in range(3):
            ep.add(_temporal_transition())
        seq = ep.get_sequence(2, 10)
        assert len(seq) == 1


# ---------------------------------------------------------------------------
# TemporalReplayBuffer
# ---------------------------------------------------------------------------

class TestTemporalReplayBuffer:
    def _make_episode(self, n: int, reward: float = 1.0) -> Episode:
        ep = Episode()
        for _ in range(n):
            ep.add(_temporal_transition(reward=reward))
        return ep

    def test_add_episode_accepted_when_long_enough(self):
        """Episodes with length >= seq_len are stored."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(5))
        assert len(buf) == 1

    def test_add_episode_rejected_when_too_short(self):
        """Episodes shorter than seq_len are silently dropped."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(2))
        assert len(buf) == 0

    def test_add_transitions_creates_episode(self):
        """add_transitions wraps transitions in an Episode."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        transitions = [_temporal_transition() for _ in range(4)]
        buf.add_transitions(transitions)
        assert len(buf) == 1

    def test_add_transitions_rejects_too_few(self):
        """add_transitions with fewer than seq_len transitions is dropped."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_transitions([_temporal_transition()])
        assert len(buf) == 0

    def test_sample_returns_list_of_dicts(self):
        """sample() returns a list of dicts."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        for _ in range(5):
            buf.add_episode(self._make_episode(5))
        result = buf.sample(3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, dict) for item in result)

    def test_sample_empty_returns_empty(self):
        """Sampling empty buffer returns []."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        assert buf.sample(5) == []

    def test_sample_has_transitions_key(self):
        """Each sample dict has 'transitions' key."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(5))
        result = buf.sample(1)
        assert "transitions" in result[0]

    def test_clear_removes_all_episodes(self):
        """clear() empties the buffer."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(5))
        buf.clear()
        assert len(buf) == 0

    def test_total_transitions_sums_correctly(self):
        """total_transitions sums lengths across episodes."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(4))
        buf.add_episode(self._make_episode(6))
        assert buf.total_transitions() == 10

    def test_is_ready_false_below_min(self):
        """is_ready returns False when fewer episodes than min."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        buf.add_episode(self._make_episode(5))
        assert buf.is_ready(10) is False

    def test_is_ready_true_at_min(self):
        """is_ready returns True when enough episodes."""
        buf = TemporalReplayBuffer(capacity=100, seq_len=3)
        for _ in range(10):
            buf.add_episode(self._make_episode(5))
        assert buf.is_ready(10) is True

    def test_capacity_overflow_removes_oldest(self):
        """Buffer drops oldest episode when capacity exceeded."""
        buf = TemporalReplayBuffer(capacity=3, seq_len=2)
        for _ in range(5):
            buf.add_episode(self._make_episode(3))
        assert len(buf) == 3
