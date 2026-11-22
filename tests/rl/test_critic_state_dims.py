"""The attention/GAT critics must accept the package's own state encoders.

``ma_state.encode_state`` (per-agent) and ``ma_state.encode_global`` produce
different widths, so a critic must size its per-agent and global inputs
independently (``state_dim`` and ``global_state_dim``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.ml, pytest.mark.regression]

torch = pytest.importorskip("torch")

NODES = ("sam2", "nvdiffrec", "material_backend")
EDGES = [[0, 1], [1, 2]]


def _encoded_inputs():
    from transformation_portal.rl.ma_state import encode_global, encode_state, get_global_dim, get_state_dim

    metrics, diff = {"score": 0.5}, {"changes": []}
    agent_states = torch.tensor(np.stack([encode_state({}, metrics, diff, n) for n in NODES])).float().unsqueeze(0)
    global_state = torch.tensor(encode_global(metrics, diff)).float().unsqueeze(0)
    assert agent_states.shape[-1] == get_state_dim()
    assert global_state.shape[-1] == get_global_dim()
    assert get_state_dim() != get_global_dim()
    return get_state_dim(), get_global_dim(), agent_states, global_state


def test_gat_qmix_critic_accepts_package_encoders():
    from transformation_portal.rl.ma_gat_critic import GATQMIXCritic, create_gat_critic

    agent_dim, global_dim, agent_states, global_state = _encoded_inputs()
    edges = torch.tensor(EDGES)
    for critic in (
        GATQMIXCritic(agent_dim, len(NODES), global_state_dim=global_dim),
        create_gat_critic(agent_dim, len(NODES), global_state_dim=global_dim),
    ):
        q = critic.forward(torch.zeros(1, len(NODES)), agent_states, global_state, edges)
        assert q.shape == (1, 1)
        q_attn, _ = critic.forward_with_attention(torch.zeros(1, len(NODES)), agent_states, global_state, edges)
        assert torch.allclose(q, q_attn)


def test_temporal_gat_critic_accepts_package_encoders():
    from transformation_portal.rl.temporal_gat_critic import TemporalGATCritic

    agent_dim, global_dim, agent_states, global_state = _encoded_inputs()
    critic = TemporalGATCritic(agent_dim, len(NODES), seq_len=1, global_state_dim=global_dim)

    q = critic.forward(
        torch.zeros(1, 1, len(NODES)), agent_states.unsqueeze(1), global_state.unsqueeze(1), torch.tensor(EDGES)
    )

    assert q.shape == (1, 1)


def test_attention_critic_accepts_package_encoders():
    from transformation_portal.rl.ma_qmix_attn import CentralCriticQMIXAttn, create_attention_critic

    agent_dim, global_dim, agent_states, global_state = _encoded_inputs()
    for critic in (
        CentralCriticQMIXAttn(len(NODES), agent_dim, global_state_dim=global_dim),
        create_attention_critic(len(NODES), agent_dim, global_state_dim=global_dim),
    ):
        q, attn = critic.forward(torch.zeros(1, len(NODES)), agent_states, global_state)
        assert q.shape == (1, 1)
        assert attn.shape[-1] == len(NODES)


def test_global_state_dim_defaults_to_state_dim():
    """Backward compatible: without ``global_state_dim`` both inputs share ``state_dim``."""
    from transformation_portal.rl.ma_gat_critic import GATQMIXCritic
    from transformation_portal.rl.ma_qmix_attn import CentralCriticQMIXAttn
    from transformation_portal.rl.temporal_gat_critic import TemporalGATCritic

    gat = GATQMIXCritic(8, 3, embed_dim=8, gat_heads=2)
    temporal = TemporalGATCritic(8, 3, embed_dim=8, gat_heads=2, temporal_heads=2)
    attn = CentralCriticQMIXAttn(3, 8, embed_dim=8, n_heads=2)

    assert gat.global_state_dim == temporal.global_state_dim == attn.global_state_dim == 8
    assert gat.hyper_b1.in_features == temporal.hyper_b1.in_features == attn.mixer.hyper_b1.in_features == 8
    q = gat.forward(torch.zeros(2, 3), torch.randn(2, 3, 8), torch.randn(2, 8), torch.tensor(EDGES))
    assert q.shape == (2, 1)
