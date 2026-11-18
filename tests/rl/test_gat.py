"""Tests for the pure-PyTorch GAT layers and the GAT critics built on them."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml, pytest.mark.regression]

torch = pytest.importorskip("torch")

FAN_OUT = {"nodes": [{"id": "depth"}, {"id": "sam2", "deps": ["depth"]}, {"id": "nvdiffrec", "deps": ["depth"]}]}
CHAIN = {"nodes": [{"id": "a"}, {"id": "b", "deps": ["a"]}, {"id": "c", "deps": ["b"]}]}


def _own_state_influence(fn, num_nodes: int, in_dim: int) -> list[bool]:
    """For each node i, does perturbing node i's input change node i's output?"""
    base = torch.randn(num_nodes, in_dim)
    with torch.no_grad():
        ref = fn(base)
        influence = []
        for node in range(num_nodes):
            pert = base.clone()
            pert[node] += 5.0
            influence.append(bool((fn(pert)[node] - ref[node]).abs().max() > 1e-6))
    return influence


@pytest.mark.parametrize("pipeline", [FAN_OUT, CHAIN], ids=["fan_out", "chain"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_gat_layer_node_output_depends_on_own_features(pipeline, version):
    from transformation_portal.rl.gat import create_gat_layer
    from transformation_portal.rl.graph_utils import build_edge_index

    torch.manual_seed(0)
    edges = build_edge_index(pipeline)
    layer = create_gat_layer(8, 8, version=version, heads=2)

    assert _own_state_influence(lambda x: layer.forward(x, edges), 3, 8) == [True, True, True]


@pytest.mark.parametrize("pipeline", [FAN_OUT, CHAIN], ids=["fan_out", "chain"])
def test_gat_qmix_critic_context_depends_on_every_node_state(pipeline):
    from transformation_portal.rl.graph_utils import build_edge_index
    from transformation_portal.rl.ma_gat_critic import GATQMIXCritic

    torch.manual_seed(0)
    edges = build_edge_index(pipeline)
    critic = GATQMIXCritic(state_dim=8, n_agents=3, embed_dim=8, gat_heads=2)

    influence = _own_state_influence(lambda s: critic._run_gat(critic.node_embed(s), edges), 3, 8)

    assert influence == [True, True, True]


def test_self_loops_appended_without_duplicating_existing_ones():
    from transformation_portal.rl.gat import _with_self_loops

    edges = torch.tensor([[0, 1, 2], [1, 1, 0]])

    out = _with_self_loops(edges, 3)

    assert torch.equal(out[:, :3], edges)
    assert sorted(out[:, 3:].T.tolist()) == [[0, 0], [2, 2]]


def test_attention_rows_cover_input_edges_then_added_loops():
    from transformation_portal.rl.gat import GATLayer

    torch.manual_seed(0)
    layer = GATLayer(4, 4, heads=2)
    edges = torch.tensor([[0, 1], [1, 2]])

    _, attn = layer.forward(torch.randn(3, 4), edges, return_attention=True)

    # 2 input edges + 3 self-loops; incoming attention per node sums to one.
    assert attn.shape == (5, 2)
    dst = torch.cat([edges[1], torch.arange(3)])
    for node in range(3):
        assert torch.allclose(attn[dst == node].sum(dim=0), torch.ones(2))


def test_self_loops_can_be_disabled():
    from transformation_portal.rl.gat import GATLayer

    torch.manual_seed(0)
    layer = GATLayer(4, 4, heads=2, add_self_loops=False)
    edges = torch.tensor([[0, 1], [1, 2]])

    out, attn = layer.forward(torch.randn(3, 4), edges, return_attention=True)

    assert attn.shape == (2, 2)
    # Root node 0 has no incoming edge, so without self-loops only the bias remains.
    assert torch.allclose(out[0], layer.bias[:4])
