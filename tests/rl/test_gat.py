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
    assert isinstance(attn, torch.Tensor)
    assert attn.shape == (5, 2)
    dst = torch.cat([edges[1], torch.arange(3)])
    for node in range(3):
        incoming = torch.index_select(attn, 0, torch.nonzero(dst == node).flatten())
        assert torch.allclose(incoming.sum(dim=0), torch.ones(2))


def test_self_loops_can_be_disabled():
    from transformation_portal.rl.gat import GATLayer

    torch.manual_seed(0)
    layer = GATLayer(4, 4, heads=2, add_self_loops=False)
    edges = torch.tensor([[0, 1], [1, 2]])

    out, attn = layer.forward(torch.randn(3, 4), edges, return_attention=True)

    assert attn.shape == (2, 2)
    # Root node 0 has no incoming edge, so without self-loops only the bias remains.
    assert torch.allclose(out[0], layer.bias[:4])


def test_gat_layer_with_dropout_runs_and_eval_disables_dropout():
    """R6: GATLayer(dropout>0) crashed reading self.training on a plain class."""
    from transformation_portal.rl.gat import GATLayer

    torch.manual_seed(0)
    layer = GATLayer(8, 8, heads=2, dropout=0.5)
    x = torch.randn(3, 8)
    edges = torch.tensor([[0, 1], [1, 2]])

    assert layer.training is True
    _, attn_train = layer.forward(x, edges, return_attention=True)
    assert attn_train.shape[1] == 2

    assert layer.eval() is layer
    assert layer.training is False
    with torch.no_grad():
        first = layer.forward(x, edges)
        second = layer.forward(x, edges)
    assert torch.equal(first, second)

    layer.train()
    assert layer.training is True


def test_gat_qmix_critic_with_gat_dropout_forwards_and_eval_is_deterministic():
    """R6: GATQMIXCritic(gat_dropout>0).forward crashed with AttributeError."""
    from transformation_portal.rl.ma_gat_critic import GATQMIXCritic

    torch.manual_seed(0)
    critic = GATQMIXCritic(8, 3, embed_dim=8, gat_heads=2, gat_dropout=0.5)
    args = (torch.zeros(1, 3), torch.randn(1, 3, 8), torch.randn(1, 8), torch.tensor([[0, 1], [1, 2]]))

    q_total = critic.forward(*args)
    assert q_total.shape == (1, 1)
    assert torch.isfinite(q_total).all()

    critic.eval()
    assert all(not gat.training for gat in critic.gat_layers)
    with torch.no_grad():
        assert torch.equal(critic.forward(*args), critic.forward(*args))


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_gat_layers_expose_train_eval_modes(version):
    from transformation_portal.rl.gat import create_gat_layer

    layer = create_gat_layer(4, 4, version=version, heads=2, dropout=0.1)
    assert layer.training is True
    assert layer.eval().training is False
    assert layer.train(True).training is True


# --- R8: vectorised per-destination softmax / aggregation and batched critics ---


def _loop_segment_softmax(e, dst, num_nodes):
    """The original per-node Python loop, kept as the reference."""
    F = torch.nn.functional
    attn = torch.zeros_like(e)
    for i in range(num_nodes):
        mask = dst == i
        if mask.any():
            attn[mask] = F.softmax(e[mask], dim=0)
    return attn


def _loop_scatter_sum(messages, dst, num_nodes):
    out = torch.zeros((num_nodes,) + tuple(messages.shape[1:]), dtype=messages.dtype)
    for i in range(messages.size(0)):
        out[dst[i]] += messages[i]
    return out


R8_EDGES = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 5, 6, 3]])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_vectorised_gat_layer_matches_loop_reference(version, monkeypatch):
    import transformation_portal.rl.gat as gat_mod

    torch.manual_seed(0)
    layer = gat_mod.create_gat_layer(6, 5, version=version, heads=3)
    x = torch.randn(10, 6, requires_grad=True)

    out = layer.forward(x, R8_EDGES)
    (grad_vec,) = torch.autograd.grad((out * torch.arange(5.0)).sum(), x)

    monkeypatch.setattr(gat_mod, "_segment_softmax", _loop_segment_softmax)
    monkeypatch.setattr(gat_mod, "_scatter_sum", _loop_scatter_sum)
    ref = layer.forward(x, R8_EDGES)
    (grad_ref,) = torch.autograd.grad((ref * torch.arange(5.0)).sum(), x)

    assert torch.allclose(out, ref, rtol=0, atol=1e-6)
    assert torch.allclose(grad_vec, grad_ref, rtol=0, atol=1e-6)


def test_segment_softmax_is_stable_for_large_scores():
    from transformation_portal.rl.gat import _segment_softmax

    e = torch.tensor([[1000.0], [999.0], [-1000.0]])
    dst = torch.tensor([0, 0, 1])

    attn = _segment_softmax(e, dst, 3)

    assert torch.isfinite(attn).all()
    assert torch.allclose(attn, _loop_segment_softmax(e, dst, 3), rtol=0, atol=1e-6)


def test_batched_gat_qmix_critic_matches_per_item_loop():
    from transformation_portal.rl.ma_gat_critic import GATQMIXCritic

    torch.manual_seed(0)
    critic = GATQMIXCritic(state_dim=7, n_agents=10, embed_dim=8, gat_heads=2)
    args = (torch.randn(4, 10), torch.randn(4, 10, 7), torch.randn(4, 7), R8_EDGES)

    batched = critic.forward(*args)
    looped, _ = critic.forward_with_attention(*args)  # still runs one graph per batch item

    assert torch.allclose(batched, looped, rtol=0, atol=1e-6)


def test_batched_temporal_gat_critic_matches_per_item_loop():
    from transformation_portal.rl.temporal_gat_critic import TemporalGATCritic

    torch.manual_seed(0)
    critic = TemporalGATCritic(7, 10, embed_dim=8, seq_len=3, gat_heads=2, temporal_heads=2)
    args = (torch.randn(4, 3, 10), torch.randn(4, 3, 10, 7), torch.randn(4, 3, 7), R8_EDGES)

    batched = critic.forward(*args)
    looped, _, _ = critic.forward_with_attention(*args)  # still loops over (timestep, batch item)

    assert torch.allclose(batched, looped, rtol=0, atol=1e-6)


def test_batch_edge_index_offsets_copies_and_rejects_out_of_range_nodes():
    from transformation_portal.rl.gat import batch_edge_index

    edges = torch.tensor([[0, 1], [1, 2]])

    assert batch_edge_index(edges, 2, 3).tolist() == [[0, 1, 3, 4], [1, 2, 4, 5]]
    assert batch_edge_index(torch.zeros(2, 0, dtype=torch.long), 4, 3).numel() == 0
    for bad in (torch.tensor([[0, 3], [1, 2]]), torch.tensor([[0, -1], [1, 2]])):
        with pytest.raises(ValueError, match="outside"):
            batch_edge_index(bad, 2, 3)
