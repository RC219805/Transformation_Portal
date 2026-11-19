"""Tests for the Pipeline Foundation Model trainer."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml, pytest.mark.regression]

torch = pytest.importorskip("torch")


def _tiny_trainer(lr: float, warmup_steps: int):
    from transformation_portal.rl.pipeline_foundation import (
        PipelineFoundationConfig,
        PipelineFoundationModel,
        PipelineFoundationTrainer,
    )

    torch.manual_seed(0)
    cfg = PipelineFoundationConfig(
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=32,
        dropout=0.0,
        max_nodes=4,
        max_seq_len=8,
        node_state_dim=8,
        action_dim=4,
        metric_dim=3,
    )
    model = PipelineFoundationModel(cfg)
    trainer = PipelineFoundationTrainer(model, lr=lr, warmup_steps=warmup_steps)
    batch_size, seq_len, n_nodes = 2, 3, 3
    batch = {
        "node_states": torch.randn(batch_size, n_nodes, cfg.node_state_dim),
        "actions": torch.randn(batch_size, seq_len, cfg.action_dim),
        "rewards": torch.randn(batch_size, seq_len, 1),
        "target_value": torch.randn(batch_size),
    }
    return model, trainer, batch


def test_warmup_is_linear_and_reaches_the_base_lr():
    """R4: warm-up must scale the base lr, not compound onto the scaled lr."""
    base_lr = 1e-3
    _, trainer, batch = _tiny_trainer(base_lr, warmup_steps=4)

    seen = []
    for _ in range(7):
        trainer.train_step(batch)
        seen.append(trainer.optimizer.param_groups[0]["lr"])

    expected = [base_lr * min(1.0, step / 4) for step in range(1, 8)]
    assert seen == pytest.approx(expected, rel=1e-12)


def test_weights_keep_updating_after_warmup():
    model, trainer, batch = _tiny_trainer(1e-3, warmup_steps=3)
    for _ in range(5):
        trainer.train_step(batch)
    before = model.value_head[0].weight.detach().clone()
    trainer.train_step(batch)
    assert not torch.equal(before, model.value_head[0].weight.detach())


def test_zero_warmup_uses_the_base_lr_immediately():
    _, trainer, batch = _tiny_trainer(5e-4, warmup_steps=0)
    trainer.train_step(batch)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def _chain_adjacency(batch_size: int, n_nodes: int):
    adj = torch.zeros(batch_size, n_nodes, n_nodes)
    for i in range(n_nodes - 1):
        adj[:, i, i + 1] = 1.0
    return adj


def _pretrain_batch(n: int, positive_idx: list[int]):
    return {
        "node_states": torch.zeros(n, 3, 8),
        "actions": torch.zeros(n, 3, 4),
        "rewards": torch.zeros(n, 3, 1),
        "positive_idx": torch.tensor(positive_idx),
    }


def test_batched_adjacency_as_documented_runs_and_is_finite():
    """R5: a [B, N, N] adjacency crashed (3-D attn_mask not repeated per head)."""
    model, _, batch = _tiny_trainer(1e-3, warmup_steps=0)
    model.eval()
    adj = _chain_adjacency(2, 3)

    with torch.no_grad():
        out = model(batch["node_states"], batch["actions"], batch["rewards"], adjacency=adj)

    assert all(torch.isfinite(v).all() for v in out.values())


def test_dag_adjacency_does_not_produce_nan_embeddings():
    """R5: the sink row of a DAG adjacency was fully masked, so its softmax was NaN."""
    model, _, batch = _tiny_trainer(1e-3, warmup_steps=0)
    model.eval()
    adj = _chain_adjacency(1, 3)[0]

    with torch.no_grad():
        nodes, graph = model.graph_encoder(batch["node_states"][:1], adjacency=adj)

    assert torch.isfinite(nodes).all()
    assert torch.isfinite(graph).all()


def test_batched_adjacency_is_applied_per_sample():
    model, _, batch = _tiny_trainer(1e-3, warmup_steps=0)
    model.eval()
    states = batch["node_states"]
    adj = torch.stack([_chain_adjacency(1, 3)[0], torch.ones(3, 3)])

    with torch.no_grad():
        nodes_batched, _ = model.graph_encoder(states, adjacency=adj)
        nodes_0, _ = model.graph_encoder(states[:1], adjacency=adj[0])
        nodes_1, _ = model.graph_encoder(states[1:], adjacency=adj[1])

    assert torch.allclose(nodes_batched[0], nodes_0[0], atol=1e-5)
    assert torch.allclose(nodes_batched[1], nodes_1[0], atol=1e-5)


def test_adjacency_with_wrong_shape_is_rejected():
    model, _, batch = _tiny_trainer(1e-3, warmup_steps=0)
    with pytest.raises(ValueError, match="adjacency"):
        model.graph_encoder(batch["node_states"], adjacency=torch.ones(4, 4))


def test_contrastive_loss_excludes_self_similarity():
    """R5: InfoNCE kept the diagonal, so perfectly aligned pairs could not go below log(2)."""
    model, _, _ = _tiny_trainer(1e-3, warmup_steps=0)
    embeddings = torch.eye(4)[[0, 0, 1, 1]]
    model.forward = lambda *args, **kwargs: {"contrastive": embeddings}

    losses = model.pretrain_step(_pretrain_batch(4, [1, 0, 3, 2]))

    assert losses["contrastive"].item() < 1e-3


def test_contrastive_positive_pointing_at_itself_is_rejected():
    model, _, _ = _tiny_trainer(1e-3, warmup_steps=0)
    model.forward = lambda *args, **kwargs: {"contrastive": torch.eye(2)}

    with pytest.raises(ValueError, match="itself"):
        model.pretrain_step(_pretrain_batch(2, [0, 0]))


def test_batch_without_targets_fails_with_a_clear_error():
    """R5: with no loss term, the total was the int 0 and backward() crashed."""
    _, trainer, batch = _tiny_trainer(1e-3, warmup_steps=0)
    del batch["target_value"]
    with pytest.raises(ValueError, match="no training target"):
        trainer.train_step(batch)
