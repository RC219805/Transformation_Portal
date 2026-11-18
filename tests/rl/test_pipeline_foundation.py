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
