#!/usr/bin/env python3
"""
Depth Estimation Trainer

Orchestrates training for Depth Anything V2 fine-tuning:
- Training and validation loops
- Checkpoint management (best/last/periodic)
- Metric tracking and TensorBoard logging
- Early stopping
- Learning rate scheduling with warmup
- Resume from checkpoint

Author: Transformation Portal Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .losses import CombinedDepthLoss
from .metrics import DepthMetricCalculator, visualize_depth_comparison
from .utils import (
    AverageMeter,
    cleanup_checkpoints,
    get_lr,
    get_num_params,
    save_checkpoint,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for depth model training.

    Attributes:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Number of steps for gradient accumulation
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs
        mixed_precision: Mixed precision mode ('fp16', 'bf16', or None)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        save_dir: Directory for checkpoints
        log_dir: Directory for logs
        save_every_n_epochs: Save checkpoint every N epochs
        keep_last_n: Number of recent checkpoints to keep
        save_best: Whether to save best model
        monitor_metric: Metric to monitor for best model
        mode: 'min' or 'max' for monitor metric
        early_stopping_patience: Patience for early stopping (0 to disable)
        log_every_n_steps: Log training loss every N steps
        save_images_every_n_epochs: Save visualization every N epochs
    """
    # Training
    num_epochs: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    mixed_precision: Optional[str] = "fp16"
    num_workers: int = 8
    pin_memory: bool = True

    # Checkpointing
    save_dir: str = "checkpoints/depth_anything_v2"
    log_dir: str = "logs/depth_anything_v2"
    save_every_n_epochs: int = 5
    keep_last_n: int = 3
    save_best: bool = True
    monitor_metric: str = "val_rmse"
    mode: str = "min"

    # Early stopping
    early_stopping_patience: int = 10

    # Logging
    log_every_n_steps: int = 50
    save_images_every_n_epochs: int = 1
    tensorboard: bool = True


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.

    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.early_stop = False


class DepthTrainer:
    """Trainer for Depth Anything V2 fine-tuning.

    Handles the complete training workflow including:
    - Training and validation loops
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint management
    - TensorBoard logging
    - Early stopping

    Example:
        >>> config = TrainingConfig(num_epochs=50, batch_size=8)
        >>> trainer = DepthTrainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: "nn.Module",
        config: TrainingConfig,
        loss_fn: Optional["nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        scheduler: Optional[Any] = None,
        device: Optional["torch.device"] = None,
    ):
        """Initialize trainer.

        Args:
            model: Depth estimation model
            config: Training configuration
            loss_fn: Loss function (default: CombinedDepthLoss)
            optimizer: Optimizer (default: AdamW)
            scheduler: LR scheduler (default: cosine with warmup)
            device: Training device
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for training. "
                "Install with: pip install torch"
            )

        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        # Move model to device
        self.model = model.to(self.device)
        logger.info(f"Model parameters: {get_num_params(model):,}")

        # Loss function
        self.loss_fn = loss_fn or CombinedDepthLoss()

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler (will be set up in fit())
        self.scheduler = scheduler

        # Mixed precision scaler
        self.scaler = None
        if config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Using FP16 mixed precision")

        # Early stopping
        self.early_stopping = None
        if config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode=config.mode,
            )

        # TensorBoard writer
        self.writer = None
        if config.tensorboard and TENSORBOARD_AVAILABLE:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(config.log_dir)
            logger.info(f"TensorBoard logging to {config.log_dir}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if config.mode == "min" else float("-inf")
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
        }

        # Create checkpoint directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader: "DataLoader",
        val_loader: Optional["DataLoader"] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from

        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if specified
        if resume_from:
            self._resume_checkpoint(resume_from)

        # Set up scheduler if not provided
        if self.scheduler is None:
            total_steps = len(train_loader) * self.config.num_epochs
            warmup_steps = len(train_loader) * self.config.warmup_epochs
            self.scheduler = self._create_scheduler(total_steps, warmup_steps)

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)
            self.training_history["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(val_loader, epoch)
                self.training_history["val_loss"].append(val_metrics["loss"])
                self.training_history["val_rmse"].append(val_metrics["rmse"])

                # Check for best model
                current_metric = val_metrics.get(
                    self.config.monitor_metric.replace("val_", ""),
                    val_metrics["rmse"]
                )
                is_best = self._is_best(current_metric)

                if is_best:
                    self.best_metric = current_metric
                    logger.info(f"New best {self.config.monitor_metric}: {current_metric:.4f}")

                # Save checkpoint
                if self.config.save_best and is_best:
                    self._save_checkpoint(epoch, is_best=True)

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(current_metric):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Clean up old checkpoints
            cleanup_checkpoints(
                self.config.save_dir,
                keep_last_n=self.config.keep_last_n,
            )

        # Final save
        self._save_checkpoint(self.current_epoch, is_best=False, final=True)

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        logger.info("Training complete!")
        logger.info(f"Best {self.config.monitor_metric}: {self.best_metric:.4f}")

        return self.training_history

    def _train_epoch(
        self,
        train_loader: "DataLoader",
        epoch: int,
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        loss_meter = AverageMeter()

        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )
        else:
            pbar = train_loader

        accumulation_steps = self.config.gradient_accumulation_steps

        for batch_idx, (images, depths) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            depths = depths.to(self.device)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.loss_fn(predictions, depths)
                    loss = loss / accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                # Standard precision
                predictions = self.model(images)
                loss, loss_dict = self.loss_fn(predictions, depths)
                loss = loss / accumulation_steps

                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

            # Update metrics
            loss_meter.update(loss.item() * accumulation_steps, images.size(0))
            self.global_step += 1

            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "lr": f"{get_lr(self.optimizer):.2e}",
                })

            # Log to TensorBoard
            if (
                self.writer is not None
                and self.global_step % self.config.log_every_n_steps == 0
            ):
                self.writer.add_scalar(
                    "train/loss", loss_meter.val, self.global_step
                )
                self.writer.add_scalar(
                    "train/lr", get_lr(self.optimizer), self.global_step
                )
                for name, value in loss_dict.items():
                    if name != "total":
                        self.writer.add_scalar(
                            f"train/{name}", value.item(), self.global_step
                        )

        logger.info(f"Epoch {epoch + 1} - Train Loss: {loss_meter.avg:.4f}")
        return loss_meter.avg

    def _validate(
        self,
        val_loader: "DataLoader",
        epoch: int,
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        loss_meter = AverageMeter()
        metric_calculator = DepthMetricCalculator()

        with torch.no_grad():
            for images, depths in val_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(images)
                        loss, _ = self.loss_fn(predictions, depths)
                else:
                    predictions = self.model(images)
                    loss, _ = self.loss_fn(predictions, depths)

                loss_meter.update(loss.item(), images.size(0))
                metric_calculator.update(predictions, depths)

        # Compute metrics
        metrics = metric_calculator.compute()

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss_meter.avg, epoch)
            for name, value in metrics.to_dict().items():
                self.writer.add_scalar(f"val/{name}", value, epoch)

        # Save visualization
        if (epoch + 1) % self.config.save_images_every_n_epochs == 0:
            self._save_visualization(val_loader, epoch)

        logger.info(
            f"Epoch {epoch + 1} - Val Loss: {loss_meter.avg:.4f}, "
            f"RMSE: {metrics.rmse:.4f}, "
            f"Abs Rel: {metrics.abs_rel:.4f}, "
            f"δ<1.25: {metrics.delta1:.4f}"
        )

        return {
            "loss": loss_meter.avg,
            "rmse": metrics.rmse,
            "abs_rel": metrics.abs_rel,
            "delta1": metrics.delta1,
        }

    def _save_visualization(
        self,
        val_loader: "DataLoader",
        epoch: int,
    ) -> None:
        """Save depth visualization.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch
        """
        self.model.eval()

        # Get first batch
        images, depths = next(iter(val_loader))
        images = images.to(self.device)

        with torch.no_grad():
            predictions = self.model(images)

        # Save first sample
        save_dir = Path(self.config.log_dir) / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)

        visualize_depth_comparison(
            predictions[0],
            depths[0],
            images[0],
            save_path=str(save_dir / f"epoch_{epoch + 1:03d}.png"),
        )

    def _is_best(self, current_metric: float) -> bool:
        """Check if current metric is best.

        Args:
            current_metric: Current metric value

        Returns:
            True if current is best
        """
        if self.config.mode == "min":
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final: bool = False,
    ) -> None:
        """Save training checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
            final: Whether this is the final checkpoint
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config.__dict__,
        }

        if final:
            save_path = Path(self.config.save_dir) / "final_model.pth"
        else:
            save_path = Path(self.config.save_dir) / f"checkpoint_epoch_{epoch + 1}.pth"

        best_path = Path(self.config.save_dir) / "best_model.pth" if is_best else None

        save_checkpoint(state, save_path, is_best=is_best, best_path=best_path)

    def _resume_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.training_history = checkpoint.get("training_history", self.training_history)

        logger.info(f"Resumed from epoch {self.current_epoch}")

    def _create_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ) -> Any:
        """Create learning rate scheduler with warmup.

        Args:
            total_steps: Total training steps
            warmup_steps: Number of warmup steps

        Returns:
            LR scheduler
        """
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)
