#!/usr/bin/env python3
"""
Training Utilities

Provides utility functions for training:
- Configuration loading and validation
- Device setup (CPU, GPU, MPS, DDP)
- Seed setting for reproducibility
- Logging configuration
- Progress tracking

Author: Transformation Portal Team
Version: 1.0.0
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import yaml
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try to import torch
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML required for config loading. "
            "Install with: pip install pyyaml"
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    logger.info(f"Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate training configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []

    # Required sections
    required_sections = ["model", "training", "data"]
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing required section: {section}")

    # Model validation
    if "model" in config:
        model_cfg = config["model"]
        if "name" not in model_cfg:
            warnings.append("Missing model.name")

    # Training validation
    if "training" in config:
        train_cfg = config["training"]
        if train_cfg.get("batch_size", 0) <= 0:
            warnings.append("Invalid batch_size (must be > 0)")
        if train_cfg.get("learning_rate", 0) <= 0:
            warnings.append("Invalid learning_rate (must be > 0)")

    # Data validation
    if "data" in config:
        data_cfg = config["data"]
        if "train_dir" not in data_cfg:
            warnings.append("Missing data.train_dir")

    for warning in warnings:
        logger.warning(f"Config validation: {warning}")

    return warnings


def setup_device(
    use_cuda: bool = True,
    use_mps: bool = True,
    distributed: bool = False,
    local_rank: int = 0,
) -> Tuple["torch.device", Optional[int]]:
    """Set up compute device.

    Args:
        use_cuda: Whether to use CUDA if available
        use_mps: Whether to use MPS (Apple Silicon) if available
        distributed: Whether to use distributed training
        local_rank: Local rank for distributed training

    Returns:
        Tuple of (device, world_size)
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch required for device setup. "
            "Install with: pip install torch"
        )

    world_size = None

    # Check for distributed training
    if distributed:
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(
            f"Distributed training: rank {local_rank}/{world_size}"
        )
        return device, world_size

    # Single-device training
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device, world_size


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Sets seed for Python random, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def create_logger(
    name: str = "training",
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Create configured logger.

    Args:
        name: Logger name
        log_dir: Optional directory for log files
        level: Logging level
        console: Whether to log to console

    Returns:
        Configured logger
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    # Remove existing handlers
    log.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log


def get_num_params(model: "torch.nn.Module", trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_size(num_bytes: int) -> str:
    """Format byte size to human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def get_lr(optimizer: "torch.optim.Optimizer") -> float:
    """Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


class AverageMeter:
    """Computes and stores the average and current value.

    Example:
        >>> meter = AverageMeter()
        >>> for loss in losses:
        ...     meter.update(loss)
        >>> print(f"Average loss: {meter.avg:.4f}")
    """

    def __init__(self):
        """Initialize meter."""
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics.

        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressTracker:
    """Track training progress with ETA estimation.

    Example:
        >>> tracker = ProgressTracker(total_steps=1000)
        >>> for step in range(1000):
        ...     tracker.update()
        ...     print(tracker.format_progress())
    """

    def __init__(
        self,
        total_steps: int,
        start_step: int = 0,
    ):
        """Initialize progress tracker.

        Args:
            total_steps: Total number of steps
            start_step: Starting step (for resume)
        """
        import time
        self.total_steps = total_steps
        self.current_step = start_step
        self.start_time = time.time()
        self.step_times: List[float] = []
        self._last_time = self.start_time

    def update(self, steps: int = 1) -> None:
        """Update progress.

        Args:
            steps: Number of steps completed
        """
        import time
        self.current_step += steps

        current_time = time.time()
        step_time = current_time - self._last_time
        self.step_times.append(step_time)

        # Keep only last 100 step times for ETA
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]

        self._last_time = current_time

    def get_eta(self) -> float:
        """Get estimated time remaining in seconds.

        Returns:
            Estimated time remaining
        """
        if len(self.step_times) == 0:
            return 0.0

        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        return avg_step_time * remaining_steps

    def format_progress(self) -> str:
        """Format progress as string.

        Returns:
            Formatted progress string
        """
        progress = self.current_step / self.total_steps * 100
        eta_seconds = self.get_eta()

        # Format ETA
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        if hours > 0:
            eta_str = f"{hours}h {minutes}m"
        elif minutes > 0:
            eta_str = f"{minutes}m {seconds}s"
        else:
            eta_str = f"{seconds}s"

        return f"{self.current_step}/{self.total_steps} ({progress:.1f}%) ETA: {eta_str}"


def save_checkpoint(
    state: Dict[str, Any],
    save_path: Union[str, Path],
    is_best: bool = False,
    best_path: Optional[Union[str, Path]] = None,
) -> None:
    """Save training checkpoint.

    Args:
        state: State dictionary to save
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
        best_path: Path to save best model copy
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for checkpoint saving")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

    if is_best:
        best_path = best_path or save_path.parent / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional["torch.nn.Module"] = None,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load tensors to

    Returns:
        Loaded checkpoint dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for checkpoint loading")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    map_location = device if device else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Restore model
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restored model state")

    # Restore optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state")

    # Restore scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored scheduler state")

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def cleanup_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    keep_best: bool = True,
) -> None:
    """Clean up old checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    # Sort by epoch number
    def get_epoch(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            logger.warning(f"Could not parse epoch from checkpoint: {path.name}")
            return 0

    checkpoints.sort(key=get_epoch, reverse=True)

    # Keep last N checkpoints
    to_delete = checkpoints[keep_last_n:]

    for checkpoint in to_delete:
        checkpoint.unlink()
        logger.debug(f"Deleted old checkpoint: {checkpoint}")

    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old checkpoints")
