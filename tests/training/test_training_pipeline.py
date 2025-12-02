#!/usr/bin/env python3
"""
Tests for Depth Training Pipeline

Comprehensive tests covering:
- Augmentations
- Dataset loading
- Loss functions
- Metric calculation
- Trainer functionality

Author: Transformation Portal Team
Version: 1.0.0
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - training tests require ML dependencies"
)


class TestTrainingConfig:
    """Test training configuration."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig()

        assert config.num_epochs == 50
        assert config.batch_size == 8
        assert config.learning_rate == 1e-5
        assert config.mixed_precision == "fp16"
        assert config.save_best is True

    def test_training_config_custom(self):
        """Test TrainingConfig accepts custom values."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(
            num_epochs=100,
            batch_size=16,
            learning_rate=5e-6,
        )

        assert config.num_epochs == 100
        assert config.batch_size == 16
        assert config.learning_rate == 5e-6


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_init(self):
        """Test EarlyStopping initialization."""
        from src.training.trainer import EarlyStopping

        es = EarlyStopping(patience=5, min_delta=0.001, mode="min")

        assert es.patience == 5
        assert es.min_delta == 0.001
        assert es.mode == "min"
        assert es.counter == 0
        assert not es.early_stop

    def test_early_stopping_improves(self):
        """Test that improving metrics reset counter."""
        from src.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3, mode="min")

        # Improving values should not trigger early stop
        assert not es(0.5)
        assert not es(0.4)
        assert not es(0.3)
        assert es.counter == 0

    def test_early_stopping_triggers(self):
        """Test that non-improving metrics trigger early stop."""
        from src.training.trainer import EarlyStopping

        es = EarlyStopping(patience=3, mode="min")

        es(0.3)  # Best value
        assert not es(0.4)  # Worse
        assert es.counter == 1
        assert not es(0.4)  # Worse
        assert es.counter == 2
        assert es(0.4)  # Worse - triggers!
        assert es.early_stop

    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode."""
        from src.training.trainer import EarlyStopping

        es = EarlyStopping(patience=2, mode="max")

        assert not es(0.5)
        assert not es(0.6)  # Improvement
        assert es.counter == 0
        assert not es(0.55)  # Worse
        assert es.counter == 1
        assert es(0.54)  # Worse - triggers!
        assert es.early_stop


class TestDepthTrainer:
    """Test trainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock depth model."""
        model = Mock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        model.to.return_value = model
        model.train.return_value = None
        model.eval.return_value = None
        model.state_dict.return_value = {}
        return model

    def test_trainer_init(self, mock_model):
        """Test trainer initialization."""
        from src.training.trainer import DepthTrainer, TrainingConfig

        config = TrainingConfig()
        trainer = DepthTrainer(mock_model, config)

        assert trainer.model is mock_model
        assert trainer.config is config
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_trainer_is_best_min(self, mock_model):
        """Test _is_best with min mode."""
        from src.training.trainer import DepthTrainer, TrainingConfig

        config = TrainingConfig(mode="min")
        trainer = DepthTrainer(mock_model, config)
        trainer.best_metric = 0.5

        assert trainer._is_best(0.4)  # Better
        assert not trainer._is_best(0.6)  # Worse

    def test_trainer_is_best_max(self, mock_model):
        """Test _is_best with max mode."""
        from src.training.trainer import DepthTrainer, TrainingConfig

        config = TrainingConfig(mode="max")
        trainer = DepthTrainer(mock_model, config)
        trainer.best_metric = 0.5

        assert trainer._is_best(0.6)  # Better
        assert not trainer._is_best(0.4)  # Worse


class TestDataLoading:
    """Test data loading utilities."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data."""
        tmpdir = tempfile.mkdtemp()
        train_dir = Path(tmpdir) / "train"
        images_dir = train_dir / "images"
        depth_dir = train_dir / "depth"

        images_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        # Create sample images and depth maps
        for i in range(5):
            # Image
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            from PIL import Image as PILImage
            PILImage.fromarray(image).save(images_dir / f"image_{i:03d}.png")

            # Depth
            depth = np.random.rand(64, 64).astype(np.float32) * 100
            np.save(depth_dir / f"image_{i:03d}.npy", depth)

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_dataset_loading(self, sample_data_dir):
        """Test ArchitecturalDepthDataset loads correctly."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "train"),  # Use same for test
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")

        assert len(dataset) == 5

        # Test loading a sample
        image, depth = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(depth, torch.Tensor)
        assert image.shape[0] == 3  # Channels
        assert depth.shape[0] == 1  # Single channel

    def test_dataset_stats(self, sample_data_dir):
        """Test dataset statistics computation."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "train"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")
        stats = dataset.get_stats()

        assert "num_samples" in stats
        assert stats["num_samples"] == 5
        assert "depth_min" in stats
        assert "depth_max" in stats


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        from src.training.utils import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "epoch": 5,
                "model_state_dict": {"weight": torch.randn(10)},
                "optimizer_state_dict": {},
            }

            save_path = Path(tmpdir) / "checkpoint.pth"
            save_checkpoint(state, save_path)

            assert save_path.exists()

    def test_save_best_checkpoint(self):
        """Test best checkpoint is saved separately."""
        from src.training.utils import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"epoch": 5}
            save_path = Path(tmpdir) / "checkpoint.pth"
            best_path = Path(tmpdir) / "best.pth"

            save_checkpoint(state, save_path, is_best=True, best_path=best_path)

            assert save_path.exists()
            assert best_path.exists()

    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        from src.training.utils import save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            state = {
                "epoch": 10,
                "best_metric": 0.123,
            }
            save_path = Path(tmpdir) / "checkpoint.pth"
            save_checkpoint(state, save_path)

            # Load
            loaded = load_checkpoint(save_path)

            assert loaded["epoch"] == 10
            assert loaded["best_metric"] == 0.123

    def test_cleanup_checkpoints(self):
        """Test old checkpoint cleanup."""
        from src.training.utils import cleanup_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create several checkpoints
            for i in range(10):
                (Path(tmpdir) / f"checkpoint_epoch_{i}.pth").touch()

            cleanup_checkpoints(tmpdir, keep_last_n=3)

            remaining = list(Path(tmpdir).glob("checkpoint_epoch_*.pth"))
            assert len(remaining) == 3


class TestUtilities:
    """Test training utilities."""

    def test_average_meter(self):
        """Test AverageMeter."""
        from src.training.utils import AverageMeter

        meter = AverageMeter()

        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)

        assert meter.avg == 2.0
        assert meter.sum == 6.0
        assert meter.count == 3

    def test_average_meter_weighted(self):
        """Test AverageMeter with weights."""
        from src.training.utils import AverageMeter

        meter = AverageMeter()

        meter.update(1.0, n=2)  # 2 samples at 1.0
        meter.update(2.0, n=2)  # 2 samples at 2.0

        assert meter.avg == 1.5
        assert meter.count == 4

    def test_set_seed(self):
        """Test seed setting produces reproducible results."""
        from src.training.utils import set_seed

        set_seed(42)
        a = torch.rand(10)

        set_seed(42)
        b = torch.rand(10)

        assert torch.allclose(a, b)

    def test_get_lr(self):
        """Test getting learning rate from optimizer."""
        from src.training.utils import get_lr

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        lr = get_lr(optimizer)

        assert lr == 0.001

    def test_get_num_params(self):
        """Test parameter counting."""
        from src.training.utils import get_num_params

        model = torch.nn.Linear(10, 20, bias=True)
        # 10*20 weights + 20 biases = 220

        total = get_num_params(model, trainable_only=False)
        assert total == 220

    def test_format_size(self):
        """Test size formatting."""
        from src.training.utils import format_size

        assert format_size(1024) == "1.0 KB"
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_config(self):
        """Test YAML config loading."""
        from src.training.utils import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("model:\n  name: test\ntraining:\n  epochs: 10\n")
            f.flush()

            config = load_config(f.name)

            assert config["model"]["name"] == "test"
            assert config["training"]["epochs"] == 10

        Path(f.name).unlink()

    def test_load_config_not_found(self):
        """Test config loading with missing file."""
        from src.training.utils import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_validate_config(self):
        """Test configuration validation."""
        from src.training.utils import validate_config

        # Valid config
        config = {
            "model": {"name": "test"},
            "training": {"batch_size": 8, "learning_rate": 1e-5},
            "data": {"train_dir": "data/train"},
        }

        warnings = validate_config(config)
        assert len(warnings) == 0

        # Invalid config
        config = {"model": {}}  # Missing required fields

        warnings = validate_config(config)
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
