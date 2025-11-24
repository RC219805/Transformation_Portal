"""
Tests for model training infrastructure
Validates that training components work correctly
"""

# pylint: disable=possibly-used-before-assignment  # Conditional imports for optional dependencies

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
from packaging import version

# Check if PyTorch is available (required for training infrastructure)
try:
    import torch  # noqa: F401 - imported for availability check only
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
# Try to import training modules - may fail if torch.nn not available
if TORCH_AVAILABLE:
    try:
        from enhancements.train_hyper_reality import (
            TrainingConfig,
            SyntheticDataGenerator,
            EnhancementDataset,
            HyperRealityTrainer,
            configure_device
        )
    except ImportError:
        # torch exists but nn module or other dependencies not available
        TORCH_AVAILABLE = False

# Skip all tests in this module if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - training tests require ML dependencies"
)


class TestTrainingConfig:
    """Test training configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.batch_size == 4
        assert config.num_epochs == 50
        assert config.learning_rate == 1e-4
        assert config.val_split == 0.1
        assert config.checkpoint_dir == "weights/hyper_reality"

    def test_custom_config(self):
        """Test custom configuration"""
        config = TrainingConfig(
            batch_size=8,
            num_epochs=100,
            learning_rate=5e-5,
            checkpoint_dir="custom/path"
        )
        
        assert config.batch_size == 8
        assert config.num_epochs == 100
        assert config.learning_rate == 5e-5
        assert config.checkpoint_dir == "custom/path"


class TestSyntheticDataGenerator:
    """Test synthetic data generation"""

    def test_generator_initialization(self):
        """Test generator can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(tmpdir, num_pairs=10)
            assert generator.output_dir == Path(tmpdir)
            assert generator.num_pairs == 10

    def test_generate_small_dataset(self):
        """Test generating a small dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(tmpdir, num_pairs=5)
            generator.generate_training_data()
            
            # Check directories exist
            low_quality_dir = Path(tmpdir) / "low_quality"
            high_quality_dir = Path(tmpdir) / "high_quality"
            
            assert low_quality_dir.exists()
            assert high_quality_dir.exists()
            
            # Check images were created
            low_images = list(low_quality_dir.glob("*.png"))
            high_images = list(high_quality_dir.glob("*.png"))
            
            assert len(low_images) == 5
            assert len(high_images) == 5


class TestEnhancementDataset:
    """Test dataset loading"""

    @pytest.fixture
    def sample_dataset(self):
        """Create a small sample dataset"""
        tmpdir = tempfile.mkdtemp()
        generator = SyntheticDataGenerator(tmpdir, num_pairs=5)
        generator.generate_training_data()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_dataset_loading(self, sample_dataset):
        """Test dataset can load images"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        low_quality_dir = Path(sample_dataset) / "low_quality"
        high_quality_dir = Path(sample_dataset) / "high_quality"
        
        dataset = EnhancementDataset(low_quality_dir, high_quality_dir, transform)
        
        assert len(dataset) == 5
        
        # Test loading a sample
        low, high = dataset[0]
        assert low.shape == (3, 256, 256)  # RGB, 256x256
        assert high.shape == (3, 256, 256)


class TestDeviceConfiguration:
    """Test device configuration"""

    def test_configure_device(self):
        """Test device configuration works"""
        device = configure_device()
        
        # Should return a valid device (cpu, cuda, or mps)
        assert str(device) in ['cpu', 'cuda', 'mps', 'cuda:0']


class TestTrainerInitialization:
    """Test trainer initialization"""

    def test_trainer_creation(self):
        """Test trainer can be created"""
        config = TrainingConfig(
            num_epochs=1,
            batch_size=2,
            checkpoint_dir="weights/test"
        )
        
        trainer = HyperRealityTrainer(config)
        
        # Check models were created
        assert 'caustics' in trainer.models
        assert 'atmosphere' in trainer.models
        assert 'materials' in trainer.models
        assert 'harmonics' in trainer.models
        
        # Check optimizer was created
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None


class TestTrainingDemo:
    """Test actual training (minimal)"""

    @pytest.mark.slow
    def test_minimal_training(self):
        """Test training loop with minimal data (integration test)"""
        # Skip in CI unless explicitly requested
        import os
        if not os.environ.get('RUN_SLOW_TESTS'):
            pytest.skip("Slow test - set RUN_SLOW_TESTS=1 to run")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate tiny dataset
            generator = SyntheticDataGenerator(tmpdir, num_pairs=4)
            generator.generate_training_data()
            
            # Configure minimal training
            config = TrainingConfig(
                data_dir=tmpdir,
                batch_size=2,
                num_epochs=1,
                checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
                save_frequency=1,
                val_split=0.25,  # 1 validation sample
                num_workers=0,  # No multiprocessing for test
                use_mixed_precision=False
            )
            
            # Create dataset and dataloaders
            from torchvision import transforms
            from torch.utils.data import DataLoader
            import torch
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            low_quality_dir = Path(tmpdir) / "low_quality"
            high_quality_dir = Path(tmpdir) / "high_quality"
            
            dataset = EnhancementDataset(low_quality_dir, high_quality_dir, transform)
            
            # Split dataset
            val_size = 1
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Train for 1 epoch
            trainer = HyperRealityTrainer(config)
            initial_loss = trainer.best_val_loss
            
            trainer.train(train_loader, val_loader)
            
            # Check that training happened
            assert len(trainer.training_history) > 0
            assert trainer.best_val_loss < initial_loss or initial_loss == float('inf')
            
            # Check checkpoint was saved
            checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_epoch_1.pth"
            assert checkpoint_path.exists()


def test_imports():
    """Test that all required modules can be imported"""
    import torch
    import torchvision
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    
    # Verify versions using proper semantic versioning
    assert version.parse(torch.__version__) >= version.parse("2.0.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
