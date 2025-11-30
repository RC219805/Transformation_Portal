#!/usr/bin/env python3
"""
Tests for Hyper-Reality Enhancement Training Infrastructure

Tests cover:
- Synthetic data generation
- Dataset loading
- Loss functions (including LPIPS)
- Training loop
- Model checkpoint saving/loading
"""

# pylint: disable=possibly-used-before-assignment  # Conditional imports for optional dependencies

import sys
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Check if PyTorch is available (required for training infrastructure)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# pylint: disable=wrong-import-position
# Try to import training modules - may fail if torch.nn not available
LPIPS_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from enhancements.train_hyper_reality import (
            SyntheticDataGenerator,
            EnhancementDataset,
            PerceptualLoss,
            StyleLoss,
            VGGFeatureExtractor,
            HyperRealityTrainer,
            TrainingConfig,
            LPIPS_AVAILABLE
        )
        from enhancements.model_loader import ModelLoader, load_pretrained_weights
        from enhancements.hyper_reality_enhancement import (
            CausticGenerator,
            AtmosphericSynthesizer,
            MaterialTranscendence,
            SpatialHarmonics,
        )
    except ImportError:
        # torch exists but nn module or other dependencies not available
        TORCH_AVAILABLE = False

# Skip all tests in this module if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - training infrastructure tests require ML dependencies"
)


class TestSyntheticDataGeneration:
    """Test synthetic training data generation"""

    def test_create_synthetic_image(self):
        """Test synthetic image creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(tmpdir, num_pairs=5)

            # Generate single image
            img = generator._create_synthetic_image(size=(256, 256))

            assert img.shape == (256, 256, 3)
            assert img.dtype == np.uint8
            assert img.min() >= 0
            assert img.max() <= 255

    def test_degrade_image(self):
        """Test image degradation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(tmpdir, num_pairs=5)

            # Create high quality image
            high_quality = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

            # Degrade it
            low_quality = generator._degrade_image(high_quality)

            assert low_quality.shape == high_quality.shape
            assert low_quality.dtype == np.uint8

            # Degraded should have lower contrast (roughly)
            assert low_quality.std() < high_quality.std() * 1.2

    def test_generate_training_data(self):
        """Test full training data generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticDataGenerator(tmpdir, num_pairs=5)
            generator.generate_training_data()

            # Check directories created
            low_dir = Path(tmpdir) / "low_quality"
            high_dir = Path(tmpdir) / "high_quality"

            assert low_dir.exists()
            assert high_dir.exists()

            # Check correct number of images
            low_images = list(low_dir.glob("*.png"))
            high_images = list(high_dir.glob("*.png"))

            assert len(low_images) == 5
            assert len(high_images) == 5

            # Check names match
            low_names = {p.name for p in low_images}
            high_names = {p.name for p in high_images}
            assert low_names == high_names


class TestEnhancementDataset:
    """Test dataset loading"""

    @pytest.fixture
    def sample_dataset(self):
        """Create temporary dataset"""
        tmpdir = tempfile.mkdtemp()

        # Generate sample data
        generator = SyntheticDataGenerator(tmpdir, num_pairs=10)
        generator.generate_training_data()

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_dataset_loading(self, sample_dataset):
        """Test dataset can load image pairs"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        low_dir = Path(sample_dataset) / "low_quality"
        high_dir = Path(sample_dataset) / "high_quality"

        dataset = EnhancementDataset(low_dir, high_dir, transform)

        assert len(dataset) == 10

        # Test loading single item
        low_img, high_img = dataset[0]

        assert low_img.shape == (3, 128, 128)
        assert high_img.shape == (3, 128, 128)
        assert isinstance(low_img, torch.Tensor)
        assert isinstance(high_img, torch.Tensor)

    def test_dataset_iteration(self, sample_dataset):
        """Test dataset iteration"""
        from torchvision import transforms
        from torch.utils.data import DataLoader

        transform = transforms.ToTensor()
        low_dir = Path(sample_dataset) / "low_quality"
        high_dir = Path(sample_dataset) / "high_quality"

        dataset = EnhancementDataset(low_dir, high_dir, transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        batch = next(iter(loader))
        low_batch, high_batch = batch

        assert low_batch.shape[0] == 2  # batch size
        assert high_batch.shape[0] == 2


class TestLossFunctions:
    """Test loss functions with pretrained VGG features"""

    @pytest.fixture(autouse=True)
    def check_torchvision(self):
        """Check if torchvision is available for VGG-based tests"""
        try:
            import torchvision  # noqa: F401
            self.has_torchvision = True
        except ImportError:
            self.has_torchvision = False

    def test_vgg_feature_extractor(self):
        """Test VGG feature extraction"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        # Test with specific layers
        extractor = VGGFeatureExtractor(layers=[2, 7, 12])

        # Create dummy input
        x = torch.randn(1, 3, 64, 64)

        # Extract features
        features = extractor(x)

        assert len(features) == 3
        for feat in features:
            assert isinstance(feat, torch.Tensor)
            assert feat.ndim == 4  # batch, channels, height, width

    def test_perceptual_loss(self):
        """Test perceptual loss computation with VGG features"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        loss_fn = PerceptualLoss()

        # Create dummy tensors
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_style_loss(self):
        """Test style loss computation with VGG features"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        loss_fn = StyleLoss()

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_loss_backprop(self):
        """Test loss can backpropagate through VGG features"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        loss_fn = PerceptualLoss()

        # Create tensor with gradient tracking
        pred = torch.randn(1, 3, 32, 32, requires_grad=True)
        target = torch.randn(1, 3, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_perceptual_loss_identical_images(self):
        """Test that identical images have zero perceptual loss"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        loss_fn = PerceptualLoss()

        # Create identical tensors
        img = torch.randn(1, 3, 64, 64)
        loss = loss_fn(img, img.clone())

        # Loss should be very close to zero for identical images
        assert loss.item() < 1e-6

    def test_style_loss_identical_images(self):
        """Test that identical images have zero style loss"""
        if not self.has_torchvision:
            pytest.skip("torchvision not available")

        loss_fn = StyleLoss()

        # Create identical tensors
        img = torch.randn(1, 3, 64, 64)
        loss = loss_fn(img, img.clone())

        # Loss should be very close to zero for identical images
        assert loss.item() < 1e-6


class TestModelCheckpoints:
    """Test model checkpoint saving and loading"""

    def test_checkpoint_saving(self):
        """Test checkpoint can be saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint_dir=tmpdir,
                num_epochs=2,
                batch_size=2
            )

            trainer = HyperRealityTrainer(config)

            # Save checkpoint
            trainer._save_checkpoint(epoch=0, is_best=True)

            # Check files created
            checkpoint_path = Path(tmpdir) / "checkpoint_epoch_1.pth"
            best_path = Path(tmpdir) / "best_model.pth"

            assert checkpoint_path.exists()
            assert best_path.exists()

    def test_checkpoint_loading(self):
        """Test checkpoint can be loaded"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a checkpoint
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)
            trainer._save_checkpoint(epoch=0, is_best=True)

            # Load it back
            loader = ModelLoader(tmpdir)
            checkpoint = loader.load_best_model()

            assert checkpoint is not None
            assert 'models' in checkpoint
            assert 'epoch' in checkpoint
            assert checkpoint['epoch'] == 0

    def test_model_weights_loading(self):
        """Test loading weights into models"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save models
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)
            trainer._save_checkpoint(epoch=0, is_best=True)

            # Create fresh models
            from enhancements.hyper_reality_enhancement import EnhancementConfig
            enhancement_config = EnhancementConfig()

            models = {
                'caustics': CausticGenerator(enhancement_config.quantum_caustics),
                'atmosphere': AtmosphericSynthesizer(enhancement_config.neural_atmosphere),
                'materials': MaterialTranscendence(enhancement_config.material_transcendence),
                'harmonics': SpatialHarmonics(enhancement_config.spatial_harmonics),
            }

            # Load weights
            loader = ModelLoader(tmpdir)
            success = loader.load_model_weights(models)

            assert success is True


class TestTrainingIntegration:
    """Integration tests for training pipeline"""

    @pytest.mark.slow
    def test_minimal_training_run(self):
        """Test training can run for 1 epoch (integration test)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate minimal dataset
            generator = SyntheticDataGenerator(tmpdir, num_pairs=8)
            generator.generate_training_data()

            # Create dataset
            from torchvision import transforms
            from torch.utils.data import DataLoader

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

            low_dir = Path(tmpdir) / "low_quality"
            high_dir = Path(tmpdir) / "high_quality"
            dataset = EnhancementDataset(low_dir, high_dir, transform)

            # Split train/val
            train_size = 6
            val_size = 2
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            train_loader = DataLoader(train_dataset, batch_size=2)
            val_loader = DataLoader(val_dataset, batch_size=2)

            # Train for 1 epoch
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            config = TrainingConfig(
                checkpoint_dir=str(checkpoint_dir),
                num_epochs=1,
                batch_size=2,
                save_frequency=1
            )

            trainer = HyperRealityTrainer(config)

            # Run training (should not crash)
            trainer.train(train_loader, val_loader)

            # Check checkpoint saved
            assert (checkpoint_dir / "checkpoint_epoch_1.pth").exists()
            assert trainer.current_epoch == 0  # 0-indexed


class TestDepthNormalsIntegration:
    """Test depth/normals integration in training pipeline"""

    def test_estimate_depth(self):
        """Test depth estimation helper method"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Create test input
            img = torch.rand(2, 3, 64, 64)

            # Estimate depth
            depth = trainer._estimate_depth(img)

            assert depth.shape == (2, 1, 64, 64)
            assert depth.min() >= 0
            assert depth.max() <= 1

    def test_compute_normals(self):
        """Test normal computation from depth"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Create test depth
            depth = torch.rand(2, 1, 64, 64)

            # Compute normals
            normals = trainer._compute_normals(depth)

            assert normals.shape == (2, 3, 64, 64)
            # Normals should be unit vectors (approximately)
            norms = torch.norm(normals, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=0.01)

    def test_training_uses_harmonics_model(self):
        """Test that training loop uses SpatialHarmonics model with normals"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Verify harmonics model exists and is included in training
            assert 'harmonics' in trainer.models
            assert isinstance(trainer.models['harmonics'], SpatialHarmonics)

            # Verify harmonics parameters are in optimizer
            harmonics_params = set(id(p) for p in trainer.models['harmonics'].parameters())
            optimizer_params = set(id(p) for pg in trainer.optimizer.param_groups for p in pg['params'])

            assert harmonics_params.issubset(optimizer_params), (
                "SpatialHarmonics parameters should be included in optimizer"
            )

    def test_caustics_receives_depth(self):
        """Test that caustics model receives depth during forward pass"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Create test input
            img = torch.rand(1, 3, 64, 64)
            depth = trainer._estimate_depth(img)

            # Call caustics with depth (should not raise)
            caustics = trainer.models['caustics'](img, depth)

            assert caustics.shape == img.shape


class TestModelLoader:
    """Test model loader functionality"""

    def test_get_available_checkpoints_empty(self):
        """Test getting checkpoints from empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ModelLoader(tmpdir)
            checkpoints = loader.get_available_checkpoints()

            assert not checkpoints

    def test_get_available_checkpoints(self):
        """Test getting available checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some dummy checkpoints
            (Path(tmpdir) / "best_model.pth").touch()
            (Path(tmpdir) / "checkpoint_epoch_5.pth").touch()
            (Path(tmpdir) / "checkpoint_epoch_10.pth").touch()

            loader = ModelLoader(tmpdir)
            checkpoints = loader.get_available_checkpoints()

            assert "best_model" in checkpoints
            assert "checkpoint_epoch_5" in checkpoints
            assert "checkpoint_epoch_10" in checkpoints

    def test_checkpoint_info(self):
        """Test extracting checkpoint info"""
        checkpoint = {
            'epoch': 25,
            'best_val_loss': 0.0123,
            'config': {'learning_rate': 1e-4},
            'models': {'caustics': {}, 'atmosphere': {}},
        }

        loader = ModelLoader()
        info = loader.checkpoint_info(checkpoint)

        assert info['epoch'] == 25
        assert info['best_val_loss'] == 0.0123
        assert 'caustics' in info['models']
        assert 'atmosphere' in info['models']


def test_load_pretrained_weights_fallback():
    """Test fallback when no weights available"""
    with tempfile.TemporaryDirectory() as tmpdir:
        from enhancements.hyper_reality_enhancement import EnhancementConfig
        enhancement_config = EnhancementConfig()

        models = {
            'caustics': CausticGenerator(enhancement_config.quantum_caustics),
        }

        # Should return False when no weights found
        success = load_pretrained_weights(models, tmpdir, verbose=False)
        assert success is False


class TestLPIPSIntegration:
    """Test LPIPS loss integration"""

    def test_trainer_has_lpips_config(self):
        """Test that TrainingConfig has lpips_weight parameter"""
        config = TrainingConfig()
        assert hasattr(config, 'lpips_weight')
        assert config.lpips_weight == 1.0

    def test_trainer_history_includes_lpips(self):
        """Test that training history tracks LPIPS loss"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Training history should have lpips key
            assert 'lpips' in trainer.training_history

    def test_trainer_lpips_fn_attribute(self):
        """Test that trainer has lpips_fn attribute"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # lpips_fn should exist (may be None if lpips not installed)
            assert hasattr(trainer, 'lpips_fn')

    @pytest.mark.skipif(not LPIPS_AVAILABLE, reason="LPIPS not installed")
    def test_lpips_loss_computation(self):
        """Test LPIPS loss computation when available"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = HyperRealityTrainer(config)

            # Create test tensors
            img1 = torch.rand(1, 3, 64, 64)
            img2 = torch.rand(1, 3, 64, 64)

            # LPIPS expects [-1, 1] range
            img1_scaled = img1 * 2 - 1
            img2_scaled = img2 * 2 - 1

            # Compute LPIPS loss
            lpips_loss = trainer.lpips_fn(img1_scaled, img2_scaled).mean()

            assert isinstance(lpips_loss, torch.Tensor)
            assert lpips_loss.ndim == 0  # scalar
            assert lpips_loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
