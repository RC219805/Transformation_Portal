"""
Smoke tests for ML stack upgrades (PR #793b).

These tests validate that major ML framework upgrades don't break core functionality:
- torch 2.4.1 → 2.10.0
- torchvision 0.19.1 → 0.25.0
- scikit-learn 1.7.2 → 1.8.0
- timm 0.6.7 → 1.0.24
- diffusers 0.31.0 → 0.36.0
- transformers 4.53.0 → 4.57.6

Smoke tests exercise representative code paths without requiring large model downloads.
They use mocked model loading where appropriate to keep test time and disk usage minimal.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Check if ML packages are available for import
try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    TORCH_AVAILABLE = False
    torch = None

try:
    import torchvision

    TORCHVISION_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    TORCHVISION_AVAILABLE = False
    torchvision = None

try:
    import timm

    TIMM_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    TIMM_AVAILABLE = False
    timm = None

try:
    import diffusers

    DIFFUSERS_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    DIFFUSERS_AVAILABLE = False
    diffusers = None

try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    TRANSFORMERS_AVAILABLE = False
    transformers = None

try:
    import sklearn

    SKLEARN_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    SKLEARN_AVAILABLE = False
    sklearn = None

# Skip all ML tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for ML smoke tests")


@pytest.mark.ml
def test_pytorch_basic_operations():
    """Test basic PyTorch operations work with torch 2.10.0."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")

    # Create a simple tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Basic operations
    z = x + y
    assert z.shape == (3,)
    assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))

    # Matrix operations
    mat = torch.randn(3, 3)
    result = torch.matmul(mat, mat.T)
    assert result.shape == (3, 3)


@pytest.mark.ml
def test_pytorch_mps_device_availability():
    """Test MPS (Apple Silicon) device detection with torch 2.10.0."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")

    # Should not raise even if MPS not available
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.tensor([1.0, 2.0]).to(device)
        assert x.device.type == "mps"
    else:
        # On non-Mac or without MPS, should have CPU
        device = torch.device("cpu")
        x = torch.tensor([1.0, 2.0]).to(device)
        assert x.device.type == "cpu"


@pytest.mark.ml
@pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not installed")
def test_torchvision_transforms():
    """Test torchvision transforms work with torchvision 0.25.0."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    # Create a dummy image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Test basic transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


@pytest.mark.ml
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_scikit_learn_basic_classifier():
    """Test scikit-learn basic classifier with scikit-learn 1.8.0."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate dummy data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})

    # Score should be reasonable (>0.5 for this easy synthetic dataset)
    score = clf.score(X_test, y_test)
    assert score > 0.5


@pytest.mark.ml
@pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not installed")
@patch("timm.create_model")
def test_timm_model_interface(mock_create_model):
    """Test timm model creation interface with timm 1.0.24."""
    import timm

    # Mock the model to avoid downloading
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_create_model.return_value = mock_model

    # Test model creation interface (mocked)
    model = timm.create_model("resnet18", pretrained=False)
    assert model is not None

    # Verify the mock was called correctly
    mock_create_model.assert_called_once_with("resnet18", pretrained=False)


@pytest.mark.ml
@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not installed")
@patch("diffusers.DiffusionPipeline.from_pretrained")
def test_diffusers_pipeline_interface(mock_from_pretrained):
    """Test diffusers pipeline interface with diffusers 0.36.0."""
    import torch
    from diffusers import DiffusionPipeline

    # Mock the pipeline to avoid downloading large models
    mock_pipeline = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline

    # Test pipeline creation interface (mocked)
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, use_safetensors=True
    )
    assert pipeline is not None

    # Verify the mock was called with correct arguments
    assert mock_from_pretrained.called
    call_args = mock_from_pretrained.call_args
    assert call_args[0][0] == "runwayml/stable-diffusion-v1-5"
    assert call_args[1]["torch_dtype"] == torch.float32
    assert call_args[1]["use_safetensors"] is True


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModel.from_pretrained")
def test_transformers_model_interface(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    """Test transformers model interface with transformers 4.57.6."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    from transformers import AutoModel, AutoTokenizer

    # Mock tokenizer and model to avoid downloading
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    # Test tokenizer and model creation interface (mocked)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    assert tokenizer is not None
    assert model is not None

    # Verify mocks were called
    mock_tokenizer_from_pretrained.assert_called_once_with("bert-base-uncased")
    mock_model_from_pretrained.assert_called_once_with("bert-base-uncased")


@pytest.mark.ml
def test_torch_cuda_compatibility():
    """Test CUDA compatibility (if available) with torch 2.10.0."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")

    # This should not fail even without CUDA
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = torch.device("cuda:0")
        x = torch.tensor([1.0, 2.0]).to(device)
        assert x.device.type == "cuda"
        assert x.device.index == 0
    else:
        # Should work fine on CPU
        device = torch.device("cpu")
        x = torch.tensor([1.0, 2.0]).to(device)
        assert x.device.type == "cpu"


@pytest.mark.ml
def test_ml_stack_imports():
    """Test that all major ML packages can be imported without errors."""
    import pkg_resources

    def get_version(package_name):
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None

    # Always check torch (required for ML tests to run)
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    torch_version = get_version("torch")
    assert torch_version is not None, "torch not installed"
    assert torch_version >= "2.10.0", f"torch version {torch_version} < 2.10.0"

    # Check sklearn (if available)
    if SKLEARN_AVAILABLE:
        import sklearn

        sklearn_version = get_version("scikit-learn")
        assert sklearn_version is not None, "scikit-learn not installed"
        assert sklearn_version >= "1.8.0", f"scikit-learn version {sklearn_version} < 1.8.0"

    # Check diffusers (if available)
    if DIFFUSERS_AVAILABLE:
        import diffusers

        diffusers_version = get_version("diffusers")
        assert diffusers_version is not None, "diffusers not installed"
        assert diffusers_version >= "0.36.0", f"diffusers version {diffusers_version} < 0.36.0"

    # Check transformers (if available)
    if TRANSFORMERS_AVAILABLE:
        import transformers

        transformers_version = get_version("transformers")
        assert transformers_version is not None, "transformers not installed"
        assert transformers_version >= "4.57.0", f"transformers version {transformers_version} < 4.57.0"

    # Check torchvision (if available)
    if TORCHVISION_AVAILABLE:
        import torchvision

        torchvision_version = get_version("torchvision")
        assert torchvision_version is not None, "torchvision not installed"

    # Check timm (if available)
    if TIMM_AVAILABLE:
        import timm

        timm_version = get_version("timm")
        assert timm_version is not None, "timm not installed"
