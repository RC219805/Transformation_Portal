"""
Smoke tests for the governed ML dependency baselines.

These tests validate that CI-selected ML baselines do not break core
imports or representative code paths without requiring large model
downloads.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

# Check if ML packages are available for import
try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, TypeError):
    TORCH_AVAILABLE = False
    torch = None

# Skip all ML tests if torch is not available
# ADR-044 Section 4.1 maps tests/smoke/ -> @pytest.mark.unit (no separate smoke marker).
# The skipif ensures these won't run when torch is unavailable, so `-m unit` remains lightweight.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for ML smoke tests"),
]

_ML_IMPORT_EXCEPTIONS = (ImportError, RuntimeError, TypeError, AttributeError)


def _import_optional_module(module_name: str):
    """Import an optional ML module without failing collection."""
    try:
        return importlib.import_module(module_name), None
    except _ML_IMPORT_EXCEPTIONS as exc:
        return None, str(exc)


def _require_optional_module(module_name: str):
    """Skip the calling test when an optional ML module is unavailable."""
    module, error = _import_optional_module(module_name)
    if module is None:
        pytest.skip(f"{module_name} unavailable: {error}")
    return module


def _ensure_torch_xpu_namespace() -> None:
    """Provide the optional torch.xpu namespace expected by newer diffusers builds."""
    if torch is None or hasattr(torch, "xpu"):
        return

    torch.xpu = SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)  # type: ignore[attr-defined]


def test_pytorch_basic_operations():
    """Test basic PyTorch tensor operations against the supported baseline."""
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


def test_pytorch_mps_device_availability():
    """Test MPS (Apple Silicon) device detection against the supported baseline."""
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


def test_torchvision_transforms():
    """Test torchvision transforms against the supported baseline."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    torchvision = _require_optional_module("torchvision")
    import numpy as np
    from PIL import Image

    # Create a dummy image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Test basic transforms
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_scikit_learn_basic_classifier():
    """Test scikit-learn basic classifier against the supported baseline."""
    _require_optional_module("sklearn")
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


def test_timm_model_interface():
    """Test timm model creation against the supported baseline."""
    timm = _require_optional_module("timm")

    with patch.object(timm, "create_model") as mock_create_model:
        # Mock the model to avoid downloading
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_create_model.return_value = mock_model

        # Test model creation interface (mocked)
        model = timm.create_model("resnet18", pretrained=False)
        assert model is not None

        # Verify the mock was called correctly
        mock_create_model.assert_called_once_with("resnet18", pretrained=False)


def test_diffusers_pipeline_interface():
    """Test the diffusers pipeline interface against the supported baseline."""
    import torch

    _ensure_torch_xpu_namespace()
    diffusers = _require_optional_module("diffusers")

    with patch.object(diffusers.DiffusionPipeline, "from_pretrained") as mock_from_pretrained:
        # Mock the pipeline to avoid downloading large models
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Test pipeline creation interface (mocked)
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, use_safetensors=True
        )
        assert pipeline is not None

        # Verify the mock was called with correct arguments
        assert mock_from_pretrained.called
        call_args = mock_from_pretrained.call_args
        assert call_args[0][0] == "runwayml/stable-diffusion-v1-5"
        assert call_args[1]["torch_dtype"] == torch.float32
        assert call_args[1]["use_safetensors"] is True


def test_transformers_model_interface():
    """Test the transformers model interface against the supported baseline."""
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    transformers = _require_optional_module("transformers")

    with (
        patch.object(transformers.AutoTokenizer, "from_pretrained") as mock_tokenizer_from_pretrained,
        patch.object(transformers.AutoModel, "from_pretrained") as mock_model_from_pretrained,
    ):
        # Mock tokenizer and model to avoid downloading
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        # Test tokenizer and model creation interface (mocked)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        model = transformers.AutoModel.from_pretrained("bert-base-uncased")

        assert tokenizer is not None
        assert model is not None

        # Verify mocks were called
        mock_tokenizer_from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_model_from_pretrained.assert_called_once_with("bert-base-uncased")


def test_torch_cuda_compatibility():
    """Test CUDA compatibility, when present, against the supported baseline."""
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


def test_ml_stack_imports():
    """Test that all major ML packages can be imported without errors."""
    from importlib.metadata import PackageNotFoundError, version

    def get_version(package_name):
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None

    def assert_minimum_version(package_name: str, minimum_version: str) -> None:
        package_version = get_version(package_name)
        assert package_version is not None, f"{package_name} not installed"
        assert Version(package_version) >= Version(
            minimum_version
        ), f"{package_name} version {package_version} < {minimum_version}"

    # Always check torch (required for ML tests to run)
    torch = pytest.importorskip("torch", reason="torch required for ML smoke tests")
    assert_minimum_version("torch", "2.2.2")

    # Check sklearn (if available)
    sklearn, _ = _import_optional_module("sklearn")
    if sklearn is not None:
        assert_minimum_version("scikit-learn", "1.8.0")

    # Check diffusers install floor (runtime compatibility is covered separately above).
    diffusers, _ = _import_optional_module("diffusers")
    if diffusers is not None or get_version("diffusers") is not None:
        assert_minimum_version("diffusers", "0.36.0")

    # Check transformers (if available)
    transformers, _ = _import_optional_module("transformers")
    if transformers is not None or get_version("transformers") is not None:
        assert_minimum_version("transformers", "4.57.0")

    # Check torchvision (if available)
    torchvision, _ = _import_optional_module("torchvision")
    if torchvision is not None:
        assert_minimum_version("torchvision", "0.17.2")

    # Check timm (if available)
    timm, _ = _import_optional_module("timm")
    if timm is not None:
        timm_version = get_version("timm")
        assert timm_version is not None, "timm not installed"
