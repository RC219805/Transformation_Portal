"""Unit tests for foundation.tensor_processor (CPU only, torch required)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit, pytest.mark.ml]

CPU_DEVICE = torch.device("cpu")


def _processor():
    from transformation_portal.foundation.tensor_processor import PrecisionMode, TensorConfig, TensorProcessor

    return TensorProcessor(TensorConfig(precision=PrecisionMode.FP32), CPU_DEVICE)


@pytest.mark.regression
class TestAllocateRequiresGrad:
    """Audit M-19: requires_grad was set before the channels_last copy."""

    def test_nchw_allocation_with_grad_is_an_optimisable_leaf(self):
        tensor = _processor().allocate((2, 3, 8, 8), requires_grad=True, fill_value=0.0)
        assert tensor.requires_grad
        assert tensor.is_leaf
        assert tensor.grad_fn is None
        assert tensor.is_contiguous(memory_format=torch.channels_last)
        torch.optim.SGD([tensor], lr=0.1)  # raises ValueError for a non-leaf tensor

    def test_non_image_allocation_with_grad_is_a_leaf(self):
        tensor = _processor().allocate((4, 4), requires_grad=True)
        assert tensor.requires_grad
        assert tensor.is_leaf


@pytest.mark.regression
class TestNormalizeIntegerInput:
    """Audit M-19: mean/std were cast to the input dtype (uint8 std -> 0 -> inf)."""

    def test_uint8_input_is_normalised_in_floating_point(self):
        image = torch.full((3, 4, 4), 128, dtype=torch.uint8)
        out = _processor().normalize(image)
        assert out.is_floating_point()
        assert torch.isfinite(out).all()
        expected = (128.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        assert torch.allclose(out[:, 0, 0], expected)

    def test_uint8_below_mean_does_not_wrap(self):
        image = torch.full((1, 2, 2), 100, dtype=torch.uint8)
        out = _processor().normalize(image, mean=(128.0,), std=(64.0,))
        assert torch.allclose(out, torch.full((1, 2, 2), -0.4375))

    def test_inplace_on_integer_tensor_is_rejected(self):
        image = torch.full((3, 4, 4), 128, dtype=torch.uint8)
        with pytest.raises(TypeError, match="floating-point"):
            _processor().normalize(image, inplace=True)
        assert int(image[0, 0, 0]) == 128

    def test_float_input_behaviour_unchanged(self):
        image = torch.full((1, 3, 2, 2), 0.5)
        out = _processor().normalize(image)
        expected = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        assert torch.allclose(out[0, :, 0, 0], expected)
