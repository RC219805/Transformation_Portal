"""Unit tests for processors.material_response.core math helpers and primitives.

Phase 6 coverage. Targets the pure-math and dataclass surfaces of
``core.py`` that the existing cognitive/aesthetic tests don't reach:
sequence coercion, the manual DFT and radial-frequency machinery,
linear regression, keyword extraction, the ``MaterialResponsePrinciple``
documentation surface, and the ``compose_operations`` /
``apply_transformation_tensor`` tensor pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from transformation_portal.processors.material_response import core as core_module
from transformation_portal.processors.material_response.core import (
    MaterialResponseExample,
    MaterialResponsePrinciple,
    _clamp,
    _coerce_diagonal,
    _coerce_matrix,
    _dft2,
    _energy_by_band,
    _ensure_matrix,
    _extract_keywords,
    _fft_frequency,
    _flatten,
    _is_sequence,
    _linear_regression,
    _median,
    _radial_frequency_grid,
    apply_transformation_tensor,
    compose_operations,
)

pytestmark = pytest.mark.unit


class TestIsSequence:
    """Tests for the _is_sequence sentinel."""

    @pytest.mark.parametrize("value", [[], [1, 2], (1, 2), range(3)])
    def test_treats_collections_as_sequences(self, value: Any) -> None:
        assert _is_sequence(value) is True

    @pytest.mark.parametrize("value", ["abc", b"abc", bytearray(b"abc"), 5, 1.0, None, {1: 2}])
    def test_rejects_strings_bytes_and_scalars(self, value: Any) -> None:
        assert _is_sequence(value) is False


class TestCoerceMatrix:
    """Tests for _coerce_matrix."""

    def test_1d_sequence_becomes_single_row(self) -> None:
        assert _coerce_matrix([1, 2, 3]) == [[1.0, 2.0, 3.0]]

    def test_2d_sequence_is_validated_and_coerced_to_floats(self) -> None:
        assert _coerce_matrix([[1, 2], [3, 4]]) == [[1.0, 2.0], [3.0, 4.0]]

    def test_rejects_non_sequence_input(self) -> None:
        with pytest.raises(TypeError):
            _coerce_matrix(42)  # type: ignore[arg-type]

    def test_rejects_empty_outer_sequence(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            _coerce_matrix([])

    def test_rejects_empty_inner_row(self) -> None:
        with pytest.raises(ValueError, match="rows cannot be empty"):
            _coerce_matrix([[], []])  # row_length resolves to 0

    def test_rejects_non_sequence_row(self) -> None:
        with pytest.raises(TypeError, match="rows must be sequences"):
            _coerce_matrix([[1, 2], 3])  # type: ignore[list-item]

    def test_rejects_jagged_rows(self) -> None:
        with pytest.raises(ValueError, match="rectangular"):
            _coerce_matrix([[1, 2], [3]])


class TestFlattenAndMedian:
    """Tests for _flatten and _median."""

    def test_flatten_concatenates_rows_in_order(self) -> None:
        assert _flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_median_of_odd_length(self) -> None:
        assert _median([3, 1, 2]) == 2

    def test_median_of_even_length_averages_middle_pair(self) -> None:
        assert _median([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_median_of_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _median([])


class TestFftHelpers:
    """Tests for _fft_frequency and _radial_frequency_grid."""

    def test_fft_frequency_returns_zero_at_origin(self) -> None:
        assert _fft_frequency(0, 8) == 0.0

    def test_fft_frequency_wraps_to_negative_for_upper_half(self) -> None:
        # In a size-4 signal, index 3 represents frequency -1/4.
        assert _fft_frequency(3, 4) == pytest.approx(-0.25)

    def test_fft_frequency_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _fft_frequency(0, 0)

    def test_radial_frequency_grid_origin_is_zero(self) -> None:
        grid = _radial_frequency_grid((4, 4))
        assert grid[0][0] == 0.0

    def test_radial_frequency_grid_dimensions_match_shape(self) -> None:
        grid = _radial_frequency_grid((3, 5))
        assert len(grid) == 3
        assert all(len(row) == 5 for row in grid)


class TestDft2AndEnergyByBand:
    """Tests for _dft2 and _energy_by_band."""

    def test_dft2_of_constant_matrix_concentrates_at_dc(self) -> None:
        # For a constant matrix, all energy collapses to the DC bin.
        result = _dft2([[1.0, 1.0], [1.0, 1.0]])
        assert abs(result[0][0]) == pytest.approx(4.0)
        for u in range(2):
            for v in range(2):
                if (u, v) != (0, 0):
                    assert abs(result[u][v]) == pytest.approx(0.0, abs=1e-9)

    def test_energy_by_band_partitions_low_and_high(self) -> None:
        dft = _dft2([[1.0, 1.0], [1.0, 1.0]])
        radii = _radial_frequency_grid((2, 2))

        low = _energy_by_band(dft, "low", cutoff=0.1, radii=radii)
        high = _energy_by_band(dft, "high", cutoff=0.1, radii=radii)

        # All energy is at DC (radius 0), which is in the low band.
        assert low == pytest.approx(16.0)
        assert high == pytest.approx(0.0, abs=1e-9)


class TestLinearRegression:
    """Tests for _linear_regression."""

    def test_recovers_slope_and_intercept(self) -> None:
        # y = 2x + 1
        slope, intercept = _linear_regression([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_returns_mean_intercept_when_xs_are_constant(self) -> None:
        slope, intercept = _linear_regression([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        assert slope == 0.0
        assert intercept == pytest.approx(2.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            _linear_regression([1, 2], [1, 2, 3])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="requires data"):
            _linear_regression([], [])


class TestPrincipleDocumentation:
    """Tests for the MaterialResponsePrinciple / Example documentation surface."""

    def test_example_as_dict_roundtrips_fields(self) -> None:
        example = MaterialResponseExample(
            material="oak",
            lighting="morning",
            challenge="contrast",
            response="midtone",
            outcome="luminous",
        )
        assert example.as_dict() == {
            "material": "oak",
            "lighting": "morning",
            "challenge": "contrast",
            "response": "midtone",
            "outcome": "luminous",
        }

    def test_principle_describe_combines_name_and_focus(self) -> None:
        principle = MaterialResponsePrinciple()
        description = principle.describe()
        assert description.startswith("Material Response:")

    def test_principle_guidelines_is_a_copy(self) -> None:
        principle = MaterialResponsePrinciple()
        guidelines = principle.guidelines()

        assert guidelines == principle.tenets
        guidelines.append("mutated")
        # Mutating the returned list must not leak into the principle's tenets.
        assert "mutated" not in principle.tenets

    def test_principle_generate_examples_returns_serialisable_dicts(self) -> None:
        principle = MaterialResponsePrinciple()
        examples = principle.generate_examples()

        assert len(examples) >= 1
        for entry in examples:
            assert set(entry) == {"material", "lighting", "challenge", "response", "outcome"}
            for value in entry.values():
                assert isinstance(value, str) and value


class TestKeywordsAndClamp:
    """Tests for _extract_keywords and _clamp."""

    def test_extract_keywords_lowercases_and_filters_short_and_stopwords(self) -> None:
        text = "The Quick brown fox keeps jumping IN between"
        keywords = _extract_keywords(text)
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        # Stopwords and short tokens are filtered out.
        assert "the" not in keywords
        assert "in" not in keywords
        assert "between" not in keywords

    @pytest.mark.parametrize(
        "value,expected",
        [(-1.0, 0.0), (0.5, 0.5), (2.0, 1.0)],
    )
    def test_clamp_default_unit_range(self, value: float, expected: float) -> None:
        assert _clamp(value) == pytest.approx(expected)

    def test_clamp_custom_range(self) -> None:
        assert _clamp(7.0, minimum=-1.0, maximum=5.0) == pytest.approx(5.0)

    def test_clamp_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="minimum cannot be greater than maximum"):
            _clamp(0.0, minimum=1.0, maximum=0.0)


class TestComposeOperationsAndMatrixCoercion:
    """Tests for compose_operations and the matrix-coercion helpers."""

    def test_no_operations_returns_identity(self) -> None:
        out = compose_operations(size=3)
        assert np.allclose(out, np.eye(3))

    def test_scalar_operation_scales_all_channels(self) -> None:
        out = compose_operations(2.0, size=3)
        assert np.allclose(out, 2.0 * np.eye(3))

    def test_vector_operation_becomes_diagonal(self) -> None:
        out = compose_operations([1.0, 2.0, 3.0], size=3)
        assert np.allclose(out, np.diag([1.0, 2.0, 3.0]))

    def test_matrix_operation_is_applied_directly(self) -> None:
        matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = compose_operations(matrix, size=3)
        assert np.allclose(out, matrix)

    def test_mapping_with_scale_key_uses_diagonal(self) -> None:
        out = compose_operations({"scale": [0.5, 0.5, 0.5]}, size=3)
        assert np.allclose(out, 0.5 * np.eye(3))

    def test_mapping_with_weights_key_uses_diagonal(self) -> None:
        out = compose_operations({"weights": [1.0, 0.0, 1.0]}, size=3)
        assert np.allclose(out, np.diag([1.0, 0.0, 1.0]))

    def test_mapping_with_matrix_key_uses_full_matrix(self) -> None:
        m = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        out = compose_operations({"matrix": m, "name": "warm-tone"}, size=3)
        assert np.allclose(out, np.diag([1.0, 2.0, 3.0]))

    def test_mapping_with_mix_key_uses_full_matrix(self) -> None:
        m = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        out = compose_operations({"mix": m}, size=3)
        assert np.allclose(out, np.array(m))

    def test_named_operations_coalesce_by_multiplication(self) -> None:
        # Two named "warm" operations should be multiplied, not duplicated.
        op = {"matrix": np.diag([2.0, 1.0, 1.0]).tolist(), "name": "warm"}
        out = compose_operations(op, op, size=3)
        # Composition multiplies diagonals: 2 * 2 == 4 on the first channel.
        assert out[0, 0] == pytest.approx(4.0)

    def test_callable_operation_is_invoked_with_size(self) -> None:
        def double(_size: int) -> float:
            return 2.0

        out = compose_operations(double, size=3)
        assert np.allclose(out, 2.0 * np.eye(3))

    def test_none_operations_are_skipped(self) -> None:
        out = compose_operations(None, 2.0, None, size=3)
        assert np.allclose(out, 2.0 * np.eye(3))

    def test_mapping_without_recognised_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain"):
            compose_operations({"unknown": 1}, size=3)

    def test_compose_returns_requested_dtype(self) -> None:
        out = compose_operations(2.0, size=3, dtype=np.float64)
        assert out.dtype == np.float64

    def test_coerce_diagonal_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            _coerce_diagonal(None, 3)

    def test_coerce_diagonal_rejects_higher_dimensions(self) -> None:
        with pytest.raises(ValueError, match="scalar or a 1-D"):
            _coerce_diagonal([[1, 2], [3, 4]], 3)

    def test_coerce_diagonal_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expects 3 coefficients"):
            _coerce_diagonal([1.0, 2.0], 3)

    def test_ensure_matrix_rejects_higher_than_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _ensure_matrix(np.zeros((2, 2, 2)), 3)


class TestApplyTransformationTensor:
    """Tests for apply_transformation_tensor."""

    def test_identity_operation_returns_input(self) -> None:
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)
        out = apply_transformation_tensor(image)
        assert np.allclose(out, image)

    def test_clips_into_unit_range_by_default(self) -> None:
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)
        # A 4x scale would push values to 2.0; clipping must bring them to 1.0.
        out = apply_transformation_tensor(image, 4.0)
        assert out.max() == pytest.approx(1.0)

    def test_clip_false_preserves_out_of_range_values(self) -> None:
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)
        out = apply_transformation_tensor(image, 4.0, clip=False)
        assert out.max() == pytest.approx(2.0)

    def test_rejects_non_3d_input(self) -> None:
        with pytest.raises(ValueError, match="H.W.C"):
            apply_transformation_tensor(np.zeros((2, 2)))
