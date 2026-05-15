"""Unit tests for material_response.core validator + quantum surfaces.

Phase 7 coverage. Continues the math-helper work from Phase 6 by
targeting the two heavier classes in ``core.py`` that the existing
test suites don't exercise:

- ``MaterialResponseValidator`` -- the Fourier-band energy ratio
  (low/high bands, shape and zero-energy guards) and the box-counting
  Hausdorff estimator.
- ``QuantumMaterialResponse`` -- contextual wavefunction normalisation,
  entanglement/coherence pipeline, conflict resolution, and the
  collapse-guidance narrative branches.

All surfaces are pure numpy / math -- no ML, GPU, or network.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from transformation_portal.processors.material_response.core import (
    MaterialResponseValidator,
    QuantumMaterialResponse,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# MaterialResponseValidator
# ---------------------------------------------------------------------------


class TestFourierEnergyRatio:
    """Tests for MaterialResponseValidator._fourier_energy_ratio."""

    def test_rejects_unknown_band(self) -> None:
        with pytest.raises(ValueError, match="band must be"):
            MaterialResponseValidator._fourier_energy_ratio(
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                band="mid",
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            MaterialResponseValidator._fourier_energy_ratio(
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                band="high",
            )

    def test_returns_unity_when_reference_energy_is_zero(self) -> None:
        # Constant matrices have all energy at DC -> the high band sum is 0.
        ratio = MaterialResponseValidator._fourier_energy_ratio(
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            band="high",
        )
        assert ratio == pytest.approx(1.0)

    def test_low_band_ratio_for_uniform_inputs(self) -> None:
        # Both matrices are constant -> all energy lives in the low band, so
        # the ratio is the squared magnitude ratio of the DC bins.
        ratio = MaterialResponseValidator._fourier_energy_ratio(
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            band="low",
        )
        # |DC_after|^2 / |DC_before|^2 == (8.0)^2 / (4.0)^2 == 4.0.
        assert ratio == pytest.approx(4.0)


class TestHausdorffAndBoxcount:
    """Tests for _calculate_hausdorff_dimension and _boxcount."""

    def test_dimension_is_unity_for_small_surfaces(self) -> None:
        # min(rows, cols) == 2 -> max_exponent == 1 -> the helper short-circuits.
        dim = MaterialResponseValidator._calculate_hausdorff_dimension([[0.0, 1.0], [1.0, 0.0]])
        assert dim == pytest.approx(1.0)

    def test_dimension_is_finite_for_larger_surface(self) -> None:
        rng = np.random.default_rng(0)
        surface = rng.random((8, 8)).tolist()

        dim = MaterialResponseValidator._calculate_hausdorff_dimension(surface)

        assert math.isfinite(dim)
        assert dim >= 0.0

    def test_dimension_handles_constant_surface(self) -> None:
        # A constant surface should not crash on the zero-range normalisation.
        constant = [[0.5] * 8 for _ in range(8)]
        dim = MaterialResponseValidator._calculate_hausdorff_dimension(constant)
        assert math.isfinite(dim)
        assert dim >= 0.0

    def test_boxcount_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MaterialResponseValidator._boxcount([[True, False], [False, True]], 0)

    def test_boxcount_returns_zero_for_empty_input(self) -> None:
        assert MaterialResponseValidator._boxcount([], 1) == 0
        assert MaterialResponseValidator._boxcount([[]], 1) == 0

    def test_boxcount_returns_zero_when_size_exceeds_dimensions(self) -> None:
        # Trimming a 2x2 grid to a multiple of size=4 yields no boxes.
        assert MaterialResponseValidator._boxcount([[True, True], [True, True]], 4) == 0

    def test_boxcount_counts_each_occupied_block(self) -> None:
        # 4x4 grid divided into 4 size-2 boxes; each box has at least one True.
        binary = [
            [True, False, True, False],
            [False, False, False, False],
            [False, False, True, True],
            [True, False, False, False],
        ]
        assert MaterialResponseValidator._boxcount(binary, 2) == 4

    def test_boxcount_skips_fully_empty_blocks(self) -> None:
        binary = [
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
        assert MaterialResponseValidator._boxcount(binary, 2) == 1


# ---------------------------------------------------------------------------
# QuantumMaterialResponse
# ---------------------------------------------------------------------------


class TestQuantumInit:
    """Tests for QuantumMaterialResponse.__init__."""

    def test_default_coherence_gain(self) -> None:
        q = QuantumMaterialResponse()
        assert q.coherence_gain == pytest.approx(0.68)
        # The cognitive subsystem is wired up eagerly.
        assert q.cognitive_engine is not None

    def test_rejects_out_of_range_gain(self) -> None:
        with pytest.raises(ValueError, match="coherence_gain"):
            QuantumMaterialResponse(coherence_gain=1.5)
        with pytest.raises(ValueError, match="coherence_gain"):
            QuantumMaterialResponse(coherence_gain=-0.1)


class TestContextualWavefunction:
    """Tests for QuantumMaterialResponse._contextual_wavefunction."""

    def test_normalises_numeric_weights(self) -> None:
        wf = QuantumMaterialResponse._contextual_wavefunction({"awe": 2.0, "comfort": 3.0})
        assert wf["awe"] == pytest.approx(0.4)
        assert wf["comfort"] == pytest.approx(0.6)
        assert sum(wf.values()) == pytest.approx(1.0)

    def test_text_components_contribute_keyword_weights(self) -> None:
        wf = QuantumMaterialResponse._contextual_wavefunction({"narrative": "Polished marble luminous foyer"})
        # The extracted keywords sum (with their split weights) plus zero
        # numeric input must still normalise to 1.0.
        assert sum(wf.values()) == pytest.approx(1.0)
        assert "marble" in wf or "luminous" in wf or "polished" in wf

    def test_sequence_values_extend_textual_components(self) -> None:
        wf = QuantumMaterialResponse._contextual_wavefunction({"tags": ["polished", "marble", "luminous"]})
        assert sum(wf.values()) == pytest.approx(1.0)
        assert "marble" in wf

    def test_non_recognised_value_falls_back_to_constant_weight(self) -> None:
        # Booleans land in the catch-all branch and get 0.1 weight.
        wf = QuantumMaterialResponse._contextual_wavefunction({"flag": object()})
        assert wf == {"flag": pytest.approx(1.0)}

    def test_empty_context_returns_baseline(self) -> None:
        assert QuantumMaterialResponse._contextual_wavefunction({}) == {"baseline": pytest.approx(1.0)}


class TestEntangleSurfaces:
    """Tests for QuantumMaterialResponse.entangle_surfaces."""

    def test_rejects_empty_tensor(self) -> None:
        q = QuantumMaterialResponse()
        with pytest.raises(ValueError, match="cannot be empty"):
            q.entangle_surfaces([], {"awe": 1.0})

    def test_rejects_non_mapping_context(self) -> None:
        q = QuantumMaterialResponse()
        with pytest.raises(TypeError, match="must be a mapping"):
            q.entangle_surfaces([[0.5, 0.5]], "not-a-mapping")  # type: ignore[arg-type]

    def test_returns_expected_payload_for_2d_input_with_numeric_context(self) -> None:
        q = QuantumMaterialResponse()
        result = q.entangle_surfaces(
            [[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]],
            {"awe": 1.0},
        )

        assert set(result) == {
            "superposition_states",
            "contextual_weights",
            "coherence_map",
            "entanglement_matrix",
            "conflict_resolution",
            "collapse_guidance",
        }
        assert isinstance(result["coherence_map"], list)
        assert isinstance(result["collapse_guidance"], str)
        assert result["contextual_weights"]["awe"] == pytest.approx(1.0)

    def test_returns_expected_payload_for_2d_input(self) -> None:
        q = QuantumMaterialResponse()
        result = q.entangle_surfaces(
            [[0.2, 0.4, 0.6], [0.8, 0.5, 0.3]],
            {"awe": 0.5, "comfort": 0.5},
        )

        assert isinstance(result["superposition_states"], list)
        assert len(result["superposition_states"]) == 2

    def test_handles_all_zero_tensor(self) -> None:
        # A flat-zero amplitude triggers the math.isclose(amplitude_sum, 0)
        # branch, returning a zero coherence map without dividing by zero.
        q = QuantumMaterialResponse()
        result = q.entangle_surfaces([[0.0, 0.0], [0.0, 0.0]], {"awe": 1.0})

        assert all(value == 0.0 for row in result["coherence_map"] for value in row)


class TestConflictResolutionAndGuidance:
    """Tests for identify_and_resolve_conflicts and _collapse_guidance."""

    def test_empty_coherence_map_raises(self) -> None:
        q = QuantumMaterialResponse()
        with pytest.raises(ValueError, match="cannot be empty"):
            q.identify_and_resolve_conflicts(np.zeros((0, 0)), {"awe": 1.0})

    def test_flat_map_yields_no_conflicts_and_unit_stability(self) -> None:
        q = QuantumMaterialResponse()
        flat = np.full((3, 3), 0.5)

        result = q.identify_and_resolve_conflicts(flat, {"awe": 1.0})

        assert result["conflicts"] == []
        assert result["resolutions"] == []
        assert result["stability_index"] == pytest.approx(1.0)

    def test_high_deviation_produces_conflicts_and_resolutions(self) -> None:
        q = QuantumMaterialResponse()
        # One cell deviates well above the default threshold of 0.12.
        spiky = np.array([[0.5, 0.5], [0.5, 1.0]])

        result = q.identify_and_resolve_conflicts(spiky, {"awe": 0.8, "comfort": 0.2})

        assert len(result["conflicts"]) >= 1
        # Resolutions reference the dominant context keyword (sorted descending).
        assert any("awe" in r for r in result["resolutions"])
        assert 0.0 <= result["stability_index"] <= 1.0

    def test_resolution_falls_back_when_context_is_empty(self) -> None:
        q = QuantumMaterialResponse()
        spiky = np.array([[0.5, 0.5], [0.5, 1.0]])

        result = q.identify_and_resolve_conflicts(spiky, {})

        assert any("decoherence damping" in r for r in result["resolutions"])

    def test_collapse_guidance_emphasises_pivot_when_conflicts_present(self) -> None:
        guidance = QuantumMaterialResponse._collapse_guidance(
            {"conflicts": ["surface[0][0] dominates"]},
            {"awe": 0.7, "comfort": 0.3},
        )
        assert "awe" in guidance
        assert "Stabilise decoherence" in guidance

    def test_collapse_guidance_uses_top_context_when_smooth(self) -> None:
        guidance = QuantumMaterialResponse._collapse_guidance(
            {"conflicts": []},
            {"comfort": 0.9, "awe": 0.1},
        )
        assert "comfort" in guidance
        assert "anchor the reveal" in guidance

    def test_collapse_guidance_neutral_when_nothing_to_anchor(self) -> None:
        guidance = QuantumMaterialResponse._collapse_guidance({"conflicts": []}, {})
        assert "neutral cultural frame" in guidance


class TestQuantumInternals:
    """Tests for the quantum class's lower-level helpers."""

    def test_apply_coherence_returns_zeros_for_zero_amplitude(self) -> None:
        q = QuantumMaterialResponse()
        out = q._apply_coherence(np.zeros((4,)), {"awe": 0.5})
        assert out.shape == (4,)
        assert out.max() == pytest.approx(0.0)

    def test_apply_coherence_clips_into_unit_range(self) -> None:
        q = QuantumMaterialResponse(coherence_gain=1.0)
        amplitude = np.array([0.1, 0.2, 0.7])
        out = q._apply_coherence(amplitude, {"awe": 1.0})
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_entanglement_matrix_returns_zeros_for_zero_input(self) -> None:
        out = QuantumMaterialResponse._entanglement_matrix(np.zeros((2, 2)), np.zeros((2, 2), dtype=complex))
        assert np.all(out == 0.0)
        assert out.shape == (2, 2)

    def test_entanglement_matrix_is_unit_normalised(self) -> None:
        coherence = np.array([[0.25, 0.5], [0.75, 1.0]])
        frequency = np.fft.fft2(coherence)
        out = QuantumMaterialResponse._entanglement_matrix(coherence, frequency)
        assert np.max(np.abs(out)) == pytest.approx(1.0)

    def test_superposition_states_summarise_each_surface(self) -> None:
        matrix = np.array([[0.2, 0.4, 0.6], [0.8, 0.5, 0.3]])
        coherence = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        states = QuantumMaterialResponse._superposition_states(matrix, coherence)

        assert len(states) == 2
        for idx, state in enumerate(states):
            assert state["surface_index"] == idx
            assert "mean_reflectance" in state
            assert "phase_variance" in state
            assert "coherence" in state

    def test_superposition_states_handles_shorter_coherence_map(self) -> None:
        matrix = np.array([[0.2, 0.4], [0.6, 0.8]])
        coherence = np.array([[0.1, 0.2]])  # only one row of coherence

        states = QuantumMaterialResponse._superposition_states(matrix, coherence)

        # The second surface falls back to coherence == 0.0.
        assert states[1]["coherence"] == pytest.approx(0.0)
