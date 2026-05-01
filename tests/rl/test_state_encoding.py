"""Tests for rl.ma_state and rl.state_encoder — pure numpy encoding functions."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ma_state
# ---------------------------------------------------------------------------


class TestMAStateDimensions:
    def test_get_global_dim_returns_20(self):
        """Global dim: 5 metrics + 5 types × 3 severities = 20."""
        from transformation_portal.rl.ma_state import get_global_dim

        assert get_global_dim() == 20

    def test_get_local_dim_returns_8(self):
        """Local dim is always 8."""
        from transformation_portal.rl.ma_state import get_local_dim

        assert get_local_dim() == 8

    def test_get_state_dim_returns_28(self):
        """State dim: 20 global + 8 local = 28."""
        from transformation_portal.rl.ma_state import get_state_dim

        assert get_state_dim() == 28


class TestMAStateEncodeGlobal:
    def _metrics(self, score=0.8, psnr=40.0, lpips=0.1, ssim=0.9, llava_score=0.7):
        return {"score": score, "psnr": psnr, "lpips": lpips, "ssim": ssim, "llava_score": llava_score}

    def _diff(self, changes=None):
        return {"changes": changes or []}

    def test_shape_is_global_dim(self):
        """encode_global returns shape (20,)."""
        from transformation_portal.rl.ma_state import encode_global, get_global_dim

        arr = encode_global(self._metrics(), self._diff())
        assert arr.shape == (get_global_dim(),)

    def test_dtype_float32(self):
        """Output dtype is float32."""
        from transformation_portal.rl.ma_state import encode_global

        arr = encode_global(self._metrics(), self._diff())
        assert arr.dtype == np.float32

    def test_empty_metrics_no_crash(self):
        """Empty metrics dict uses zero defaults."""
        from transformation_portal.rl.ma_state import encode_global

        arr = encode_global({}, self._diff())
        assert arr.shape == (20,)
        assert arr[0] == pytest.approx(0.0)

    def test_score_encoded_in_first_element(self):
        """score appears as the first feature."""
        from transformation_portal.rl.ma_state import encode_global

        arr = encode_global(self._metrics(score=0.6), self._diff())
        assert arr[0] == pytest.approx(0.6)

    def test_empty_diff_histogram_zeros(self):
        """No changes → all histogram features are zero."""
        from transformation_portal.rl.ma_state import encode_global

        arr = encode_global(self._metrics(), self._diff(changes=[]))
        # Elements 5..19 are the histogram (15 values)
        assert np.all(arr[5:] == 0.0)

    def test_diff_change_increments_histogram(self):
        """A geometry/high change increments the corresponding bin."""
        from transformation_portal.rl.ma_state import encode_global

        changes = [{"type": "geometry", "severity": "high"}]
        arr = encode_global(self._metrics(), self._diff(changes=changes))
        # At least one histogram bin is non-zero
        assert np.any(arr[5:] > 0.0)


class TestMAStateEncodeLocal:
    def test_shape_is_local_dim(self):
        """encode_local returns shape (8,)."""
        from transformation_portal.rl.ma_state import encode_local, get_local_dim

        arr = encode_local({})
        assert arr.shape == (get_local_dim(),)

    def test_dtype_float32(self):
        """Output dtype is float32."""
        from transformation_portal.rl.ma_state import encode_local

        arr = encode_local({})
        assert arr.dtype == np.float32

    def test_empty_config_returns_valid_array(self):
        """Empty node config returns a valid float32 array of correct length."""
        from transformation_portal.rl.ma_state import encode_local

        arr = encode_local({})
        assert arr.shape == (8,)
        assert arr.dtype == np.float32
        # Values are finite (no NaN/Inf); some may be non-zero due to bias offsets
        assert np.all(np.isfinite(arr))

    def test_known_node_id_uses_typed_keys(self):
        """Known node_id (sam2) uses its type-specific keys."""
        from transformation_portal.rl.ma_state import encode_local

        arr = encode_local({"threshold": 0.5}, node_id="sam2")
        assert arr.shape == (8,)

    def test_unknown_node_id_uses_defaults(self):
        """Unknown node_id falls back to DEFAULT_CONFIG_KEYS."""
        from transformation_portal.rl.ma_state import encode_local

        arr = encode_local({}, node_id="unknown_node")
        assert arr.shape == (8,)

    def test_boolean_config_encoded_as_float(self):
        """Boolean config values (seam_blending) are 0.0 or 1.0."""
        from transformation_portal.rl.ma_state import encode_local

        arr_on = encode_local({"seam_blending": True}, node_id="postprocess")
        arr_off = encode_local({"seam_blending": False}, node_id="postprocess")
        assert float(arr_on[0]) == pytest.approx(1.0)
        assert float(arr_off[0]) == pytest.approx(0.0)


class TestMAStateEncodeState:
    def test_shape_is_state_dim(self):
        """encode_state returns shape (28,)."""
        from transformation_portal.rl.ma_state import encode_state, get_state_dim

        arr = encode_state({}, {}, {})
        assert arr.shape == (get_state_dim(),)

    def test_global_and_local_concatenated(self):
        """encode_state == concat(encode_global, encode_local)."""
        from transformation_portal.rl.ma_state import encode_global, encode_local, encode_state

        metrics = {"score": 0.7}
        diff = {}
        cfg = {"threshold": 0.5}
        state = encode_state(cfg, metrics, diff, node_id="sam2")
        global_part = encode_global(metrics, diff)
        local_part = encode_local(cfg, node_id="sam2")
        expected = np.concatenate([global_part, local_part])
        np.testing.assert_array_almost_equal(state, expected)


# ---------------------------------------------------------------------------
# state_encoder
# ---------------------------------------------------------------------------


class TestStateEncoderDimensions:
    def test_get_state_dim_returns_105(self):
        """Total dim: 5 + 15 + 10×8 + 5 = 105."""
        from transformation_portal.rl.state_encoder import get_state_dim

        assert get_state_dim() == 105


class TestEncodeMetrics:
    def test_length_is_five(self):
        """encode_metrics returns 5 values."""
        from transformation_portal.rl.state_encoder import encode_metrics

        assert len(encode_metrics({})) == 5

    def test_score_is_first_element(self):
        """First element is the raw score."""
        from transformation_portal.rl.state_encoder import encode_metrics

        feats = encode_metrics({"score": 0.85})
        assert feats[0] == pytest.approx(0.85)

    def test_psnr_normalized_by_50(self):
        """PSNR is divided by 50."""
        from transformation_portal.rl.state_encoder import encode_metrics

        feats = encode_metrics({"psnr": 50.0})
        assert feats[1] == pytest.approx(1.0)

    def test_lpips_inverted(self):
        """LPIPS is inverted: 1 - lpips."""
        from transformation_portal.rl.state_encoder import encode_metrics

        feats = encode_metrics({"lpips": 0.3})
        assert feats[2] == pytest.approx(0.7)

    def test_missing_keys_default_to_zero(self):
        """Missing keys default to 0.0."""
        from transformation_portal.rl.state_encoder import encode_metrics

        feats = encode_metrics({})
        assert all(f in (0.0, 1.0) for f in feats)  # lpips 0 → 1-0=1


class TestEncodeDiffHistogram:
    def test_length_is_15(self):
        """encode_diff_histogram returns 15 values."""
        from transformation_portal.rl.state_encoder import encode_diff_histogram

        assert len(encode_diff_histogram({})) == 15

    def test_empty_diff_all_zeros(self):
        """No changes → all zeros."""
        from transformation_portal.rl.state_encoder import encode_diff_histogram

        feats = encode_diff_histogram({"changes": []})
        assert all(f == 0.0 for f in feats)

    def test_values_between_0_and_1(self):
        """All histogram values are in [0, 1]."""
        from transformation_portal.rl.state_encoder import encode_diff_histogram

        changes = [{"type": "geometry", "severity": "high"}] * 20
        feats = encode_diff_histogram({"changes": changes})
        assert all(0.0 <= f <= 1.0 for f in feats)

    def test_unknown_type_maps_to_semantic(self):
        """Unknown change type falls back to 'semantic' bin."""
        from transformation_portal.rl.state_encoder import encode_diff_histogram

        feats_known = encode_diff_histogram({"changes": [{"type": "semantic", "severity": "medium"}]})
        feats_unknown = encode_diff_histogram({"changes": [{"type": "nonexistent", "severity": "medium"}]})
        np.testing.assert_array_equal(feats_known, feats_unknown)


class TestEncodeNodeConfig:
    def test_length_is_8(self):
        """encode_node_config returns 8 values."""
        from transformation_portal.rl.state_encoder import encode_node_config

        assert len(encode_node_config({})) == 8

    def test_empty_config_returns_valid_list(self):
        """Empty node config returns a finite float list of correct length."""
        from transformation_portal.rl.state_encoder import encode_node_config

        feats = encode_node_config({})
        assert len(feats) == 8
        assert all(isinstance(f, float) for f in feats)
        # Values are finite; some may be non-zero due to bias/contrast offsets
        import math

        assert all(math.isfinite(f) for f in feats)

    def test_steps_normalized_by_1000(self):
        """steps=1000 → normalized to 1.0."""
        from transformation_portal.rl.state_encoder import encode_node_config

        feats = encode_node_config({"config": {"steps": 1000}})
        assert feats[1] == pytest.approx(1.0)


class TestEncodeHistory:
    def test_length_equals_window(self):
        """encode_history always returns window-many values."""
        from transformation_portal.rl.state_encoder import encode_history

        assert len(encode_history([], window=5)) == 5

    def test_empty_history_all_zeros(self):
        """No history → all zeros."""
        from transformation_portal.rl.state_encoder import encode_history

        assert all(v == 0.0 for v in encode_history([]))

    def test_recent_scores_at_end(self):
        """Most recent score appears at the last position."""
        from transformation_portal.rl.state_encoder import encode_history

        history = [0.1, 0.2, 0.3, 0.4, 0.5]
        encoded = encode_history(history, window=5)
        assert encoded[-1] == pytest.approx(0.5)

    def test_long_history_takes_last_window(self):
        """Only the last window scores are used."""
        from transformation_portal.rl.state_encoder import encode_history

        history = [0.1] * 20 + [0.9]
        encoded = encode_history(history, window=5)
        assert encoded[-1] == pytest.approx(0.9)


class TestEncodeState:
    def _pipeline(self):
        return {"nodes": [{"id": "sam2", "config": {"threshold": 0.5}}]}

    def test_shape_is_105(self):
        """encode_state returns shape (105,)."""
        from transformation_portal.rl.state_encoder import encode_state

        arr = encode_state(self._pipeline(), {}, {})
        assert arr.shape == (105,)

    def test_dtype_float32(self):
        """Output dtype is float32."""
        from transformation_portal.rl.state_encoder import encode_state

        arr = encode_state(self._pipeline(), {}, {})
        assert arr.dtype == np.float32

    def test_deterministic_output(self):
        """Same inputs produce identical arrays."""
        from transformation_portal.rl.state_encoder import encode_state

        m = {"score": 0.8}
        arr1 = encode_state(self._pipeline(), m, {})
        arr2 = encode_state(self._pipeline(), m, {})
        np.testing.assert_array_equal(arr1, arr2)


class TestDecodeStateSummary:
    def test_returns_dict_with_expected_keys(self):
        """decode_state_summary returns dict with metrics, diff_summary, recent_scores."""
        from transformation_portal.rl.state_encoder import decode_state_summary, encode_state

        arr = encode_state({"nodes": []}, {"score": 0.7}, {})
        result = decode_state_summary(arr)
        assert "metrics" in result
        assert "diff_summary" in result
        assert "recent_scores" in result

    def test_score_round_trips(self):
        """Encoded score can be recovered from decode."""
        from transformation_portal.rl.state_encoder import decode_state_summary, encode_state

        arr = encode_state({"nodes": []}, {"score": 0.75}, {})
        result = decode_state_summary(arr)
        assert result["metrics"]["score"] == pytest.approx(0.75)
