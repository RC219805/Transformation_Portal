"""Tests for reference view selection."""

import pytest
import numpy as np
from lux_depth_v3.reference_view import ReferenceViewSelector, RefViewStrategy, select_reference_view, RefViewSelectionResult


def test_first_strategy():
    """Test first strategy always returns 0."""
    selector = ReferenceViewSelector(strategy=RefViewStrategy.FIRST)
    result = selector.select(num_views=5)
    assert result.selected_index == 0
    assert result.strategy == RefViewStrategy.FIRST


def test_middle_strategy():
    """Test middle strategy returns middle index."""
    selector = ReferenceViewSelector(strategy=RefViewStrategy.MIDDLE)

    # Odd number of views
    result = selector.select(num_views=5)
    assert result.selected_index == 2

    # Even number of views
    result = selector.select(num_views=6)
    assert result.selected_index == 3


def test_saddle_balanced_strategy():
    """Test saddle_balanced strategy."""
    num_views = 5
    feature_dim = 768

    # Create deterministic class tokens for reproducibility
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    result = selector.select(num_views, class_tokens)

    assert 0 <= result.selected_index < num_views
    assert result.strategy == RefViewStrategy.SADDLE_BALANCED
    assert result.scores is not None
    assert result.metrics is not None
    assert "similarity_scores" in result.metrics
    assert "feature_norms" in result.metrics
    assert "feature_variances" in result.metrics


def test_saddle_sim_range_strategy():
    """Test saddle_sim_range strategy."""
    num_views = 5
    feature_dim = 768

    # Create deterministic class tokens
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_SIM_RANGE)
    result = selector.select(num_views, class_tokens)

    assert 0 <= result.selected_index < num_views
    assert result.strategy == RefViewStrategy.SADDLE_SIM_RANGE
    assert result.scores is not None
    assert result.metrics is not None
    assert "similarity_ranges" in result.metrics


def test_no_reordering_for_few_views():
    """Test that selection is skipped for < 3 views."""
    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)

    # 1 view
    result = selector.select(num_views=1)
    assert result.selected_index == 0
    assert "num_views < 3" in result.metrics["reason"]

    # 2 views
    result = selector.select(num_views=2)
    assert result.selected_index == 0
    assert "num_views < 3" in result.metrics["reason"]


def test_class_tokens_required_for_saddle():
    """Test that class_tokens are required for saddle strategies."""
    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)

    with pytest.raises(ValueError, match="class_tokens required"):
        selector.select(num_views=5, class_tokens=None)


def test_convenience_function():
    """Test convenience function."""
    num_views = 5

    # Create deterministic class tokens
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, 768)

    result = select_reference_view(num_views=num_views, strategy="saddle_balanced", class_tokens=class_tokens)

    assert isinstance(result, RefViewSelectionResult)
    assert 0 <= result.selected_index < num_views


def test_result_str():
    """Test result string representation."""
    result = RefViewSelectionResult(selected_index=2, strategy=RefViewStrategy.SADDLE_BALANCED)

    result_str = str(result)
    assert "view 2" in result_str
    assert "saddle_balanced" in result_str


def test_saddle_balanced_metrics_normalized():
    """Test that saddle_balanced produces normalized metrics."""
    num_views = 5
    feature_dim = 768

    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    result = selector.select(num_views, class_tokens)

    # Check normalized metrics are in [0, 1]
    norm_similarity = np.array(result.metrics["normalized_similarity"])
    norm_norms = np.array(result.metrics["normalized_norms"])
    norm_variances = np.array(result.metrics["normalized_variances"])

    assert np.all(norm_similarity >= 0) and np.all(norm_similarity <= 1)
    assert np.all(norm_norms >= 0) and np.all(norm_norms <= 1)
    assert np.all(norm_variances >= 0) and np.all(norm_variances <= 1)


def test_saddle_sim_range_selects_max_range():
    """Test that saddle_sim_range selects view with max similarity range."""
    num_views = 4
    feature_dim = 768

    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_SIM_RANGE)
    result = selector.select(num_views, class_tokens)

    # Verify the selected index has the maximum similarity range
    similarity_ranges = result.metrics["similarity_ranges"]
    max_range_idx = np.argmax(similarity_ranges)

    assert result.selected_index == max_range_idx


def test_different_strategies_can_select_different_views():
    """Test that different strategies may select different reference views."""
    num_views = 5
    feature_dim = 768

    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    # Get selections from different strategies
    balanced_selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    sim_range_selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_SIM_RANGE)
    middle_selector = ReferenceViewSelector(strategy=RefViewStrategy.MIDDLE)

    balanced_result = balanced_selector.select(num_views, class_tokens)
    sim_range_result = sim_range_selector.select(num_views, class_tokens)
    middle_result = middle_selector.select(num_views)

    # All should be valid indices
    assert 0 <= balanced_result.selected_index < num_views
    assert 0 <= sim_range_result.selected_index < num_views
    assert 0 <= middle_result.selected_index < num_views

    # Middle should always be predictable
    assert middle_result.selected_index == 2


def test_identical_tokens_saddle_balanced():
    """Test saddle_balanced with identical class tokens."""
    num_views = 5
    feature_dim = 768

    # All views have identical features
    class_tokens = np.ones((num_views, feature_dim))

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    result = selector.select(num_views, class_tokens)

    # Should still select a valid view
    assert 0 <= result.selected_index < num_views


def test_zero_norm_tokens():
    """Test handling of zero-norm tokens."""
    num_views = 5
    feature_dim = 768

    # Create tokens with one zero-norm view
    class_tokens = np.random.randn(num_views, feature_dim)
    class_tokens[2] = 0  # Zero-norm token

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    result = selector.select(num_views, class_tokens)

    # Should handle gracefully (due to epsilon in normalization)
    assert 0 <= result.selected_index < num_views


def test_strategy_enum_conversion():
    """Test strategy string to enum conversion."""
    # Valid strategies
    assert RefViewStrategy("saddle_balanced") == RefViewStrategy.SADDLE_BALANCED
    assert RefViewStrategy("saddle_sim_range") == RefViewStrategy.SADDLE_SIM_RANGE
    assert RefViewStrategy("first") == RefViewStrategy.FIRST
    assert RefViewStrategy("middle") == RefViewStrategy.MIDDLE

    # Invalid strategy should raise ValueError
    with pytest.raises(ValueError):
        RefViewStrategy("invalid_strategy")


def test_convenience_function_all_strategies():
    """Test convenience function with all strategies."""
    num_views = 7
    feature_dim = 768

    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    # Test all strategies
    strategies = ["saddle_balanced", "saddle_sim_range", "first", "middle"]

    for strategy in strategies:
        if strategy in ["saddle_balanced", "saddle_sim_range"]:
            result = select_reference_view(num_views, strategy, class_tokens)
        else:
            result = select_reference_view(num_views, strategy)

        assert 0 <= result.selected_index < num_views
        assert result.strategy.value == strategy
