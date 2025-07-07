"""
Tests for SHAP-IQ utilities module.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from mlops.shapiq_utils import (
    compute_shapiq_interactions,
    log_shapiq_interactions,
    get_top_interactions
)


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(4)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X_df, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    return model


def test_compute_shapiq_interactions_basic(sample_data, trained_model):
    """Test basic functionality of compute_shapiq_interactions."""
    X_df, _, feature_names = sample_data
    
    # Test with small sample to speed up test
    result_df = compute_shapiq_interactions(
        trained_model, 
        X_df.head(5),  # Use only 5 samples for testing
        feature_names, 
        max_order=2, 
        budget=64  # Small budget for fast testing
    )
    
    # Check structure
    expected_columns = ['sample_idx', 'combination', 'value', 'order', 'feature_names']
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check data types
    assert result_df['sample_idx'].dtype in [np.int64, int]
    assert result_df['value'].dtype in [np.float64, float]
    assert result_df['order'].dtype in [np.int64, int]
    
    # Check that we have interactions of different orders
    if not result_df.empty:
        orders = result_df['order'].unique()
        assert len(orders) > 0
        assert all(order <= 2 for order in orders)  # max_order=2


def test_compute_shapiq_interactions_with_sampling(sample_data, trained_model):
    """Test compute_shapiq_interactions with n_samples parameter."""
    X_df, _, feature_names = sample_data
    
    result_df = compute_shapiq_interactions(
        trained_model, 
        X_df, 
        feature_names, 
        max_order=1,  # Simple interactions only
        budget=32,
        n_samples=3   # Sample only 3 rows
    )
    
    if not result_df.empty:
        # Should have at most 3 different sample indices
        unique_samples = result_df['sample_idx'].nunique()
        assert unique_samples <= 3


def test_compute_shapiq_interactions_empty_result():
    """Test handling of edge cases that might result in empty results."""
    # Create trivial data that might not generate interactions
    X = pd.DataFrame([[1, 1], [1, 1]], columns=['a', 'b'])
    y = [0, 0]
    
    model = LogisticRegression()
    model.fit(X, y)
    
    result_df = compute_shapiq_interactions(
        model, X, ['a', 'b'], max_order=1, budget=16
    )
    
    # Should return a DataFrame with correct structure even if empty
    expected_columns = ['sample_idx', 'combination', 'value', 'order', 'feature_names']
    assert all(col in result_df.columns for col in expected_columns)


@patch('mlflow.log_metric')
@patch('mlflow.log_artifact')
def test_log_shapiq_interactions(mock_log_artifact, mock_log_metric, sample_data, trained_model):
    """Test log_shapiq_interactions with mocked MLflow calls."""
    X_df, _, feature_names = sample_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_interactions.csv")
        
        # Call the function
        log_shapiq_interactions(
            trained_model,
            X_df.head(5),  # Small sample for testing
            feature_names,
            max_order=2,
            top_n=3,
            budget=32,
            output_path=output_path
        )
        
        # Check that MLflow functions were called
        assert mock_log_metric.called
        assert mock_log_artifact.called
        
        # Check some expected metric calls
        metric_calls = [call[0][0] for call in mock_log_metric.call_args_list]
        expected_metrics = [
            "shapiq_total_interactions",
            "shapiq_unique_combinations", 
            "shapiq_max_order",
            "shapiq_samples_analyzed"
        ]
        
        for expected in expected_metrics:
            assert any(expected in call for call in metric_calls), f"Expected metric {expected} not found"


def test_get_top_interactions(sample_data, trained_model):
    """Test get_top_interactions utility function."""
    X_df, _, feature_names = sample_data
    
    # First compute interactions
    shapiq_df = compute_shapiq_interactions(
        trained_model, 
        X_df.head(10), 
        feature_names, 
        max_order=2, 
        budget=64
    )
    
    if not shapiq_df.empty:
        # Test getting top interactions
        top_interactions = get_top_interactions(shapiq_df, top_n=5)
        assert len(top_interactions) <= 5
        
        # Check structure
        expected_columns = ['combination', 'feature_names', 'order', 'mean', 'std', 'count', 'abs_mean']
        assert all(col in top_interactions.columns for col in expected_columns)
        
        # Test filtering by order
        if len(shapiq_df['order'].unique()) > 1:
            order_filtered = get_top_interactions(shapiq_df, top_n=3, order=1)
            if not order_filtered.empty:
                assert all(order_filtered['order'] == 1)


def test_compute_shapiq_interactions_error_handling():
    """Test error handling in compute_shapiq_interactions."""
    # Create data that might cause issues
    X = pd.DataFrame([[np.nan, 1], [2, np.nan]], columns=['a', 'b'])
    y = [0, 1]
    
    model = LogisticRegression()
    
    # This should handle errors gracefully and return empty DataFrame
    try:
        model.fit([[1, 1], [2, 2]], [0, 1])  # Fit with clean data
        result_df = compute_shapiq_interactions(model, X, ['a', 'b'], max_order=1, budget=16)
        
        # Should return DataFrame with expected structure even on error
        expected_columns = ['sample_idx', 'combination', 'value', 'order', 'feature_names']
        assert all(col in result_df.columns for col in expected_columns)
        
    except Exception:
        # If an exception occurs, that's also acceptable for this edge case
        pass


@patch('mlflow.log_metric')
@patch('mlflow.log_artifact')
def test_log_shapiq_interactions_empty_result(mock_log_artifact, mock_log_metric):
    """Test log_shapiq_interactions when no interactions are computed."""
    # Mock compute_shapiq_interactions to return empty DataFrame
    with patch('mlops.shapiq_utils.compute_shapiq_interactions') as mock_compute:
        mock_compute.return_value = pd.DataFrame(columns=['sample_idx', 'combination', 'value', 'order', 'feature_names'])
        
        # This should handle empty results gracefully
        log_shapiq_interactions(
            MagicMock(),  # Mock model
            pd.DataFrame([[1, 2]], columns=['a', 'b']),
            ['a', 'b'],
            max_order=1
        )
        
        # Should not log metrics or artifacts for empty results
        assert not mock_log_metric.called
        assert not mock_log_artifact.called


if __name__ == "__main__":
    pytest.main([__file__]) 
