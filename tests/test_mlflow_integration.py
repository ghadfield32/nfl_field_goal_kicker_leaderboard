
"""Tests for MLflow integration modules."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlops.experiment import setup_mlflow_experiment
from mlops.training import (
    load_and_prepare_iris_data, 
    train_logistic_regression
)
from mlops.model_registry import load_model_from_run


def test_data_loading():
    """Test that data loading works correctly."""
    data = load_and_prepare_iris_data()
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = data
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(feature_names) == 4
    assert len(target_names) == 3
    assert X_train.shape[1] == 4  # 4 features


def test_experiment_setup():
    """Test that MLflow experiment setup works."""
    # This should not raise an exception
    setup_mlflow_experiment("test_experiment")
    

def test_model_training_and_loading():
    """Test end-to-end model training and loading."""
    # Load data
    data = load_and_prepare_iris_data()
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = data
    
    # Train a simple model
    run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test,
        feature_names, target_names,
        run_name="test_lr",
        register=False  # Don't register for tests
    )
    
    assert run_id is not None
    assert len(run_id) > 0
    
    # Load the model back
    model = load_model_from_run(run_id, "model")
    
    # Test prediction
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    
    # Check accuracy is reasonable (should be > 0.8 for iris)
    accuracy = (predictions == y_test).mean()
    assert accuracy > 0.8


if __name__ == "__main__":
    # Run tests
    test_data_loading()
    print("✓ Data loading test passed")
    
    test_experiment_setup()
    print("✓ Experiment setup test passed")
    
    test_model_training_and_loading()
    print("✓ Model training and loading test passed")
    
    print("\nAll tests passed! 🎉") 
