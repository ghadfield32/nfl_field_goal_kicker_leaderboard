
"""Training utilities with MLflow integration."""
import mlflow
import optuna
from typing import Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .config import RANDOM_STATE, TEST_SIZE
from .experiment_utils import setup_mlflow_experiment

# Re-export for convenience
__all__ = ['setup_mlflow_experiment', 'load_and_prepare_iris_data',
           'train_logistic_regression', 'train_random_forest_with_optimization']
from .logging import (
    log_model_metrics,
    log_confusion_matrix,
    log_feature_importance,
    log_dataset_info,
    log_parameters
)


# ─────────────────────────── src/mlops/training.py (excerpt) ───────────────
def load_and_prepare_iris_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> DatasetTuple:
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ✅ re-wrap as DataFrame so feature names propagate downstream
    import pandas as pd
    feat_names = iris.feature_names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feat_names)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=feat_names)

    return (X_train_df, X_test_df, y_train, y_test,
            feat_names, list(iris.target_names), scaler)

