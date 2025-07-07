"""Training utilities with MLflow integration."""
import mlflow
from mlflow import sklearn  # type: ignore
from mlflow import models  # type: ignore
import optuna
from optuna.integration.mlflow import MLflowCallback
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Callable, cast, Any, Dict, TypeAlias
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch

from .config import RANDOM_STATE, TEST_SIZE
from .experiment_utils import setup_mlflow_experiment
from .logging import (
    log_full_metrics,
    log_confusion_matrix,
    log_feature_importance,
    log_dataset_info,
    log_parameters
)
from .shapiq_utils import log_shapiq_interactions

from src.mlops.cleanup import prune_experiment, prune_model_versions

# Type aliases for complex types
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
DatasetTuple: TypeAlias = Tuple[FloatArray, FloatArray, IntArray, IntArray, List[str], List[str], StandardScaler]


def load_and_prepare_iris_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> DatasetTuple:
    """
    Load and prepare the Iris dataset.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, 
                 feature_names, target_names, scaler)
    """
    # Load dataset
    iris: Any = load_iris()
    X: NDArray[np.float64] = cast(NDArray[np.float64], iris.data)
    y: NDArray[np.int64] = cast(NDArray[np.int64], iris.target)
    feature_names: List[str] = list(iris.feature_names)
    target_names: List[str] = list(iris.target_names)
    
    # Split data
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    X_train, X_test, y_train, y_test = cast(
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
        train_test_split(X, y, test_size=test_size, random_state=random_state)
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled: NDArray[np.float64] = cast(NDArray[np.float64], scaler.fit_transform(X_train))
    X_test_scaled: NDArray[np.float64] = cast(NDArray[np.float64], scaler.transform(X_test))
    
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            feature_names, target_names, scaler)


# === (A) LOGISTIC REGRESSION (training only, NO dashboard) ================
def train_logistic_regression(
    X_train, y_train, X_test, y_test, feature_names, target_names,
    *, run_name: str = "lr_baseline", register: bool = True
) -> str:
    """Train logistic regression model without dashboard integration."""
    setup_mlflow_experiment()
    mlflow.sklearn.autolog(log_models=True)
    
    with mlflow.start_run(run_name=run_name) as run:
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1_000).fit(
            X_train, y_train
        )

        y_pred = model.predict(X_test)
        log_full_metrics(y_test, y_pred)
        log_confusion_matrix(y_test, y_pred, class_names=target_names)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        sklearn.log_model(
            model, "model",
            registered_model_name="iris_logreg" if register else None,
            signature=signature, input_example=X_test[:5],
        )
        
        # SHAP-IQ: compute & log feature interaction values
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        log_shapiq_interactions(model, X_test_df, feature_names, max_order=2)
        
        return run.info.run_id


def _create_rf_objective(X_train, y_train, X_test, y_test) -> Callable[[optuna.trial.Trial], float]:
    """Create Optuna objective function for Random Forest optimization."""
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }
        m = RandomForestClassifier(**params).fit(X_train, y_train)
        return float(accuracy_score(y_test, m.predict(X_test)))
    return objective


# === (B) RANDOM-FOREST + Optuna (training only) ===========================
def train_random_forest_optimized(
    X_train, y_train, X_test, y_test, feature_names, target_names,
    *, n_trials: int = 50, run_name: str = "rf_optimized", register: bool = True
) -> str:
    """Train optimized Random Forest model without dashboard integration."""
    setup_mlflow_experiment()
    mlflow.sklearn.autolog(disable=True)        # Optuna will log

    with mlflow.start_run(run_name=run_name) as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(_create_rf_objective(X_train, y_train, X_test, y_test), n_trials=n_trials,
                       callbacks=[MLflowCallback(
                           tracking_uri=mlflow.get_tracking_uri(),
                           metric_name="accuracy", mlflow_kwargs={"nested": True}
                       )])

        best = RandomForestClassifier(**study.best_params).fit(X_train, y_train)
        y_pred = best.predict(X_test)
        log_full_metrics(y_test, y_pred)
        log_confusion_matrix(y_test, y_pred, class_names=target_names)
        log_feature_importance(feature_names, best.feature_importances_)
        mlflow.log_metric("best_accuracy", study.best_value)

        signature = mlflow.models.infer_signature(X_train, best.predict(X_train))
        sklearn.log_model(
            best, "model",
            registered_model_name="iris_random_forest" if register else None,
            signature=signature, input_example=X_test[:5],
        )
        
        # SHAP-IQ: compute & log feature interaction values
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        log_shapiq_interactions(best, X_test_df, feature_names, max_order=2)
        
        return run.info.run_id


# === (C) ONE-STOP helper: train both models ===============================
def run_all_trainings(*,
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE, n_trials: int = 50) -> None:
    """Train both logistic regression and random forest models."""
    X_tr, X_te, y_tr, y_te, feats, tgts, _ = load_and_prepare_iris_data(
        test_size, random_state
    )
    train_logistic_regression(
        X_tr, y_tr, X_te, y_te, feats, tgts, run_name="lr_baseline"
    )
    train_random_forest_optimized(
        X_tr, y_tr, X_te, y_te, feats, tgts,
        n_trials=n_trials, run_name="rf_optimized"
    )
    
    # Add pruning after training finishes
    prune_experiment("iris_classification", metric="accuracy", top_k=1)
    prune_model_versions("iris_classifier", metric="accuracy", top_k=1)


# === (D) Robust comparator ===============================================
def compare_models(
    experiment_name: Optional[str] = None,
    metric_key: str = "accuracy",
    maximize: bool = True,
) -> None:
    """
    Print the best run according to *metric_key* while gracefully
    falling-back to common alternates when the preferred key is missing.
    """
    from .experiment_utils import get_best_run

    fallback_keys = ["accuracy_score", "best_accuracy"]
    try:
        best = get_best_run(experiment_name, metric_key, maximize)
        rid = best["run_id"]

        # choose first key that exists
        score = best.get(f"metrics.{metric_key}")
        if score is None:
            for alt in fallback_keys:
                score = best.get(f"metrics.{alt}")
                if score is not None:
                    metric_key = alt
                    break

        model_type = best.get("params.model_type", "unknown")
        print(f"🏆 Best run: {rid}")
        print(f"📈 {metric_key}: {score if score is not None else 'N/A'}")
        print(f"🔖 Model type: {model_type}")
    except Exception as err:
        print(f"❌ Error comparing models: {err}")


# Legacy compatibility functions (with dashboard support)
train_logistic_regression_autolog = train_logistic_regression
train_random_forest_with_optimization = train_random_forest_optimized


if __name__ == "__main__":
    run_all_trainings()

