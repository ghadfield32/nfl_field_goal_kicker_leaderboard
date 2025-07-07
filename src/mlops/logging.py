"""
Extended MLflow logging helpers.
"""
from __future__ import annotations
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Sequence, Optional
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
)


def _log_fig(fig: Figure, name: str) -> None:
    """Log a Matplotlib figure directly without temp files."""
    mlflow.log_figure(fig, artifact_file=name)
    plt.close(fig)


def log_full_metrics(
    y_true, y_pred, *, label_list: Optional[Sequence[int]] = None, prefix: str = ""
) -> Dict[str, float]:
    """
    Compute & log *all* useful classification metrics.

    Returns a flat dict so callers can unit-test easily.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        label_list: Optional list of label integers (for compatibility)
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of all calculated metrics
    """
    # (1) macro metrics ------------------------------------------------------
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division="warn"
    )
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
    }

    # (2) per-class ----------------------------------------------------------
    report = classification_report(y_true, y_pred, output_dict=True, zero_division="warn")
    if isinstance(report, dict):
        for klass, d in report.items():
            if isinstance(klass, str) and klass.isdigit():  # skip 'accuracy', 'macro avg', …
                k = int(klass)
                if isinstance(d, dict):
                    precision_val = d.get("precision", 0.0)
                    recall_val = d.get("recall", 0.0)
                    f1_val = d.get("f1-score", 0.0)
                    support_val = d.get("support", 0.0)
                    
                    metrics[f"precision_{k}"] = float(precision_val) if precision_val is not None else 0.0
                    metrics[f"recall_{k}"] = float(recall_val) if recall_val is not None else 0.0
                    metrics[f"f1_{k}"] = float(f1_val) if f1_val is not None else 0.0
                    metrics[f"support_{k}"] = float(support_val) if support_val is not None else 0.0

    # (3) derived – try/except so we never crash ----------------------------
    try:
        metrics["roc_auc_ovr_weighted"] = roc_auc_score(
            y_true, pd.get_dummies(y_pred), multi_class="ovr", average="weighted"
        )
    except Exception:
        pass
    try:
        metrics["log_loss"] = log_loss(y_true, pd.get_dummies(y_pred))
    except Exception:
        pass
    try:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        pass

    # (4) optional prefix for nested CV, etc. -------------------------------
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    mlflow.log_metrics(metrics)
    return metrics


def log_confusion_matrix(
    y_true, y_pred, *, class_names: Optional[Sequence[str]] = None, artifact_name: str = "confusion_matrix.png"
) -> None:
    """Create + log confusion matrix using mlflow.log_figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    _log_fig(fig, artifact_name)


def log_feature_importance(
    feature_names: list, importances: list, artifact_name: str = "feature_importance.png"
):
    """Bar plot logged via mlflow.log_figure (no disk I/O)."""
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importances")
    _log_fig(fig, artifact_name)


def log_parameters(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter names and values
    """
    mlflow.log_params(params)


def log_dataset_info(X_train, X_test, y_train, y_test) -> None:
    """
    Log dataset information as parameters.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    dataset_params = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": (X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])),
        "n_classes": (len(set(y_train)) if hasattr(y_train, "__iter__") else 1),
    }

    log_parameters(dataset_params)


# Legacy compatibility - keep old function name as alias
log_model_metrics = log_full_metrics 
