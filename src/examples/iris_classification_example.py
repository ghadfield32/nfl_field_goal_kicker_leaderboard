#!/usr/bin/env python3
"""
Iris Classification Example (argparse-free, notebook-safe).

Configuration:
    • export EXPLAINER_DASHBOARD=1   # launch dashboard
    • export EXPLAINER_PORT=8150     # optional port override
"""

from __future__ import annotations
import os, logging

from src.mlops.utils import ensure_src_on_path
ensure_src_on_path()

from src.mlops.training import (
    load_and_prepare_iris_data,
    train_logistic_regression_autolog,
    train_random_forest_with_optimization,
    compare_models,
)
from src.mlops.model_registry import load_model_from_run
from src.mlops.experiment_utils import get_best_run

logging.basicConfig(level=logging.INFO)


def _bool_env(var: str, default: bool = False) -> bool:
    v = os.getenv(var)
    return default if v is None else v.lower() in {"1", "true", "yes"}


def main(*, dashboard: bool = False, dashboard_port: int | None = None) -> None:
    print("🌸 Iris Classification with MLflow\n" + "=" * 50)

    # 1 Load data ------------------------------------------------------------
    X_train, X_test, y_train, y_test, feat_names, tgt_names, _ = (
        load_and_prepare_iris_data()
    )
    print(f"✓ Training samples: {len(X_train)} | Test: {len(X_test)}")

    # 2 Logistic Regression --------------------------------------------------
    lr_run = train_logistic_regression_autolog(
        X_train,
        y_train,
        X_test,
        y_test,
        feat_names,
        tgt_names,
        run_name="lr_baseline",
        register=True,
        dashboard=dashboard,
        dashboard_port=dashboard_port,
    )
    print(f"✓ Logistic run {lr_run[:8]}")

    # 3 Random Forest + Optuna ----------------------------------------------
    rf_run = train_random_forest_with_optimization(
        X_train,
        y_train,
        X_test,
        y_test,
        feat_names,
        tgt_names,
        n_trials=20,
        run_name="rf_optimized",
        register=True,
        dashboard=dashboard,
        dashboard_port=dashboard_port,
    )
    print(f"✓ RF run {rf_run[:8]}")

    # 4 Compare & test best --------------------------------------------------
    compare_models()
    best = get_best_run()
    mdl = load_model_from_run(best["run_id"])
    acc = (mdl.predict(X_test) == y_test).mean()
    print(f"🏆 Best model accuracy: {acc:.4f}")

    if dashboard:
        port = dashboard_port or int(os.getenv("EXPLAINER_PORT", "8050"))
        print(f"\n🚀 ExplainerDashboard running on http://localhost:{port}")


if __name__ == "__main__":
    main(
        dashboard=_bool_env("EXPLAINER_DASHBOARD", False),
        dashboard_port=int(os.getenv("EXPLAINER_PORT", "8050")),
    )
