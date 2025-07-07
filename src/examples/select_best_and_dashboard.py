#!/usr/bin/env python3
"""
Select best model and launch dashboard (training is done elsewhere).

Usage:
    python src/examples/select_best_and_dashboard.py
"""

from __future__ import annotations

from src.mlops.utils import add_project_root_to_sys_path
PROJECT_ROOT = add_project_root_to_sys_path()

from src.mlops.experiment_utils import get_best_run
from src.mlops.model_registry import load_model_from_run
from src.mlops.explainer import dashboard_best_run

# Configuration variables
METRIC = "accuracy"  # Metric to optimize (e.g., 'accuracy', 'f1')
PORT = 8050           # Port for the dashboard
MAXIMIZE = True       # Whether to maximize (True) or minimize (False) the metric

def main() -> None:
    print(f"🔍 Searching MLflow runs by {METRIC}…")

    # Retrieve the best run based on the specified metric
    best = get_best_run(metric_key=METRIC, maximize=MAXIMIZE)
    run_id = best["run_id"]
    score = best.get(f"metrics.{METRIC}", "N/A")

    print(f"🏆 Best run: {run_id[:8]} — {METRIC}: {score}")

    # Load the model from the run registry
    model = load_model_from_run(run_id)
    if model is None:
        raise RuntimeError("Model could not be loaded from registry")

    print("✓ Model loaded – launching dashboard")
    # Launch the explainer dashboard for the best model
    dashboard_best_run(METRIC, maximize=MAXIMIZE, port=PORT)

if __name__ == "__main__":
    main()

