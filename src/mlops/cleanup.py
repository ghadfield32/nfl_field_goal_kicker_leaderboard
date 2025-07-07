# src/mlops/cleanup.py
from __future__ import annotations
import logging, shutil, pathlib
from typing import Sequence, Optional
import mlflow
from mlflow.tracking import MlflowClient

_logger = logging.getLogger(__name__)

def _runs_by_metric(
    experiment_id: str,
    metric: str = "accuracy",
    maximize: bool = True
) -> Sequence[mlflow.entities.Run]:
    client = MlflowClient()
    order = "DESC" if maximize else "ASC"
    return client.search_runs(
        [experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=10_000,  # enough for 99 % of cases
    )

def prune_experiment(
    experiment_name: str,
    metric: str = "accuracy",
    top_k: int = 1,
    ascending: bool = False
) -> None:
    """
    Prune an experiment by keeping only the top K runs based on a metric.
    
    Args:
        experiment_name: Name of the experiment to prune
        metric: Metric to sort by
        top_k: Number of top runs to keep
        ascending: Whether to sort in ascending order (default: False)
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment {experiment_name} not found")
        return
        
    # Get all runs for the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
    )
    
    # Keep top K runs
    runs_to_delete = runs[top_k:]
    
    # Delete the rest
    for run in runs_to_delete:
        client.delete_run(run.info.run_id)
        print(f"Deleted run {run.info.run_id} with {metric}={run.data.metrics.get(metric)}")

def prune_model_versions(
    model_name: str,
    metric: str = "accuracy",
    top_k: int = 1,
    ascending: bool = False
) -> None:
    """
    Prune model versions by keeping only the top K versions based on a metric.
    
    Args:
        model_name: Name of the registered model to prune
        metric: Metric to sort by
        top_k: Number of top versions to keep
        ascending: Whether to sort in ascending order (default: False)
    """
    client = MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        print(f"Model {model_name} not found")
        return
        
    # Get metrics for each version
    version_metrics = []
    for version in versions:
        try:
            if version.run_id is not None:  # Handle None run_id
                run = client.get_run(version.run_id)
                metric_value = run.data.metrics.get(metric)
                if metric_value is not None:
                    version_metrics.append((version, metric_value))
        except Exception:
            continue
            
    # Sort by metric
    version_metrics.sort(key=lambda x: x[1], reverse=not ascending)
    
    # Keep top K versions
    versions_to_delete = version_metrics[top_k:]
    
    # Delete the rest
    for version, metric_value in versions_to_delete:
        client.delete_model_version(model_name, version.version)
        print(f"Deleted version {version.version} with {metric}={metric_value}")

