"""
SHAP-IQ (Shapley Interaction) utilities for MLflow integration.

This module provides functions to compute and log Shapley interaction values
for machine learning models. Shapley interactions help understand how features
work together to influence model predictions.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import mlflow
from shapiq import TabularExplainer
from typing import Optional, Sequence, Union
import logging

logger = logging.getLogger(__name__)


def compute_shapiq_interactions(
    model,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    max_order: int = 2,
    budget: int = 256,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Robust wrapper around shapiq.TabularExplainer to return a tidy DataFrame
    with Shapley-interaction values.  Handles the two public APIs:
      •  .dict_values   (mapping)
      •  .values        (np.ndarray)  →  use  .to_dict()
    """
    logger.info(
        "Computing SHAP-IQ (max_order=%s, budget=%s, n_samples=%s)",
        max_order,
        budget,
        n_samples,
    )

    X_sample = (
        X.sample(n=n_samples, random_state=42) if n_samples and len(X) > n_samples else X
    )

    explainer = TabularExplainer(
        model=model,
        data=X_sample.values,
        index="k-SII",
        max_order=max_order,
    )

    rows: list[dict[str, Any]] = []
    for i, vec in enumerate(X_sample.values):
        try:
            iv = explainer.explain(vec, budget=budget)

            # --- unify both APIs ------------------------------------------------
            if hasattr(iv, "dict_values"):                    # shapiq ≥ 0.4
                items = iv.dict_values.items()
            elif hasattr(iv, "to_dict"):                      # fallback
                items = iv.to_dict().items()
            else:
                # last resort – try attribute access
                items = dict(iv.values).items()

            for combo, val in items:
                rows.append(
                    {
                        "sample_idx": i,
                        "combination": combo,
                        "value": float(val),
                        "order": len(combo),
                        "feature_names": tuple(feature_names[j] for j in combo)
                        if combo
                        else (),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP-IQ failed on sample %s: %s", i, exc)

    df = pd.DataFrame(rows)
    logger.info("✓ %s interaction rows computed", len(df))
    return df



def log_shapiq_interactions(
    model,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    max_order: int = 2,
    top_n: int = 10,
    budget: int = 256,
    n_samples: Optional[int] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Compute Shapley interaction values and log them to MLflow.

    This function:
    1. Computes interactions using compute_shapiq_interactions
    2. Logs the top N interactions as MLflow metrics
    3. Saves the full interaction table as CSV and logs as artifact

    Args:
        model: Trained sklearn-like model.
        X: DataFrame of features.
        feature_names: List of feature column names.
        max_order: Maximum interaction order (default: 2).
        top_n: Number of top interactions to log as metrics (default: 10).
        budget: Evaluation budget for interaction approximation (default: 256).
        n_samples: If provided, sample this many rows for computation.
        output_path: Optional path for CSV output (default: "shapiq_interactions.csv").
    """
    logger.info("Starting SHAP-IQ interaction logging")
    
    # Compute interactions
    df = compute_shapiq_interactions(
        model, X, feature_names, max_order, budget, n_samples
    )
    
    if df.empty:
        logger.warning("No interactions computed - skipping logging")
        return
    
    # Aggregate: mean absolute value per combination across all samples
    agg = (
        df.groupby(['combination', 'feature_names', 'order'])['value']
          .apply(lambda x: x.abs().mean())
          .reset_index()
          .sort_values('value', ascending=False)
    )
    
    # Log summary statistics
    mlflow.log_metric("shapiq_total_interactions", len(df))
    mlflow.log_metric("shapiq_unique_combinations", len(agg))
    mlflow.log_metric("shapiq_max_order", max_order)
    mlflow.log_metric("shapiq_samples_analyzed", len(X) if n_samples is None else min(n_samples, len(X)))
    
    # Log top N interactions as metrics
    logger.info(f"Logging top {top_n} interactions as MLflow metrics")
    for idx, row in agg.head(top_n).iterrows():
        combo = row['combination']
        feature_combo = row['feature_names'] 
        value = row['value']
        order = row['order']
        
        # Create metric name from feature names or indices
        if feature_combo:
            name = f"shapiq_order{order}_{'_x_'.join(feature_combo)}"
        else:
            name = f"shapiq_order{order}_{'_'.join(map(str, combo))}"
        
        # Sanitize metric name (MLflow has restrictions)
        name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')[:250]
        
        mlflow.log_metric(name, float(value))
    
    # Log order-specific summaries
    order_summary = df.groupby('order')['value'].agg(['count', 'mean', 'std']).fillna(0)
    for order_val in order_summary.index:
        mlflow.log_metric(f"shapiq_order{order_val}_count", order_summary.loc[order_val, 'count'])
        mlflow.log_metric(f"shapiq_order{order_val}_mean_abs", abs(order_summary.loc[order_val, 'mean']))
        if order_summary.loc[order_val, 'std'] > 0:
            mlflow.log_metric(f"shapiq_order{order_val}_std", order_summary.loc[order_val, 'std'])
    
    # Save and log full DataFrame as artifact
    output_file = output_path or "shapiq_interactions.csv"
    
    try:
        # Add readable feature names to the full DataFrame
        df_export = df.copy()
        df_export['feature_names_str'] = df_export['feature_names'].apply(lambda x: ' x '.join(x) if x else 'baseline')
        
        df_export.to_csv(output_file, index=False)
        mlflow.log_artifact(output_file)
        logger.info(f"Logged SHAP-IQ interactions artifact: {output_file}")
        
        # Also create and log a summary file
        summary_file = output_path.replace('.csv', '_summary.csv') if output_path else "shapiq_interactions_summary.csv"
        agg_export = agg.copy()
        agg_export['feature_names_str'] = agg_export['feature_names'].apply(lambda x: ' x '.join(x) if x else 'baseline')
        agg_export.to_csv(summary_file, index=False)
        mlflow.log_artifact(summary_file)
        logger.info(f"Logged SHAP-IQ summary artifact: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving SHAP-IQ artifacts: {e}")
    
    logger.info("SHAP-IQ interaction logging completed")


def get_top_interactions(
    shapiq_df: pd.DataFrame,
    top_n: int = 10,
    order: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract top interactions from a SHAP-IQ DataFrame.
    
    Args:
        shapiq_df: DataFrame returned by compute_shapiq_interactions.
        top_n: Number of top interactions to return.
        order: If provided, filter to interactions of this order only.
    
    Returns:
        DataFrame with top interactions, aggregated across samples.
    """
    df = shapiq_df.copy()
    
    if order is not None:
        df = df[df['order'] == order]
    
    if df.empty:
        return df
    
    # Aggregate and sort by absolute mean value
    agg = (
        df.groupby(['combination', 'feature_names', 'order'])['value']
          .agg(['mean', 'std', 'count'])
          .reset_index()
    )
    agg['abs_mean'] = agg['mean'].abs()
    agg = agg.sort_values('abs_mean', ascending=False)
    
    return agg.head(top_n)
