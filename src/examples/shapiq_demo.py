#!/usr/bin/env python3
"""
SHAP-IQ Integration Demo

This script demonstrates the new SHAP-IQ (Shapley Interaction) functionality
integrated into the MLOps pipeline. It shows how Shapley interaction values
are computed and logged alongside regular model metrics.

Usage:
    python src/examples/shapiq_demo.py
"""

from __future__ import annotations
import logging

# ─── Path setup ─────────────────────────────────────────────────────────────
from src.mlops.utils import add_project_root_to_sys_path
PROJECT_ROOT = add_project_root_to_sys_path(levels_up=2)  # safe in both .py and interactive :contentReference[oaicite:8]{index=8}

# ─── Imports ────────────────────────────────────────────────────────────────
from src.mlops.training import (
    load_and_prepare_iris_data,
    train_logistic_regression,
    train_random_forest_optimized
)
from src.mlops.shapiq_utils import (
    compute_shapiq_interactions,
    log_shapiq_interactions,
    get_top_interactions
)
from src.mlops.experiment_utils import setup_mlflow_experiment, get_best_run
import mlflow
import pandas as pd

# ─── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_standalone_shapiq():
    """Demonstrate standalone SHAP-IQ computation without MLflow logging."""
    print("🔬 SHAP-IQ Standalone Demo")
    print("=" * 50)
    
    # Load data and train a simple model
    X_train, X_test, y_train, y_test, feature_names, target_names, _ = load_and_prepare_iris_data()
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"✓ Trained RandomForest on {len(X_train)} samples")
    print(f"✓ Test accuracy: {model.score(X_test, y_test):.3f}")
    
    # Compute SHAP-IQ interactions
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    print(f"\n🧮 Computing SHAP-IQ interactions...")
    
    shapiq_df = compute_shapiq_interactions(
        model, 
        X_test_df.head(10),  # Use subset for demo
        feature_names, 
        max_order=2,
        budget=128
    )
    
    if not shapiq_df.empty:
        print(f"✓ Computed {len(shapiq_df)} interaction values")
        
        # Show top interactions
        top_interactions = get_top_interactions(shapiq_df, top_n=5)
        print(f"\n🏆 Top 5 Feature Interactions:")
        print("-" * 60)
        
        for idx, row in top_interactions.iterrows():
            feature_combo = ' × '.join(row['feature_names'])
            if not feature_combo:
                feature_combo = "baseline"
            print(f"  {feature_combo:30} | Order {row['order']} | {row['abs_mean']:.4f}")
        
        # Show order breakdown
        order_counts = shapiq_df['order'].value_counts().sort_index()
        print(f"\n📊 Interaction Order Breakdown:")
        for order, count in order_counts.items():
            if order == 0:
                print(f"  Order {order} (main effects):     {count:4d} values")
            elif order == 1:
                print(f"  Order {order} (individual):       {count:4d} values")
            elif order == 2:
                print(f"  Order {order} (pairwise):         {count:4d} values")
            else:
                print(f"  Order {order} (higher-order):     {count:4d} values")
    else:
        print("⚠️  No interactions computed (this can happen with simple models/data)")


def demo_integrated_training():
    """Demonstrate SHAP-IQ integration in the training pipeline."""
    print("\n\n🚀 SHAP-IQ Integrated Training Demo") 
    print("=" * 50)
    
    # Setup MLflow experiment
    setup_mlflow_experiment("shapiq_demo")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names, _ = load_and_prepare_iris_data()
    print(f"✓ Loaded Iris dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model with SHAP-IQ integration
    print(f"\n🤖 Training Logistic Regression with SHAP-IQ...")
    lr_run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test, 
        feature_names, target_names,
        run_name="lr_with_shapiq"
    )
    print(f"✓ Logistic Regression complete: {lr_run_id[:8]}")
    
    print(f"\n🌲 Training Random Forest with SHAP-IQ...")
    rf_run_id = train_random_forest_optimized(
        X_train, y_train, X_test, y_test,
        feature_names, target_names,
        n_trials=10,  # Reduced for demo
        run_name="rf_with_shapiq"
    )
    print(f"✓ Random Forest complete: {rf_run_id[:8]}")
    
    # Show logged SHAP-IQ metrics
    print(f"\n📊 SHAP-IQ Metrics from MLflow:")
    print("-" * 50)
    
    try:
        # Get the latest run (Random Forest)
        with mlflow.start_run(run_id=rf_run_id):
            run_data = mlflow.get_run(rf_run_id)
            metrics = run_data.data.metrics
            
            # Filter SHAP-IQ metrics
            shapiq_metrics = {k: v for k, v in metrics.items() if k.startswith('shapiq_')}
            
            if shapiq_metrics:
                print(f"Found {len(shapiq_metrics)} SHAP-IQ metrics:")
                for metric, value in sorted(shapiq_metrics.items()):
                    if 'order' in metric and 'count' not in metric:
                        print(f"  {metric:35} = {value:.6f}")
                    elif 'total' in metric or 'unique' in metric or 'max' in metric:
                        print(f"  {metric:35} = {int(value)}")
            else:
                print("  No SHAP-IQ metrics found (may take longer to compute)")
                
    except Exception as e:
        print(f"  Error retrieving metrics: {e}")
    
    # Compare models
    print(f"\n🏆 Comparing Models:")
    print("-" * 30)
    try:
        best_run = get_best_run("accuracy", maximize=True)
        run_id = best_run["run_id"]
        accuracy = best_run.get("metrics.accuracy", "N/A")
        print(f"Best model: {run_id[:8]} (accuracy: {accuracy})")
        
        # Check if SHAP-IQ metrics are available for best model
        shapiq_count = best_run.get("metrics.shapiq_total_interactions")
        if shapiq_count:
            print(f"SHAP-IQ interactions: {int(shapiq_count)} computed")
        
    except Exception as e:
        print(f"Error comparing models: {e}")


def demo_manual_shapiq_logging():
    """Demonstrate manual SHAP-IQ logging outside of training."""
    print(f"\n\n🔧 Manual SHAP-IQ Logging Demo")
    print("=" * 50)
    
    # Load data and train model
    X_train, X_test, y_train, y_test, feature_names, target_names, _ = load_and_prepare_iris_data()
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Manual MLflow run with SHAP-IQ logging
    setup_mlflow_experiment("shapiq_demo") 
    
    with mlflow.start_run(run_name="manual_shapiq_demo"):
        # Log basic metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log SHAP-IQ interactions
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        print("Computing and logging SHAP-IQ interactions...")
        
        log_shapiq_interactions(
            model, 
            X_test_df,
            feature_names,
            max_order=2,
            top_n=5,
            budget=64,
            n_samples=15  # Sample for faster computation
        )
        
        current_run = mlflow.active_run()
        print(f"✓ SHAP-IQ logged to run: {current_run.info.run_id[:8]}")


def main():
    """Run all SHAP-IQ demos."""
    print("🌟 SHAP-IQ Integration Demonstration")
    print("=" * 60)
    print("This demo shows how Shapley interactions are computed and logged")
    print("in the MLOps pipeline to understand feature interactions.")
    print()
    
    try:
        # Demo 1: Standalone computation
        demo_standalone_shapiq()
        
        # Demo 2: Integrated training
        demo_integrated_training()
        
        # Demo 3: Manual logging
        demo_manual_shapiq_logging()
        
        print(f"\n\n🎉 SHAP-IQ Demo Complete!")
        print("=" * 60)
        print("✓ Standalone SHAP-IQ computation")
        print("✓ Integrated training with automatic SHAP-IQ logging") 
        print("✓ Manual SHAP-IQ logging")
        print()
        print("🔍 Check MLflow UI to see logged SHAP-IQ metrics and artifacts:")
        print("   - Metrics: shapiq_order1_*, shapiq_order2_*, etc.")
        print("   - Artifacts: shapiq_interactions.csv, shapiq_interactions_summary.csv")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("This might be due to SHAP-IQ dependency issues or data problems.")


if __name__ == "__main__":
    main() 
