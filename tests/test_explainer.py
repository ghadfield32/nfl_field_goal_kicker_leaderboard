import sys
import os
import pytest
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_yaml_roundtrip(tmp_path):
    """Test that a dashboard can be saved to YAML and reloaded."""
    from src.mlops.explainer import build_and_log_dashboard, load_dashboard_yaml
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import mlflow
    import pandas as pd

    iris = load_iris()
    X, y = iris.data, iris.target
    X_df = pd.DataFrame(X, columns=iris.feature_names)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    with mlflow.start_run():
        yaml_path = build_and_log_dashboard(
            model, X_df, y,
            serve=False,
            save_yaml=True,
            output_dir=tmp_path
        )
        # Reload
        dash = load_dashboard_yaml(yaml_path)
        assert dash.explainer.model.__class__.__name__ == "LogisticRegression"


def test_build_dashboard(tmp_path):
    """Test that a dashboard can be built and saved."""
    from src.mlops.explainer import build_and_log_dashboard
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import mlflow
    import pandas as pd

    iris = load_iris()
    X, y = iris.data, iris.target
    X_df = pd.DataFrame(X, columns=iris.feature_names)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    with mlflow.start_run():
        html = build_and_log_dashboard(
            model, X_df, y,
            serve=False,
            save_yaml=False,
            output_dir=tmp_path
        )
        assert html.exists() and html.suffix == ".html" 
