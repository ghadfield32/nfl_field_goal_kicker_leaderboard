"""Central MLflow configuration for consistent experiment tracking."""
import os

# ─── MLflow configuration ──────────────────────────────────────────────────
# Use Docker service-name so this works inside the compose network
# Falls back to local file store for standalone usage
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "iris_classification"
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

# ─── Model registry ────────────────────────────────────────────────────────
MODEL_NAME = "iris_classifier"
MODEL_STAGE_PRODUCTION = "Production"
MODEL_STAGE_STAGING = "Staging"

# ─── Dataset defaults ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2 

