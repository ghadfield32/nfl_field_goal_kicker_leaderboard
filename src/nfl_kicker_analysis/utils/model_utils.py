"""
Model persistence utilities for NFL Kicker Analysis.

This module provides functions to save and load fitted models (Bayesian or scikit-learn)
to/from disk using joblib, avoiding Streamlit caching issues with un-picklable objects.
"""

from __future__ import annotations
import json
import hashlib
import datetime as _dt
from pathlib import Path
from typing import Any, Optional, Union, cast, Dict
import joblib  # fast, compressed persistence
import mlflow
import shutil
from mlflow.tracking import MlflowClient
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema
from mlflow.exceptions import MlflowException
from mlflow import sklearn as mlflow_sklearn
from src.nfl_kicker_analysis.config import config
# Ensure experiments directory or tracking server is initialized,
# and bind to a named experiment (creates it if missing).
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
from mlflow.models import infer_signature
from mlflow.pyfunc import load_model as mlflow_load_model
import pandas as pd
import numpy as np
import cloudpickle

# Use config paths instead of hardcoded ones
DEFAULT_DIR = config.MODELS_DIR / "mlruns" / "models"  # Point estimate models go here
MLRUNS_DIR = config.PROJECT_ROOT / "mlruns"

def cleanup_all_models():
    """
    Clean up all existing models and MLflow runs.
    """
    # ─── Helper for rmtree errors ───────────────────────────────────
    def _on_rm_error(func, path, exc_info):
        err = exc_info[1]
        if isinstance(err, OSError) and err.errno == 16:
            print(f"⚠️  Could not remove {path!r} (resource busy), skipping.")
        else:
            raise

    # ─── Remove local models directory ─────────────────────────────
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR, onerror=_on_rm_error)
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Remove MLflow runs directory ──────────────────────────────
    if MLRUNS_DIR.exists():
        try:
            shutil.rmtree(MLRUNS_DIR, onerror=_on_rm_error)
        except Exception as e:
            print(f"⚠️  Unexpected error removing mlruns: {e}")
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / ".trash").mkdir(parents=True, exist_ok=True)

    # ─── Delete registered models from MLflow registry ────────────
    registry_dir = MLRUNS_DIR / "models"
    if registry_dir.exists():
        client = MlflowClient()
        for rm in client.search_registered_models():
            client.delete_registered_model(rm.name)
    else:
        print(f"ℹ️  MLflow registry folder {registry_dir!r} not found; skipping registry cleanup.")


def get_best_metrics(name: str) -> dict[str, float] | None:
    """
    Get the best metrics for `name`:
      1. Try MLflow registry (preferred).
      2. If not found or MLflow error, scan local version folders for metrics.json.
      3. Otherwise return None.
    """
    client = MlflowClient()

    # 1) MLflow lookup
    try:
        versions = client.get_latest_versions(name)
        if versions:
            run_id = versions[0].run_id
            if run_id:
                run = client.get_run(run_id)
                return dict(run.data.metrics)
    except Exception:
        pass

    # 2) Local fallback: look inside version subdirectories
    model_dir = DEFAULT_DIR / name
    if model_dir.exists():
        # list timestamped subfolders
        version_dirs = sorted(
            [d for d in model_dir.iterdir() if d.is_dir()]
        )
        if version_dirs:
            latest = version_dirs[-1]
            meta_path = latest / "metrics.json"
            if meta_path.exists():
                try:
                    return json.load(meta_path.open("r"))
                except Exception:
                    pass

    # 3) Nothing found
    return None



def _timestamp() -> str:
    """Helper for version folders - returns current timestamp as string."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _hash_dict(d: dict) -> str:
    """Create stable 8-char SHA-1 hash of dict for cache keys."""
    raw = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()[:8]

def get_model_metadata(
    name: str,
    version: str | None = "latest",
    base_dir: Path = DEFAULT_DIR
) -> dict[str, Any]:
    """
    Load metadata for a saved model without loading the model itself.
    
    Args:
        name: Model name
        version: Specific version or "latest"
        base_dir: Base directory for model storage
        
    Returns:
        Metadata dictionary
    """
    model_dir = base_dir / name
    
    if version == "latest":
        version_dirs = sorted(model_dir.iterdir(), reverse=True)
        if not version_dirs:
            raise FileNotFoundError(f"No saved model found for '{name}'")
        model_dir = version_dirs[0]
    else:
        if version is None:
            raise ValueError("Version cannot be None when not 'latest'")
        model_dir = model_dir / version

    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return {}
        
    with meta_path.open("r") as fp:
        return json.load(fp)

def list_registered_models() -> dict[str, list[str]]:
    """
    Return a mapping of all MLflow-registered model names → list of version strings.
    """
    client = MlflowClient()
    out: dict[str, list[str]] = {}
    # search_registered_models is the correct method in recent MLflow releases
    for rm in client.search_registered_models():
        name = rm.name
        # get_latest_versions still works
        versions = client.get_latest_versions(name)
        out[name] = [v.version for v in versions]
    return out

def list_saved_models(base_dir: Path = DEFAULT_DIR) -> dict[str, list[str]]:
    """
    List all saved filesystem models and their version subdirectories.
    """
    if not base_dir.exists():
        return {}
    models = {}
    for d in base_dir.iterdir():
        if d.is_dir():
            versions = [v.name for v in d.iterdir() if v.is_dir()]
            if versions:
                models[d.name] = sorted(versions, reverse=True)
    return models

def _save_leaderboard(df: pd.DataFrame):
    """
    Persist the dynamic Bayesian leaderboard to disk.
    """
    out = df.reset_index().rename(columns={"index": "player_id"})
    out.to_csv(config.LEADERBOARD_FILE, index=False)

class _WrappedModel(PythonModel):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame
    ) -> pd.DataFrame:
        """
        context: MLflow-provided context (unused here)
        model_input: pandas DataFrame of features
        returns: pandas DataFrame or Series of predictions
        """
        # Delegate to the underlying model
        return pd.DataFrame(self._model.predict(model_input))

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from yaml.representer import RepresenterError

def save_model(
    model: Any,
    name: str,
    metrics: dict[str, float],
    meta: dict | None = None,
    register: bool = True,
    stage: str = "Staging"
) -> str | None:
    """
    Persist model to MLflow and register it if its metrics improve.
    Catches YAML serialization errors during stage transition.
    """
    if "accuracy" not in metrics:
        raise ValueError("Metrics must include 'accuracy'")

    best = get_best_metrics(name)
    if best and metrics["accuracy"] <= best.get("accuracy", 0):
        print(f"⚠️ New accuracy ({metrics['accuracy']:.4f}) ≤ best ({best['accuracy']:.4f}); skipping save.")
        return None

    try:
        with mlflow.start_run(nested=True) as run:
            mlflow.log_metrics(metrics)
            kwargs = {
                "artifact_path": name,
                "registered_model_name": None,
                "metadata": meta or {},
            }
            if hasattr(model, "predict") and "sklearn" in type(model).__module__:
                mlflow_sklearn.log_model(sk_model=model, **kwargs)
            else:
                mlflow.pyfunc.log_model(
                    python_model=_WrappedModel(model),
                    **kwargs
                )
            artifact_uri = f"runs:/{run.info.run_id}/{name}"
            print(f"✅ Logged model to {artifact_uri}")

            if register:
                result = mlflow.register_model(
                    model_uri=artifact_uri,
                    name=name
                )
                version = result.version
                print(f"✅ Registered model '{name}' as version {version}")

                client = MlflowClient()
                try:
                    client.transition_model_version_stage(
                        name, version, stage,
                        archive_existing_versions=False
                    )
                    uri = f"models:/{name}/{stage}"
                    print(f"✅ Transitioned version {version} to stage '{stage}'")
                    return uri
                except RepresenterError as rep_err:
                    print(f"⚠️ Could not serialize model-version metadata: {rep_err}; skipping stage transition.")
                    return f"models:/{name}/{version}"

            return artifact_uri

    except MlflowException as e:
        # Fallback to local filesystem
        ts = _timestamp()
        version_dir = DEFAULT_DIR / name / ts
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics.json
        metrics_path = version_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        path = version_dir / "model.joblib"
        joblib.dump(model, path)
        print(f"⚠️ MLflow error ({e}); saved locally to {path}")
        return str(path)

# ── New imports ─────────────────────────────────────────────
from mlflow.sklearn import load_model as sklearn_load_model
from mlflow.pyfunc import load_model as pyfunc_load_model
import yaml
# ──────────────────────────────────────────────────────────

def load_model(
    name: str,
    version: str | None = "latest",
    base_dir: Path = DEFAULT_DIR
) -> Any:
    """
    Load a saved model with enhanced MLflow registry support:
      1) Try the sklearn flavor (full API w/ predict_proba).
      2) Fallback to the pyfunc flavor (generic predict-only).
      3) Fallback to local filesystem using MLflow registry metadata.
      4) Final fallback to legacy local filesystem (joblib/cloudpickle).
    """
    model_uri = f"models:/{name}/{version or 'latest'}"
    
    print(f"[DEBUG] Loading model '{name}' (version: {version or 'latest'})")

    # 1. sklearn flavor
    try:
        model = sklearn_load_model(model_uri)
        print(f"[SUCCESS] Loaded sklearn model '{name}' from '{model_uri}'")
        return model
    except Exception as e:
        print(f"[WARNING] sklearn flavor load failed ({e}); trying pyfunc...")

    # 2. pyfunc flavor
    try:
        model = pyfunc_load_model(model_uri)
        print(f"[SUCCESS] Loaded pyfunc model '{name}' from '{model_uri}'")
        return model
    except Exception as e:
        print(f"[WARNING] pyfunc flavor load failed ({e}); trying MLflow registry metadata...")

    # 3. MLflow registry metadata fallback
    try:
        model = _load_model_from_registry_metadata(name, version, base_dir)
        if model is not None:
            print(f"[SUCCESS] Loaded model '{name}' from MLflow registry metadata")
            return model
    except Exception as e:
        print(f"[WARNING] MLflow registry metadata load failed ({e}); trying legacy filesystem...")

    # 4. Legacy filesystem fallback
    model_root = base_dir / name
    if not model_root.exists():
        raise FileNotFoundError(f"No model directory for '{name}' at {model_root}")

    # […existing local-loading logic…]
    versions = sorted(d for d in model_root.iterdir() if d.is_dir())
    if not versions:
        raise FileNotFoundError(f"No versions for '{name}' in {model_root}")
    model_dir = versions[-1]

    joblib_path = model_dir / "model.joblib"
    if joblib_path.exists():
        print(f"[SUCCESS] Loaded model '{name}' from legacy joblib: {joblib_path}")
        return joblib.load(joblib_path)

    pkl_path = model_dir / "model.pkl"
    if pkl_path.exists():
        import cloudpickle
        with open(pkl_path, "rb") as f:
            print(f"[SUCCESS] Loaded model '{name}' from legacy pickle: {pkl_path}")
            return cloudpickle.load(f)

    raise FileNotFoundError(f"No model file in {model_dir}. Expected 'model.joblib' or 'model.pkl'.")


def _load_model_from_registry_metadata(
    name: str,
    version: str | None = "latest",
    base_dir: Path = DEFAULT_DIR
) -> Any:
    """
    Load a model by reading MLflow registry metadata to find the actual storage location.
    
    Args:
        name: Model name
        version: Version (or "latest")
        base_dir: Base directory for model registry
        
    Returns:
        Loaded model or None if not found
    """
    # Find the model registry directory
    models_registry_dir = config.PROJECT_ROOT / "mlruns" / "models" / name
    if not models_registry_dir.exists():
        raise FileNotFoundError(f"No model registry directory for '{name}' at {models_registry_dir}")
    
    # Find version directories
    version_dirs = [d for d in models_registry_dir.iterdir() if d.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"No version directories for '{name}' in {models_registry_dir}")
    
    # Select version directory
    if version == "latest" or version is None:
        # Sort by version number (extract number from version-X)
        version_dirs.sort(key=lambda d: int(d.name.split('-')[1]) if '-' in d.name else 0, reverse=True)
        version_dir = version_dirs[0]
    else:
        version_dir = models_registry_dir / f"version-{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory '{version_dir}' not found")
    
    print(f"[DEBUG] Reading metadata from: {version_dir}")
    
    # Read metadata
    meta_file = version_dir / "meta.yaml"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    with open(meta_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Extract storage location
    storage_location = metadata.get('storage_location', '')
    if not storage_location:
        raise ValueError(f"No storage_location found in metadata for '{name}'")
    
    print(f"[DEBUG] Storage location: {storage_location}")
    
    # Parse storage location to find artifacts
    if storage_location.startswith('file://'):
        # Remove file:// prefix and handle path conversion
        artifacts_path = storage_location[7:]  # Remove 'file://'
        
        # Convert workspace path to actual path
        if artifacts_path.startswith('/workspace/'):
            # Replace /workspace/ with the actual project root
            relative_path = artifacts_path[11:]  # Remove '/workspace/'
            artifacts_path = config.PROJECT_ROOT / relative_path
        else:
            # Handle other file:// paths
            artifacts_path = Path(artifacts_path)
        
        print(f"[DEBUG] Artifacts path: {artifacts_path}")
        
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")
        
        # Try to load the model from artifacts
        # First try model.pkl (most common for sklearn models)
        pkl_path = artifacts_path / "model.pkl"
        if pkl_path.exists():
            print(f"[DEBUG] Loading from: {pkl_path}")
            import cloudpickle
            with open(pkl_path, "rb") as f:
                return cloudpickle.load(f)
        
        # Then try model.joblib
        joblib_path = artifacts_path / "model.joblib"
        if joblib_path.exists():
            print(f"[DEBUG] Loading from: {joblib_path}")
            return joblib.load(joblib_path)
        
        # Check what files are actually available
        available_files = list(artifacts_path.iterdir())
        print(f"[WARNING] Available files in artifacts: {[f.name for f in available_files]}")
        
        raise FileNotFoundError(f"No model file found in {artifacts_path}. Expected 'model.pkl' or 'model.joblib'")
    
    else:
        raise ValueError(f"Unsupported storage location format: {storage_location}")

from mlflow.tracking import MlflowClient

def get_best_model_info(name: str) -> tuple[str, float | None]:
    """
    Return the latest registered version and its accuracy for `name`.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(name)
    if not versions:
        raise ValueError(f"No registered versions found for '{name}'")
    latest_ver = versions[0].version
    metrics = get_best_metrics(name)
    acc = metrics.get("accuracy") if metrics else None
    return latest_ver, acc



if __name__ == "__main__":
    # Clean up existing models first
    cleanup_all_models()
    print("🧹 Cleaned up existing models and MLflow runs")
    # # Re-initialize MLflow now that mlruns/ is fresh
    # mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    import numpy as np
    import pymc as pm

    # 1️⃣ Dummy test with metrics
    class Dummy:
        def __init__(self, x):
            self.x = x
            
        def predict(self, X):
            return np.array([self.x] * len(X))

    dummy = Dummy(42)
    uri = save_model(dummy, name="dummy_model", metrics=config.metrics)
    
    # Try saving a worse model - should be rejected
    dummy2 = Dummy(43)
    worse_metrics = {"accuracy": 0.80, "f1": 0.78}
    uri2 = save_model(dummy2, name="dummy_model", metrics=worse_metrics)
    assert uri2 is None, "Worse model should have been rejected"
    
    # Try saving a better model - should be accepted
    dummy3 = Dummy(44)
    better_metrics = {"accuracy": 0.90, "f1": 0.88}
    uri3 = save_model(dummy3, name="dummy_model", metrics=better_metrics)
    assert uri3 is not None, "Better model should have been accepted"
    
    loaded_dummy = load_model("dummy_model")
    #assert isinstance(loaded_dummy, Dummy)
    print("✅ Dummy save/load with metrics passed!")

    # 2️⃣ LogisticRegression test with metrics
    from sklearn.linear_model import LogisticRegression
    X, y = [[0,0],[1,1],[1,0],[0,1]], [0,1,1,0]
    model = LogisticRegression().fit(X, y)
    metrics = {"accuracy": 0.75, "f1": 0.73}
    save_model(model, name="logreg_test", metrics=metrics)
    reloaded = load_model("logreg_test")
    assert (reloaded.predict(X) == model.predict(X)).all()
    print("✅ LogisticRegression save/load with metrics passed!")

    # 3️⃣ RandomForestClassifier test with metrics
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    metrics = {"accuracy": 0.85, "f1": 0.83}
    save_model(rf, name="rf_test", metrics=metrics)
    rf2 = load_model("rf_test")
    assert (rf2.predict(X) == rf.predict(X)).all()
    print("✅ RandomForestClassifier save/load with metrics passed!")


