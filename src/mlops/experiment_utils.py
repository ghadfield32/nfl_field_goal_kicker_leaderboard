"""MLflow experiment utilities."""
import os
import pathlib
import mlflow
import mlflow.tracking
from typing import Optional, Dict, Any
import requests

from src.mlops.config import EXPERIMENT_NAME, TRACKING_URI

import re, shutil, logging

_HEALTH_ENDPOINTS = ("/health", "/version")
_hex32 = re.compile(r"^[0-9a-f]{32}$", re.I)
logger = logging.getLogger(__name__)

def _ping_tracking_server(uri: str, timeout: float = 2.0) -> bool:
    """Return True iff an HTTP MLflow server is reachable at *uri*."""
    if not uri.startswith("http"):
        return False                        # file store – nothing to ping
    try:
        # Use new health endpoints
        for ep in _HEALTH_ENDPOINTS:
            response = requests.get(uri.rstrip("/") + ep, timeout=timeout)
            response.raise_for_status()
        return True
    except Exception:
        return False


# ─────────────────────────── src/mlops/experiment_utils.py ──────────────────


def _sanitize_mlruns_dir(root: pathlib.Path) -> None:
    """
    Remove or archive directories inside *root* that cannot possibly be valid
    MLflow experiments (file-store experiments MUST be numeric).
    """
    for p in root.iterdir():
        if p.is_dir() and _hex32.match(p.name) and not (p / "meta.yaml").exists():
            logging.warning("🧹 Removing orphan MLflow dir %s", p)
            shutil.rmtree(p, ignore_errors=True)

def _fallback_uri() -> str:
    """Local file-store *outside* the default ./mlruns to avoid collisions."""
    local = pathlib.Path.cwd() / "mlruns_local"
    local.mkdir(exist_ok=True)
    _sanitize_mlruns_dir(local)          # one-time clean-up
    return f"file:{local}"

# ---------------------------------------------------------------------
# Add this just below the imports at module top (once per module)


# ---------------------------------------------------------------------
def setup_mlflow_experiment(experiment_name: Optional[str] = None) -> None:
    """
    Resolve a reachable MLflow tracking URI and make sure the experiment exists.
    Falls back to a local file store if the remote /health or /version ping fails.
    """
    from .config import EXPERIMENT_NAME, TRACKING_URI

    exp_name = experiment_name or EXPERIMENT_NAME
    uri = TRACKING_URI

    def _ping(u: str) -> bool:
        if not u.startswith("http"):
            return False
        try:
            for ep in ("/health", "/version"):
                r = requests.get(u.rstrip("/") + ep, timeout=2)
                r.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.debug("MLflow server ping failed: %s", exc)
            return False

    if not _ping(uri):
        uri = _fallback_uri()
        logger.warning("⚠️  MLflow server unreachable – using local store %s", uri)

    mlflow.set_tracking_uri(uri)

    # guarantee the experiment exists
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=f"{uri}/artifacts")

    mlflow.set_experiment(exp_name)
    logger.info("🗂  Using MLflow experiment '%s' @ %s", exp_name, uri)


def get_best_run(
    experiment_name: Optional[str] = None,
    metric_key: str = "accuracy",
    maximize: bool = True,
) -> Dict[str, Any]:
    """
    Return a *shallow* dict with run_id, metrics.*, and params.* keys
    so downstream code can use predictable dotted paths.
    """
    exp_name = experiment_name or EXPERIMENT_NAME
    setup_mlflow_experiment(exp_name)

    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(f"Experiment '{exp_name}' not found")

    order = "DESC" if maximize else "ASC"
    run = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric_key} {order}"],
        max_results=1,
    )[0]

    # Build a *flat* mapping -------------------------------------------------
    flat: Dict[str, Any] = {"run_id": run.info.run_id}

    # Metrics
    for k, v in run.data.metrics.items():
        flat[f"metrics.{k}"] = v

    # Params
    for k, v in run.data.params.items():
        flat[f"params.{k}"] = v

    # Tags (optional but handy)
    for k, v in run.data.tags.items():
        flat[f"tags.{k}"] = v

    return flat

