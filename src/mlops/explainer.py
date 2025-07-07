from __future__ import annotations
import os
import socket
import logging
from pathlib import Path
from typing import Any, Sequence, Optional
from contextlib import closing

import mlflow
import psutil  # lightweight; already added to pyproject deps
from sklearn.utils.multiclass import type_of_target
from explainerdashboard import (
    ClassifierExplainer,
    RegressionExplainer,
    ExplainerDashboard,
)

logging.basicConfig(level=logging.INFO)

__all__ = ["build_and_log_dashboard", "load_dashboard_yaml", "dashboard_best_run", "_first_free_port", "_port_details"]


# ---------------------------------------------------------------------------
def _port_details(port: int) -> str:
    """
    Return a one-line string with PID & cmdline of the process
    listening on *port*, or '' if none / not discoverable.
    """
    for c in psutil.net_connections(kind="tcp"):
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
            try:
                p = psutil.Process(c.pid)
                return f"[PID {p.pid} – {p.name()}] cmd={p.cmdline()}"
            except psutil.Error:
                return f"[PID {c.pid}] (no detail)"
    return ""

def _first_free_port(start: int = 8050, tries: int = 50) -> int:
    """Return first free TCP port ≥ *start* on localhost."""
    for port in range(start, start + tries):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.05)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            # Port is in use, try next one
            continue
    raise RuntimeError("⚠️  No free ports found in range")

def _next_free_port(start: int = 8050, tries: int = 50) -> int:
    """Return the first free TCP port ≥ *start*. (Alias for backward compatibility)"""
    return _first_free_port(start, tries)

def _port_in_use(port: int) -> bool:
    """Check if a port is already in use on any interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.05)
        # Check both localhost and 0.0.0.0 to be thorough
        try:
            # First check localhost (127.0.0.1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
            # Also check if anything is bound to all interfaces
            if s.connect_ex(("0.0.0.0", port)) == 0:
                return True
        except (socket.gaierror, OSError):
            # If we can't connect, assume port is free
            pass
        return False


# ---------------------------------------------------------------------------
# -------------------------------------------------------------- #
#  src/mlops/explainer.py (only this function changed)           #
# -------------------------------------------------------------- #
def build_and_log_dashboard(
    model: Any,
    X_test,
    y_test,
    *,
    # ---- explainer kwargs (unchanged) -------------------------
    cats: Optional[Sequence[str]] = None,
    idxs: Optional[Sequence[Any]] = None,
    descriptions: Optional[dict[str, str]] = None,
    target: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    X_background=None,
    model_output: str = "probability",
    shap: str = "guess",
    shap_interaction: bool = True,
    simple: bool = False,
    mode: str = "dash",         # 🆕 safest default for docker
    title: str = "Model Explainer",
    # ---- infra -----------------------------------------------
    run: mlflow.ActiveRun | None = None,
    port: int | None = None,
    serve: bool = False,
    server_backend: str = "waitress",   # 🆕 waitress|gunicorn|jupyterdash
    conflict_strategy: str = "next",
    max_tries: int = 20,
    save_yaml: bool = True,
    output_dir: os.PathLike | str | None = None,
) -> Path:
    """
    Build + (optionally) serve the dashboard.

    server_backend
        'waitress'    – production WSGI server (binds 0.0.0.0)  
        'gunicorn'    – spawn via subprocess (needs gunicorn installed)  
        'jupyterdash' – fallback; use only for notebook demos
    """
    # ------------ build explainer (unchanged) ------------------
    problem = type_of_target(y_test)
    ExplainerCls = RegressionExplainer if problem.startswith("continuous") else ClassifierExplainer
    expl_kwargs = dict(
        cats=cats, idxs=idxs, descriptions=descriptions, target=target,
        labels=labels, X_background=X_background, model_output=model_output, shap=shap,
    )
    expl_kwargs = {k: v for k, v in expl_kwargs.items() if v is not None}
    explainer = ExplainerCls(model, X_test, y_test, **expl_kwargs)

    dash = ExplainerDashboard(
        explainer, title=title, shap_interaction=shap_interaction,
        simple=simple, mode=mode,
    )

    out_dir = Path(output_dir or "."); out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "explainer_dashboard.html"; dash.save_html(html_path); mlflow.log_artifact(str(html_path))
    if save_yaml:
        yaml = out_dir / "dashboard.yaml"; dash.to_yaml(yaml); mlflow.log_artifact(str(yaml))

    # ------------ serve ----------------------------------------
    if not serve:
        return html_path

    chosen = port or _first_free_port()
    attempts = 0
    while _port_in_use(chosen):
        if conflict_strategy == "raise":
            raise RuntimeError(f"Port {chosen} in use {_port_details(chosen)}")
        if conflict_strategy == "kill":
            pid = int((_port_details(chosen) or "PID 0").split()[1]); psutil.Process(pid).terminate()
            break
        attempts += 1
        if attempts >= max_tries:
            raise RuntimeError(f"No free port after {max_tries} tries")
        chosen += 1

    logging.info("🌐 Dashboard on http://0.0.0.0:%s via %s", chosen, server_backend)

    if server_backend == "waitress":
        dash.run(chosen, host="0.0.0.0", use_waitress=True, mode="dash")
    elif server_backend == "gunicorn":
        import subprocess, shlex
        cmd = f"gunicorn -w 3 -b 0.0.0.0:{chosen} dashboard:app"
        subprocess.Popen(shlex.split(cmd), cwd=str(out_dir))
    else:  # jupyterdash
        dash.run(chosen, host="0.0.0.0")

    return html_path




# ---------------------------------------------------------------------------
def load_dashboard_yaml(path: os.PathLike | str) -> ExplainerDashboard:
    """Reload a YAML config – unchanged but kept for public API."""
    return ExplainerDashboard.from_config(path) 


# ────────────────────────────────────────────────────────────────────────────
def dashboard_best_run(metric: str = "accuracy",
                       maximize: bool = True,
                       *, port: int | None = None) -> None:
    """
    Load the *best* run (by `metric`) from the active experiment and
    launch an ExplainerDashboard **once** for that model.

    Example
    -------
    >>> from mlops.explainer import dashboard_best_run
    >>> dashboard_best_run("accuracy")      # opens http://0.0.0.0:8050
    """
    from .experiment_utils import get_best_run
    from .model_registry  import load_model_from_run
    from sklearn.datasets import load_iris
    import pandas as pd

    best = get_best_run(metric_key=metric, maximize=maximize)
    run_id = best["run_id"]
    model  = load_model_from_run(run_id)

    iris = load_iris()
    X_df  = pd.DataFrame(iris.data, columns=iris.feature_names)
    build_and_log_dashboard(
        model, X_df, iris.target,
        labels=list(iris.target_names),
        run=None, serve=True, port=port or 8050
    )

   
