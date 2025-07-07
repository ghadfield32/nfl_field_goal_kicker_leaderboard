"""
Time-Series Bayesian Models for NFL Kicker Analysis.

Provides hierarchical dynamic-linear model (level + trend) and SARIMA
built on PyMC 5 (+ pymc-experimental).
"""
from __future__ import annotations
import warnings
from typing import Dict, Optional, List, Union, Any

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from arviz.data.inference_data import InferenceData

import jax
import numpyro

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

try:
    from pymc_experimental.statespace import SARIMA  # type: ignore
except ImportError:
    warnings.warn("pymc_experimental not installed, SARIMA model will not be available")
    SARIMA = None

def _choose_chain_config(requested: int, use_jax: bool) -> dict:
    """Configure PyMC/JAX sampler chains based on backend and requested chains."""
    if use_jax:
        return dict(chains=max(2, requested), chain_method="vectorized")
    else:
        return dict(chains=max(2, requested), cores=min(4, max(2, requested)))

class TimeSeriesBayesianModelSuite:
    """Hierarchical DLM / SARIMA model for weekly kicker make-rates."""
    def __init__(
        self,
        freq: str = "W-MON",
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.95,
        random_seed: Optional[int] = None,
        use_sarima: bool = False
    ):
        self.freq = freq
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.use_sarima = use_sarima and SARIMA is not None
        self._trace: Optional[InferenceData] = None
        self._model: Optional[pm.Model] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the time series model."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if self.use_sarima:
            self._fit_sarima(df)
        else:
            self._fit_dlm(df)

    def _fit_dlm(self, df: pd.DataFrame) -> None:
        """Fit dynamic linear model."""
        with pm.Model() as model:
            level_sd = pm.HalfNormal("level_sd", 0.1)
            trend_sd = pm.HalfNormal("trend_sd", 0.01)

            level = pm.GaussianRandomWalk("level", sigma=level_sd, shape=len(df))
            trend = pm.GaussianRandomWalk("trend", sigma=trend_sd, shape=len(df))

            mu = level + trend
            pm.Normal("obs", mu=mu, sigma=0.1, observed=df["success_rate"])

            self._trace = pm.sample(  # type: ignore
                draws=self.draws,
                tune=self.tune,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                **_choose_chain_config(4, use_jax=False)
            )
            self._model = model

    def _fit_sarima(self, df: pd.DataFrame) -> None:
        """Fit SARIMA model."""
        if SARIMA is None:
            raise ImportError("pymc_experimental not installed, SARIMA model not available")
            
        with pm.Model() as model:
            sarima = SARIMA(
                df["success_rate"].values,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 52)
            )
            self._trace = pm.sample(  # type: ignore
                draws=self.draws,
                tune=self.tune,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                **_choose_chain_config(4, use_jax=True)
            )
            self._model = model

    def forecast(self, steps: int = 6) -> pd.DataFrame:
        """Generate forecasts for the specified number of steps."""
        if self._trace is None or self._model is None:
            raise ValueError("Model not fitted")

        with self._model:
            post_pred = pm.sample_posterior_predictive(  # type: ignore
                self._trace,
                extend_kwargs={"steps": steps},
                random_seed=self.random_seed
            )

        y_pred = post_pred.posterior_predictive["obs"].mean(("chain", "draw"))  # type: ignore
        y_std = post_pred.posterior_predictive["obs"].std(("chain", "draw"))  # type: ignore

        return pd.DataFrame({
            "step": range(1, steps + 1),
            "p_mean": y_pred[-steps:],
            "p_std": y_std[-steps:]
        })

    def diagnostics(self, thin: int = 5) -> Dict[str, float]:
        """Calculate model diagnostics (R-hat and ESS statistics)."""
        if self._trace is None:
            raise ValueError("Model not fitted")

        rhat_max = 1.0
        ess_min = float("inf")

        for var in self._trace.posterior.data_vars:  # type: ignore
            if var.startswith("_"):
                continue
            data = self._trace.posterior[var]  # type: ignore
            if thin > 1 and data.ndim > 2:
                slc = (slice(None), slice(None)) + (slice(None, None, thin),) * (data.ndim - 2)
                data = data[slc]
            rhat = az.rhat(data)  # type: ignore
            if hasattr(rhat, "to_array"):
                rhat_max = max(rhat_max, float(rhat.to_array().max()))
            else:
                rhat_max = max(rhat_max, float(rhat))
            ess = az.ess(data, method="bulk")  # type: ignore
            if hasattr(ess, "to_array"):
                ess_min = min(ess_min, float(ess.to_array().min()))
            else:
                ess_min = min(ess_min, float(ess))
        return {"rhat_max": float(rhat_max), "ess_min": float(ess_min)}

if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer

    df_raw = DataLoader().load_complete_dataset()
    df_feat = FeatureEngineer().create_all_features(df_raw)
    df_feat = pd.DataFrame(df_feat)  # Ensure DataFrame type

    ts = TimeSeriesBayesianModelSuite(
        freq="W-MON",
        draws=250,
        tune=250,
        use_sarima=True,
        target_accept=0.85
    )
    ts.fit(df_feat[df_feat["season"] <= 2018])
    fcst = ts.forecast(steps=6)
    print(fcst.head())

    diag = ts.diagnostics()
    print(f"R-hat: {diag['rhat_max']:.3f} | ESS: {diag['ess_min']:.0f}")




