"""
Bayesian models for NFL kicker analysis.
Provides hierarchical Bayesian logistic regression and evaluation utilities using PyMC.
"""
from __future__ import annotations

import numpy as np                             # Core numerical arrays
import pandas as pd                            # DataFrame handling
import pymc as pm                              # Bayesian modeling
import arviz as az                             # Posterior analysis
from arviz.data.inference_data import InferenceData
import xarray as xr                            # 📥 Required for posterior casting :contentReference[oaicite:6]{index=6}
import matplotlib.pyplot as plt                # Plot utilities
from matplotlib.axes import Axes
from scipy import stats                        # ECDF and stats functions
import time                                    # Timestamps for file naming
import json                                    # JSON metadata
from pathlib import Path
from typing import (
    Dict, Any, TYPE_CHECKING, Optional, Union,
    List, Tuple, cast, Protocol
)

from src.nfl_kicker_analysis.utils.metrics import ModelEvaluator
from src.nfl_kicker_analysis.config import config, FEATURE_LISTS  # Move import to module level
from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
from src.nfl_kicker_analysis.eda import _FIELD_GOAL_RESULT_SUCCESS  # NEW: Import success constant


from xarray import Dataset
import xarray as xr
from numpy.typing import ArrayLike

__all__ = [
    "BayesianModelSuite",
]

# Add debug flag
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("🔍 DEBUG:", *args, **kwargs)

class HasPosterior(Protocol):
    """Protocol for objects with posterior attribute."""
    posterior: Any

class HasPosteriorPredictive(Protocol):
    """Protocol for objects with posterior_predictive attribute."""
    posterior_predictive: Any

def _get_posterior_mean(trace: InferenceData, var_name: str) -> Union[float, np.ndarray]:
    """
    Safely extract posterior means from an ArviZ InferenceData object.
    """
    if not hasattr(trace, "posterior"):
        raise RuntimeError("InferenceData object has no posterior group")
    # Access posterior using getattr to handle dynamic attributes
    posterior = getattr(trace, "posterior")
    if var_name not in posterior:
        raise KeyError(f"Variable {var_name} not found in posterior")
    # Collapse over chain & draw dims to get the mean
    result = posterior[var_name].mean(("chain", "draw")).values
    return float(result) if isinstance(result, (float, np.floating)) else result

def _get_posterior_predictive(trace: InferenceData, var_name: str) -> np.ndarray:
    """
    Safely extract posterior predictive samples from an ArviZ InferenceData object.
    """
    if not hasattr(trace, "posterior_predictive"):
        raise RuntimeError("InferenceData object has no posterior_predictive group")
    pp = getattr(trace, "posterior_predictive")
    if var_name not in pp:
        raise KeyError(f"Variable {var_name} not found in posterior_predictive")
    return np.asarray(pp[var_name].values)


def _plot_comparison(
    ax: Axes,
    x_vals: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    xlabel: str,
    title: str
) -> None:
    """Helper to plot actual vs predicted values."""
    ax.plot(np.asarray(x_vals), np.asarray(actual), marker="o", label="Actual", linewidth=2)
    ax.plot(np.asarray(x_vals), np.asarray(predicted), marker="s", label="Posterior mean", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FG make probability")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

class BayesianModelSuite:
    """Hierarchical Bayesian logistic‑regression models for kicker analysis."""

    def __init__(
        self,
        *,
        draws: int = 1_000,
        tune: int = 1_000,
        target_accept: float = 0.95,  # Increased from 0.9 to reduce divergences
        include_random_slope: bool = False,
        random_seed: Optional[int] = 42,
    ) -> None:
        debug_print("Initializing BayesianModelSuite")
        
        # Validate configuration
        required_config = [
            'BAYESIAN_MCMC_SAMPLES',
            'BAYESIAN_TUNE',
            'BAYESIAN_CHAINS',
            'MIN_DISTANCE',
            'MAX_DISTANCE',
            'MIN_KICKER_ATTEMPTS',
            'SEASON_TYPES'
        ]
        
        missing = [attr for attr in required_config if not hasattr(config, attr)]
        if missing:
            raise ValueError(f"Missing required configuration attributes: {missing}")
            
        # Override defaults with config values if provided
        self.draws = config.BAYESIAN_MCMC_SAMPLES if draws == 1_000 else draws
        self.tune = config.BAYESIAN_TUNE if tune == 1_000 else tune
        self.target_accept = target_accept
        self.include_random_slope = include_random_slope
        self.random_seed = random_seed

        # Model components - set during fit()
        self._model: Optional[pm.Model] = None
        self._trace: Optional[InferenceData] = None
        self._kicker_map: Dict[int, int] = {}
        self._distance_mu: float = 0.0
        self._distance_sigma: float = 1.0
        self.baseline_probs: Dict[int, float] = {}  # For consistent EPA baselines with EPACalculator
        self.evaluator = ModelEvaluator()
        debug_print("Initialization complete")

    def _bootstrap_distances(
        self,
        distances: ArrayLike,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Bootstrap sample distances from an empirical distribution.
        
        Parameters
        ----------
        distances : array of actual distances to sample from
        n_samples : number of bootstrap samples to draw
        rng : numpy random number generator
        
        Returns
        -------
        ndarray of shape (n_samples,)
            Resampled distances with replacement
        """
        return rng.choice(np.asarray(distances), size=n_samples, replace=True)

    # ---------------------------------------------------------------------
    # 🛠️  Helper utilities
    # ---------------------------------------------------------------------
    def _standardize(self, x: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if fit:
            self._distance_mu = float(x.mean())
            self._distance_sigma = float(x.std())
        return (x - self._distance_mu) / self._distance_sigma

    def _encode_kicker(self, raw_ids: ArrayLike, *, fit: bool = False,
                       unknown_action: str = "average") -> np.ndarray:
        """
        Map raw kicker IDs → compact indices (kicker_idx).
        """
        raw_ids_arr = np.asarray(raw_ids)
        if fit:
            unique_ids = np.unique(raw_ids_arr)
            debug_print(f"Fitting kicker map with {len(unique_ids)} unique IDs")
            debug_print(f"Raw IDs: {unique_ids.tolist()}")
            self._kicker_map = {int(pid): i for i, pid in enumerate(unique_ids)}
            debug_print(f"First few mappings: {dict(list(self._kicker_map.items())[:3])}")
        
        # Convert to list of ints for safer lookup
        raw_ids_list = [int(pid) for pid in raw_ids_arr]
        debug_print(f"Looking up {len(raw_ids_list)} IDs")
        debug_print(f"Sample raw IDs: {raw_ids_list[:5]}")
        debug_print(f"Kicker map size: {len(self._kicker_map)}")
        
        # Map IDs to indices, using -1 for unknown
        idx = np.array([self._kicker_map.get(pid, -1) for pid in raw_ids_list], dtype=int)
        n_unseen = (idx == -1).sum()
        
        if n_unseen > 0:
            msg = f"{n_unseen} unseen kicker IDs – mapped to league mean."
            if unknown_action == "raise":
                raise ValueError(msg)
            elif unknown_action == "warn":
                debug_print("⚠️ " + msg)
                debug_print(f"Unmapped IDs: {[pid for pid in raw_ids_list if pid not in self._kicker_map]}")

        return idx

    # ---------------------------------------------------------------------
    # 🔨  Model construction
    # ---------------------------------------------------------------------
    def _build_model(
        self,
        distance_std: np.ndarray,
        age_c: np.ndarray,           # <-- NEW: centered age
        age_c2: np.ndarray,          # <-- NEW: quadratic age
        exp_std: np.ndarray,
        success: np.ndarray,
        kicker_idx: np.ndarray,
        n_kickers: int,
    ) -> pm.Model:
        with pm.Model() as model:
            # Population-level effects
            alpha = pm.Normal("alpha", 1.5, 1.0)
            beta_dist = pm.Normal("beta_dist", -1.5, 0.8)
            
            # Age effects (linear + quadratic)
            beta_age  = pm.Normal("beta_age",  0.0, 0.5)
            beta_age2 = pm.Normal("beta_age2", 0.0, 0.5)
            beta_exp  = pm.Normal("beta_exp",  0.0, 0.5)

            # Per-kicker random intercepts (non-centered)
            σ_u   = pm.HalfNormal("sigma_u", 0.8)
            u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=n_kickers)
            u     = pm.Deterministic("u", σ_u * u_raw)

            # Per-kicker random aging slopes (optional enhancement)
            if self.include_random_slope:
                σ_age = pm.HalfNormal("sigma_age", 0.5)
                a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=n_kickers)
                a_k   = pm.Deterministic("a_k", σ_age * a_raw)
                age_slope_effect = a_k[kicker_idx] * age_c
            else:
                age_slope_effect = 0.0

            # Linear predictor
            lin_pred = (
                alpha
                + (beta_dist * distance_std)
                + (beta_age * age_c) + age_slope_effect
                + (beta_age2 * age_c2)
                + (beta_exp * exp_std)
                + u[kicker_idx]
            )

            θ = pm.Deterministic("theta", pm.invlogit(lin_pred))
            pm.Bernoulli("obs", p=θ, observed=success)
        return model

    # ---------------------------------------------------------------------
    # 📈  Public API
    # ---------------------------------------------------------------------
    def fit(self, df, *, preprocessor=None):
        debug_print(f"Starting fit with preprocessor: {preprocessor is not None}")
        debug_print(f"Input DataFrame rows: {len(df)}")
        debug_print(f"DataFrame engineered flag: {df.attrs.get('engineered', False)}")
        debug_print(f"DataFrame columns: {df.columns.tolist()}")
        
        # ------------------------------------------------------------------
        # 0️⃣  Exactly one preprocessing pass
        if df.attrs.get("engineered", False):
            processed = df.copy()
            debug_print("Using pre-engineered data")
        elif preprocessor is not None:
            debug_print("Using provided preprocessor")
            debug_print("Preprocessor config:", preprocessor.__dict__)
            processed = preprocessor.preprocess_slice(df)
        else:
            # 🎯 AUTO-CREATE BAYESIAN-MINIMAL PREPROCESSOR 
            # Using module-level imports instead of local imports
            debug_print("Creating minimal preprocessor")
            bayes_preprocessor = DataPreprocessor()
            bayes_preprocessor.update_config(
                min_distance=config.MIN_DISTANCE,
                max_distance=config.MAX_DISTANCE, 
                min_kicker_attempts=config.MIN_KICKER_ATTEMPTS,
                season_types=config.SEASON_TYPES,
                include_performance_history=False,  # Not needed for Bayesian
                include_statistical_features=False,  # Avoid complex features
                include_player_status=True,  # Enable player status features
                performance_window=12
            )
            bayes_preprocessor.update_feature_lists(**FEATURE_LISTS)
            processed = bayes_preprocessor.preprocess_slice(df)
            debug_print("Minimal preprocessing complete")

        debug_print(f"Processed DataFrame rows: {len(processed)}")
        debug_print(f"Processed columns: {processed.columns.tolist()}")

        # ------------------------------------------------------------------
        # 1️⃣  Predictors
        debug_print("Preparing predictors")
        dist_std = self._standardize(processed["attempt_yards"].to_numpy(float), fit=True)
        debug_print(f"Distance stats - mean: {self._distance_mu:.2f}, std: {self._distance_sigma:.2f}")
        
        # Age variables (centered & scaled)
        age_c  = processed["age_c"].to_numpy(float) if "age_c" in processed.columns else np.zeros(len(processed), dtype=float)
        age_c2 = processed["age_c2"].to_numpy(float) if "age_c2" in processed.columns else np.zeros(len(processed), dtype=float)
        debug_print(f"Age features present: age_c={age_c is not None}, age_c2={age_c2 is not None}")

        # Handle experience standardization
        if "exp_100" in processed.columns:
            exp_std = (
                (processed["exp_100"] - processed["exp_100"].mean()) /
                processed["exp_100"].std()
            ).to_numpy(float)
            debug_print("Experience features included")
        else:
            exp_std = np.zeros(len(processed), dtype=float)
            debug_print("No experience features found")
            
        success    = processed["success"].to_numpy(int)
        kicker_idx = self._encode_kicker(processed["kicker_id"].to_numpy(int), fit=True)
        n_kickers  = len(self._kicker_map)
        debug_print(f"Number of kickers: {n_kickers}")
        debug_print(f"Success rate: {success.mean():.3f}")

        # ---- model & sampling -------------------------------------------
        debug_print("Building model")
        self._model = self._build_model(
            dist_std, age_c, age_c2, exp_std, success, kicker_idx, n_kickers
        )
        
        # ── FIX: ensure pm.sample knows which model to use ────────────
        debug_print("Starting MCMC sampling")
        with self._model:
            self._trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=config.BAYESIAN_CHAINS,  # Use config value
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True
            )
        debug_print("Sampling complete")

        from src.nfl_kicker_analysis.models.bayes_model_utils import save_pymc_inference
        # ── ENSURE model directory exists & write full suite ────────────────
        try:
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            suite_dir = config.MODEL_DIR / f"bayesian_suite_{ts}"
            self.save_suite(suite_dir)
            debug_print(f"✅ Full Bayesian suite saved to {suite_dir!r}")
        except Exception as e:
            debug_print(f"⚠️ Failed to persist full suite: {e}")

        return self

    def predict(
        self,
        df: pd.DataFrame,
        *,
        return_ci: bool = False,
        return_proba: bool = True,
        preprocessor=None
    ):
        """
        Predict FG make probabilities or credible intervals on new data.
        
        Parameters
        ----------
        df : DataFrame to predict on
        return_ci : bool, default False
            If True, returns (mean, lower, upper) credible intervals
        return_proba : bool, default True
            If True, returns probabilities; if False, returns 0/1 predictions
        preprocessor : optional preprocessor to use
        
        Returns
        -------
        If return_ci=False:
            pd.Series of probabilities (if return_proba=True) or 0/1 (if False)
        If return_ci=True:
            Tuple of (mean, lower, upper) Series
        """
        if self._trace is None:
            raise RuntimeError("Model not yet fitted or loaded.")

        original_idx = df.index
        # 1️⃣ Preprocess
        proc = preprocessor.preprocess_slice(df) if preprocessor else df.copy()

        # 2️⃣ Standardize distances using helper
        dist_std = self._standardize(
            proc["attempt_yards"].to_numpy(float),
            fit=False
        )

        # 3️⃣ Map kickers → indices
        kicker_idx = self._encode_kicker(
            proc["kicker_id"].to_numpy(int),
            fit=False,
            unknown_action="average"
        )

        # 4️⃣ Get age and experience features
        age_c = proc["age_c"].to_numpy(float) if "age_c" in proc.columns else np.zeros(len(proc))
        age_c2 = proc["age_c2"].to_numpy(float) if "age_c2" in proc.columns else np.zeros(len(proc))
        
        if "exp_100" in proc.columns:
            exp_std = (
                (proc["exp_100"] - proc["exp_100"].mean()) /
                proc["exp_100"].std()
            ).to_numpy(float)
        else:
            exp_std = np.zeros(len(proc))

        # 5️⃣ Collapse posterior to means using helper
        a_mean = float(_get_posterior_mean(self._trace, "alpha"))
        b_mean = float(_get_posterior_mean(self._trace, "beta_dist"))
        u_mean_result = _get_posterior_mean(self._trace, "u")
        
        # Get age and experience coefficients
        beta_age_mean = float(_get_posterior_mean(self._trace, "beta_age"))
        beta_age2_mean = float(_get_posterior_mean(self._trace, "beta_age2"))
        beta_exp_mean = float(_get_posterior_mean(self._trace, "beta_exp"))

        # 6️⃣ Point predictions
        if not return_ci:
            # Handle u_mean indexing - ensure it's an array
            if isinstance(u_mean_result, (int, float)):
                u_effects = np.full(len(kicker_idx), u_mean_result)
            else:
                u_mean_arr = np.asarray(u_mean_result)
                u_effects = u_mean_arr[kicker_idx]
            
            # Full linear predictor with all effects
            logit = (
                a_mean
                + b_mean * dist_std
                + beta_age_mean * age_c
                + beta_age2_mean * age_c2
                + beta_exp_mean * exp_std
                + u_effects
            )
            
            # Convert to probabilities
            probs = 1 / (1 + np.exp(-logit))
            
            # Handle NaN values consistently
            probs = np.where(np.isnan(logit), np.nan, probs)
            
            # Return probabilities or classifications
            result = pd.Series(probs, index=proc.index).reindex(original_idx)
            if not return_proba:
                result = (result > 0.5).astype(float)
            return result

        # 7️⃣ Credible intervals via posterior predictive
        ppc = pm.sample_posterior_predictive(
            self._trace,
            model=self._model,
            var_names=["obs"],
            random_seed=self.random_seed,
            progressbar=False,
            return_inferencedata=True
        )
        draws = _get_posterior_predictive(ppc, "obs")
        mean_preds = draws.mean(axis=(0, 1))
        lower, upper = np.percentile(draws, [2.5, 97.5], axis=(0, 1))

        s_mean = pd.Series(mean_preds, index=proc.index).reindex(original_idx)
        s_lower = pd.Series(lower, index=proc.index).reindex(original_idx)
        s_upper = pd.Series(upper, index=proc.index).reindex(original_idx)
        return s_mean, s_lower, s_upper

    def evaluate(
        self, 
        df: pd.DataFrame, 
        *, 
        preprocessor=None
    ) -> Dict[str, float]:
        """Compute AUC, Brier score & log‑loss on provided data.
        
        Args:
            df: Data to evaluate on
            preprocessor: Optional DataPreprocessor instance. If provided, will
                         use it to preprocess the data before evaluation.
        """
        # Apply preprocessing if provided  
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)
                
        y_true = df["success"].to_numpy(dtype=int)
        y_pred_result = self.predict(df)  # predict() will handle its own preprocessing if needed
        
        # Handle both single prediction and CI tuple returns
        if isinstance(y_pred_result, tuple):
            y_pred = y_pred_result[0]  # Just use mean predictions for evaluation
        else:
            y_pred = y_pred_result
            
        return self.evaluator.calculate_classification_metrics(y_true, y_pred)

    def diagnostics(self, *, return_scalars: bool = False) -> Dict[str, Any]:
        """
        Compute and return MCMC diagnostics.

        Parameters
        ----------
        return_scalars : bool, default False
            If True, also include convenience keys
            ``rhat_max`` and ``ess_min`` for quick threshold checks.

        Returns
        -------
        dict
            Keys: rhat, ess (xarray.Dataset), rhat_vals, ess_vals (np.ndarray),
            summary_ok (bool), and optionally rhat_max, ess_min (float).
        """
        if self._trace is None:
            raise RuntimeError("Model not yet fitted.")

        # ArviZ calls (collapse chain/draw)
        rhats = cast(Dataset, az.rhat(self._trace))
        ess   = cast(Dataset, az.ess(self._trace))

        # Flatten → numpy for easy thresholding
        rhat_vals = rhats.to_array().values.ravel()
        ess_vals  = ess.to_array().values.ravel()

        summary_ok = (rhat_vals <= 1.01).all() and (ess_vals >= 100).all()
        if not summary_ok:
            print("⚠️  Sampling diagnostics outside recommended thresholds.")

        out = {
            "rhat": rhats,
            "ess": ess,
            "rhat_vals": rhat_vals,
            "ess_vals": ess_vals,
            "summary_ok": summary_ok,
        }
        if return_scalars:
            out["rhat_max"] = float(rhat_vals.max())
            out["ess_min"] = float(ess_vals.min())
        return out

    # -----------------------------------------------------------------
    # 🌟  NEW 1: kicker-level credible interval
    # -----------------------------------------------------------------
    def kicker_interval(
        self,
        kicker_id: int,
        distance: float | None = None,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Return mean, lower, upper success probability for a *single* kicker.

        Args
        ----
        kicker_id : raw ID as in dataframe
        distance  : yards; if None, uses the empirical mean distance of
                    training data, transformed with stored μ/σ.
        ci        : central credible-interval mass (default 0.95)
        """
        if self._trace is None:
            raise RuntimeError("Model must be fitted first")

        # 1 → index or league-mean column
        k_idx = self._kicker_map.get(kicker_id, -1)
        pad_col = len(self._kicker_map)   # after pad in predict()

        # 2 → choose distance
        if distance is None:
            distance_std = 0.0            # z-score of mean is 0
        else:
            distance_std = (distance - self._distance_mu) / self._distance_sigma

        posterior = getattr(self._trace, "posterior")
        a = posterior["alpha"].values.flatten()
        
        # Robust lookup for the distance slope parameter (handles naming changes)
        slope_name = "beta_dist" if "beta_dist" in posterior else "beta"
        b = posterior[slope_name].values.flatten()
        
        u = posterior["u"].values.reshape(a.size, -1)

        # pad league-mean
        u = np.pad(u, ((0, 0), (0, 1)), constant_values=0.0)
        idx = pad_col if k_idx == -1 else k_idx

        logit_p = a + b * distance_std + u[:, idx]
        p = 1 / (1 + np.exp(-logit_p))

        lower, upper = np.quantile(p, [(1-ci)/2, 1-(1-ci)/2])
        return {"mean": p.mean(), "lower": lower, "upper": upper,
                "n_draws": p.size, "distance_std": distance_std}

    # -----------------------------------------------------------------
    # 🌟  NEW 2: posterior-predictive plot across 5-yd bins
    # -----------------------------------------------------------------
    def plot_distance_ppc(
        self,
        df: pd.DataFrame,
        *,
        bin_width: int = 5,
        preprocessor = None,
        ax: Optional[Axes] = None
    ) -> Axes:
        """
        Bin attempts by distance and overlay actual vs posterior mean make-rate.
        """
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        # 1 Actual success by bin
        df = df.copy()
        df["bin"] = (df["attempt_yards"] // bin_width) * bin_width
        actual = df.groupby("bin")["success"].mean()

        # 2 Posterior mean per attempt → group
        preds = self.predict(df)
        df["pred"] = preds
        posterior = df.groupby("bin")["pred"].mean()

        # 3 Plot
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        
        _plot_comparison(
            ax,
            x_vals=np.asarray(actual.index),
            actual=np.asarray(actual),
            predicted=np.asarray(posterior),
            xlabel="Distance bin (yards)",
            title=f"Posterior-Predictive Check ({bin_width}-yd bins)"
        )
        return ax

    # -----------------------------------------------------------------
    # 🌟  NEW 3: age-binned posterior-predictive check
    # -----------------------------------------------------------------
    def plot_age_ppc(
        self,
        df: pd.DataFrame,
        *,
        bin_width: float = 2.0,
        preprocessor = None,
        ax: Optional[Axes] = None
    ) -> Axes:
        """
        Bin attempts by age and overlay actual vs posterior mean make-rate.
        """
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        # Use raw age for binning (more interpretable)
        age_col = "age_at_attempt" if "age_at_attempt" in df.columns else "age_c"
        df = df.copy()
        
        if age_col == "age_c":
            # Convert back to raw age for binning
            df["age_bin"] = ((df["age_c"] * 10 + 30) // bin_width) * bin_width
        else:
            df["age_bin"] = (df[age_col] // bin_width) * bin_width
            
        # Actual success by age bin
        actual = df.groupby("age_bin")["success"].mean()

        # Posterior mean per attempt → group by age
        preds = self.predict(df)
        df["pred"] = preds
        posterior = df.groupby("age_bin")["pred"].mean()

        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
            
        _plot_comparison(
            ax,
            x_vals=np.asarray(actual.index),
            actual=np.asarray(actual),
            predicted=np.asarray(posterior),
            xlabel="Age bin (years)",
            title=f"Age-Based Posterior-Predictive Check ({bin_width:.1f}-yr bins)"
        )
        return ax

    # ───────────────────────────────────────────────────────────
    # Helper: draw-level EPA simulation  (fully replaced)
    # ───────────────────────────────────────────────────────────
    def _epa_fg_plus_draws(
        self,
        league_df: pd.DataFrame,
        *,
        kicker_ids: ArrayLike,
        n_samples: int,                 # renamed for clarity
        rng: np.random.Generator,
        distance_strategy: str = "kicker",
        τ: float = 20.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Bootstrap EPA-FG⁺ draws for a set of kickers.
        
        Parameters
        ----------
        league_df : DataFrame with at least ['player_id', 'attempt_yards', 'success']
        kicker_ids : array of kicker IDs to compute for
        n_samples : number of bootstrap samples per kicker
        rng : numpy random number generator
        distance_strategy : how to sample distances ('kicker' or 'league')
        τ : shrinkage parameter for empirical Bayes
        **kwargs : future-proofing for additional options
        
        Returns
        -------
        ndarray of shape (n_samples, len(kicker_ids))
            Each column is the EPA-FG⁺ draws for one kicker
        """
        n_draws = n_samples                # ← backward compatibility shim
        kicker_ids_arr = np.asarray(kicker_ids)
        
        # Validate inputs
        if not isinstance(league_df, pd.DataFrame):
            raise TypeError("league_df must be a pandas DataFrame")
        if not isinstance(n_draws, int) or n_draws <= 0:
            raise ValueError("n_draws must be a positive integer")
            
        # Get baseline probabilities if not already computed
        if not self.baseline_probs:
            from src.nfl_kicker_analysis.utils.metrics import EPACalculator
            self.baseline_probs = EPACalculator().calculate_baseline_probs(league_df)
            
        # Initialize output array
        epa_draws = np.zeros((n_draws, len(kicker_ids_arr)))
        
        # For each kicker...
        for k, kid in enumerate(kicker_ids_arr):
            # Get this kicker's attempts
            mask = league_df['player_id'] == kid
            k_data = league_df[mask]
            
            if len(k_data) == 0:
                continue  # Skip if no data (shouldn't happen given filtering)
                
            # Bootstrap distances based on strategy
            if distance_strategy == "kicker":
                # Sample from this kicker's empirical distribution
                distances = self._bootstrap_distances(
                    k_data['attempt_yards'],
                    n_draws,
                    rng=rng
                )
            else:
                # Sample from league-wide distribution
                distances = self._bootstrap_distances(
                    league_df['attempt_yards'],
                    n_draws,
                    rng=rng
                )
                
            # Get baseline probabilities for these distances
            baseline_probs = np.array([
                self.baseline_probs.get(int(d), 0.5) for d in distances
            ])
            
            # Compute actual success rate (with Bayesian shrinkage)
            n_attempts = len(k_data)
            raw_rate = k_data['success'].mean()
            
            # Empirical Bayes shrinkage toward league average
            shrinkage = n_attempts / (n_attempts + τ)
            success_rate = (shrinkage * raw_rate + 
                          (1 - shrinkage) * np.mean(list(self.baseline_probs.values())))
            
            # EPA = (actual - expected) × points
            epa_draws[:, k] = 3 * (success_rate - baseline_probs)
            
        return epa_draws

    # ───────────────────────────────────────────────────────────────
    # PUBLIC – EPA-FG⁺ leaderboard with robust column checks
    # ───────────────────────────────────────────────────────────────
    def epa_fg_plus(
        self,
        df: pd.DataFrame,
        *,
        min_attempts: int = 20,
        n_samples: int | None = None,
        return_ci: bool = False,
    ) -> pd.DataFrame:
        """
        Build an EPA-FG⁺ leaderboard, optionally with bootstrap CIs and certainty levels.

        Returns a DataFrame indexed by player_id, with columns:
        - rank, player_name, total_attempts, epa_fg_plus_mean
        - (if return_ci=True) hdi_lower, hdi_upper, certainty_level
        - plus raw_success_rate, avg_distance, player_status, last_age, seasons_exp
        """
        debug_print(f"epa_fg_plus called with min_attempts={min_attempts}, "
                    f"n_samples={n_samples}, return_ci={return_ci}")

        # 1️⃣ Defaults & RNG
        if n_samples is None:
            n_samples = config.BAYESIAN_MCMC_SAMPLES
        rng = np.random.default_rng(self.random_seed or 42)

        # 2️⃣ Ensure EPA columns present
        from src.nfl_kicker_analysis.data.feature_engineering import ensure_epa_columns
        work = ensure_epa_columns(df.copy())

        # 3️⃣ Baseline probabilities cache
        if not self.baseline_probs:
            from src.nfl_kicker_analysis.utils.metrics import EPACalculator
            self.baseline_probs = EPACalculator().calculate_baseline_probs(work)

        # 4️⃣ Bootstrap draws
        kicker_ids = work["player_id"].unique()
        draws = self._epa_fg_plus_draws(
            work, kicker_ids=kicker_ids, n_samples=n_samples, rng=rng
        )

        # 5️⃣ Summarize draws
        means  = draws.mean(axis=0)
        lowers, uppers = np.percentile(draws, [2.5, 97.5], axis=0)

        # 6️⃣ Build base summary
        summary = pd.DataFrame({
            "player_id":   kicker_ids,
            "epa_fg_plus": means,
            "hdi_lower":   lowers,
            "hdi_upper":   uppers,
        })

        # 7️⃣ Enrich with metadata
        meta = (
            work.groupby("player_id")
                .agg(
                    player_name       = ("player_name", "first"),
                    attempts          = ("success", "size"),
                    raw_success_rate  = ("success", "mean"),
                    avg_distance      = ("attempt_yards", "mean"),
                    player_status     = ("player_status", "first"),
                    last_age          = ("age_at_attempt", "max"),
                    seasons_exp       = ("season", "nunique"),
                )
                .reset_index()
        )
        summary = (
            summary
            .merge(meta, on="player_id", how="left")
            .query("attempts >= @min_attempts")
            .sort_values("epa_fg_plus", ascending=False)
            .reset_index(drop=True)
            .assign(rank=lambda d: d.index + 1)
        )

        # 8️⃣ Rename for downstream code
        summary = summary.rename(columns={
            "epa_fg_plus": "epa_fg_plus_mean",
            "attempts":    "total_attempts",
        })

        # 9️⃣ ✨ NEW: set raw player_id as the DataFrame index
        summary = summary.set_index("player_id", drop=True)
        debug_print("After set_index, summary.index:", summary.index[:5])

        # 🔟 Compute certainty if requested
        if return_ci:
            ci_widths = summary["hdi_upper"] - summary["hdi_lower"]
            q33, q66 = np.percentile(ci_widths, [33.3, 66.6])
            def label(w):
                if w <= q33:    return "high"
                if w <= q66:    return "medium"
                return "low"
            summary["certainty_level"] = ci_widths.map(label)

        # 1️⃣1️⃣ Final column order
        cols = [
            "rank", "player_name", "total_attempts", "epa_fg_plus_mean",
            "raw_success_rate", "avg_distance", "player_status",
            "last_age", "seasons_exp",
        ]
        if return_ci:
            cols += ["hdi_lower", "hdi_upper", "certainty_level"]

        debug_print("Final EPA table columns:", cols)
        return summary[cols]

    # ---------------------------------------------------------------------
    # 🔍  Helper methods for kicker ID/name conversion
    # ---------------------------------------------------------------------
    def get_kicker_id_by_name(self, df: pd.DataFrame, player_name: str) -> int | None:
        """
        Get kicker_id for a given player_name from the dataset.
        
        Args:
            df: DataFrame containing kicker_id and player_name columns
            player_name: Name of the kicker to look up
            
        Returns:
            kicker_id if found, None otherwise
        """
        debug_print(f"Looking up kicker: {player_name}")
        debug_print(f"DataFrame columns: {df.columns.tolist()}")
        debug_print(f"Number of unique kickers: {df['player_name'].nunique()}")
        
        matches = df[df["player_name"] == player_name]["kicker_id"]
        debug_print(f"Found matches: {matches.tolist()}")
        
        if len(matches) == 0:
            debug_print("No matches found")
            return None
            
        unique_matches = matches.unique()
        debug_print(f"Unique kicker IDs: {unique_matches.tolist()}")
        
        # Check if this ID exists in our mapping
        raw_id = int(unique_matches[0])
        mapped_idx = self._kicker_map.get(raw_id)
        debug_print(f"Raw ID: {raw_id}, Mapped index: {mapped_idx}")
        debug_print(f"Kicker map size: {len(self._kicker_map)}")
        debug_print(f"Available mappings: {self._kicker_map}")
        
        return mapped_idx
    
    def get_kicker_name_by_id(self, df: pd.DataFrame, kicker_id: int) -> str | None:
        """
        Get player_name for a given kicker_id from the dataset.
        
        Args:
            df: DataFrame containing kicker_id and player_name columns
            kicker_id: ID of the kicker to look up
            
        Returns:
            player_name if found, None otherwise
        """
        matches = df[df["kicker_id"] == kicker_id]["player_name"].unique()
        return str(matches[0]) if len(matches) > 0 else None
    
    def kicker_interval_by_name(
        self,
        df: pd.DataFrame,
        player_name: str,
        distance: float | None = None,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Return mean, lower, upper success probability for a kicker by name.
        
        Args:
            df: DataFrame containing kicker mappings
            player_name: Name of the kicker
            distance: yards; if None, uses empirical mean
            ci: central credible-interval mass (default 0.95)
        """
        kicker_id = self.get_kicker_id_by_name(df, player_name)
        if kicker_id is None:
            raise ValueError(f"Kicker '{player_name}' not found in dataset")
        return self.kicker_interval(kicker_id, distance, ci)

    def save_suite(self, dirpath: Path | str) -> None:
        """
        Persist the trained suite to *dirpath* in a **fully deterministic**
        fashion so that a reload followed by ``predict`` returns *bit-for-bit*
        identical probabilities.

        What we store
        -------------
        1. The full ArviZ ``InferenceData`` object → ``trace.nc`` (NetCDF).
        2. A JSON side-car ``meta.json`` containing:
           • distance mean / std  (rounded to 12 dp to avoid JSON drift)
           • kicker_map – serialised as **string keys**, **stable order** (sorted
             by the *value* which is the contiguous index used by the sampler).
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # 1️⃣ Save the InferenceData exactly as returned by PyMC/ArviZ.
        trace_file = dirpath / "trace.nc"
        if self._trace is not None:
            self._trace.to_netcdf(str(trace_file))  # Convert Path to str for to_netcdf

        # 2️⃣ Build a JSON-serialisable metadata dict.
        # Sort by value (index) for stable ordering
        ordered_map = {
            str(k): int(v)           # keys → str for JSON, values stay int
            for k, v in sorted(self._kicker_map.items(), key=lambda kv: kv[1])
        }

        # Round floats to avoid JSON drift
        meta = {
            "version": "1.0.0",  # Add version for future compatibility
            "distance_mu": round(float(self._distance_mu), 12),
            "distance_sigma": round(float(self._distance_sigma), 12),
            "kicker_map": ordered_map,
        }

        # Debug prints
        debug_print("Saving suite metadata:")
        debug_print(f"  distance_mu: {meta['distance_mu']}")
        debug_print(f"  distance_sigma: {meta['distance_sigma']}")
        debug_print(f"  kicker_map (first 3): {dict(list(ordered_map.items())[:3])}")

        with open(dirpath / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))   # compact, stable

        debug_print(f"✅ Suite saved deterministically to {dirpath!r}")

    def save_metrics(self, dirpath: Path | str, metrics: dict) -> None:
        """
        Persist evaluation metrics alongside trace.nc and meta.json.
        
        Parameters
        ----------
        dirpath : Path or str
            Directory where the suite is saved
        metrics : dict
            Dictionary of evaluation metrics to persist
            
        Notes
        -----
        This ensures metrics computed during training are preserved exactly
        and can be loaded by the Streamlit app without recomputation.
        """
        dirpath = Path(dirpath)
        metrics_file = dirpath / "metrics.json"
        # Write with indentation for readability
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        debug_print(f"✅ Metrics saved to {metrics_file!r}")

    @classmethod
    def load_suite(cls, dirpath: Path | str) -> "BayesianModelSuite":
        """
        Reload a suite that was persisted by :py:meth:`save_suite`.
        The returned instance is guaranteed to yield **identical** predictions
        to the original suite (within IEEE-754 floating point rules).
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            raise FileNotFoundError(f"No saved suite at {dirpath}")

        # 1️⃣ Create a fresh instance so that __init__ validations run.
        suite = cls()

        # 2️⃣ Load metadata
        with open(dirpath / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # Check version compatibility
        version = meta.get("version", "0.0.0")  # Default for older saves
        if version != "1.0.0":
            debug_print(f"⚠️ Warning: Loading model version {version}, current version is 1.0.0")

        suite._distance_mu = float(meta["distance_mu"])
        suite._distance_sigma = float(meta["distance_sigma"])
        
        # Reconstruct the kicker_map **exactly** as before (value-sorted)
        suite._kicker_map = {
            int(k): int(v) for k, v in sorted(meta["kicker_map"].items(), key=lambda kv: int(kv[1]))
        }

        # Debug prints
        debug_print("Loading suite metadata:")
        debug_print(f"  distance_mu: {suite._distance_mu}")
        debug_print(f"  distance_sigma: {suite._distance_sigma}")
        debug_print(f"  kicker_map (first 3): {dict(list(suite._kicker_map.items())[:3])}")

        # 3️⃣ Load the trace
        suite._trace = az.from_netcdf(dirpath / "trace.nc")

        debug_print(f"✅ Suite loaded from {dirpath!r} — distance μ/σ = {suite._distance_mu:.4f}/{suite._distance_sigma:.4f}; {len(suite._kicker_map)} kickers.")
        return suite

    def verify_persistence(self, test_data: pd.DataFrame, *, preprocessor=None) -> bool:
        """
        Verify that model persistence is working correctly by saving and loading
        the model, then comparing predictions within floating-point tolerances.
        """
        import tempfile
        from pathlib import Path

        debug_print("Verifying model persistence...")

        # Original predictions
        preds_orig = self.predict(test_data, preprocessor=preprocessor)
        orig_vals = preds_orig.values

        # Save & reload
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.save_suite(tmp_path)
            loaded = self.load_suite(tmp_path)
            preds_loaded = loaded.predict(test_data, preprocessor=preprocessor)
            loaded_vals = preds_loaded.values

        # Tolerance-based comparison
        cmp = np.allclose(
            orig_vals,
            loaded_vals,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True
        )

        if not cmp:
            # Identify and debug first few mismatches
            diff_mask = ~np.isclose(
                orig_vals,
                loaded_vals,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True
            )
            idx = np.where(diff_mask)[0]
            debug_print(f"⚠️  Persistence tolerance check failed at {len(idx)} positions")
            if idx.size > 0:
                i = idx[0]
                debug_print(f"First mismatch at idx {i}: orig={orig_vals[i]}, loaded={loaded_vals[i]}")
            return False

        debug_print("✅ Predictions match within tolerance")
        return True


# -------------------------------------------------------------------------
# CLI smoke test entrypoint with leaderboard integration
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    from src.nfl_kicker_analysis.utils.metrics import train_test_split_by_season
    from src.nfl_kicker_analysis import config
    from src.nfl_kicker_analysis.utils.model_utils import _save_leaderboard
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # 1️⃣ Load & engineer features
    print("\n1️⃣ Loading and engineering data...")
    df_raw  = DataLoader().load_complete_dataset()
    df_feat = FeatureEngineer().create_all_features(df_raw)

    # 1a️⃣ Persist engineered features for Streamlit
    feat_path = config.MODEL_DATA_FILE
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"1a️⃣ Persisting engineered features to {feat_path}")
    df_feat.to_csv(feat_path, index=False)
    print("🔍 Contents of OUTPUT_DIR:", list(config.OUTPUT_DIR.iterdir()))

    # 2️⃣ Season split
    print("\n2️⃣ Splitting data by season...")
    train_raw, test_raw = train_test_split_by_season(df_feat)
    print(f"  Train size: {len(train_raw):,}")
    print(f"  Test size: {len(test_raw):,}")

    # 3️⃣ Configure preprocessor
    print("\n3️⃣ Configuring preprocessor...")
    pre = DataPreprocessor()
    pre.update_config(
        min_distance=config.MIN_DISTANCE,
        max_distance=config.MAX_DISTANCE,
        min_kicker_attempts=config.MIN_KICKER_ATTEMPTS,
        season_types=config.SEASON_TYPES,
        include_performance_history=False,
        include_statistical_features=False,
        include_player_status=True,
        performance_window=12,
    )
    pre.update_feature_lists(**config.FEATURE_LISTS)

    # 4️⃣ Fit Bayesian model
    print("\n4️⃣ Fitting Bayesian model...")
    suite = BayesianModelSuite(
        draws=config.BAYESIAN_MCMC_SAMPLES,
        tune=config.BAYESIAN_TUNE,
        include_random_slope=False,
        random_seed=42,
    )
    suite.fit(train_raw, preprocessor=pre)

    # ── Persist the full suite ──
    suite_dir = config.MODEL_DIR / f"bayesian_suite_{int(time.time())}"
    suite.save_suite(suite_dir)
    print(f"✅ Full suite saved at {suite_dir}")

    # 5. Evaluate metrics
    metrics = suite.evaluate(test_raw, preprocessor=pre)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save metrics alongside the suite
    suite.save_metrics(suite_dir, metrics)
    print(f"✅ Metrics saved to {suite_dir / 'metrics.json'}")

    # Validation checks
    print("\n✅ Running validation checks...")
    try:
        cid = suite.kicker_interval_by_name(df_feat, "JUSTIN TUCKER", distance=40)
        assert cid["lower"] <= cid["mean"] <= cid["upper"], f"Credible interval ordering failed: {cid}"
        print("• Credible interval check passed.")
    except Exception as e:
        print(f"• WARNING: Credible interval check failed: {e}")

    try:
        ax = suite.plot_distance_ppc(test_raw, bin_width=5, preprocessor=pre)
        df_ppc = pre.preprocess_slice(test_raw).copy()
        df_ppc["bin"] = (df_ppc["attempt_yards"] // 5) * 5
        actual = df_ppc.groupby("bin")["success"].mean().values
        posterior = df_ppc.assign(pred=suite.predict(df_ppc)) \
                       .groupby("bin")["pred"].mean().values
        corr = np.corrcoef(np.asarray(actual), np.asarray(posterior))[0, 1]
        assert corr > 0.9, f"PPC correlation too low: {corr:.3f}"
        print(f"• PPC correlation check passed (r={corr:.3f}).")
    except Exception as e:
        print(f"• WARNING: PPC correlation check failed: {e}")

    # EPA-FG+ leaderboard
    try:
        epa_tbl = suite.epa_fg_plus(df_feat, n_samples=500, return_ci=True)
        print("\nTop 5 Active Kickers by EPA-FG+:")
        display_cols = [
            "player_name", "epa_fg_plus_mean", "raw_success_rate",
            "total_attempts", "avg_distance", "player_status",
            "last_age", "seasons_exp", "certainty_level"
        ]
        active_kickers = epa_tbl[epa_tbl["player_status"] != "Retired/Injured"]
        print(active_kickers[display_cols].head().to_string())

        print("\nKicker Status Breakdown:")
        for status, count in epa_tbl["player_status"].value_counts().items():
            print(f"  {status}: {count}")

        # Persist leaderboard
        epa_no_ci = epa_tbl.reset_index()[["player_id","player_name","epa_fg_plus_mean","rank"]]
        _save_leaderboard(epa_no_ci.set_index("player_id"))
        print(f"✓ Wrote updated leaderboard to {config.LEADERBOARD_FILE}")
    except Exception as e:
        print(f"• WARNING: EPA leaderboard generation failed: {e}")

    # MCMC diagnostics
    try:
        diag = suite.diagnostics(return_scalars=True)
        assert diag["summary_ok"], f"Diagnostics failed: R-hat={diag['rhat_max']:.3f}, ESS={diag['ess_min']:.0f}"
        print("• Diagnostics check passed.")
    except Exception as e:
        print(f"• WARNING: Diagnostics check failed: {e}")

    # Test visualization with Justin Tucker
    try:
        print("\nTesting visualization with Justin Tucker...")
        
        # Get Tucker's interval
        tucker_interval = suite.kicker_interval_by_name(df_feat, "JUSTIN TUCKER")
        print(f"Tucker's P(make) at mean distance: {tucker_interval['mean']:.3f}")
        print(f"95% CI: ({tucker_interval['lower']:.3f}, {tucker_interval['upper']:.3f})")
        
        # Get posterior draws for visualization
        k_idx = suite.get_kicker_id_by_name(df_feat, "JUSTIN TUCKER")
        if k_idx is not None:
            posterior = suite._trace.posterior
            if posterior is None:
                raise ValueError("No posterior samples available")
                
            a_draws = np.asarray(posterior["alpha"].values.flatten())
            b_draws = np.asarray(posterior["beta_dist"].values.flatten())
            u_draws = np.asarray(posterior["u"].values.reshape(-1, posterior["u"].shape[-1]))
            
            # Get kicker's random effects
            u_k = u_draws[:, k_idx]
            
            # Compute P(make) at mean distance
            logit = a_draws + u_k  # distance=0 (mean-centered)
            p_make = 1 / (1 + np.exp(-logit))
            
            print(f"\nVisualization test successful!")
            print(f"P(make) range: [{float(p_make.min()):.3f}, {float(p_make.max()):.3f}]")
            print(f"Mean: {float(p_make.mean()):.3f}")
            print(f"95% CI: [{float(np.percentile(p_make, 2.5)):.3f}, {float(np.percentile(p_make, 97.5)):.3f}]")
        else:
            print("❌ Failed to get kicker index for Tucker")
            
    except Exception as e:
        print(f"• WARNING: Visualization tests failed: {e}")

    print("\n🎉 Core functionality complete!")

def test_tucker_visualization(suite: BayesianModelSuite, df: pd.DataFrame) -> None:
    """
    Debug test for Justin Tucker visualization.
    """
    print("\n=== Testing Tucker Visualization ===")
    print("1. DataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of unique kickers: {df['player_name'].nunique()}")
    print(f"Sample of kicker names: {df['player_name'].unique()[:5].tolist()}")
    
    print("\n2. Kicker Map Info:")
    print(f"Size of kicker map: {len(suite._kicker_map)}")
    print(f"Sample mappings: {dict(list(suite._kicker_map.items())[:3])}")
    
    print("\n3. Looking up Tucker:")
    tucker_rows = df[df["player_name"] == "JUSTIN TUCKER"]
    print(f"Found {len(tucker_rows)} rows for Tucker")
    if len(tucker_rows) > 0:
        print(f"Tucker's kicker_id values: {tucker_rows['kicker_id'].unique().tolist()}")
        print(f"Tucker's player_status: {tucker_rows['player_status'].iloc[0]}")
        
    k_idx = suite.get_kicker_id_by_name(df, "JUSTIN TUCKER")
    print(f"Mapped index for Tucker: {k_idx}")
    
    print("\n4. Posterior Info:")
    posterior = suite._trace.posterior
    print(f"Posterior groups: {list(posterior.groups)}")
    print(f"U parameter shape: {posterior['u'].shape}")
    
    if k_idx is not None:
        print(f"\n5. Checking index bounds:")
        u = posterior["u"].values.reshape(-1, posterior["u"].shape[-1])
        print(f"Reshaped U matrix: {u.shape}")
        print(f"Attempting to access column {k_idx}")
        if 0 <= k_idx < u.shape[1]:
            print("✅ Index is within bounds")
            u_k = u[:, k_idx]
            print(f"Successfully extracted kicker effects: shape={u_k.shape}")
        else:
            print(f"❌ Index {k_idx} is out of bounds for shape {u.shape}")
    
    print("\n=== Test Complete ===")


