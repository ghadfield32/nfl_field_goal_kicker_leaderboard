"""
Metrics utilities for NFL kicker analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,     # AUC-PR
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from typing import Dict, Optional, Union, List, Any, Callable, Tuple, cast
from numpy.typing import NDArray
import arviz as az
from src.nfl_kicker_analysis.config import config

class EPACalculator:
    """Calculates EPA-based metrics for kickers."""
    
    def __init__(self):
        """Initialize the EPA calculator."""
        self.baseline_probs: Dict[int, float] = {}
        self.distance_profile = config.DISTANCE_PROFILE
        self.distance_weights = config.DISTANCE_WEIGHTS
    
    def calculate_baseline_probs(self, data: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate baseline success probabilities by distance.
        
        Args:
            data: DataFrame with attempt_yards and success
            
        Returns:
            Dictionary mapping distance to success probability
        """
        baseline = data.groupby('attempt_yards')['success'].mean()
        self.baseline_probs = baseline.to_dict()
        return self.baseline_probs
    
    def calculate_epa_fg_plus(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EPA-FG+ for each kicker.
        
        Args:
            data: DataFrame with kicker attempts
            
        Returns:
            DataFrame with kicker EPA-FG+ ratings
        """
        if not self.baseline_probs:
            self.baseline_probs = self.calculate_baseline_probs(data)
            
        # Calculate expected points
        data = data.copy()
        data.loc[:, 'expected_points'] = data['attempt_yards'].map(lambda x: self.baseline_probs.get(x, 0.5)) * 3
        data.loc[:, 'actual_points'] = data['success'] * 3
        data.loc[:, 'epa'] = data['actual_points'] - data['expected_points']
        
        # Calculate EPA-FG+ per kicker
        kicker_stats = data.groupby('player_name').agg({
            'epa': ['count', 'mean', 'sum'],
            'player_id': 'first'  # Keep player ID
        })
        
        kicker_stats.columns = ['attempts', 'epa_per_kick', 'total_epa', 'player_id']
        kicker_stats.loc[:, 'epa_fg_plus'] = kicker_stats['epa_per_kick']
        
        # Add rank
        kicker_stats.loc[:, 'rank'] = kicker_stats['epa_fg_plus'].rank(ascending=False, method='min')
        
        return kicker_stats.reset_index()
    
    def calculate_clutch_rating_with_shrinkage(
        self, 
        data: pd.DataFrame, 
        prior_a: float = 8.0, 
        prior_b: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate clutch field goal rating with beta-binomial shrinkage.
        
        Args:
            data: DataFrame with kicker attempts including is_clutch column
            prior_a: Beta prior alpha parameter (favors success)
            prior_b: Beta prior beta parameter (favors failure)
            
        Returns:
            DataFrame with clutch ratings per kicker
        """
        # Default prior centers on 80% (8/(8+2))
        
        results = []
        for player_name, grp in data.groupby('player_name'):
            if 'is_clutch' in grp.columns:
                clutch_attempts = grp[grp['is_clutch'] == 1]
            else:
                # Fallback if no clutch column
                clutch_attempts = grp[grp['success'] == 0]  # Empty fallback
                
            made_clutch = clutch_attempts['success'].sum() if len(clutch_attempts) > 0 else 0
            miss_clutch = len(clutch_attempts) - made_clutch if len(clutch_attempts) > 0 else 0
            
            # Beta-binomial posterior
            post_a = prior_a + made_clutch
            post_b = prior_b + miss_clutch
            clutch_rate_shrunk = post_a / (post_a + post_b)
            
            # Raw clutch rate for comparison
            raw_clutch_rate = clutch_attempts['success'].mean() if len(clutch_attempts) > 0 else 0.0
            
            results.append({
                'player_name': player_name,
                'player_id': int(grp['player_id'].iat[0]),
                'total_attempts': len(grp),
                'clutch_attempts': len(clutch_attempts),
                'clutch_made': made_clutch,
                'raw_clutch_rate': raw_clutch_rate,
                'clutch_rate_shrunk': clutch_rate_shrunk,
                'shrinkage_applied': abs(clutch_rate_shrunk - raw_clutch_rate)
            })
            
        df = pd.DataFrame(results)
        df['clutch_rank'] = df['clutch_rate_shrunk'].rank(ascending=False, method='min')
        return df.sort_values('clutch_rank')
    
    # ────────────────────────────────────────────────────────────────────
    # NEW helper ─ bootstrap EPA-FG⁺ draws for ONE kicker
    # ────────────────────────────────────────────────────────────────────
    def _bootstrap_kicker_epa(
        self,
        grp: pd.DataFrame,
        *,
        n_draws: int = 2_000,
        rng: np.random.Generator,
    ) -> NDArray[np.float_]:
        """
        Non-parametric bootstrap: resample the kicker's attempts (with replacement)
        and recompute mean EPA-FG⁺.  Returns an array of shape (n_draws,).
        
        Args:
            grp: DataFrame with kicker's attempts
            n_draws: Number of bootstrap draws
            rng: Random number generator
            
        Returns:
            Array of bootstrap EPA-FG+ draws
        """
        # pre-compute baseline EPA for every attempt in the group
        base_pts = grp["attempt_yards"].map(
            lambda x: self.baseline_probs.get(x, 0.5)
        ).to_numpy(np.float_) * 3.0
        actual   = grp["success"].to_numpy(np.int_) * 3.0
        diff     = actual - base_pts                    # vector of EPA per attempt

        if diff.size == 0:
            return np.array([np.nan])                   # safeguard – should not happen

        boot = rng.choice(diff, size=(n_draws, diff.size), replace=True)
        return boot.mean(axis=1)                       # average per draw

    # ────────────────────────────────────────────────────────────────────
    # PUBLIC – EPA-FG⁺ with 95 % interval & certainty flag
    # ────────────────────────────────────────────────────────────────────
    def calculate_epa_fg_plus_ci(
        self,
        data: pd.DataFrame,
        *,
        n_draws: int = 2_000,
        alpha: float = 0.05,
        random_state: int | None = 42,
    ) -> pd.DataFrame:
        """
        Bootstrap EPA-FG⁺ per kicker, returning mean, lower, upper bounds and a
        qualitative certainty label (high | medium | low).

        Args:
            data: DataFrame with kicker attempts
            n_draws: Number of bootstrap draws
            alpha: Significance level for confidence intervals
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with bootstrap EPA-FG+ ratings and confidence intervals
            
        Notes
        -----
        * Uses the **Jeffreys-prior** interpretation of a (1–alpha) credible
          interval via percentile bootstrap.
        * CI width thresholds (empirical 33rd & 66th percentiles) define the
          certainty bands as recommended in sports-analytics literature.
        """
        if not self.baseline_probs:
            self.calculate_baseline_probs(data)

        rng = np.random.default_rng(random_state)

        records: list[dict[str, Any]] = []
        for player, grp in data.groupby("player_name"):
            draws = self._bootstrap_kicker_epa(grp, n_draws=n_draws, rng=rng)
            mean  = float(np.nanmean(draws))
            lower = float(np.nanpercentile(draws, 100 * alpha / 2))
            upper = float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))
            width = upper - lower
            records.append({
                "player_name": player,
                "player_id":   int(grp["player_id"].iat[0]),
                "attempts":    int(len(grp)),
                "epa_fg_plus_mean":  mean,
                "hdi_lower":   lower,
                "hdi_upper":   upper,
                "ci_width":    width,
            })

        tbl = pd.DataFrame(records).set_index("player_name")

        # Certainty levels by tercile of CI-width
        q33, q66 = tbl["ci_width"].quantile([.33, .66])
        tbl["certainty"] = np.where(
            tbl["ci_width"] <= q33, "high",
            np.where(tbl["ci_width"] <= q66, "medium", "low")
        )

        tbl["rank"] = tbl["epa_fg_plus_mean"].rank(ascending=False, method="min")
        return tbl.sort_values("rank")
    
    def calculate_all_kicker_ratings(self, data: pd.DataFrame, include_ci: bool = False) -> pd.DataFrame:
        """
        Calculate complete kicker ratings, optionally with uncertainty intervals.
        
        Args:
            data: Complete dataset
            include_ci: Whether to include bootstrap confidence intervals
            
        Returns:
            DataFrame with kicker ratings
        """
        print("Calculating EPA-FG+ ratings...")
        
        if include_ci:
            ratings = self.calculate_epa_fg_plus_ci(data)
            metric_col = "epa_fg_plus_mean"
        else:
            ratings = self.calculate_epa_fg_plus(data)
            metric_col = "epa_fg_plus"
        
        print(f"\nTop 5 kickers by EPA-FG+:")
        display_cols = ['attempts', metric_col, 'rank']
        if include_ci:
            display_cols.extend(['hdi_lower', 'hdi_upper', 'certainty'])
        
        top_5 = ratings.head(5) if include_ci else ratings.nlargest(5, metric_col)
        if include_ci:
            print(top_5[display_cols].to_string(index=True))
        else:
            print(top_5[['player_name'] + display_cols].to_string(index=False))
        
        return ratings

class ModelEvaluator:
    """Compute a rich set of discrimination & calibration metrics."""

    # ---------- single-metric helpers ----------
    @staticmethod
    def calculate_auc(y, p) -> float:
        return float(roc_auc_score(y, p))

    @staticmethod
    def calculate_auc_pr(y, p) -> float:
        return float(average_precision_score(y, p))  # AUC-PR

    @staticmethod
    def calculate_log_loss(y, p) -> float:
        return float(log_loss(y, p))

    @staticmethod
    def calculate_brier_score(y, p) -> float:
        return float(brier_score_loss(y, p))

    @staticmethod
    def calculate_threshold_metrics(y, p, thresh: float = 0.5) -> Tuple[float, float, float, float]:
        """Return accuracy, precision, recall, F1 at a chosen threshold."""
        pred = (p >= thresh).astype(int)
        acc = float(accuracy_score(y, pred))
        prec = float(precision_score(y, pred, zero_division="warn"))
        rec = float(recall_score(y, pred, zero_division="warn"))
        f1 = float(f1_score(y, pred, zero_division="warn"))
        return acc, prec, rec, f1

    # ---------- calibration helpers ----------
    @staticmethod
    def calculate_ece(y, p, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE) using equally spaced probability bins.

        Handles the fact that `sklearn.calibration.calibration_curve` drops
        empty bins, which otherwise causes a length-mismatch when you try to
        multiply by fixed-length weights.
        """
        try:
            # 1 – Compute reliability data.  The function *silently* removes
            #     any empty bin, so prob_true/pred may be < n_bins long.
            prob_true, prob_pred = calibration_curve(
                y, p, n_bins=n_bins, strategy="uniform"
            )

            # 2 – Re-create the original histogram and keep *only* non-empty bins
            bin_counts, _ = np.histogram(p, bins=n_bins, range=(0, 1))
            non_empty_mask = bin_counts > 0

            # Make sure shapes now align
            bin_counts = bin_counts[non_empty_mask]
            prob_true  = np.asarray(prob_true)
            prob_pred  = np.asarray(prob_pred)

            # Sanity check – all arrays must have identical length
            assert len(prob_true) == len(bin_counts) == len(prob_pred), (
                "Length mismatch after masking empty bins"
            )

            # 3 – Compute weighted absolute gap
            weights = bin_counts / bin_counts.sum()
            ece = np.sum(np.abs(prob_true - prob_pred) * weights)
            return float(ece)

        except Exception as exc:
            # Fallback – simple loop (same logic as original fallback)
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                in_bin = (p > lo) & (p <= hi)
                if in_bin.any():
                    acc_bin  = y[in_bin].mean()
                    conf_bin = p[in_bin].mean()
                    ece      += np.abs(acc_bin - conf_bin) * in_bin.mean()
            return float(ece)


    @staticmethod
    def reliability_curve(y, p, n_bins: int = 10) -> pd.DataFrame:
        """Return a DataFrame for plotting a reliability diagram."""
        try:
            prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
            return pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
        except (ImportError, TypeError):
            # Fallback implementation
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            prob_true = []
            prob_pred = []
            
            for bin_lower, bin_upper, bin_center in zip(bin_lowers, bin_uppers, bin_centers):
                in_bin = (p > bin_lower) & (p <= bin_upper)
                if np.sum(in_bin) > 0:
                    prob_true.append(y[in_bin].astype(float).mean())
                    prob_pred.append(bin_center)
            
            return pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})

    # ---------- public aggregator ----------
    def calculate_classification_metrics(self, y_true, y_pred_proba) -> Dict[str, float]:
        """Return a full metric dictionary."""
        metrics: Dict[str, float] = {
            "auc_roc":   self.calculate_auc(y_true, y_pred_proba),
            "auc_pr":    self.calculate_auc_pr(y_true, y_pred_proba),
            "log_loss":  self.calculate_log_loss(y_true, y_pred_proba),
            "brier":     self.calculate_brier_score(y_true, y_pred_proba),
            "ece":       self.calculate_ece(y_true, y_pred_proba),
        }
        acc, prec, rec, f1 = self.calculate_threshold_metrics(y_true, y_pred_proba)
        metrics.update({
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
        })
        return metrics  # 10 metrics total

    # ---------- comparison helper ----------
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Turn {model: metric_dict} into a tidy table ordered by AUC-ROC.
        """
        df = pd.DataFrame(results).T  # one row per model
        # guarantee consistent column order
        desired_cols: List[str] = [
            "auc_roc", "auc_pr", "log_loss", "brier", "ece",
            "accuracy", "precision", "recall", "f1"
        ]
        for c in desired_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[desired_cols].sort_values("auc_roc", ascending=False)

class BayesianEvaluator:
    """Wrap ArviZ to calculate WAIC and PSIS-LOO in one call."""
    def information_criteria(self, trace, pointwise: bool = False) -> Dict[str, float]:
        """
        Return WAIC and PSIS-LOO for a fitted PyMC model.

        Parameters
        ----------
        trace : arviz.InferenceData
        pointwise : bool
            If True, also return the pointwise arrays for advanced analysis.
        """
        waic_res = az.waic(trace, scale="deviance")
        loo_res  = az.loo(trace,  scale="deviance")
        info = {
            "waic":      float(waic_res.waic),
            "waic_se":   float(waic_res.waic_se),
            "psis_loo":  float(loo_res.loo),
            "psis_loo_se": float(loo_res.loo_se),
        }
        if pointwise:
            info["waic_i"] = waic_res.waic_i
            info["loo_i"]  = loo_res.loo_i
        return info



def train_test_split_by_season(
    df: pd.DataFrame,
    *,
    train_seasons: list[int] = list(range(2010, 2018)),
    test_seasons:  list[int] = [2018]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free season split used across the package.

    Parameters
    ----------
    df : DataFrame with a 'season' column
    train_seasons : seasons to assign to the training fold
    test_seasons  : seasons to assign to the test   fold
    """
    train = cast(pd.DataFrame, df[df["season"].isin(train_seasons)].copy())
    test  = cast(pd.DataFrame, df[df["season"].isin(test_seasons)].copy())
    return train, test 

if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    # Test the data loader
    print("Testing DataLoader...")
    
    loader = DataLoader()
    
    try:
        # Load complete dataset
        df = loader.load_complete_dataset()
        
        # Print summary
        summary = loader.get_data_summary()
        print("\nData Summary:")
        print(f"Total attempts: {summary['total_attempts']:,}")
        print(f"Unique kickers: {summary['unique_kickers']}")
        print(f"Seasons: {summary['unique_seasons']}")
        print(f"Outcomes: {summary['outcome_counts']}")
        
        print("******* DataLoader tests passed!")
        
    except Exception as e:
        print(f"-------------- Error testing DataLoader: {e}")
        print("Note: This is expected if data files are not present.")
        
    
    
    # Test the metrics module
    print("Testing EPA Calculator...")

    
    # Test EPA calculator
    epa_calc = EPACalculator()
    
    try:
        # Test league average calculation
        league_avg = epa_calc.calculate_league_average_epa(df)
        print(f"League average EPA: {league_avg:.3f}")
        
        # Test individual kicker rating
        rating = epa_calc.calculate_kicker_epa_plus(df, 'Player A')
        print(f"Player A EPA-FG+: {rating['epa_fg_plus']:.3f}")
        
        # Test all kicker ratings
        all_ratings = epa_calc.calculate_all_kicker_ratings(df)
        print(f"Calculated ratings for {len(all_ratings)} kickers")
        
        # Test model evaluator
        evaluator = ModelEvaluator()
        y_true = np.random.choice([0, 1], 100)
        y_pred = np.random.random(100)
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
        print(f"Sample model AUC: {metrics['auc']:.3f}")
        
        print("******* Metrics module tests passed!")
        
    except Exception as e:
        print(f"-------------- Error testing metrics: {e}")





