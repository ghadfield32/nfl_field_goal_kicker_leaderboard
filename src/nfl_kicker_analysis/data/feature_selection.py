"""
New in v0.2.0
-------------
* Added Random-Forest impurity importance (`compute_rf_importance`)
* Added tri-modal merge and multicollinearity pruning
* Re-worked `select_final_features` to call these helpers

New in v0.3.0
-------------
* Added mutable `FEATURE_LISTS` dictionary for flexible schema management
* Added `DynamicSchema` class to replace hardcoded `_ColumnSchema` 
* Added `make_feature_matrix` helper for consistent X/y construction
* Updated functions to accept explicit schema parameter
"""

import pandas as pd
import numpy as np

# ── NEW: model and importance imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import minmax_scale
import shap
from pathlib import Path
import shapiq
from sklearn.utils import resample
from itertools import combinations
import json

# ── NEW: dataclass and typing imports for DynamicSchema
from dataclasses import dataclass, field
from typing import List, Dict



# ------------------------------------------------------------------
# 🧩 Lightweight runtime schema object
# ------------------------------------------------------------------
@dataclass
class DynamicSchema:
    lists: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def numerical(self):   return self.lists.get("numerical", [])
    @property
    def ordinal(self):     return self.lists.get("ordinal", [])
    @property
    def nominal(self):     return self.lists.get("nominal", [])
    @property
    def target(self):      return self.lists.get("y_variable", [])[0]
    def all_features(self): return (
        self.numerical + self.ordinal + self.nominal
    )

# ── Schema-aware utilities ──────────────────────────────────────────────
def restrict_to_numerical(df: pd.DataFrame, schema: DynamicSchema) -> pd.DataFrame:
    """
    Return a view of `df` that contains only the columns listed under
    schema.numerical. Trust the schema's numerical list as the source of truth.
    """
    return df[schema.numerical].copy()


def update_schema_numerical(schema: DynamicSchema, new_numericals: list[str]) -> None:
    """
    In-place replacement of the numerical list inside the DynamicSchema.
    Keeps a copy of the old list for logging/debugging.
    """
    old = schema.numerical
    schema.lists["numerical"] = sorted(new_numericals)
    added   = sorted(set(new_numericals) - set(old))
    removed = sorted(set(old)            - set(new_numericals))
    print(f"🔄  Schema update → numerical features now = {len(new_numericals)} columns")
    if added:   print(f"   ➕ added   : {added}")
    if removed: print(f"   ➖ removed : {removed}")


def make_feature_matrix(df: pd.DataFrame,
                        schema: DynamicSchema,
                        numeric_only: bool = True
                       ) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) based on the supplied schema."""
    X: pd.DataFrame = restrict_to_numerical(df, schema) if numeric_only else df[schema.all_features()].copy()
    y: pd.Series = df[schema.target]
    return X, y

def train_baseline_model(X, y):
    """
    Fit a RandomForestRegressor on X, y.
    Returns the fitted model.
    """
    # You can adjust hyperparameters as needed
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model



def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    n_jobs: int = 1,
    max_samples: float | int | None = None,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute permutation importances with controlled resource usage.
    
    Parameters
    ----------
    model : estimator
        Fitted model implementing .predict and .score.
    X : pd.DataFrame
        Features.
    y : pd.Series or array
        Target.
    n_repeats : int
        Number of shuffles per feature.
    n_jobs : int
        Number of parallel jobs (avoid -1 on Windows).
    max_samples : float or int, optional
        If float in (0,1], fraction of rows to sample.
        If int, absolute number of rows to sample.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print debug info if True.
        
    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_mean, importance_std.
        Sorted descending by importance_mean.
    """
    if verbose:
        print(f"⏳ Computing permutation importances on {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   n_repeats={n_repeats}, n_jobs={n_jobs}, max_samples={max_samples}")

    # Subsample if requested
    X_sel, y_sel = X, y
    if max_samples is not None:
        if isinstance(max_samples, float):
            nsamp = int(len(X) * max_samples)
        else:
            nsamp = int(max_samples)
        if verbose:
            print(f"   Subsampling to {nsamp} rows for speed")
        X_sel, y_sel = resample(X, y, replace=False, n_samples=nsamp, random_state=random_state)

    try:
        result = permutation_importance(
            model,
            X_sel, y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except OSError as e:
        # Graceful fallback to single job
        if verbose:
            print(f"⚠️  OSError ({e}). Retrying with n_jobs=1")
        result = permutation_importance(
            model,
            X_sel, y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

    # Build and sort DataFrame
    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    if verbose:
        print("✅ Permutation importances computed.")
    return importance_df


def compute_shap_importance(model, X, nsamples=100):
    """
    Compute mean absolute SHAP values per feature.
    Returns a DataFrame sorted by importance.
    """
    # DeepExplainer or TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    # sample for speed
    X_sample = X.sample(n=min(nsamples, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    # For regression, shap_values is a 2D array
    mean_abs_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0),
    })
    mean_abs_shap = mean_abs_shap.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    return mean_abs_shap


def compute_rf_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return impurity-based RF importances."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model has no feature_importances_ attribute")
    return (
        pd.DataFrame(
            {"feature": feature_names,
             "rf_importance": model.feature_importances_}
        )
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )


def merge_and_score_importances(perm_df, shap_df, rf_df) -> pd.DataFrame:
    """Merge the three importance tables and add a combined_score column."""
    merged = (
        perm_df.merge(shap_df, on="feature", how="outer")
               .merge(rf_df,  on="feature", how="outer")
               .fillna(0.0)
    )
    # Min-max normalise each column so weights are comparable
    for col in ["importance_mean", "shap_importance", "rf_importance"]:
        merged[f"norm_{col}"] = minmax_scale(merged[col].values)
    merged["combined_score"] = merged[
        ["norm_importance_mean", "norm_shap_importance", "norm_rf_importance"]
    ].mean(axis=1)
    return merged.sort_values("combined_score", ascending=False).reset_index(drop=True)


def drop_multicollinear(X: pd.DataFrame,
                        ranked_feats: pd.DataFrame,
                        corr_threshold: float = 0.85,
                        method: str = "pearson") -> list[str]:
    """
    Remove the lower-scoring feature from each highly correlated pair.
    """
    corr = X[ranked_feats["feature"]].corr().abs()
    to_drop = set()
    for f1, f2 in combinations(ranked_feats["feature"], 2):
        if corr.loc[f1, f2] > corr_threshold:
            # keep the one with higher combined_score
            better = f1 if ranked_feats.set_index("feature").loc[f1,"combined_score"] \
                     >= ranked_feats.set_index("feature").loc[f2,"combined_score"] else f2
            worse  = f2 if better == f1 else f1
            to_drop.add(worse)
    return [f for f in ranked_feats["feature"] if f not in to_drop]


def filter_permutation_features(
    importance_df: pd.DataFrame,
    threshold: float
) -> list[str]:
    """
    Return features whose permutation importance_mean exceeds threshold.
    """
    kept = importance_df.loc[
        importance_df["importance_mean"] > threshold, "feature"
    ]
    return kept.tolist()


def filter_shap_features(
    importance_df: pd.DataFrame,
    threshold: float
) -> list[str]:
    """
    Return features whose shap_importance exceeds threshold.
    """
    kept = importance_df.loc[
        importance_df["shap_importance"] > threshold, "feature"
    ]
    return kept.tolist()


def select_final_features(perm_df,
                          shap_df,
                          rf_df,
                          X: pd.DataFrame,
                          schema: DynamicSchema,
                          perm_thresh: float = 0.01,
                          shap_thresh: float = 0.01,
                          rf_thresh: float = 0.01,
                          corr_threshold: float = 0.85) -> list[str]:
    """
    Return the final feature list after:
      1. Individual thresholding on each importance type
      2. Union/intersection logic (here: *intersection*)
      3. Combined-score ranking
      4. Multicollinearity pruning
    """
    # --- step 1: individual screens ---
    keep_perm = perm_df.loc[perm_df["importance_mean"] > perm_thresh, "feature"]
    keep_shap = shap_df.loc[shap_df["shap_importance"]   > shap_thresh, "feature"]
    keep_rf   = rf_df.loc[rf_df["rf_importance"]         > rf_thresh,   "feature"]
    intersect = set(keep_perm) & set(keep_shap) & set(keep_rf)

    # --- step 2: combine & rank only those ---
    merged = merge_and_score_importances(
        perm_df[perm_df["feature"].isin(intersect)],
        shap_df[shap_df["feature"].isin(intersect)],
        rf_df[rf_df["feature"].isin(intersect)]
    )

    # --- step 3: drop multicollinear ---
    final_feats = drop_multicollinear(X, merged, corr_threshold)

    return final_feats


# ============== BACKWARD COMPATIBILITY HELPERS ==============

def select_final_features_legacy(
    perm_feats: list[str],
    shap_feats: list[str],
    mode: str = "intersection"
) -> list[str]:
    """
    DEPRECATED: Legacy version for backward compatibility.
    Use select_final_features with DataFrame inputs for enhanced functionality.
    """
    import warnings
    warnings.warn(
        "select_final_features_legacy is deprecated. Use select_final_features with DataFrame inputs.", 
        DeprecationWarning,
        stacklevel=2
    )
    set_perm = set(perm_feats)
    set_shap = set(shap_feats)
    if mode == "union":
        final = set_perm | set_shap
    else:
        final = set_perm & set_shap
    return sorted(final)


# ==============================================================


def load_final_features(
    file_path: str = "data/models/features/final_features.txt"
) -> list[str]:
    """
    Read the newline-delimited feature names file and return as a list.
    """
    with open(file_path, "r") as fp:
        return [line.strip() for line in fp if line.strip()]


def filter_to_final_features(df: pd.DataFrame,
                             final_feats_file: str,
                             schema: DynamicSchema,
                             id_cols: list[str] = None
                            ) -> pd.DataFrame:
    """Return df[id + final + y] using the dynamic schema."""
    final_feats = load_final_features(final_feats_file)
    id_cols = id_cols or []
    missing = set(id_cols + final_feats + [schema.target]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after filtering: {missing}")
    return df[id_cols + final_feats + [schema.target]].copy()


if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
    import os

    # ------------------------------------------------------------------
    # 🔧 Single source of truth for column roles – edit freely
    # ------------------------------------------------------------------
    FEATURE_LISTS = {
        "numerical": [
            "attempt_yards", "age_at_attempt", "distance_squared",
            "career_length_years", "season_progress", "rolling_success_rate",
            "current_streak", "distance_zscore", "distance_percentile",
        ],
        "ordinal":  ["season", "week", "month", "day_of_year"],
        "nominal":  [
            "kicker_id", "is_long_attempt", "is_very_long_attempt",
            "is_rookie_attempt", "distance_category", "experience_category",
        ],
        "y_variable": ["success"],
    }

    # ➊  Build schema from the dict
    schema = DynamicSchema(FEATURE_LISTS)

    # ➋  Load & feature-engineer
    loader = DataLoader()
    df_raw = loader.load_complete_dataset()
    engineer = FeatureEngineer()
    df_feat = engineer.create_all_features(df_raw)
    print("---------------df_feat---------------")
    print(df_feat.head())
    print("---------------df_feat.columns---------------")
    print(df_feat.columns)
    print("---------------schema.all_features()---------------")
    print(schema.all_features())

    # ➌  Make X / y using the new helper
    X, y = make_feature_matrix(df_feat, schema)
    
    print(f"\n📊 Starting enhanced feature selection with {X.shape[1]} features")
    print(f"   Schema contains {len(schema.all_features())} total features defined")

    # ➍  Run the tri-modal importance pipeline exactly as before
    print("\n🌲 Training Random Forest model...")
    model = train_baseline_model(X, y)
    
    print("\n⚡ Computing permutation importance...")
    perm_df = compute_permutation_importance(model, X, y, max_samples=0.3)
    
    print("\n🔍 Computing SHAP importance...")
    shap_df = compute_shap_importance(model, X, nsamples=100)
    
    print("\n🌳 Computing Random Forest importance...")
    rf_df = compute_rf_importance(model, X.columns.tolist())

    # 📊 Display top 10 features for each importance metric
    print("\n📈 Top 10 Features by Importance Metric:")
    print("\nPermutation Importance:")
    print(perm_df.head(10)[["feature", "importance_mean"]].to_string(index=False))
    
    print("\nSHAP Importance:")
    print(shap_df.head(10)[["feature", "shap_importance"]].to_string(index=False))
    
    print("\nRandom Forest Importance:")
    print(rf_df.head(10)[["feature", "rf_importance"]].to_string(index=False))

    # 🔬 Select & de-correlate with detailed output
    print("\n🔍 Analyzing feature correlations...")
    corr = X[X.columns].corr().abs()
    
    # Find highly correlated pairs before feature selection
    high_corr_pairs = []
    for f1, f2 in combinations(X.columns, 2):
        if corr.loc[f1, f2] > 0.85:  # Using same threshold as drop_multicollinear
            high_corr_pairs.append({
                'feature1': f1,
                'feature2': f2,
                'correlation': corr.loc[f1, f2]
            })
    
    if high_corr_pairs:
        print("\n📊 Highly correlated feature pairs (correlation > 0.85):")
        for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
            print(f"\n{pair['feature1']} ↔️ {pair['feature2']}")
            print(f"Correlation: {pair['correlation']:.3f}")
            
            # Get importance scores for both features
            f1_scores = {
                'perm': float(perm_df[perm_df.feature == pair['feature1']].importance_mean.iloc[0]),
                'shap': float(shap_df[shap_df.feature == pair['feature1']].shap_importance.iloc[0]),
                'rf': float(rf_df[rf_df.feature == pair['feature1']].rf_importance.iloc[0])
            }
            f2_scores = {
                'perm': float(perm_df[perm_df.feature == pair['feature2']].importance_mean.iloc[0]),
                'shap': float(shap_df[shap_df.feature == pair['feature2']].shap_importance.iloc[0]),
                'rf': float(rf_df[rf_df.feature == pair['feature2']].rf_importance.iloc[0])
            }
            
            print(f"{pair['feature1']} importance scores:")
            print(f"  Permutation: {f1_scores['perm']:.4f}")
            print(f"  SHAP: {f1_scores['shap']:.4f}")
            print(f"  RF: {f1_scores['rf']:.4f}")
            
            print(f"{pair['feature2']} importance scores:")
            print(f"  Permutation: {f2_scores['perm']:.4f}")
            print(f"  SHAP: {f2_scores['shap']:.4f}")
            print(f"  RF: {f2_scores['rf']:.4f}")
            
            # Calculate average importance
            f1_avg = sum(f1_scores.values()) / 3
            f2_avg = sum(f2_scores.values()) / 3
            
            keeper = pair['feature1'] if f1_avg >= f2_avg else pair['feature2']
            dropped = pair['feature2'] if keeper == pair['feature1'] else pair['feature1']
            print(f"\n➡️ Decision: Keep {keeper} (avg importance: {max(f1_avg, f2_avg):.4f})")
            print(f"❌ Drop {dropped} (avg importance: {min(f1_avg, f2_avg):.4f})")
    else:
        print("No highly correlated feature pairs found.")

    # Run feature selection
    final_features = select_final_features(
        perm_df, shap_df, rf_df, X, schema,
        perm_thresh=0.005, shap_thresh=0.005, rf_thresh=0.005
    )
    print(f"---------------final_features---------------")
    print(final_features)
    # output final_features to final_features.txt
    with open("data/models/features/final_features.txt", "w") as f:
        for feat in final_features:
            f.write(feat + "\n")
            
    
    # read final_features.txt
    with open("data/models/features/final_features.txt", "r") as f:
        final_features = [line.strip() for line in f]
    print(f"---------------final_features---------------")
    print(final_features)
    numeric_final = [f for f in final_features if f in schema.numerical]

    print(f"\n✨ Final feature count: {len(numeric_final)}")
    print("Selected features:")
    for feat in numeric_final:
        print(f"  • {feat}")

    # 🔄 Push into schema so every later stage sees the new list
    update_schema_numerical(schema, numeric_final)

    # output final_features from schema
    FEATURE_LISTS = schema.lists
    print(f"---------------FEATURE_LISTS---------------")
    print(FEATURE_LISTS)
    
    
