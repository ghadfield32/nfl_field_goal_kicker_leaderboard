"""
Data preprocessing module for NFL kicker analysis.
Handles filtering, feature selection, and preparation for modeling.

New in v0.4.0
--------------
* Added **inverse‑preprocessing** utilities so that any matrix produced by the
  fitted `ColumnTransformer` can be projected back into human‑readable feature
  space.  This is handy for debugging, error analysis, or piping model outputs
  into post‑processing code that expects the raw‑scale feature values.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union, cast, Any, TypeVar, Protocol
from datetime import datetime

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse

from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
from src.nfl_kicker_analysis.data.feature_schema import FeatureSchema
from src.nfl_kicker_analysis import config


# Type variables for better type hints
T = TypeVar('T')
MatrixType = Union[np.ndarray, sparse.spmatrix]


class Transformer(Protocol):
    """Protocol for scikit-learn transformers."""
    def fit_transform(self, X: Any, y: Any = None) -> MatrixType: ...
    def transform(self, X: Any) -> MatrixType: ...
    def inverse_transform(self, X: MatrixType) -> np.ndarray: ...


class DataPreprocessor:
    """Handles data preprocessing, feature engineering, and inversion."""
    
    def __init__(self):
        """Create a preprocessor with defaults from central config."""
        # 1️⃣ Initialize all attributes (as before)
        self.MIN_DISTANCE: int | None = None
        self.MAX_DISTANCE: int | None = None
        self.MIN_KICKER_ATTEMPTS: int | None = None
        self.SEASON_TYPES: list[str] | None = None
        
        self.NUMERICAL_FEATURES: list[str] = []
        self.ORDINAL_FEATURES: list[str] = []
        self.NOMINAL_FEATURES: list[str] = []
        self.TARGET: str = "success"
        
        self.INCLUDE_PERFORMANCE_HISTORY: bool | None = None
        self.INCLUDE_STATISTICAL_FEATURES: bool | None = None
        self.INCLUDE_PLAYER_STATUS: bool | None = None
        self.PERFORMANCE_WINDOW: int | None = None
        
        # Runtime artifacts
        self.feature_engineer = FeatureEngineer()
        self.raw_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.schema: FeatureSchema | None = None
        self.column_transformer_: ColumnTransformer | None = None
        self._feature_cols_: List[str] | None = None
        
        # 2️⃣ Immediately inject defaults from the global config instance
        from src.nfl_kicker_analysis import config
        defaults = config  # config is already the Config() instance
        
        self.update_config(
            min_distance=defaults.MIN_DISTANCE,
            max_distance=defaults.MAX_DISTANCE,
            min_kicker_attempts=defaults.MIN_KICKER_ATTEMPTS,
            season_types=list(defaults.SEASON_TYPES),
            include_performance_history=True,
            include_statistical_features=False,
            include_player_status=True,
            performance_window=12
        )
        
        # 3️⃣ Inject the feature lists from your central FEATURE_LISTS
        from src.nfl_kicker_analysis.config import FEATURE_LISTS
        self.update_feature_lists(
            numerical=FEATURE_LISTS["numerical"],
            ordinal=FEATURE_LISTS["ordinal"],
            nominal=FEATURE_LISTS["nominal"],
            y_variable=FEATURE_LISTS["y_variable"]
        )
        
        print("******* DataPreprocessor initialized with defaults")



    def update_feature_lists(self, 
                           numerical: Optional[List[str]] = None,
                           ordinal: Optional[List[str]] = None,
                           nominal: Optional[List[str]] = None,
                           y_variable: Optional[List[str]] = None):
        """
        Update feature lists for easy experimentation.
        
        Args:
            numerical: List of numerical features to use
            ordinal: List of ordinal features to use  
            nominal: List of nominal categorical features to use
            y_variable: List containing target variable name
        """
        if numerical is not None:
            self.NUMERICAL_FEATURES = numerical
        if ordinal is not None:
            self.ORDINAL_FEATURES = ordinal
        if nominal is not None:
            self.NOMINAL_FEATURES = nominal
        if y_variable is not None:
            self.TARGET = y_variable[0]  # Single target assumed
            
        print("******* Feature lists updated")
    
    def update_config(self, 
                     min_distance: Optional[int] = None,
                     max_distance: Optional[int] = None,
                     min_kicker_attempts: Optional[int] = None,
                     season_types: Optional[List[str]] = None,
                     include_performance_history: Optional[bool] = None,
                     include_statistical_features: Optional[bool] = None,
                     include_player_status: Optional[bool] = None,
                     performance_window: Optional[int] = None):
        """
        Update preprocessing configuration.
        
        Args:
            min_distance: Minimum field goal distance to include
            max_distance: Maximum field goal distance to include
            min_kicker_attempts: Minimum attempts required per kicker
            season_types: List of season types to include
            include_performance_history: Whether to include performance history features
            include_statistical_features: Whether to include statistical features
            performance_window: Window size for rolling performance features
        """
        if min_distance is not None:
            self.MIN_DISTANCE = min_distance
        if max_distance is not None:
            self.MAX_DISTANCE = max_distance
        if min_kicker_attempts is not None:
            self.MIN_KICKER_ATTEMPTS = min_kicker_attempts
        if season_types is not None:
            self.SEASON_TYPES = season_types
        if include_performance_history is not None:
            self.INCLUDE_PERFORMANCE_HISTORY = include_performance_history
        if include_statistical_features is not None:
            self.INCLUDE_STATISTICAL_FEATURES = include_statistical_features
        if include_player_status is not None:
            self.INCLUDE_PLAYER_STATUS = include_player_status
        if performance_window is not None:
            self.PERFORMANCE_WINDOW = performance_window
            
        print("******* Configuration updated")
    
    def _validate_config(self):
        """Validate that required configuration is set."""
        missing = []
        if self.MIN_DISTANCE is None:
            missing.append("MIN_DISTANCE")
        if self.MAX_DISTANCE is None:
            missing.append("MAX_DISTANCE")
        if self.MIN_KICKER_ATTEMPTS is None:
            missing.append("MIN_KICKER_ATTEMPTS")
        if self.SEASON_TYPES is None:
            missing.append("SEASON_TYPES")
        if self.INCLUDE_PERFORMANCE_HISTORY is None:
            missing.append("INCLUDE_PERFORMANCE_HISTORY")
        if self.INCLUDE_STATISTICAL_FEATURES is None:
            missing.append("INCLUDE_STATISTICAL_FEATURES")
        if self.INCLUDE_PLAYER_STATUS is None:
            missing.append("INCLUDE_PLAYER_STATUS")
        if self.PERFORMANCE_WINDOW is None:
            missing.append("PERFORMANCE_WINDOW")
            
        if missing:
            raise ValueError(f"Configuration not set. Please call update_config() first. Missing: {missing}")
    
    def filter_season_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data by season type."""
        self._validate_config()
        assert self.SEASON_TYPES is not None  # guaranteed after validation
        filtered_df = cast(pd.DataFrame, df[df['season_type'].isin(self.SEASON_TYPES)].copy())
        print(f"******* Filtered to {self.SEASON_TYPES} season(s): {len(filtered_df):,} attempts")
        return filtered_df
    
    def filter_blocked_field_goals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out blocked field goals since they don't reflect kicker skill."""
        filtered_df = cast(pd.DataFrame, df[df['field_goal_result'] != 'Blocked'].copy())
        removed = len(df) - len(filtered_df)
        print(f"******* Filtered out blocked field goals")
        print(f"   Removed {removed} blocked attempts, kept {len(filtered_df):,}")
        return filtered_df
    
    def filter_retired_injured_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally drop Retired/Injured attempts (status created upstream).
        Controlled by config.FILTER_RETIRED_INJURED.
        """
        if "player_status" not in df.columns or not config.FILTER_RETIRED_INJURED:
            return df                # no-op if flag is False or column missing

        keep_mask = df["player_status"] != "Retired/Injured"
        dropped   = (~keep_mask).sum()
        players   = df.loc[~keep_mask, "player_name"].nunique()

        print(f"🗑️  Filtered {dropped} attempts from {players} retired/injured players")
        return cast(pd.DataFrame, df[keep_mask].copy())
    
    def filter_distance_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data by distance range to remove outliers."""
        self._validate_config()
        filtered_df = cast(pd.DataFrame, df[
            (df['attempt_yards'] >= self.MIN_DISTANCE) &
            (df['attempt_yards'] <= self.MAX_DISTANCE)
        ].copy())
        
        removed = len(df) - len(filtered_df)
        print(f"******* Filtered distance range {self.MIN_DISTANCE}-{self.MAX_DISTANCE} yards")
        print(f"   Removed {removed} extreme attempts, kept {len(filtered_df):,}")
        return filtered_df
    
    def filter_min_attempts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out kickers with too few attempts."""
        self._validate_config()
        # Count attempts per kicker
        kicker_counts = df['player_name'].value_counts()
        valid_kickers = kicker_counts[kicker_counts >= self.MIN_KICKER_ATTEMPTS]
        # Convert index to list safely using pandas Series methods
        valid_kicker_names = cast(pd.Series, valid_kickers).index.tolist()
        
        # Filter to valid kickers only
        filtered_df = cast(pd.DataFrame, df[df['player_name'].isin(valid_kicker_names)].copy())
        
        removed_kickers = len(kicker_counts) - len(valid_kickers)
        print(f"******* Filtered kickers with <{self.MIN_KICKER_ATTEMPTS} attempts")
        print(f"   Removed {removed_kickers} kickers, kept {len(valid_kickers)}")
        print(f"   Final dataset: {len(filtered_df):,} attempts")
        
        return filtered_df
    
    def _get_selected_features(self) -> list[str]:
        """
        Return the columns that will actually be fed into the model.

        Priority order
        --------------
        1. If self.schema has been built, trust **only** the columns that the
           schema says are present in `processed_data`.  This guarantees that
           every returned name exists after filtering & feature-engineering.
        2. Otherwise fall back to the raw configuration lists (old behaviour).

        The list is deduplicated while preserving order.
        """
        if self.schema is not None:
            feats = (
                self.schema.numerical +
                self.schema.ordinal   +
                self.schema.nominal
            )
        else:  # happens only before preprocess_complete()
            feats = (
                self.NUMERICAL_FEATURES +
                self.ORDINAL_FEATURES   +
                self.NOMINAL_FEATURES
            )
        # Preserve order, drop dups
        return list(dict.fromkeys(feats))

    
    def _build_schema(self, df: pd.DataFrame) -> FeatureSchema:
        """Build the feature schema based on selected features."""
        selected_features = self._get_selected_features()
        
        # Filter feature lists to only include features that exist in the dataframe
        available_features = set(df.columns)
        
        numerical = [f for f in self.NUMERICAL_FEATURES if f in available_features]
        ordinal = [f for f in self.ORDINAL_FEATURES if f in available_features]
        nominal = [f for f in self.NOMINAL_FEATURES if f in available_features]
        
        schema = FeatureSchema(
            numerical=numerical,
            binary=[],  # Binary features are now in nominal
            ordinal=ordinal,
            nominal=nominal,
            target=self.TARGET,
        )
        
        # Validate schema
        try:
            schema.assert_in_dataframe(df)
        except AssertionError as e:
            print(f"Warning: Schema validation failed: {e}")
            print("Available features:", list(available_features))
            print("Requested features:", selected_features)
        
        return schema

    def make_column_transformer(self) -> ColumnTransformer:
        """
        Return a scikit-learn ColumnTransformer based on the feature schema.
        Call after `preprocess_complete()`.
        """
        if self.schema is None:
            raise AttributeError("Run preprocess_complete() before building transformers.")

        # Numeric pipeline - standard scaling
        numeric_pipe = Pipeline(
            steps=[
                ("scale", StandardScaler())
            ]
        )

        # Build transformers list
        transformers = []
        
        if self.schema.numerical:
            transformers.append(("num_scaled", numeric_pipe, self.schema.numerical))
            
        if self.schema.binary:
            transformers.append(("binary_passthrough", "passthrough", self.schema.binary))
            
        if self.schema.ordinal:
            transformers.append(("ordinal_passthrough", "passthrough", self.schema.ordinal))
            
        if self.schema.nominal:
            transformers.append(("nominal_onehot", 
                               OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                               self.schema.nominal))

        # Categorical features are now included in nominal features
        # No separate categorical handling needed

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ct
    
    def preprocess_slice(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run the exact notebook filters on an arbitrary slice *without* mutating
           global state (so train/test can differ)."""
        self._validate_config()

        df = raw_df.copy()
        print(f"\n🔍 Preprocessing slice of shape {df.shape}")
        
        # Check if features are already engineered (to avoid double processing)
        features_already_present = 'success' in df.columns and 'player_status' in df.columns
        print(f"Features already engineered: {features_already_present}")
        
        if not features_already_present:
            # Only run feature engineering if not already done
            print("Running feature engineering...")
            df = self.feature_engineer.create_all_features(
                df,
                include_performance_history=self.INCLUDE_PERFORMANCE_HISTORY,
                performance_window=self.PERFORMANCE_WINDOW,
                include_player_status=self.INCLUDE_PLAYER_STATUS,
            )
            if self.INCLUDE_STATISTICAL_FEATURES:
                df = self.feature_engineer.create_statistical_features(df)
            print(f"After feature engineering: {df.shape}")
        
        # Apply filtering steps
        print("\nApplying filters:")
        print(f"Initial rows: {len(df)}")
        
        df = self.filter_season_type(df)
        print(f"After season type filter: {len(df)}")
        
        df = self.filter_blocked_field_goals(df)
        print(f"After blocked FG filter: {len(df)}")
        
        df = self.filter_distance_range(df)
        print(f"After distance range filter: {len(df)}")
        
        df = self.filter_min_attempts(df)
        print(f"After min attempts filter: {len(df)}") 
        
        # Filter retired/injured players (needs player_status column)
        df = self.filter_retired_injured_players(df)
        print(f"After retired/injured filter: {len(df)}")

        self._build_schema(df)
        print(f"\nFinal preprocessed shape: {df.shape}")
        print(f"Number of kickers: {df['kicker_id'].nunique()}")
        print(f"Success rate: {df['success'].mean():.4f}")
        return df


    def preprocess_complete(self, raw_df: pd.DataFrame, *, inplace: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw_df: Raw merged DataFrame
            inplace: If True (default), store results in instance attributes.
                    If False, behaves like preprocess_slice() and leaves internal state untouched.
            
        Returns:
            Fully preprocessed DataFrame ready for modeling
        """
        if inplace:
            # Validate configuration first
            self._validate_config()
            
            self.raw_data = raw_df.copy()
            
            print("Starting complete preprocessing pipeline...")
            print(f"Configuration: {self.MIN_DISTANCE}-{self.MAX_DISTANCE} yards, "
                  f"min {self.MIN_KICKER_ATTEMPTS} attempts, {self.SEASON_TYPES} seasons")
            
            self.processed_data = self.preprocess_slice(raw_df)
            self.schema = self._build_schema(self.processed_data)
            
            print(f"******* Preprocessing complete: {len(self.processed_data):,} attempts ready for modeling")
            print(f"******* Features selected: {len(self._get_selected_features())} total")
            return self.processed_data
        else:
            return self.preprocess_slice(raw_df)
    
    def get_feature_summary(self) -> Dict:
        """Get summary of selected features by category."""
        if self.processed_data is None:
            raise ValueError("No processed data available")
            
        selected_features = self._get_selected_features()
        available_features = set(self.processed_data.columns)
        
        summary = {
            'total_selected': len(selected_features),
            'total_available': len(available_features),
            'numerical': [f for f in self.NUMERICAL_FEATURES if f in available_features],
            'ordinal': [f for f in self.ORDINAL_FEATURES if f in available_features],
            'nominal': [f for f in self.NOMINAL_FEATURES if f in available_features],
            'missing_features': [f for f in selected_features if f not in available_features],
        }
        return summary
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing steps and results."""
        if self.processed_data is None:
            raise ValueError("No processed data available")
            
        summary = {
            'original_size': len(self.raw_data) if self.raw_data is not None else 0,
            'final_size': len(self.processed_data),
            'unique_kickers': self.processed_data['player_name'].nunique(),
            'success_rate': self.processed_data[self.TARGET].mean(),
            'distance_range': (
                self.processed_data['attempt_yards'].min(),
                self.processed_data['attempt_yards'].max()
            ),
            'config': {
                'min_distance': self.MIN_DISTANCE,
                'max_distance': self.MAX_DISTANCE,
                'min_kicker_attempts': self.MIN_KICKER_ATTEMPTS,
                'season_types': self.SEASON_TYPES,
                'include_performance_history': self.INCLUDE_PERFORMANCE_HISTORY,
                'include_statistical_features': self.INCLUDE_STATISTICAL_FEATURES,
            },
            'features': self.get_feature_summary()
        }
        return summary

    def fit_transform_features(
        self,
        *,                     # keyword-only
        drop_missing: bool = True
    ) -> tuple[MatrixType, np.ndarray]:
        """
        Fit the ColumnTransformer and return X / y.

        Parameters
        ----------
        drop_missing : If True (default) silently removes any feature that is
            absent from `processed_data` after printing a warning.  If False,
            the old strict KeyError behaviour is retained.
        """
        if self.processed_data is None:
            raise ValueError("Run preprocess_complete() first.")

        feature_cols = self._get_selected_features()
        missing = [c for c in feature_cols if c not in self.processed_data.columns]

        if missing:
            if drop_missing:
                print(
                    f"⚠️  [DataPreprocessor] Dropping {len(missing)} "
                    f"unavailable feature(s): {missing}"
                )
                feature_cols = [c for c in feature_cols if c not in missing]
            else:
                raise KeyError(
                    f"These features are missing from processed_data: {missing}"
                )

        # Build & fit transformer on the *pruned* list
        self.column_transformer_ = self.make_column_transformer()
        self._feature_cols_      = feature_cols
        X = self.column_transformer_.fit_transform(self.processed_data[feature_cols])
        y = self.processed_data[self.TARGET].values
        return X, y


    
    def transform_features(self, df: pd.DataFrame) -> MatrixType:
        """Transform a *new* DataFrame using the already‑fitted transformer.
        
        Args:
            df: DataFrame to transform, must have same columns as training data
            
        Returns:
            Transformed feature matrix (sparse or dense)
            
        Raises:
            ValueError: If transformer hasn't been fitted yet
        """
        if self.column_transformer_ is None:
            raise ValueError("Transformer not fitted – call fit_transform_features() first.")
            
        feature_cols = self._get_selected_features()
        X = self.column_transformer_.transform(df[feature_cols])
        return cast(MatrixType, X)
    
    def invert_preprocessing(self, X_transformed: MatrixType) -> pd.DataFrame:
        """
        Version-safe inverse transform.
        Works on scikit-learn ≥0.24 (native) and ≤0.23 (manual fallback).
        """
        if self.column_transformer_ is None or self._feature_cols_ is None:
            raise ValueError("Transformer not fitted – call fit_transform_features() first.")

        # --- 1 ▸ Modern sklearn: just call the native helper -----------------
        if hasattr(self.column_transformer_, "inverse_transform"):
            X_inv = self.column_transformer_.inverse_transform(X_transformed)
            return pd.DataFrame(X_inv, columns=self._feature_cols_)

        # --- 2 ▸ Legacy sklearn: manual reconstruction -----------------------
        X_dense = X_transformed.toarray() if sparse.issparse(X_transformed) else np.asarray(X_transformed)

        col_arrays, current = [], 0
        for name, trans, cols in self.column_transformer_.transformers_:
            # -------- width calculation WITHOUT touching the transformer -----
            if name == "num_scaled":
                width = len(cols)  # numeric slice = number of original columns
            elif name == "nominal_onehot":
                enc = cast(OneHotEncoder, trans)
                width = sum(len(c) for c in enc.categories_)
            else:                           # passthrough groups
                width = len(cols)

            slice_ = X_dense[:, current: current + width]
            current += width

            # -------- inverse for each block ---------------------------------
            if name == "num_scaled":
                scaler = cast(StandardScaler, trans.named_steps["scale"])
                slice_inv = (slice_ * scaler.scale_) + scaler.mean_     # docs :contentReference[oaicite:2]{index=2}
                col_arrays.append(slice_inv)

            elif name == "nominal_onehot":
                enc = cast(OneHotEncoder, trans)
                slice_inv = enc.inverse_transform(slice_)               # docs :contentReference[oaicite:3]{index=3}
                col_arrays.append(slice_inv)

            else:                                                       # passthrough
                col_arrays.append(slice_)

        X_inv_full = np.column_stack(col_arrays)
        return pd.DataFrame(X_inv_full, columns=self._feature_cols_)

    def signature_hash(self) -> str:
        """
        Create a stable hash of the preprocessor's configuration for caching.
        
        Returns:
            SHA-1 hash (8 characters) of the configuration dict
        """
        import json
        import hashlib
        
        # Create a config dict with all the settings that affect preprocessing
        config_dict = {
            'min_distance': self.MIN_DISTANCE,
            'max_distance': self.MAX_DISTANCE,
            'min_kicker_attempts': self.MIN_KICKER_ATTEMPTS,
            'season_types': self.SEASON_TYPES,
            'include_performance_history': self.INCLUDE_PERFORMANCE_HISTORY,
            'include_statistical_features': self.INCLUDE_STATISTICAL_FEATURES,
            'include_player_status': self.INCLUDE_PLAYER_STATUS,
            'performance_window': self.PERFORMANCE_WINDOW,
            'numerical_features': self.NUMERICAL_FEATURES,
            'ordinal_features': self.ORDINAL_FEATURES,
            'nominal_features': self.NOMINAL_FEATURES,
            'target': self.TARGET,
        }
        
        # Create stable hash
        raw = json.dumps(config_dict, sort_keys=True).encode()
        return hashlib.sha1(raw).hexdigest()[:8]



if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_selection import (
        DynamicSchema,
        filter_to_final_features,
        update_schema_numerical,
    )

    # ─── 1 Load raw data ──────────────────────────────────────────
    loader = DataLoader()
    df_raw = loader.load_complete_dataset()
    
    # ─── 2 Feature engineering pass ───────────────────────────────
    engineer = FeatureEngineer()
    df_feat = engineer.create_all_features(df_raw)

    for category, details in engineer.get_available_features(df_feat).items():
        print(f"-- {category} --")
        for feat, uniques in details.items():
            print(f"   {feat}: {len(uniques)} unique | sample {uniques[:5] if uniques else '...'}")

    # ─── 3 Define all tunables in one place ─────────────────────
    CONFIG = {
        'min_distance': 20,
        'max_distance': 60,
        'min_kicker_attempts': 8,
        'season_types': ['Reg', 'Post'],  # now include playoffs
        'include_performance_history': True,
        'include_statistical_features': False,
        'include_player_status': True,  # ✅ FIX: Added missing parameter
        'performance_window': 12,
    }


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
            "kicker_id", "kicker_idx", "is_long_attempt", "is_very_long_attempt",
            "is_rookie_attempt", "distance_category", "experience_category",
        ],
        "y_variable": ["success"],
    }

    # ➊  Build schema from the dict
    schema = DynamicSchema(FEATURE_LISTS)
    
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

    pre = DataPreprocessor()
    pre.update_config(**CONFIG)
    pre.update_feature_lists(**FEATURE_LISTS)
    _ = pre.preprocess_complete(df_feat)
    X, y = pre.fit_transform_features()

    print("First 5 rows after inverse‑transform round‑trip →")
    print(pre.invert_preprocessing(X[:5]).head())





