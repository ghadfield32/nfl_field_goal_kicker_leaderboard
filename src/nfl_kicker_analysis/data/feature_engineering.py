"""
Feature engineering module for NFL kicker analysis.
Contains all feature creation functions that can be used to build features for modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Union, List
from pandas.tseries.offsets import DateOffset

# Required columns for EPA calculation
REQUIRED_EPA_COLS = {"age_at_attempt", "exp_100", "player_status"}

def ensure_epa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that df contains every column in REQUIRED_EPA_COLS.
    If any are missing, run the minimal FeatureEngineer step that
    creates just that column and merge the result back.
    Never fills with dummy values.
    """
    eng = FeatureEngineer()          # lightweight, stateless
    missing = REQUIRED_EPA_COLS.difference(df.columns)

    if "age_at_attempt" in missing:
        df = eng.create_date_features(df)
        missing -= {"age_at_attempt"}

    if {"exp_100"}.issubset(missing):           # experience needs ordering
        df = eng.create_experience_features(df)
        missing -= {"exp_100"}

    if "player_status" in missing:
        df = eng.create_player_status_features(df)
        missing -= {"player_status"}

    if missing:      # still not resolved ➜ stop early
        raise KeyError(f"Unable to generate columns: {sorted(missing)}")

    return df

class FeatureEngineer:
    """Handles all feature engineering operations."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.kicker_mapping = None

    # ---------------------------------------------------------------------
    # Target / basic temporal features
    # ---------------------------------------------------------------------
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary success target variable."""
        df = df.copy()
        df["success"] = (df["field_goal_result"] == "Made").astype(int)
        print(f"******* Created target variable: {df['success'].mean():.1%} success rate")
        return df

    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create date-related features including centered age variables.
        
        Adds:
        • age_at_attempt     – raw age in years
        • age_c              – centred (30 yrs) & scaled (÷10) age
        • age_c2             – quadratic term (for simple aging curve)
        """
        df = df.copy()

        # Ensure datetime dtypes
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["birthdate"] = pd.to_datetime(df["birthdate"])

        # Age at attempt
        df["age_at_attempt"] = (df["game_date"] - df["birthdate"]).dt.days / 365.25
        
        # Centered & scaled age (Gelman scaling)
        df["age_c"]  = (df["age_at_attempt"] - 30.0) / 10.0
        df["age_c2"] = df["age_c"] ** 2
        
        # Seasonal features
        df["day_of_year"] = df["game_date"].dt.dayofyear
        df["month"] = df["game_date"].dt.month
        print("******* Created date features (age_c, age_c2, day_of_year, month)")
        return df

    # ------------------------------------------------------------------
    # OPTIONAL – continuous spline basis for age (k = 3 knots)
    # ------------------------------------------------------------------
    def create_age_spline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add natural-cubic-spline basis age_spline_1…k (centered & scaled age).
        Use when nonlinear aging curve is desired.
        """
        try:
            import patsy as ps
        except ImportError:
            print("⚠️  patsy not available - skipping age spline features")
            return df
            
        df = df.copy()
        # Design matrix returns ndarray with intercept, drop it
        spline = ps.dmatrix("bs(age_c, df=3, degree=3, include_intercept=False)",
                            df, return_type="dataframe")
        # Rename columns nicely
        spline.columns = [f"age_spline_{i+1}" for i in range(spline.shape[1])]
        df = pd.concat([df, spline], axis=1)
        print("******* Created age spline features")
        return df

    # ------------------------------------------------------------------
    # Identifier mapping
    # ------------------------------------------------------------------
    def create_kicker_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds two columns:
          • kicker_id   – raw player_id from the source (GSIS / Stathead)
          • kicker_idx  – zero-based contiguous index used ONLY for matrix ops
        The mapping is cached so that train/test splits share the same indices.
        Also preserves player_name for human-readable analysis.
        """
        df = df.copy()

        df["kicker_id"] = df["player_id"].astype(int)        # ← raw, never mutates
        if self.kicker_mapping is None:
            unique = df["kicker_id"].unique()
            self.kicker_mapping = {pid: i for i, pid in enumerate(sorted(unique))}
        df["kicker_idx"] = df["kicker_id"].map(self.kicker_mapping).astype(int)

        # Ensure player_name is preserved for name-based operations
        # This enables the hybrid approach recommended in the roadmap
        if "player_name" not in df.columns:
            print("⚠️  WARNING: player_name column not found. Name-based operations may not work.")

        print(f"******* Created kicker mapping for {len(self.kicker_mapping)} unique kickers "
              f"(raw_id→idx) with player names preserved")
        return df


    # ------------------------------------------------------------------
    # Distance related features
    # ------------------------------------------------------------------
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based engineered features, with safe log transform."""
        df = df.copy()
        df["distance_squared"] = df["attempt_yards"] ** 2
        df["distance_cubed"]   = df["attempt_yards"] ** 3
        # Use log1p to handle zero-yard attempts without -inf :contentReference[oaicite:13]{index=13}
        df["log_distance"]     = np.log1p(df["attempt_yards"])
        df["is_long_attempt"]  = (df["attempt_yards"] >= 50).astype(int)
        df["is_very_long_attempt"] = (df["attempt_yards"] >= 55).astype(int)
        q1, q2, q3 = df["attempt_yards"].quantile([0.25, 0.5, 0.75])
        df["distance_category"] = df["attempt_yards"].apply(
            lambda dist: "Short" if dist < q1
                        else "Medium-Short" if dist < q2
                        else "Medium" if dist < q3
                        else "Long"
        )
        df["distance_from_sweet_spot"] = (df["attempt_yards"] - 35).abs()
        print("******* Created distance features (poly, log, quantile categories, flags)")
        return df


    # ------------------------------------------------------------------
    # Experience features
    # ------------------------------------------------------------------
    def create_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cumulative-experience variables.

        Columns created
        ---------------
        kicker_attempt_number : 1-indexed count (already used elsewhere)
        experience_in_kicks   : 0-indexed before current kick
        exp_100               : experience_in_kicks / 100  (for stable priors)
        is_rookie_attempt     : 1 if first ≤20 attempts
        experience_category   : Rookie / Developing / Veteran (10th & 25th pct)
        career_length_years   : yrs since first attempt
        """
        df = df.sort_values(["player_id", "game_date"]).copy()

        # cumulative counts
        df["kicker_attempt_number"] = df.groupby("player_id").cumcount() + 1
        df["experience_in_kicks"]   = df["kicker_attempt_number"] - 1
        df["exp_100"]               = df["experience_in_kicks"] / 100.0

        # career length (years)
        first_dates = df.groupby("player_id")["game_date"].transform("min")
        df["career_length_years"] = (df["game_date"] - first_dates).dt.days / 365.25

        # simple buckets
        df["is_rookie_attempt"] = (df["kicker_attempt_number"] <= 20).astype(int)
        p10, p25 = df["kicker_attempt_number"].quantile([0.1, 0.25])
        df["experience_category"] = df["kicker_attempt_number"].apply(
            lambda n: "Rookie" if n <= p10
            else ("Developing" if n <= p25 else "Veteran")
        )

        print("******* Created experience features (exp_100 added)")
        return df
    
    # ------------------------------------------------------------------
    # Situational features
    # ------------------------------------------------------------------
    def create_situational_features(
            self,
            df: pd.DataFrame,
            *,
            weight_cfg: dict | None = None
    ) -> pd.DataFrame:
        """
        Adds season / clutch flags *and* an 'importance' weight.

        Parameters
        ----------
        weight_cfg : dict, optional
            Keys: 'late', 'clutch', 'playoff'.
            Defaults = {'late': 1, 'clutch': 2, 'playoff': 4}.
        """
        w = {'late': 1, 'clutch': 2, 'playoff': 4}
        if weight_cfg:
            w.update(weight_cfg)

        df = df.copy()
        df["is_early_season"] = (df["week"] <= 4).astype(int)
        df["is_late_season"]  = (df["week"] >= 14).astype(int)
        df["is_playoffs"]     = (df["week"] >= 17).astype(int)
        df["season_progress"] = df["week"] / 16.0

        # # Optional clutch (leave at 0 if context cols absent)
        # req = ["quarter", "game_seconds_remaining", "score_differential"]
        # missing = [c for c in req if c not in df.columns]
        # if missing:
        #     df["is_clutch"] = 0
        # else:
        #     df["is_clutch"] = (
        #         (df["quarter"] >= 4) &
        #         (df["game_seconds_remaining"] <= 120) &
        #         (df["score_differential"].abs() <= 3)
        #     ).astype(int)

        # --- NEW flexible weighting ------------------------------------------
        df["importance"] = (
            1
            + w['late']     * df["is_late_season"]
            # + w['clutch']   * df["is_clutch"]
            + w['playoff']  * df["is_playoffs"]
        )
        return df


    # ------------------------------------------------------------------
    # Rolling performance history
    # ------------------------------------------------------------------
    def create_performance_history_features(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """Rolling success rates, similar-distance success, and current streaks."""
        df = df.sort_values(["player_id", "game_date"]).copy()

        # Rolling mean of success
        df["rolling_success_rate"] = (
            df.groupby("player_id")["success"]
            .transform(lambda s: s.rolling(window_size, min_periods=1).mean())
        )

        overall = df["success"].mean()

        def similar_rate(sub):
            """For each attempt, compute mean success rate of prior attempts within ±5 yards."""
            vals = []
            for i in range(len(sub)):
                prev = sub.iloc[:i]
                mask = (prev["attempt_yards"] - sub.iloc[i]["attempt_yards"]).abs() <= 5
                vals.append(prev.loc[mask, "success"].mean() if mask.any() else overall)
            return pd.Series(vals, index=sub.index)

        # Minimal deprecation fix: exclude grouping column before apply
        sim = (
            df.groupby("player_id", group_keys=False)
            .apply(similar_rate, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        df["rolling_similar_distance_success"] = sim

        # Current streak length
        df["current_streak"] = (
            df.groupby("player_id")["success"]
            .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )

        print("******* Created performance history features (rolling & streaks)")
        return df



    # ------------------------------------------------------------------
    # Statistical interaction features
    # ------------------------------------------------------------------
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["distance_zscore"] = (
            (df["attempt_yards"] - df["attempt_yards"].mean()) /
            df["attempt_yards"].std()
        )
        df["distance_percentile"] = df["attempt_yards"].rank(pct=True)
        
        # Cleaner interaction names using centered age
        df["age_dist_interact"] = df["age_c"] * df["attempt_yards"]
        df["exp_dist_interact"] = df["exp_100"] * df["attempt_yards"]
        
        print("******* Created statistical interaction features (age × distance, exp × distance)")
        return df

    def create_player_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create player status categorization based on recent activity relative to dataset's latest date.
        
        Categories:
        - Retired/Injured: 2+ years since last kick (730+ days)
        - Not Playing/Potentially Playable: 1-2 years since last kick (365-729 days)  
        - Playable: Less than 1 year since last kick (0-364 days)
        """
        df = df.copy()
        
        # Ensure datetime format
        df["game_date"] = pd.to_datetime(df["game_date"])
        
        # Find the latest date in the dataset (reference point)
        latest_date = df["game_date"].max()
        
        # Calculate last kick date for each player
        last_kick_by_player = df.groupby("player_id")["game_date"].max().reset_index()
        last_kick_by_player.columns = ["player_id", "last_kick_date"]
        
        # Calculate days since last kick
        last_kick_by_player["days_since_last_kick"] = (
            latest_date - last_kick_by_player["last_kick_date"]
        ).dt.days
        
        # Categorize player status
        def categorize_status(days_since_last):
            if days_since_last >= 730:  # 2+ years
                return "Retired/Injured"
            elif days_since_last >= 365:  # 1-2 years
                return "Not Playing/Potentially Playable"
            else:  # < 1 year
                return "Playable"
        
        last_kick_by_player["player_status"] = last_kick_by_player["days_since_last_kick"].apply(categorize_status)
        
        # Merge back to main dataframe
        df = df.merge(
            last_kick_by_player[["player_id", "player_status", "days_since_last_kick"]], 
            on="player_id", 
            how="left"
        )
        
        # Add summary statistics
        status_counts = last_kick_by_player["player_status"].value_counts()
        print("******* Created player status features based on recent activity")
        print(f"   Reference date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"   Playable: {status_counts.get('Playable', 0)} players (<1 year)")
        print(f"   Not Playing/Potentially Playable: {status_counts.get('Not Playing/Potentially Playable', 0)} players (1-2 years)")
        print(f"   Retired/Injured: {status_counts.get('Retired/Injured', 0)} players (2+ years)")
        
        return df

    # ------------------------------------------------------------------
    # Orchestration: build *all* features
    # ------------------------------------------------------------------
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_performance_history: bool = True,
        performance_window: int = 10,
        include_player_status: bool = True
    ) -> pd.DataFrame:
        print("Creating all features...")
        df = (
            df.pipe(self.create_target_variable)
              .pipe(self.create_date_features)
              .pipe(self.create_kicker_mapping)
              .pipe(self.create_distance_features)
              .pipe(self.create_experience_features)
              .pipe(self.create_situational_features)
              .pipe(self.create_statistical_features)
        )
        if include_performance_history:
            df = self.create_performance_history_features(df, performance_window)
        if include_player_status:
            df = self.create_player_status_features(df)
        print(f"******* All features created! Dataset shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Dynamic feature catalogue helper
    # ------------------------------------------------------------------
    def get_available_features(
        self,
        df: pd.DataFrame,
        include_unique: bool = True,
        max_unique_values: int = 20
    ) -> Dict[str, Union[List[str], Dict[str, List[Union[str, int, float]]]]]:
        """Return feature categories -> features (optionally with uniques) after engineering."""
        base_catalog = {
            "target": ["success"],
            "basic": ["attempt_yards", "age_at_attempt", "kicker_attempt_number", "importance"],
            "distance": [
                "distance_squared", "distance_cubed", "log_distance",
                "distance_from_sweet_spot", "distance_zscore", "distance_percentile"
            ],
            "distance_flags": ["is_long_attempt", "is_very_long_attempt"],
            "distance_categories": ["distance_category"],
            "temporal": ["day_of_year", "month", "week", "season", "season_progress"],
            "situational": ["is_early_season", "is_late_season", "is_playoffs"],
            "experience": ["career_length_years", "is_rookie_attempt", "experience_category"],
            "performance_history": [
                "rolling_success_rate", "rolling_similar_distance_success", "current_streak"
            ],
            "interactions": ["age_distance_interaction", "experience_distance_interaction"],
            "identifiers": ["kicker_id", "kicker_idx", "player_name"]
        }
        catalog: Dict[str, Union[List[str], Dict[str, List[Union[str, int, float]]]]] = {}
        for cat, feats in base_catalog.items():
            present = [f for f in feats if f in df.columns]
            if include_unique:
                detail: Dict[str, List[Union[str, int, float]]] = {}
                for f in present:
                    if df[f].dtype == "object" or df[f].nunique() <= max_unique_values:
                        detail[f] = sorted(df[f].dropna().unique().tolist())
                    else:
                        detail[f] = []
                catalog[cat] = detail
            else:
                catalog[cat] = present
        return catalog

# Function to create a summary DataFrame
def summarize(df):
    summary = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing': df.isnull().sum(),
        'unique': df.nunique(),
        'sample_values': df.apply(lambda col: col.dropna().unique()[:5] if col.nunique() > 5 else col.unique())
    })
    return summary

if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader

    loader = DataLoader()
    df_raw = loader.load_complete_dataset()
    engineer = FeatureEngineer()
    df_feat = engineer.create_all_features(df_raw)

    summary_fg = summarize(df_feat)
    print("---------------summary_fg-----------------")
    print(summary_fg)
    

    

    


    
