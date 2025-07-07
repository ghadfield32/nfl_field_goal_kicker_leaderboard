"""
Configuration module for NFL Kicker Analysis package.
Contains all constants, paths, and configuration parameters.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import os

class Config:
    """Main configuration class for the NFL Kicker Analysis package."""
    MLFLOW_EXPERIMENT_NAME="nfl_kicker_analysis"
    
    # Base paths - use relative paths that work in both local and cloud environments
    # Get the project root by going up from this config file location
    _CONFIG_DIR = Path(__file__).parent.parent.parent  # Go up to project root
    PROJECT_ROOT = _CONFIG_DIR.resolve()
    
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "models"  # Bayesian models directory
    MODEL_DIR = MODELS_DIR / "bayesian"   # Specific directory for Bayesian models
    POINT_ESTIMATE_DIR = MODELS_DIR / "mlruns" / "models"  # Point estimate models go here
    
    
    # Raw data files
    KICKERS_FILE = RAW_DATA_DIR / "kickers.csv"
    ATTEMPTS_FILE = RAW_DATA_DIR / "field_goal_attempts.csv"
    
    # Processed data files
    MODELING_DATA_FILE = PROCESSED_DATA_DIR / "field_goal_modeling_data.csv"
    LEADERBOARD_FILE = OUTPUT_DIR / "leaderboard.csv"
    MODEL_DATA_FILE: Path = OUTPUT_DIR / "bayesian_features.csv"
    
    # Analysis parameters
    MIN_DISTANCE = 20
    min_distance = 20
    MAX_DISTANCE = 60
    MIN_KICKER_ATTEMPTS = 10  # Changed from 8 to 5 to match Method B
    
    # Distance profile for EPA calculation
    DISTANCE_PROFILE = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    DISTANCE_WEIGHTS = [0.05, 0.10, 0.20, 0.20, 0.20, 0.15, 0.08, 0.02, 0.01]
    
    # Distance ranges for analysis
    DISTANCE_RANGES = [
        (18, 29, "Short (18-29 yards)"),
        (30, 39, "Medium-Short (30-39 yards)"),
        (40, 49, "Medium (40-49 yards)"),
        (50, 59, "Long (50-59 yards)"),
        (60, 75, "Extreme (60+ yards)")
    ]
    
    # Model parameters
    BAYESIAN_MCMC_SAMPLES = 500
    BAYESIAN_TUNE = 250
    BAYESIAN_CHAINS = 2
    
    # Rating thresholds
    ELITE_THRESHOLD = 0.15
    STRUGGLING_THRESHOLD = -0.20
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 100
    
    # Season types to include
    SEASON_TYPES = ['Reg', 'Post']  # Include both regular season and playoffs
    
    # ─── Feature flags ───────────────────────────────────────────
    FILTER_RETIRED_INJURED = True   # keep everyone by default
    metrics = {"accuracy": 0.85, "f1": 0.82}
    
    # 16 game seasons
    PRE_2021_SEASON_GAMES = 16
    POST_2021_SEASON_GAMES = 17
    YEAR_GAMES_INCREASED = 2021
    
    # ── NEW: maximum gap (in games) since last kick for Bayesian defaults
    MAX_GAMES_SINCE_LAST_KICK: int | None = None

    # How many Optuna trials for each tree-based model
    OPTUNA_TRIALS: Dict[str, int] = {
        "simple_logistic": 50,
        "ridge_logistic": 50,
        "random_forest": 50,
        "xgboost": 50,
        "catboost": 50
    }

    @classmethod
    def ensure_directories(cls):
        """Create all required directories if they don't exist."""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                        cls.OUTPUT_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Create global config instance
config = Config()

# ───────────────────────── Feature catalogue ─────────────────────────
# Single source of truth for column roles – centralized from all modules
FEATURE_LISTS: Dict[str, List[str]] = {
    "numerical": [
        "attempt_yards", "age_at_attempt", "distance_squared",
        "career_length_years", "season_progress", "exp_100", 
        "distance_zscore", "distance_percentile",
        "seasons_of_experience", "career_year",
        "age_c", "age_c2", 
        "importance", "days_since_last_kick",
        "age_dist_interact", "exp_dist_interact", 
    ],
    "ordinal": ["season", "week", "month", "day_of_year"],
    "nominal": [
        "kicker_id", "kicker_idx",
        "is_long_attempt", "is_very_long_attempt",
        "is_rookie_attempt", "distance_category", "experience_category",
        "is_early_season", "is_late_season", "is_playoffs",
    ],
    "y_variable": ["success"],
}

# Attach FEATURE_LISTS onto the config instance for ease of use
config.FEATURE_LISTS = FEATURE_LISTS

if __name__ == "__main__":
    # Test the configuration
    print("NFL Kicker Analysis Configuration")
    print("=" * 40)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Min distance: {config.MIN_DISTANCE}")
    print(f"Max distance: {config.MAX_DISTANCE}")
    print(f"Distance profile: {config.DISTANCE_PROFILE}")
    print(f"Elite threshold: {config.ELITE_THRESHOLD}")
    
    # Test directory creation
    config.ensure_directories()
    print("******* Configuration loaded and directories created!")





