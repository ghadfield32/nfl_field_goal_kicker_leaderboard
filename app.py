# app.py
from pathlib import Path
import pandas as pd
import streamlit as st
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, cast, List
import os

# Try to import optional dependencies
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    az = None

# Import configuration
try:
    from src.nfl_kicker_analysis.config import config
    CONFIG_AVAILABLE = True
except ImportError as e:
    st.error("Configuration module not found. Please ensure the project is set up correctly.")
    st.stop()
except Exception as e:
    st.error("Error loading configuration. Please check your project setup.")
    st.stop()

# Try to import model utilities
try:
    from src.nfl_kicker_analysis.utils.model_utils import (
        list_registered_models,
        list_saved_models,
        load_model,
        get_best_model_info,
        get_best_metrics,
    )
    MODEL_UTILS_AVAILABLE = True
except ImportError:
    MODEL_UTILS_AVAILABLE = False

# Try to import Bayesian models
try:
    from src.nfl_kicker_analysis.models.bayesian import BayesianModelSuite
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianModelSuite = None

# Try to import data modules
try:
    from src.nfl_kicker_analysis.data.loader import DataLoader
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    DataLoader = None

try:
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    DATA_PREPROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PREPROCESSOR_AVAILABLE = False
    DataPreprocessor = None

# Try to import EDA modules
try:
    from src.nfl_kicker_analysis.eda import (
        outcome_summary, 
        distance_analysis, 
        temporal_analysis, 
        kicker_performance_analysis, 
        feature_engineering
    )
    EDA_AVAILABLE = True
except ImportError:
    EDA_AVAILABLE = False

@st.cache_data
def compute_kicker_metrics(df: pd.DataFrame, player_name: str) -> dict:
    """Compute and cache kicker metrics."""
    kicker_data = df[df['player_name'] == player_name]
    return {
        'total_attempts': len(kicker_data),
        'success_rate': kicker_data['success'].mean() * 100,
        'avg_distance': kicker_data['attempt_yards'].mean(),
    }

@st.cache_data
def compute_distance_binned_stats(df: pd.DataFrame, player_name: str, bin_size: int = 5) -> tuple:
    """Compute and cache binned statistics for distance analysis."""
    kicker_data = df[df['player_name'] == player_name].copy()
    kicker_data['bin'] = (kicker_data['attempt_yards'] // bin_size) * bin_size
    actual = kicker_data.groupby('bin')['success'].mean()
    return actual.index.values, actual.values

def plot_kicker_skill_posterior(
    suite,
    df: pd.DataFrame,
    player_name: str,
    bin_width: int = 50,
) -> Figure:
    """Compute and plot the posterior distribution of P(make) for a single kicker."""
    if not ARVIZ_AVAILABLE or not BAYESIAN_AVAILABLE:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Bayesian analysis not available", ha='center', va='center')
        return fig
        
    k_idx = suite.get_kicker_id_by_name(df, player_name)
    if k_idx is None:
        raise ValueError(f"Kicker '{player_name}' not found")
    
    if suite._trace is None:
        raise ValueError("Model has not been fit yet")
    
    # Lazy import matplotlib
    import matplotlib.pyplot as plt
        
    posterior = cast(az.InferenceData, suite._trace).posterior
    a = posterior["alpha"].values.flatten()
    b = posterior["beta_dist"].values.flatten()
    u = posterior["u"].values.reshape(-1, posterior["u"].shape[-1])
    u_k = u[:, k_idx]
    logit = a + b * 0.0 + u_k
    p = 1 / (1 + np.exp(-logit))
    
    fig, ax = plt.subplots()
    ax.hist(p, bins=bin_width, density=True, alpha=0.8)
    ax.set_title(f"Posterior P(make) for {player_name}")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    return fig

def plot_distance_comparison(
    suite,
    df: pd.DataFrame,
    player_name: str,
    bin_size: int = 5
) -> Figure:
    """Plot actual vs predicted success rates by distance."""
    # Lazy import matplotlib
    import matplotlib.pyplot as plt
    
    # Use cached computation for actual values
    bins, actuals = compute_distance_binned_stats(df, player_name, bin_size)
    
    # Get predictions for this kicker's attempts
    kicker_data = df[df['player_name'] == player_name]
    preds = suite.predict(kicker_data)
    kicker_data['predicted'] = preds
    kicker_data['bin'] = (kicker_data['attempt_yards'] // bin_size) * bin_size
    predicted = kicker_data.groupby('bin')['predicted'].mean()
    
    fig, ax = plt.subplots()
    ax.plot(bins, actuals, marker='o', label='Actual', linewidth=2)
    ax.plot(predicted.index, predicted.values, marker='s', label='Predicted', linestyle='--')
    ax.set_xlabel('Distance (yards)')
    ax.set_ylabel('Make Probability')
    ax.set_title(f'Performance by Distance: {player_name}')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_kicker_epa_distribution(
    suite,
    df: pd.DataFrame,
    player_name: str,
    n_samples: int = 500,
    bin_width: int = 20,
) -> Figure:
    """
    Bootstrap EPA-FG⁺ draws for one kicker and plot histogram.
    """
    if not BAYESIAN_AVAILABLE:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Bayesian analysis not available", ha='center', va='center')
        return fig
        
    kid = suite.get_kicker_id_by_name(df, player_name)
    if kid is None:
        raise ValueError(f"Kicker '{player_name}' not found")
        
    work = df.copy()
    draws = suite._epa_fg_plus_draws(
        work,
        kicker_ids=np.array([kid]),  # Convert to numpy array
        n_samples=n_samples,
        rng=np.random.default_rng(suite.random_seed),
        distance_strategy="kicker",
    )
    epa = draws[:, 0]
    fig, ax = plt.subplots()
    ax.hist(epa, bins=bin_width, density=True, alpha=0.8)
    ax.set_title(f"EPA-FG⁺ Distribution for {player_name}")
    ax.set_xlabel("EPA-FG⁺ (points)")
    ax.set_ylabel("Density")
    return fig

POINT_ESTIMATE_MODELS = [
    "simple_logistic",
    "ridge_logistic",
    "random_forest",
    "xgboost",
    "catboost",
]

_FIELD_GOAL_RESULT_SUCCESS = "Made"
_PRESEASON_FLAG = "Pre"

# ───────────────────────── Streamlit page config ──────────────────────────────
st.set_page_config(
    page_title="NFL Kicker Analysis – Broncos Tech Assessment",
    page_icon="🎯",
    layout="wide",
)

# Custom CSS for dark theme with Broncos‑flavoured palette
BRONCOS_COLOURS = {
    "orange": "#fb4f14",
    "navy": "#0a2343",
    "steel": "#a5acaf",
}

def inject_css() -> None:
    """Inject a full dark theme with Broncos-styled custom styling."""
    st.markdown(
        f"""
        <style>
        /* Page background & text */
        .appview-container, .main, .block-container {{
            background-color: #121212 !important;
            color: #E0E0E0 !important;
            font-family: "Inter", sans-serif;
        }}
        
        /* Sidebar container and elements */
        [data-testid="stSidebar"] {{
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }}
        [data-testid="stSidebarNav"] {{
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }}
        [data-testid="stSidebar"] .css-1d391kg,
        [data-testid="stSidebar"] .css-1aw8i8e,
        [data-testid="stSidebar"] .css-1vq4p4l {{
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {{
            color: {BRONCOS_COLOURS['orange']} !important;
        }}
        
        /* Headers & titles */
        h1, h2, h3, h4, .stHeader {{
            color: #FFFFFF !important;
        }}
        
        /* Buttons */
        button[kind="primary"] {{
            background-color: {BRONCOS_COLOURS['orange']} !important;
            color: #121212 !important;
            border-radius: 8px !important;
        }}
        
        /* Metrics cards */
        .stMetric {{
            background-color: #1E1E1E !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        }}
        
        /* Tables & dataframes */
        .css-1lcbmhc .css-1d391kg, .stDataFrame {{
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }}
        .stDataFrame td, .stDataFrame th {{
            background-color: #2A2A2A !important;
        }}
        
        /* Tabs */
        div[data-baseweb="tab-list"] button {{
            background-color: transparent !important;
            color: #E0E0E0 !important;
            border-bottom: 3px solid transparent;
        }}
        
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: transparent !important;
            color: {BRONCOS_COLOURS['orange']} !important;
            border-color: {BRONCOS_COLOURS['orange']} !important;
            border-radius: 0 !important;
        }}
        
        /* Inputs & selects */
        input, select, .stSelectbox, .stTextInput {{
            background-color: #2A2A2A !important;
            color: #E0E0E0 !important;
            border: none !important;
            border-radius: 4px !important;
        }}
        
        /* Links and accents */
        a, .st-de, .st-ag {{
            color: {BRONCOS_COLOURS['orange']} !important;
        }}
        
        /* Card styling */
        .st-bb {{
            border: none;
        }}
        .st-bx {{
            border-radius: 0.75rem;
        }}
        
        /* Header spacing */
        .block-container {{
            padding-top: 2rem;
        }}
        
        /* Plot backgrounds */
        .stPlot {{
            background-color: #1E1E1E !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }}
        
        /* Code blocks */
        .stCodeBlock {{
            background-color: #2A2A2A !important;
            color: #E0E0E0 !important;
            border-radius: 8px !important;
        }}
        
        /* Dropdown menus */
        .stSelectbox > div > div {{
            background-color: #2A2A2A !important;
            color: #E0E0E0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

st.sidebar.header("⚙️ Select Model")
model_types = ["Uncertainty Interval Models", "Point Estimate Models"]
model_type = st.sidebar.selectbox("Model Type", model_types)

# ======== Model Directories Setup =========
if MODEL_UTILS_AVAILABLE:
    try:
        correct_mlflow_dir = config.PROJECT_ROOT / "mlruns" / "models"
        fs_models = {}
        if correct_mlflow_dir.exists():
            fs_models.update(list_saved_models(correct_mlflow_dir))
        old_point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"
        if old_point_estimate_dir.exists():
            fs_models.update(list_saved_models(old_point_estimate_dir))
        mlflow_models = list_registered_models()
        all_models = {**mlflow_models, **fs_models}
        point_estimate_dir = correct_mlflow_dir
        if not point_estimate_dir.exists():
            point_estimate_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Error accessing model directories: {e}")
        fs_models, mlflow_models, all_models = {}, {}, {}
        point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"
else:
    fs_models, mlflow_models, all_models = {}, {}, {}
    point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"

# ── Cache the metrics lookup so dropdown changes are fast ──
@st.cache_data
def get_metrics_df(model_name: str) -> pd.DataFrame:
    """
    Return a DataFrame of all logged metrics for `model_name`.
    """
    metrics = get_best_metrics(model_name) or {}
    # Convert {'accuracy':0.88, 'f1':0.82} → DataFrame
    df = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["Value"]
    ).reset_index().rename(columns={"index":"Metric"})
    return df

# Cache expensive resource instantiation
@st.cache_resource
def get_data_loader() -> Optional[DataLoader]:
    """Cache DataLoader instance for reuse."""
    if not DATA_LOADER_AVAILABLE:
        return None
    return DataLoader()

@st.cache_resource
def get_data_preprocessor() -> Optional[DataPreprocessor]:
    """Cache DataPreprocessor instance for reuse."""
    if not DATA_PREPROCESSOR_AVAILABLE:
        return None
    return DataPreprocessor()

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_raw_data() -> pd.DataFrame:
    """Load raw data with fallback between Parquet and CSV formats."""
    loader = get_data_loader()
    if loader is None:
        return pd.DataFrame()
        
    try:
        # Try Parquet first (faster)
        parquet_path = Path("data/raw/field_goal_attempts.parquet")
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
            
        # Fallback to CSV
        return loader.load_complete_dataset()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_preprocessed_data() -> pd.DataFrame:
    """Load preprocessed data with optimized format handling."""
    loader = get_data_loader()
    preprocessor = get_data_preprocessor()
    
    if loader is None or preprocessor is None:
        # Try to load from preprocessed files directly
        try:
            # Check Parquet first
            if hasattr(config, 'MODEL_DATA_FILE'):
                parquet_path = config.MODEL_DATA_FILE.with_suffix('.parquet')
                if parquet_path.exists():
                    return pd.read_parquet(parquet_path)
            
            # Fallback to CSV locations
            if hasattr(config, 'MODEL_DATA_FILE') and config.MODEL_DATA_FILE.exists():
                return pd.read_csv(config.MODEL_DATA_FILE)
            elif hasattr(config, 'PROCESSED_DATA_DIR'):
                csv_path = config.PROCESSED_DATA_DIR / "field_goal_modeling_data.csv"
                if csv_path.exists():
                    return pd.read_csv(csv_path)
            
            st.warning("Preprocessed data not available and modules not loaded")
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading data files: {e}")
            return pd.DataFrame()

    try:
        raw = load_raw_data()
        if raw.empty:
            return pd.DataFrame()
            
        # Update preprocessor config
        preprocessor.update_config(
            min_distance=getattr(config, 'MIN_DISTANCE', 20),
            max_distance=getattr(config, 'MAX_DISTANCE', 60),
            min_kicker_attempts=getattr(config, 'MIN_KICKER_ATTEMPTS', 10),
            season_types=getattr(config, 'SEASON_TYPES', ['Reg', 'Post']),
            include_performance_history=True,
            include_statistical_features=False,
            include_player_status=True,
            performance_window=12,
        )
        
        # Apply feature lists if available
        feature_lists = getattr(config, 'FEATURE_LISTS', {})
        if feature_lists:
            preprocessor.update_feature_lists(**feature_lists)
        
        return preprocessor.preprocess_complete(raw)
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_bayesian_metrics(suite_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    metrics_path = suite_dir / "metrics.json"
    if metrics_path.exists():
        # Load the exact metrics from training time
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                st.sidebar.success("✅ Loaded saved metrics from training")
        except Exception as e:
            st.sidebar.warning(f"Error loading metrics file: {e}")
            metrics = {}
    else:
        # Fallback: recompute on-the-fly (if modules are available)
        if not BAYESIAN_AVAILABLE or not DATA_PREPROCESSOR_AVAILABLE:
            st.sidebar.warning("⚠️ Cannot compute metrics - modules not available")
            metrics = {}
        else:
            try:
                st.sidebar.warning("⚠️ No saved metrics found - recomputing")
                suite = BayesianModelSuite.load_suite(suite_dir)
                pre = DataPreprocessor()
                
                # Safely access config attributes
                min_distance = getattr(config, 'MIN_DISTANCE', 20)
                max_distance = getattr(config, 'MAX_DISTANCE', 60)
                min_kicker_attempts = getattr(config, 'MIN_KICKER_ATTEMPTS', 10)
                season_types = getattr(config, 'SEASON_TYPES', ['Reg', 'Post'])
                feature_lists = getattr(config, 'FEATURE_LISTS', {})
                
                pre.update_config(
                    min_distance=min_distance,
                    max_distance=max_distance,
                    min_kicker_attempts=min_kicker_attempts,
                    season_types=season_types,
                    include_performance_history=False,
                    include_statistical_features=False,
                    include_player_status=True,
                    performance_window=12,
                )
                
                if feature_lists:
                    pre.update_feature_lists(**feature_lists)
                
                metrics = suite.evaluate(df, preprocessor=pre)
            except Exception as e:
                st.sidebar.error(f"Error computing metrics: {e}")
                metrics = {}

    # Convert to DataFrame for display
    dfm = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["Value"]
    ).reset_index().rename(columns={"index":"Metric"})
    return dfm

# === Enhanced setup for cloud deployment ===
def setup_for_cloud():
    """Automatically copy leaderboard files and setup paths for Streamlit Cloud compatibility."""
    try:
        project_root = Path(__file__).parent.absolute()
        source_dir = project_root / "output"
        
        # Check for missing leaderboard files in project root
        expected_leaderboards = [
            "catboost_leaderboard.csv", "random_forest_leaderboard.csv", 
            "ridge_logistic_leaderboard.csv", "simple_logistic_leaderboard.csv",
            "xgboost_leaderboard.csv"
        ]
        
        missing_leaderboards = [
            f for f in expected_leaderboards 
            if not (project_root / f).exists()
        ]
        
        # Check if we need to copy files and copy them if source exists
        if missing_leaderboards and source_dir.exists():
            import shutil
            for csv_file in source_dir.glob("*leaderboard.csv"):
                target_file = project_root / csv_file.name
                if not target_file.exists():
                    shutil.copy2(csv_file, target_file)
                    print(f"✅ Copied {csv_file.name} to project root")
        
        # === NEW: Ensure Bayesian leaderboard is available ===
        bayesian_leaderboard_files = [
            "leaderboard.csv",  # Main Bayesian leaderboard
            "bayesian_features.csv"  # Bayesian features data
        ]
        
        for filename in bayesian_leaderboard_files:
            source_file = source_dir / filename
            target_file = project_root / filename
            
            if source_file.exists() and not target_file.exists():
                import shutil
                shutil.copy2(source_file, target_file)
                print(f"✅ Copied {filename} to project root for cloud deployment")
        
        return True
    except Exception as e:
        print(f"⚠️ Warning: Setup for cloud failed: {e}")
        return False

def find_bayesian_leaderboard() -> Optional[Path]:
    """Find the Bayesian EPA leaderboard file in various locations."""
    project_root = Path(__file__).parent.absolute()
    
    # Priority order for leaderboard search
    possible_locations = [
        project_root / "leaderboard.csv",  # Cloud deployment location
        project_root / "output" / "leaderboard.csv",  # Local development location
        project_root / "data" / "processed" / "leaderboard.csv",  # Alternative location
    ]
    
    # Check config location if available
    try:
        possible_locations.append(config.LEADERBOARD_FILE)
    except:
        pass
    
    for location in possible_locations:
        if location.exists():
            return location
    
    return None

# === Enhanced Bayesian model loading functions ===
def find_bayesian_data_file() -> Optional[Path]:
    """Find the bayesian_features.csv file in multiple possible locations."""
    project_root = Path(__file__).parent.absolute()
    
    # List of possible locations for the bayesian features file
    possible_locations = [
        # Primary location from config
        getattr(config, 'MODEL_DATA_FILE', None),
        # Alternative locations
        project_root / "output" / "bayesian_features.csv",
        project_root / "bayesian_features.csv",
        project_root / "data" / "processed" / "bayesian_features.csv",
        Path("output/bayesian_features.csv"),
        Path("bayesian_features.csv"),
    ]
    
    # Filter out None values and check each location
    for location in possible_locations:
        if location is not None and Path(location).exists():
            return Path(location)
    
    return None

@st.cache_resource
def get_bayesian_suite(suite_path: Path) -> Optional[BayesianModelSuite]:
    """Cache BayesianModelSuite instance for reuse."""
    if not BAYESIAN_AVAILABLE:
        return None
    try:
        return BayesianModelSuite.load_suite(suite_path)
    except Exception as e:
        st.error(f"Error loading Bayesian suite: {e}")
        return None

@st.cache_data(ttl=3600)
def find_bayesian_suite_dirs() -> List[Path]:
    """Find Bayesian suite directories with caching."""
    project_root = Path(__file__).parent.absolute()
    
    # List of possible locations for Bayesian suites
    possible_roots = [
        getattr(config, 'MODEL_DIR', None),
        project_root / "models" / "bayesian",
        Path("models/bayesian"),
        project_root / "bayesian",
    ]
    
    suite_dirs = []
    for suite_root in possible_roots:
        if suite_root is not None and Path(suite_root).exists():
            # Find valid suite directories
            dirs = sorted(
                [d for d in Path(suite_root).iterdir()
                 if d.is_dir() and (d/"meta.json").exists() and (d/"trace.nc").exists()],
                reverse=True
            )
            suite_dirs.extend(dirs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dirs = []
    for d in suite_dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    
    return unique_dirs

@st.cache_data(ttl=3600)
def load_bayesian_data_with_fallback() -> Optional[pd.DataFrame]:
    """Load Bayesian features data with optimized caching."""
    data_file = find_bayesian_data_file()
    
    if data_file is not None:
        try:
            # Try Parquet first
            parquet_path = data_file.with_suffix('.parquet')
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)
            # Fallback to CSV
            return pd.read_csv(data_file)
        except Exception as e:
            st.error(f"Error loading Bayesian data: {e}")
    
    # Fallback: Try to generate data from raw sources
    if DATA_LOADER_AVAILABLE and DATA_PREPROCESSOR_AVAILABLE:
        try:
            # Use cached loader
            loader = get_data_loader()
            if loader is None:
                return None
            df_raw = loader.load_complete_dataset()
            
            # Create minimal preprocessor for Bayesian models
            from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            return engineer.create_all_features(df_raw)
            
        except Exception as e:
            st.error("Error generating Bayesian features. Please check data availability.")
    
    return None

# Run setup
setup_for_cloud()

# === Main Streamlit App ===

if model_type == "Point Estimate Models":
    st.sidebar.subheader("🏆 Point Estimate Models")
    if all_models:
        model_names = list(all_models.keys())
        selected_model = st.selectbox("Choose best model", model_names)
        try:
            if selected_model in fs_models:
                model = load_model(selected_model, base_dir=point_estimate_dir)
            elif selected_model in mlflow_models:
                model = load_model(selected_model)
            else:
                st.warning(f"Model '{selected_model}' not found")
                model = None
        except Exception as e:
            st.error(f"Error loading model '{selected_model}'")
            model = None
    else:
        st.info("No trained models found. Showing available data without model predictions.")
        selected_model = None
        model = None

    # Create tabs: metrics, leaderboard, and EDA
    if model:
        lb_tab, metrics_tab, eda_tab = st.tabs(
            ["🏅 Leaderboard", "📈 Model Metrics", "📊 EDA & Analytics"]
        )
    else:
        lb_tab, eda_tab = st.tabs(
            ["🏅 Leaderboard", "📊 EDA & Analytics"]
        )

    # ── Tab A: Leaderboard ─────────────────────────
    with lb_tab:
        st.header(f"{selected_model.replace('_',' ').title()} Leaderboard")
        possible_files = []
        project_root = Path(__file__).parent.absolute()
        possible_files.extend([
            project_root / "output" / f"{selected_model}_leaderboard.csv",
            project_root / "data" / "processed" / f"{selected_model}_leaderboard.csv",
            project_root / f"{selected_model}_leaderboard.csv",
        ])
        if hasattr(config, 'OUTPUT_DIR'):
            possible_files.append(config.OUTPUT_DIR / f"{selected_model}_leaderboard.csv")
        if hasattr(config, 'PROCESSED_DATA_DIR'):
            possible_files.append(config.PROCESSED_DATA_DIR / f"{selected_model}_leaderboard.csv")

        leaderboard_found = False
        for lb_file in possible_files:
            if lb_file.exists():
                try:
                    df_lb = pd.read_csv(lb_file)
                    if model is not None:
                        try:
                            version, accuracy = get_best_model_info(selected_model)
                            if accuracy is not None:
                                st.write(f"**Accuracy:** {accuracy:.3f}")
                        except Exception:
                            pass
                    st.dataframe(df_lb)
                    leaderboard_found = True
                    break
                except Exception:
                    continue

        if not leaderboard_found:
            if model is not None and not load_preprocessed_data().empty:
                st.info("Generating leaderboard from available data...")
                try:
                    data = load_preprocessed_data()
                    if 'player_name' in data.columns and 'success' in data.columns:
                        leaderboard = (
                            data.groupby('player_name')
                            .agg({
                                'success': ['count', 'sum', 'mean'],
                                'attempt_yards': 'mean'
                            })
                            .round(3)
                        )
                        leaderboard.columns = ['attempts', 'made', 'success_rate', 'avg_distance']
                        leaderboard = leaderboard.reset_index()
                        leaderboard = leaderboard[leaderboard['attempts'] >= 10]
                        leaderboard = leaderboard.sort_values('success_rate', ascending=False)
                        leaderboard['rank'] = range(1, len(leaderboard) + 1)
                        leaderboard = leaderboard[['rank', 'player_name', 'attempts', 'success_rate', 'avg_distance']]
                        st.dataframe(leaderboard)
                        leaderboard_found = True
                except Exception:
                    st.error("Error generating leaderboard from available data")
            if not leaderboard_found:
                st.warning("No leaderboard available. The model may not have been trained yet.")

    # ── Tab B: Full metrics table (only if model loaded) ────
    if model:
        with metrics_tab:
            st.header(f"{selected_model.replace('_',' ').title()} Metrics")
            try:
                df_metrics = get_metrics_df(selected_model)
                st.table(df_metrics)  # shows all logged metrics
            except Exception as e:
                st.error(f"Error loading metrics: {str(e)}")

    # ── Tab C: EDA & Analytics (always available) ───────────
    with eda_tab:
        st.header("📊 Exploratory Data Analysis & Diagnostics")
        if not EDA_AVAILABLE:
            st.info("Showing basic data information")
            try:
                data = load_preprocessed_data()
                if not data.empty:
                    st.write("Data Shape:", data.shape)
                    st.write("Data Info:")
                    st.dataframe(data.head())
                    if 'success' in data.columns:
                        success_rate = data['success'].mean()
                        st.metric("Overall Success Rate", f"{success_rate:.3f}")
            except Exception:
                st.error("Error loading data")
        else:
            try:
                data = load_preprocessed_data()
                if data.empty:
                    st.warning("No data available for analysis")
                    st.stop()

                st.subheader("Overall Outcome Distribution")
                _, fig_out = outcome_summary(data)
                st.pyplot(fig_out)

                st.subheader("Success Rate vs Distance")
                _, fig_dist = distance_analysis(data)
                st.pyplot(fig_dist)

                st.subheader("Temporal Trends & Age")
                _, fig_temp = temporal_analysis(data)
                st.pyplot(fig_temp)

                st.subheader("Kicker Performance Dashboard")
                _, fig_kick = kicker_performance_analysis(data)
                st.pyplot(fig_kick)

                st.subheader("Feature Correlation Matrix")
                fig_corr = feature_engineering(data)
                st.pyplot(fig_corr)

            except Exception:
                st.error("Error generating EDA plots")
                st.info("Please ensure the data files are available")

else:
    st.sidebar.subheader("🔬 Uncertainty Interval Models")
    suite_dirs = find_bayesian_suite_dirs()
    if not suite_dirs:
        st.info("No Bayesian model suites available. Showing EDA and Technical Paper sections.")
        eda_tab, paper_tab = st.tabs(["📊 EDA & Analytics", "📄 Technical Paper"])
        with eda_tab:
            st.header("📊 Exploratory Data Analysis & Diagnostics")
            df = load_preprocessed_data()
            if not df.empty and EDA_AVAILABLE:
                try:
                    st.subheader("Overall Outcome Distribution")
                    _, fig_out = outcome_summary(df)
                    st.pyplot(fig_out)
                    st.subheader("Success Rate vs Distance")
                    _, fig_dist = distance_analysis(df)
                    st.pyplot(fig_dist)
                    st.subheader("Temporal Trends & Age")
                    _, fig_temp = temporal_analysis(df)
                    st.pyplot(fig_temp)
                    st.subheader("Kicker Performance Dashboard")
                    _, fig_kick = kicker_performance_analysis(df)
                    st.pyplot(fig_kick)
                    st.subheader("Feature Correlation Matrix")
                    fig_corr = feature_engineering(df)
                    st.pyplot(fig_corr)
                except Exception:
                    st.error("Error generating EDA plots")
            else:
                st.warning("EDA data not available")
        with paper_tab:
            st.header("📄 Technical Paper")
            try:
                paper_file = Path("data/paper_details/FINAL_PAPER.txt")
                if paper_file.exists():
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper_content = f.read()
                    st.markdown(paper_content)
                else:
                    st.warning("Technical paper not found")
            except Exception:
                st.error("Error loading technical paper")
        st.stop()
    else:
        selected = st.sidebar.selectbox("Choose Bayesian suite", [d.name for d in suite_dirs])
        suite_path = suite_dirs[[d.name for d in suite_dirs].index(selected)]
        df = load_bayesian_data_with_fallback()
        if df is None:
            st.info("Bayesian features not available. Showing EDA and Technical Paper sections.")
            eda_tab, paper_tab = st.tabs(["📊 EDA & Analytics", "📄 Technical Paper"])
            with eda_tab:
                st.header("📊 Exploratory Data Analysis & Diagnostics")
                df_basic = load_preprocessed_data()
                if not df_basic.empty and EDA_AVAILABLE:
                    try:
                        st.subheader("Overall Outcome Distribution")
                        _, fig_out = outcome_summary(df_basic)
                        st.pyplot(fig_out)
                        st.subheader("Success Rate vs Distance")
                        _, fig_dist = distance_analysis(df_basic)
                        st.pyplot(fig_dist)
                        st.subheader("Temporal Trends & Age")
                        _, fig_temp = temporal_analysis(df_basic)
                        st.pyplot(fig_temp)
                        st.subheader("Kicker Performance Dashboard")
                        _, fig_kick = kicker_performance_analysis(df_basic)
                        st.pyplot(fig_kick)
                        st.subheader("Feature Correlation Matrix")
                        fig_corr = feature_engineering(df_basic)
                        st.pyplot(fig_corr)
                    except Exception:
                        st.error("Error generating EDA plots")
                else:
                    st.warning("EDA data not available")
            with paper_tab:
                st.header("📄 Technical Paper")
                try:
                    paper_file = Path("data/paper_details/FINAL_PAPER.txt")
                    if paper_file.exists():
                        with open(paper_file, 'r', encoding='utf-8') as f:
                            paper_content = f.read()
                        st.markdown(paper_content)
                    else:
                        st.warning("Technical paper not found")
                except Exception:
                    st.error("Error loading technical paper")
            st.stop()

        # Load Bayesian suite and create tabs
        try:
            suite = BayesianModelSuite.load_suite(suite_path)
            lb_tab, metrics_tab, analysis_tab, eda_tab, paper_tab = st.tabs([
                "🏅 EPA-FG⁺ Leaderboard",
                "📈 Model Metrics",
                "🎯 Kicker Analysis",
                "📊 EDA & Analytics",
                "📄 Technical Paper"
            ])

            # Tab A: EPA-FG⁺ Leaderboard
            with lb_tab:
                st.header("EPA-FG⁺ Leaderboard")
                try:
                    leaderboard_file = find_bayesian_leaderboard()
                    if leaderboard_file and leaderboard_file.exists():
                        df_lb = pd.read_csv(leaderboard_file)
                        st.dataframe(df_lb)
                    else:
                        st.info("Generating EPA-FG⁺ leaderboard...")
                        df_lb = suite.generate_epa_leaderboard(df)
                        st.dataframe(df_lb)
                except Exception:
                    st.error("Error generating EPA-FG⁺ leaderboard")

            # Tab B: Model Metrics
            with metrics_tab:
                st.header("Model Metrics")
                try:
                    df_metrics = get_bayesian_metrics(suite_path, df)
                    st.table(df_metrics)
                except Exception:
                    st.error("Error loading model metrics")

            # Tab C: Kicker Analysis
            with analysis_tab:
                st.header("Individual Kicker Analysis")
                try:
                    kicker_names = sorted(df['player_name'].unique())
                    selected_kicker = st.selectbox("Select Kicker", kicker_names)
                    
                    # Use cached metrics computation
                    metrics = compute_kicker_metrics(df, selected_kicker)
                    
                    # Display key metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Attempts", f"{metrics['total_attempts']:,}")
                    with col2:
                        st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
                    with col3:
                        st.metric("Avg Distance", f"{metrics['avg_distance']:.1f} yards")
                    
                    st.markdown("---")
                    
                    # Create two columns for visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.subheader("Make Probability Distribution")
                        fig_post = plot_kicker_skill_posterior(suite, df, selected_kicker)
                        st.pyplot(fig_post)
                    
                    with viz_col2:
                        st.subheader("Prediction vs Actuals by Distance")
                        fig_dist = plot_distance_comparison(suite, df, selected_kicker)
                        st.pyplot(fig_dist)
                        st.caption("Note: Predictions are binned into 5-yard intervals for clearer visualization.")
                        
                except Exception as e:
                    st.error(f"Error generating kicker analysis plots: {e}")

            # Tab D: EDA & Analytics
            with eda_tab:
                st.header("📊 Exploratory Data Analysis & Diagnostics")
                if not EDA_AVAILABLE:
                    st.info("Showing basic data information")
                    try:
                        if not df.empty:
                            st.write("Data Shape:", df.shape)
                            st.write("Data Info:")
                            st.dataframe(df.head())
                            if 'success' in df.columns:
                                success_rate = df['success'].mean()
                                st.metric("Overall Success Rate", f"{success_rate:.3f}")
                    except Exception:
                        st.error("Error loading data")
                else:
                    try:
                        if df.empty:
                            st.warning("No data available for analysis")
                            st.stop()

                        st.subheader("Overall Outcome Distribution")
                        _, fig_out = outcome_summary(df)
                        st.pyplot(fig_out)

                        st.subheader("Success Rate vs Distance")
                        _, fig_dist = distance_analysis(df)
                        st.pyplot(fig_dist)

                        st.subheader("Temporal Trends & Age")
                        _, fig_temp = temporal_analysis(df)
                        st.pyplot(fig_temp)

                        st.subheader("Kicker Performance Dashboard")
                        _, fig_kick = kicker_performance_analysis(df)
                        st.pyplot(fig_kick)

                        st.subheader("Feature Correlation Matrix")
                        fig_corr = feature_engineering(df)
                        st.pyplot(fig_corr)
                    except Exception:
                        st.error("Error generating EDA plots")

            # Tab E: Technical Paper
            with paper_tab:
                st.header("📄 Technical Paper")
                try:
                    paper_file = Path("data/paper_details/FINAL_PAPER.txt")
                    if paper_file.exists():
                        with open(paper_file, 'r', encoding='utf-8') as f:
                            paper_content = f.read()
                        st.markdown(paper_content)
                    else:
                        st.warning("Technical paper not found")
                except Exception:
                    st.error("Error loading technical paper")

        except Exception:
            st.error("Error loading Bayesian model suite")
            st.info("Showing EDA and Technical Paper sections only")
            eda_tab, paper_tab = st.tabs(["📊 EDA & Analytics", "📄 Technical Paper"])
            # ... (rest of the fallback code remains the same)

# Note: Technical Paper section is now handled within the tabs above

# ────────────────────────── Technical Paper Section ──────────────────────────
st.markdown("---")
st.header("📄 Technical Paper")

try:
    with open("data/paper_details/FINAL_PAPER.txt", "r") as f:
        paper_content = f.read()
    
    sections = paper_content.split("```mermaid")
    st.markdown(sections[0])
    
    for i, section in enumerate(sections[1:], 1):
        mermaid_and_rest = section.split("```", 2)
        if len(mermaid_and_rest) >= 2:
            mermaid_content = mermaid_and_rest[0].strip()
            st.write("")
            st.code(mermaid_content, language="mermaid")
            st.write("")
            if len(mermaid_and_rest) > 1:
                st.markdown(mermaid_and_rest[1])
    
    st.markdown("---")
    st.caption("© 2025 Geoffrey Hadfield. All rights reserved.")
    
except FileNotFoundError:
    st.error("Technical paper not found. Please ensure the paper file exists.")
except Exception:
    st.error("Error loading technical paper")
