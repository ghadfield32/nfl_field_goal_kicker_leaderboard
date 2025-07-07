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
    st.error(f"Configuration module not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error importing config: {e}")
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
except ImportError as e:
    st.warning(f"Model utilities not available: {e}")
    MODEL_UTILS_AVAILABLE = False

# Try to import Bayesian models
try:
    from src.nfl_kicker_analysis.models.bayesian import BayesianModelSuite
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    st.warning(f"Bayesian models not available: {e}")
    BAYESIAN_AVAILABLE = False
    BayesianModelSuite = None

# Try to import data modules
try:
    from src.nfl_kicker_analysis.data.loader import DataLoader
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Data loader not available: {e}")
    DATA_LOADER_AVAILABLE = False
    DataLoader = None

try:
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    DATA_PREPROCESSOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"Data preprocessor not available: {e}")
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
except ImportError as e:
    st.warning(f"EDA modules not available: {e}")
    EDA_AVAILABLE = False

def plot_kicker_skill_posterior(
    suite,
    df: pd.DataFrame,
    player_name: str,
    bin_width: int = 50,
) -> Figure:
    """
    Compute and plot the posterior distribution of P(make) for a single kicker
    at the empirical mean distance. Returns a matplotlib Figure.
    """
    if not ARVIZ_AVAILABLE or not BAYESIAN_AVAILABLE:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Bayesian analysis not available", ha='center', va='center')
        return fig
        
    k_idx = suite.get_kicker_id_by_name(df, player_name)
    if k_idx is None:
        raise ValueError(f"Kicker '{player_name}' not found")
    
    if suite._trace is None:
        raise ValueError("Model has not been fit yet")
        
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

# Custom CSS for Broncos‑flavoured palette & rounded cards
BRONCOS_COLOURS = {
    "orange": "#fb4f14",
    "navy": "#0a2343",
    "steel": "#a5acaf",
}

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* Global */
        html, body, [class*="css"]  {{
            font-family: "Inter", sans-serif;
        }}
        /* Accent colour */
        .st-bb  {{ border: none; }}
        .st-bx  {{ border-radius: 0.75rem; }}
        a, .st-de, .st-ag  {{ color: {BRONCOS_COLOURS['orange']} !important; }}
        /* Header */
        .block-container {{ padding-top: 2rem; }}
        /* Tabs style */
        div[data-baseweb="tab-list"] button  {{
            border-bottom: 3px solid transparent;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-color: {BRONCOS_COLOURS['orange']};
            color: {BRONCOS_COLOURS['orange']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

st.sidebar.header("⚙️ Select Model")
model_types = ["Point Estimate Models", "Uncertainty Interval Models"]
model_type = st.sidebar.selectbox("Model Type", model_types)

# Check if model utilities are available
if MODEL_UTILS_AVAILABLE:
    try:
        # DEBUG: Add comprehensive model directory debugging
        def debug_model_directories():
            """Debug function to show all model directory paths and contents."""
            debug_info = []
            
            # Check main MLflow directory (where models actually are)
            main_mlruns_dir = config.PROJECT_ROOT / "mlruns" / "models"
            debug_info.append(f"Main MLflow dir: {main_mlruns_dir}")
            debug_info.append(f"Main MLflow dir exists: {main_mlruns_dir.exists()}")
            if main_mlruns_dir.exists():
                debug_info.append(f"Main MLflow contents: {list(main_mlruns_dir.iterdir())}")
            
            # Check the old incorrect path
            old_point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"
            debug_info.append(f"Old point estimate dir: {old_point_estimate_dir}")
            debug_info.append(f"Old point estimate dir exists: {old_point_estimate_dir.exists()}")
            if old_point_estimate_dir.exists():
                debug_info.append(f"Old point estimate contents: {list(old_point_estimate_dir.iterdir())}")
            
            return debug_info
        
        # Show debug info in sidebar
        debug_info = debug_model_directories()
        with st.sidebar.expander("🔍 Debug: Model Directories", expanded=False):
            for info in debug_info:
                st.text(info)
        
        # FIX: Use the correct MLflow directory where models actually exist
        # Instead of config.MODELS_DIR / "mlruns" / "models" (which creates models/mlruns/models)
        # Use config.PROJECT_ROOT / "mlruns" / "models" (which is the actual MLflow location)
        correct_mlflow_dir = config.PROJECT_ROOT / "mlruns" / "models"
        
        # Try both the correct MLflow directory and the old path for backward compatibility
        fs_models = {}
        
        # 1. Check the correct MLflow directory first
        if correct_mlflow_dir.exists():
            fs_models.update(list_saved_models(correct_mlflow_dir))
            st.sidebar.success(f"✅ Found models in correct MLflow dir: {list(fs_models.keys())}")
        
        # 2. Also check the old path for any models that might be there
        old_point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"
        if old_point_estimate_dir.exists():
            old_fs_models = list_saved_models(old_point_estimate_dir)
            if old_fs_models:
                fs_models.update(old_fs_models)
                st.sidebar.info(f"ℹ️ Also found models in old path: {list(old_fs_models.keys())}")
        
        # 3. Get MLflow registered models
        mlflow_models = list_registered_models()
        if mlflow_models:
            st.sidebar.success(f"✅ Found registered models: {list(mlflow_models.keys())}")
        
        # Combine all models
        all_models = {**mlflow_models, **fs_models}
        
        # Store the correct directory for later use
        point_estimate_dir = correct_mlflow_dir
        
        # Create the directory if it doesn't exist
        if not point_estimate_dir.exists():
            point_estimate_dir.mkdir(parents=True, exist_ok=True)
            
    except Exception as e:
        st.error(f"Error accessing model directories: {e}")
        fs_models = {}
        mlflow_models = {}
        all_models = {}
        point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"  # fallback
else:
    fs_models = {}
    mlflow_models = {}
    all_models = {}
    point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"  # fallback

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

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if not DATA_LOADER_AVAILABLE:
        st.error("Data loader not available")
        return pd.DataFrame()
    loader = DataLoader()
    return loader.load_complete_dataset()

@st.cache_data(show_spinner=False)
def load_preprocessed_data() -> pd.DataFrame:
    """Load and preprocess data so we have success, kicker_id, etc."""
    if not DATA_LOADER_AVAILABLE or not DATA_PREPROCESSOR_AVAILABLE:
        # Try to load from CSV files directly if available
        try:
            # Look for preprocessed data file
            if hasattr(config, 'MODEL_DATA_FILE') and config.MODEL_DATA_FILE.exists():
                return pd.read_csv(config.MODEL_DATA_FILE)
            elif hasattr(config, 'PROCESSED_DATA_DIR') and (config.PROCESSED_DATA_DIR / "field_goal_modeling_data.csv").exists():
                return pd.read_csv(config.PROCESSED_DATA_DIR / "field_goal_modeling_data.csv")
            else:
                st.warning("Preprocessed data not available and modules not loaded")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading data files: {e}")
            return pd.DataFrame()
    
    try:
        loader = DataLoader()
        raw = loader.load_complete_dataset()
        pre = DataPreprocessor()
        
        # Safely access config attributes
        min_distance = getattr(config, 'MIN_DISTANCE', 20)
        max_distance = getattr(config, 'MAX_DISTANCE', 60)
        min_kicker_attempts = getattr(config, 'MIN_KICKER_ATTEMPTS', 10)
        season_types = getattr(config, 'SEASON_TYPES', ['Reg', 'Post'])
        feature_lists = getattr(config, 'FEATURE_LISTS', {})
        
        # Update with your config settings
        pre.update_config(
            min_distance=min_distance,
            max_distance=max_distance,
            min_kicker_attempts=min_kicker_attempts,
            season_types=season_types,
            include_performance_history=True,
            include_statistical_features=False,
            include_player_status=True,
            performance_window=12,
        )
        
        if feature_lists:
            pre.update_feature_lists(**feature_lists)
        
        # This both engineers and filters so we get a 'success' column, etc.
        return pre.preprocess_complete(raw)
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

def find_bayesian_suite_dirs() -> List[Path]:
    """Find Bayesian suite directories in multiple possible locations."""
    project_root = Path(__file__).parent.absolute()
    
    # List of possible locations for Bayesian suites
    possible_roots = [
        # Primary location from config
        getattr(config, 'MODEL_DIR', None),
        # Alternative locations
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

def load_bayesian_data_with_fallback() -> Optional[pd.DataFrame]:
    """Load Bayesian features data with comprehensive fallback options."""
    # Try to find the data file
    data_file = find_bayesian_data_file()
    
    if data_file is not None:
        try:
            df = pd.read_csv(data_file)
            st.sidebar.success(f"✅ Loaded Bayesian data from {data_file.name}")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {data_file}: {e}")
    
    # Fallback: Try to generate data from raw sources
    if DATA_LOADER_AVAILABLE and DATA_PREPROCESSOR_AVAILABLE:
        try:
            st.sidebar.warning("⚠️ Generating Bayesian features from raw data...")
            
            # Load and process data
            loader = DataLoader()
            df_raw = loader.load_complete_dataset()
            
            # Create minimal preprocessor for Bayesian models
            from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            df_feat = engineer.create_all_features(df_raw)
            
            st.sidebar.success("✅ Generated Bayesian features on-the-fly")
            return df_feat
            
        except Exception as e:
            st.sidebar.error(f"❌ Error generating features: {e}")
    
    return None

# Run setup
setup_for_cloud()

# === Main Streamlit App ===

if model_type == "Point Estimate Models":
    st.sidebar.subheader("🏆 Point Estimate Models")
    if all_models:
        model_names = list(all_models.keys())
        selected_model = st.selectbox("Choose best model", model_names)
        
        # Try to load the model
        try:
            if selected_model in fs_models:
                # Load from filesystem
                model = load_model(selected_model, base_dir=point_estimate_dir)
                st.success(f"✅ Loaded model '{selected_model}' from filesystem")
            elif selected_model in mlflow_models:
                # Load from MLflow - load_model function handles this automatically
                model = load_model(selected_model)
                st.success(f"✅ Loaded model '{selected_model}' from MLflow")
            else:
                st.warning(f"⚠️ Model '{selected_model}' not found in any location")
                model = None
                
        except Exception as e:
            st.error(f"❌ Model '{selected_model}' not available: {e}")
            
            # Enhanced debugging for model loading issues
            with st.expander("🔍 Debug: Model Loading Details", expanded=False):
                st.write(f"**Selected Model:** {selected_model}")
                st.write(f"**Available in filesystem:** {selected_model in fs_models}")
                st.write(f"**Available in MLflow:** {selected_model in mlflow_models}")
                st.write(f"**Point estimate directory:** {point_estimate_dir}")
                st.write(f"**Directory exists:** {point_estimate_dir.exists()}")
                
                # Show detailed error information
                import traceback
                st.code(traceback.format_exc(), language="python")
                
                # Show the actual model directory structure
                if point_estimate_dir.exists():
                    st.write("**Model directory contents:**")
                    for item in point_estimate_dir.iterdir():
                        if item.is_dir():
                            st.write(f"📁 {item.name}/")
                            # Show version directories
                            for sub_item in item.iterdir():
                                if sub_item.is_dir():
                                    st.write(f"  📁 {sub_item.name}/")
                                    # Show files in version directory
                                    for file in sub_item.iterdir():
                                        if file.is_file():
                                            st.write(f"    📄 {file.name}")
                                else:
                                    st.write(f"  📄 {sub_item.name}")
            
            model = None
    else:
        st.info("No trained models found. Showing available data without model predictions.")
        selected_model = None
        model = None

    # 2) Create tabs: metrics, leaderboard, and EDA (always available)
    if model:
        metrics_tab, lb_tab, eda_tab = st.tabs(
            ["📈 Model Metrics", "🏅 Leaderboard", "📊 EDA & Analytics"]
        )
    else:
        lb_tab, eda_tab = st.tabs(
            ["🏅 Leaderboard", "📊 EDA & Analytics"]
        )

    # ── Tab A: Full metrics table (only if model loaded) ────
    if model:
        with metrics_tab:
            st.header(f"{selected_model.replace('_',' ').title()} Metrics")
            try:
                df_metrics = get_metrics_df(selected_model)
                st.table(df_metrics)  # shows all logged metrics
            except Exception as e:
                st.error(f"Error loading metrics: {str(e)}")

    # ── Tab B: Existing leaderboard ─────────────────────────
    with lb_tab:
        st.header(f"{selected_model.replace('_',' ').title()} Leaderboard")
        
        # Try multiple leaderboard file locations with absolute paths
        possible_files = []
        
        # Use absolute paths based on the project root
        project_root = Path(__file__).parent.absolute()
        
        # Primary locations to check
        possible_files.extend([
            project_root / "output" / f"{selected_model}_leaderboard.csv",
            project_root / "data" / "processed" / f"{selected_model}_leaderboard.csv",
            project_root / f"{selected_model}_leaderboard.csv",
        ])
        
        # Also try config-based paths if available
        if hasattr(config, 'OUTPUT_DIR'):
            possible_files.append(config.OUTPUT_DIR / f"{selected_model}_leaderboard.csv")
        if hasattr(config, 'PROCESSED_DATA_DIR'):
            possible_files.append(config.PROCESSED_DATA_DIR / f"{selected_model}_leaderboard.csv")
        
        # Debug: Show which files we're looking for
        with st.expander("🔍 Debug: Leaderboard File Search", expanded=False):
            st.write("Looking for leaderboard files in:")
            for f in possible_files:
                exists = f.exists()
                st.write(f"{'✅' if exists else '❌'} {f} (exists: {exists})")
        
        leaderboard_found = False
        for lb_file in possible_files:
            if lb_file.exists():
                try:
                    df_lb = pd.read_csv(lb_file)
                    if model is not None:
                        # Get model info including accuracy
                        try:
                            version, accuracy = get_best_model_info(selected_model)
                            if accuracy is not None:
                                st.write(f"**Accuracy:** {accuracy:.3f}")
                        except Exception:
                            pass  # Skip accuracy display if not available
                    
                    st.success(f"✅ Loaded leaderboard from: {lb_file}")
                    st.dataframe(df_lb)
                    leaderboard_found = True
                    break
                except Exception as e:
                    st.warning(f"Error reading {lb_file}: {e}")
        
        if not leaderboard_found:
            # If no leaderboard file found, try to generate one on-the-fly if we have a model
            if model is not None and not load_preprocessed_data().empty:
                st.info(f"No pre-generated leaderboard found for {selected_model}. Generating on-the-fly...")
                try:
                    # Generate leaderboard on-the-fly
                    data = load_preprocessed_data()
                    
                    # Simple leaderboard generation
                    if 'player_name' in data.columns and 'success' in data.columns:
                        leaderboard = (
                            data.groupby('player_name')
                            .agg({
                                'success': ['count', 'sum', 'mean'],
                                'attempt_yards': 'mean'
                            })
                            .round(3)
                        )
                        
                        # Flatten column names
                        leaderboard.columns = ['attempts', 'made', 'success_rate', 'avg_distance']
                        leaderboard = leaderboard.reset_index()
                        
                        # Filter for minimum attempts
                        leaderboard = leaderboard[leaderboard['attempts'] >= 10]
                        
                        # Sort by success rate
                        leaderboard = leaderboard.sort_values('success_rate', ascending=False)
                        
                        # Add rank
                        leaderboard['rank'] = range(1, len(leaderboard) + 1)
                        
                        # Reorder columns
                        leaderboard = leaderboard[['rank', 'player_name', 'attempts', 'success_rate', 'avg_distance']]
                        
                        st.dataframe(leaderboard)
                        leaderboard_found = True
                        
                except Exception as e:
                    st.error(f"Error generating leaderboard: {e}")
            
            if not leaderboard_found:
                st.warning(f"No leaderboard available for {selected_model}. The model may not have been trained yet, or leaderboard files may not be available in this environment.")
                
                # Show available files for debugging
                with st.expander("🔍 Debug: Available Files", expanded=False):
                    st.write("Available files in project directory:")
                    try:
                        for root, dirs, files in os.walk(project_root):
                            for file in files:
                                if file.endswith('.csv'):
                                    st.write(f"📄 {Path(root) / file}")
                    except Exception as e:
                        st.write(f"Error listing files: {e}")

    # ── Tab C: EDA & Analytics (always available) ───────────
    with eda_tab:
        st.header("📊 Exploratory Data Analysis & Diagnostics")
        if not EDA_AVAILABLE:
            st.warning("EDA modules not available. Showing basic data information.")
            try:
                data = load_preprocessed_data()
                if not data.empty:
                    st.write("Data Shape:", data.shape)
                    st.write("Data Info:")
                    st.dataframe(data.head())
                    if 'success' in data.columns:
                        success_rate = data['success'].mean()
                        st.metric("Overall Success Rate", f"{success_rate:.3f}")
            except Exception as e:
                st.error(f"Error loading basic data: {str(e)}")
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

                st.markdown("---")
                st.caption(
                    "Plots generated on-the-fly using reusable utilities from the core package."
                )
            except Exception as e:
                st.error(f"Error loading EDA data: {str(e)}")
                st.info("Please ensure the data files are available in the data directory.")

else:
    st.sidebar.subheader("🔬 Uncertainty Interval Models")
    
    # Find Bayesian suite directories with enhanced search
    suite_dirs = find_bayesian_suite_dirs()

    # If there are no saved suites at all, show fallback content
    if not suite_dirs:
        st.sidebar.warning(
            "No saved Bayesian suites found.\n"
            "❗ Please run your training pipeline with "
            "`suite.save_suite(...)` targeting a subfolder of MODEL_DIR."
        )
        st.sidebar.info("📊 Showing EDA and Technical Paper sections only")
        
        # Show EDA and Technical Paper as fallback
        eda_tab, paper_tab = st.tabs(["📊 EDA & Analytics", "📄 Technical Paper"])
        
        with eda_tab:
            st.header("📊 Exploratory Data Analysis & Diagnostics")
            
            # Load preprocessed data for EDA
            df = load_preprocessed_data()
            if not df.empty and EDA_AVAILABLE:
                try:
                    # Display EDA plots
                    st.subheader("📈 Distance vs Success Analysis")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    distance_analysis(df, savefig=None)
                    st.pyplot(fig1)
                    
                    st.subheader("⏰ Temporal Analysis")
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    temporal_analysis(df, savefig=None)
                    st.pyplot(fig2)
                    
                    st.subheader("🎯 Kicker Performance Analysis")
                    fig3, ax3 = plt.subplots(figsize=(15, 12))
                    kicker_performance_analysis(df, savefig=None)
                    st.pyplot(fig3)
                    
                except Exception as e:
                    st.error(f"Error generating EDA plots: {str(e)}")
            else:
                st.warning("EDA data not available")
        
        with paper_tab:
            st.header("📄 Technical Paper")
            try:
                # Load and display technical paper content
                paper_file = Path("data/paper_details/FINAL_PAPER.txt")
                if paper_file.exists():
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper_content = f.read()
                    st.markdown(paper_content)
                else:
                    st.warning("Technical paper not found")
            except Exception as e:
                st.error(f"Error loading technical paper: {str(e)}")
        
        # Exit early - don't try to load Bayesian models
        st.stop()
    
    else:
        selected = st.sidebar.selectbox("Choose Bayesian suite", [d.name for d in suite_dirs])
        suite_path = suite_dirs[[d.name for d in suite_dirs].index(selected)]

        # Try to load Bayesian features data with enhanced fallback
        df = load_bayesian_data_with_fallback()
        
        if df is None:
            st.sidebar.warning(
                "Missing Bayesian features data.\n\n"
                "Showing EDA and Technical Paper sections only."
            )
            # Show EDA and Technical Paper even without Bayesian models
            eda_tab, paper_tab = st.tabs(["📊 EDA & Analytics", "📄 Technical Paper"])
            
            with eda_tab:
                st.header("📊 Exploratory Data Analysis & Diagnostics")
                
                # Load basic data for EDA
                df_basic = load_preprocessed_data()
                if not df_basic.empty and EDA_AVAILABLE:
                    try:
                        # Display EDA plots with basic data
                        st.subheader("📈 Distance vs Success Analysis")
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        distance_analysis(df_basic, savefig=None)
                        st.pyplot(fig1)
                        
                        st.subheader("⏰ Temporal Analysis")
                        fig2, ax2 = plt.subplots(figsize=(12, 8))
                        temporal_analysis(df_basic, savefig=None)
                        st.pyplot(fig2)
                        
                    except Exception as e:
                        st.error(f"Error generating EDA plots: {str(e)}")
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
                except Exception as e:
                    st.error(f"Error loading technical paper: {str(e)}")
            
            # Exit early - don't try to load Bayesian models
            st.stop()

        # Render metrics & leaderboard, catching errors
        try:
            # Load the suite first with enhanced error handling
            suite = BayesianModelSuite.load_suite(suite_path)
            
            # Create tabs: EPA Leaderboard, Kicker Analysis, EDA, and Model Metrics
            lb_tab, kicker_tab, eda_tab, metrics_tab = st.tabs([
                "🔬 EPA-FG⁺ Leaderboard",
                "⛹️‍♂️ Kicker Analysis",
                "📊 EDA & Analytics",
                "📈 Model Metrics"
            ])

            # Tab 1: EPA-FG⁺ Leaderboard
            with lb_tab:
                st.header("🔬 Bayesian EPA-FG⁺ Leaderboard with 95% CI")
                
                # First try to load the pre-saved leaderboard
                leaderboard_file = find_bayesian_leaderboard()
                
                if leaderboard_file is not None:
                    try:
                        st.info(f"📁 Loading pre-saved Bayesian leaderboard from {leaderboard_file.name}")
                        df_leaderboard = pd.read_csv(leaderboard_file)
                        
                        # Ensure we have the required columns
                        required_cols = ['player_name', 'epa_fg_plus_mean', 'rank']
                        if all(col in df_leaderboard.columns for col in required_cols):
                            # Sort by rank for display
                            df_leaderboard = df_leaderboard.sort_values('rank')
                            
                            # Display the leaderboard
                            st.dataframe(df_leaderboard)
                            
                            # Add download button for the leaderboard
                            csv = df_leaderboard.to_csv(index=False)
                            st.download_button(
                                label="📥 Download EPA Leaderboard as CSV",
                                data=csv,
                                file_name=f"bayesian_epa_leaderboard_{selected}.csv",
                                mime="text/csv"
                            )
                            
                            # Show summary statistics
                            st.subheader("📊 Leaderboard Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Kickers", len(df_leaderboard))
                            with col2:
                                top_epa = df_leaderboard['epa_fg_plus_mean'].max()
                                st.metric("Top EPA-FG⁺", f"{top_epa:.4f}")
                            with col3:
                                avg_epa = df_leaderboard['epa_fg_plus_mean'].mean()
                                st.metric("Average EPA-FG⁺", f"{avg_epa:.4f}")
                                
                            st.success("✅ Successfully loaded pre-saved Bayesian leaderboard")
                            
                        else:
                            st.warning(f"⚠️ Leaderboard file missing required columns. Found: {list(df_leaderboard.columns)}")
                            raise ValueError("Missing required columns")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Error loading pre-saved leaderboard: {e}")
                        st.info("Falling back to generating leaderboard on-the-fly...")
                        leaderboard_file = None  # Force fallback
                
                # Fallback: Generate leaderboard on-the-fly if no saved file or loading failed
                if leaderboard_file is None:
                    try:
                        st.info("🔄 Generating EPA leaderboard on-the-fly (this may take a moment)...")
                        with st.spinner("Computing Bayesian EPA leaderboard..."):
                            df_ci = (
                                suite.epa_fg_plus(df,
                                                n_samples=config.BAYESIAN_MCMC_SAMPLES,
                                                return_ci=True)
                                     .reset_index()
                                     .sort_values("epa_fg_plus_mean", ascending=False)
                            )
                        st.dataframe(df_ci)
                        
                        # Add download button for the leaderboard
                        csv = df_ci.to_csv(index=False)
                        st.download_button(
                            label="📥 Download EPA Leaderboard as CSV",
                            data=csv,
                            file_name=f"bayesian_epa_leaderboard_{selected}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating EPA leaderboard: {e}")
                        st.info("This may be due to missing dependencies or data incompatibility.")
                        
                        # Show debug information
                        with st.expander("🔧 Debug Information"):
                            st.write("**Suite Information:**")
                            st.write(f"- Suite path: {suite_path}")
                            st.write(f"- Data shape: {df.shape}")
                            st.write(f"- Data columns: {list(df.columns)}")
                            
                            # Show traceback
                            import traceback
                            st.code(traceback.format_exc())

            # Tab 2: Kicker Analysis
            with kicker_tab:
                st.header("⛹️‍♂️ Individual Kicker Analysis")
                
                try:
                    # Get unique kickers for selection
                    if 'player_name' in df.columns:
                        kickers = sorted(df['player_name'].unique())
                        selected_kicker = st.selectbox("Choose a kicker", kickers)
                        
                        if selected_kicker:
                            # Show kicker statistics
                            kicker_data = df[df['player_name'] == selected_kicker]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Attempts", len(kicker_data))
                            with col2:
                                success_rate = kicker_data['success'].mean()
                                st.metric("Success Rate", f"{success_rate:.1%}")
                            with col3:
                                avg_distance = kicker_data['attempt_yards'].mean()
                                st.metric("Avg Distance", f"{avg_distance:.1f} yds")
                            
                            # Generate posterior plot if possible
                            if BAYESIAN_AVAILABLE and ARVIZ_AVAILABLE:
                                try:
                                    with st.spinner("Generating skill posterior..."):
                                        fig = plot_kicker_skill_posterior(suite, df, selected_kicker)
                                        st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Could not generate posterior plot: {e}")
                            
                            # Show kicker's recent attempts
                            st.subheader("Recent Attempts")
                            recent_attempts = kicker_data.sort_values('game_date', ascending=False).head(10)
                            st.dataframe(recent_attempts[['game_date', 'attempt_yards', 'success', 'field_goal_result']])
                    
                    else:
                        st.warning("Player names not available in the dataset")
                        
                except Exception as e:
                    st.error(f"❌ Error in kicker analysis: {e}")

            # Tab 3: EDA & Analytics (same as before)
            with eda_tab:
                st.header("📊 Exploratory Data Analysis & Diagnostics")
                
                if EDA_AVAILABLE:
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

                        st.markdown("---")
                        st.caption(
                            "Plots generated on-the-fly using reusable utilities from the core package."
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating EDA plots: {str(e)}")
                else:
                    st.warning("EDA modules not available")

            # Tab 4: Model Metrics
            with metrics_tab:
                st.header("📈 Model Performance Metrics")
                
                try:
                    # Load or compute metrics
                    dfm = get_bayesian_metrics(suite_path, df)
                    
                    if not dfm.empty:
                        st.dataframe(dfm)
                        
                        # Show key metrics prominently
                        if len(dfm) > 0:
                            st.subheader("🎯 Key Performance Indicators")
                            
                            # Create columns for key metrics
                            metrics_cols = st.columns(3)
                            
                            for i, (_, row) in enumerate(dfm.head(3).iterrows()):
                                with metrics_cols[i % 3]:
                                    st.metric(
                                        label=row['Metric'].replace('_', ' ').title(),
                                        value=f"{row['Value']:.4f}"
                                    )
                    else:
                        st.warning("No metrics available")
                        
                except Exception as e:
                    st.error(f"❌ Error loading metrics: {e}")
                    
                    # Show debug information
                    with st.expander("🔧 Debug Information"):
                        st.write("**Metrics Loading Debug:**")
                        st.write(f"- Suite path: {suite_path}")
                        st.write(f"- Metrics file exists: {(suite_path / 'metrics.json').exists()}")
                        
                        import traceback
                        st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Error loading Bayesian suite: {e}")
            
            # Show comprehensive debug information
            st.subheader("🔧 Debug Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Suite Directory Structure:**")
                try:
                    if suite_path.exists():
                        files = list(suite_path.iterdir())
                        for file in files:
                            st.write(f"- {file.name} ({'dir' if file.is_dir() else 'file'})")
                    else:
                        st.write("Suite path does not exist")
                except Exception as debug_e:
                    st.write(f"Error listing files: {debug_e}")
            
            with col2:
                st.write("**Bayesian Data Search Paths:**")
                data_file = find_bayesian_data_file()
                if data_file:
                    st.write(f"✅ Found: {data_file}")
                else:
                    st.write("❌ No data file found")
                    
                st.write("**Available Suite Directories:**")
                for i, suite_dir in enumerate(suite_dirs):
                    st.write(f"{i+1}. {suite_dir}")
            
            # Show full traceback
            with st.expander("📋 Full Error Traceback"):
                import traceback
                st.code(traceback.format_exc())

# Note: Technical Paper section is now handled within the tabs above

# ────────────────────────── Technical Paper Section ──────────────────────────
st.markdown("---")  # Add a visual separator
st.header("📄 Technical Paper")

try:
    with open("data/paper_details/FINAL_PAPER.txt", "r") as f:
        paper_content = f.read()
    
    # Split content at mermaid sections
    sections = paper_content.split("```mermaid")
    
    # Display first part
    st.markdown(sections[0])
    
    # Handle each mermaid diagram
    for i, section in enumerate(sections[1:], 1):
        # Split at the end of the mermaid block
        mermaid_and_rest = section.split("```", 2)
        if len(mermaid_and_rest) >= 2:
            # Extract mermaid content
            mermaid_content = mermaid_and_rest[0].strip()
            st.write("")
            
            # Display mermaid diagram as code block for now
            # TODO: Could integrate create_diagram tool if available
            st.code(mermaid_content, language="mermaid")
            
            st.write("") 
            
            # Display the rest of the content
            if len(mermaid_and_rest) > 1:
                st.markdown(mermaid_and_rest[1])
    
    # Add citation information
    st.markdown("---")
    st.caption("© 2025 Geoffrey Hadfield. All rights reserved.")
    
except FileNotFoundError:
    st.error(
        "Technical paper file not found. Please ensure "
        "`data/paper_details/FINAL_PAPER.txt` exists."
    )
except Exception as e:
    st.error(f"Error loading technical paper: {str(e)}")



