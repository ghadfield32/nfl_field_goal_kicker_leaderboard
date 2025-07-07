# app.py
from pathlib import Path
import pandas as pd
import streamlit as st
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, cast
import arviz as az

from src.nfl_kicker_analysis.config import config
from src.nfl_kicker_analysis.utils.model_utils import (
    list_registered_models,
    list_saved_models,
    load_model,
    get_best_model_info,
    get_best_metrics,
)
from src.nfl_kicker_analysis.models.bayesian import BayesianModelSuite
from src.nfl_kicker_analysis.data.loader import DataLoader
from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
from src.nfl_kicker_analysis.eda import outcome_summary, distance_analysis, temporal_analysis, kicker_performance_analysis, feature_engineering

def plot_kicker_skill_posterior(
    suite: BayesianModelSuite,
    df: pd.DataFrame,
    player_name: str,
    bin_width: int = 50,
) -> Figure:
    """
    Compute and plot the posterior distribution of P(make) for a single kicker
    at the empirical mean distance. Returns a matplotlib Figure.
    """
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
    suite: BayesianModelSuite,
    df: pd.DataFrame,
    player_name: str,
    n_samples: int = 500,
    bin_width: int = 20,
) -> Figure:
    """
    Bootstrap EPA-FG⁺ draws for one kicker and plot histogram.
    """
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

fs_models     = list_saved_models(config.MODEL_DIR)
mlflow_models = list_registered_models()
all_models    = {**mlflow_models, **fs_models}



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
    loader = DataLoader()
    return loader.load_complete_dataset()

@st.cache_data(show_spinner=False)
def load_preprocessed_data() -> pd.DataFrame:
    """Load and preprocess data so we have success, kicker_id, etc."""
    loader = DataLoader()
    raw = loader.load_complete_dataset()
    pre = DataPreprocessor()
    # Update with your config settings
    pre.update_config(
        min_distance=config.MIN_DISTANCE,
        max_distance=config.MAX_DISTANCE,
        min_kicker_attempts=config.MIN_KICKER_ATTEMPTS,
        season_types=config.SEASON_TYPES,
        include_performance_history=True,
        include_statistical_features=False,
        include_player_status=True,
        performance_window=12,
    )
    pre.update_feature_lists(**config.FEATURE_LISTS)
    # This both engineers and filters so we get a 'success' column, etc.
    return pre.preprocess_complete(raw)

@st.cache_data(show_spinner=False)
def get_bayesian_metrics(suite_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    metrics_path = suite_dir / "metrics.json"
    if metrics_path.exists():
        # Load the exact metrics from training time
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            st.sidebar.success("✅ Loaded saved metrics from training")
    else:
        # Fallback: recompute on-the-fly
        st.sidebar.warning("⚠️ No saved metrics found - recomputing")
        suite = BayesianModelSuite.load_suite(suite_dir)
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
        metrics = suite.evaluate(df, preprocessor=pre)

    # Convert to DataFrame for display
    dfm = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["Value"]
    ).reset_index().rename(columns={"index":"Metric"})
    return dfm

if model_type == "Point Estimate Models":
    st.sidebar.subheader("🏆 Point Estimate Models")
    sel = st.sidebar.selectbox("Choose best model", POINT_ESTIMATE_MODELS)

    if sel:
        # 1) Load version & raw accuracy for the sidebar
        ver, acc = get_best_model_info(sel)
        mdl = load_model(sel, version=ver)
        st.sidebar.success(f"✅ Loaded {sel} (v{ver}) — acc {acc:.3f}")

        # 2) Create two tabs: one for metrics, one for the leaderboard
        metrics_tab, lb_tab = st.tabs(
            ["📈 Model Metrics", "🏅 Leaderboard"]
        )

        # ── Tab A: Full metrics table ───────────────────────────
        with metrics_tab:
            st.header(f"{sel.replace('_',' ').title()} Metrics")
            df_metrics = get_metrics_df(sel)
            st.table(df_metrics)  # shows all logged metrics

        # ── Tab B: Existing leaderboard ─────────────────────────
        with lb_tab:
            st.header(f"{sel.replace('_',' ').title()} Leaderboard")
            lb_file = config.OUTPUT_DIR / f"{sel}_leaderboard.csv"
            if lb_file.exists():
                df_lb = pd.read_csv(lb_file)
                st.write(f"**Accuracy:** {acc:.3f}")
                st.dataframe(df_lb)
            else:
                st.warning(f"No leaderboard found at {lb_file}")

else:
    st.sidebar.subheader("🔬 Uncertainty Interval Models")
    suite_root = Path(config.MODEL_DIR)
    suite_dirs = sorted(
        [d for d in suite_root.iterdir()
         if d.is_dir() and (d/"meta.json").exists() and (d/"trace.nc").exists()],
        reverse=True
    )

    # If there are no saved suites at all, warn and stop here.
    if not suite_dirs:
        st.sidebar.warning(
            "No saved Bayesian suites found.\n"
            "❗ Please run your training pipeline with "
            "`suite.save_suite(...)` targeting a subfolder of MODEL_DIR."
        )
    else:
        selected = st.sidebar.selectbox("Choose Bayesian suite", [d.name for d in suite_dirs])
        suite_path = suite_root / selected

        # Ensure engineered-features file exists
        data_file = config.MODEL_DATA_FILE
        if not data_file.exists():
            st.sidebar.error(
                f"Missing features file:\n  {data_file}\n\n"
                "Please run your smoke-test pipeline to generate it first."
            )
            st.stop()

        df = pd.read_csv(data_file)

        # Render metrics & leaderboard, catching errors
        try:
            # Load the suite first
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
                df_ci = (
                    suite.epa_fg_plus(df,
                                    n_samples=config.BAYESIAN_MCMC_SAMPLES,
                                    return_ci=True)
                         .reset_index()
                         .sort_values("epa_fg_plus_mean", ascending=False)
                )
                st.dataframe(df_ci)

            # Tab 2: Kicker Analysis
            with kicker_tab:
                st.header("🎯 Individual Kicker Analysis")
                # Add kicker selection
                kicker_list = df["player_name"].unique().tolist()
                selected_kicker = st.selectbox("Select a Kicker", sorted(kicker_list))

                if selected_kicker:
                    try:
                        # Get and display interval
                        interval = suite.kicker_interval_by_name(df, selected_kicker)
                        st.markdown(
                            f"**P(make)**: {interval['mean']:.3f} "
                            f"(95% CI: {interval['lower']:.3f} – {interval['upper']:.3f})"
                        )
                        
                        # Create two columns for the plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Make Probability Distribution**")
                            fig_skill = plot_kicker_skill_posterior(suite, df, selected_kicker)
                            st.pyplot(fig_skill)
                            
                        with col2:
                            st.markdown("**EPA-FG⁺ Distribution**")
                            fig_epa = plot_kicker_epa_distribution(suite, df, selected_kicker)
                            st.pyplot(fig_epa)
                            
                    except Exception as e:
                        st.error(f"Error analyzing kicker: {str(e)}")

            # Tab 3: EDA & Analytics
            with eda_tab:
                st.header("📊 Exploratory Data Analysis & Diagnostics")
                data = load_preprocessed_data()

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

            # Tab 4: Model Metrics
            with metrics_tab:
                st.header("Bayesian Model Evaluation Metrics")
                df_metrics = get_bayesian_metrics(suite_path, df)
                st.table(df_metrics)

        except Exception as e:
            st.sidebar.error(f"Failed to load suite or compute metrics: {e}")

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
            
            # Call the create_diagram tool
            try:
                result = create_diagram({"content": mermaid_content, "explanation": "Creating diagram for technical paper section"})
                st.write(result)
            except Exception as e:
                st.error(f"Failed to render diagram: {str(e)}")
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






