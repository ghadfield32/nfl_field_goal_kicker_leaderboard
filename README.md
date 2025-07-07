# NFL Field Goal Kicker Analysis

A comprehensive Python package for analyzing NFL field goal kicker performance using advanced statistical methods and machine learning, featuring an interactive Streamlit dashboard.

Streamlit Link: https://nfl-field-goal-kicker-leaderboard.streamlit.app/

## Features

- **Interactive Streamlit Dashboard**: Visualize and analyze kicker performance in real-time
  - EPA-FG+ Leaderboard with 95% Confidence Intervals
  - Individual Kicker Analysis with Performance Distributions
  - Comprehensive EDA & Analytics Suite
  - Model Metrics & Performance Tracking
- **Data Loading & Preprocessing**: Robust data pipeline for NFL kicker datasets
- **EPA-FG+ Rating System**: Expected Points Added metrics for kicker evaluation
- **Age-Aware Modeling**: Non-linear age curves with cubic splines and career progression tracking
- **Multiple Model Support**: 
  - Traditional ML models (Logistic Regression, Random Forest, XGBoost, CatBoost)
  - Advanced Bayesian approaches with uncertainty quantification
- **Comprehensive Testing**: Full unit test coverage
- **Modular Design**: Clean, maintainable code architecture

## Package Structure

```
src/nfl_kicker_analysis/
├── __init__.py              # Main package imports
├── config.py                # Configuration and constants
├── pipeline.py              # Main analysis pipeline
├── data/
│   ├── __init__.py
│   ├── loader.py           # Data loading utilities
│   └── preprocessor.py     # Data preprocessing and feature engineering
├── models/
│   ├── __init__.py
│   ├── traditional.py      # Traditional ML models
│   └── bayesian.py         # Bayesian model implementations
├── utils/
│   ├── __init__.py
│   └── metrics.py          # EPA calculations and evaluation metrics
└── visualization/
    └── __init__.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nfl_field_goal_kicker_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Analysis

### 1. Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run app.py
```

The dashboard provides:
- 🔬 EPA-FG+ Leaderboard with uncertainty intervals
- ⛹️‍♂️ Individual Kicker Analysis with performance distributions
- 📊 EDA & Analytics with interactive visualizations
- 📈 Model Metrics and performance tracking

### 2. Python API Usage

Each module can be used independently:

```python
from nfl_kicker_analysis.data import DataLoader
from nfl_kicker_analysis.models import BayesianModelSuite
from nfl_kicker_analysis.utils import EPACalculator

# Load and preprocess data
loader = DataLoader()
data = loader.load_complete_dataset()

# Run Bayesian analysis
suite = BayesianModelSuite()
results = suite.fit(data)
predictions = suite.predict(data)

# Calculate EPA ratings
epa_calc = EPACalculator()
ratings = epa_calc.calculate_all_kicker_ratings(data)
```

## Data Requirements

Place your data files in the following structure:
```
data/
├── raw/
│   ├── kickers.csv
│   └── field_goal_attempts.csv
└── processed/
    ├── field_goal_modeling_data.csv
    └── leaderboard.csv
```

## Key Metrics

- **EPA-FG+**: Expected Points Added per field goal attempt, adjusted for distance
- **Success Rate**: Raw percentage of successful field goals
- **Difficulty-Adjusted Rating**: Accounts for attempt distance distribution
- **Age Curves & Career Stage**: Non-linear age effects with 3-knot cubic splines on age and explicit career progression modeling. Empirically, league-wide make probability peaks around age 30 and declines by ~0.3 percentage points per season thereafter.

## Model Performance

The package includes multiple modeling approaches:
- Simple Logistic Regression (distance-only baseline)  
- Ridge Logistic Regression (with kicker effects)
- Random Forest (non-linear relationships)
- XGBoost & CatBoost (gradient boosting)
- Hierarchical Bayesian Models (with age splines and career effects)

### Age-Aware Enhancement
Recent upgrades include sophisticated age modeling:
- **Cubic spline basis functions** (age_spline_1/2/3) for non-linear age curves
- **Career progression tracking** (seasons_of_experience, career_year) 
- **Automatic feature preservation** in selection pipelines
- **Enhanced model performance** with typical AUC improvements of 0.005-0.010

## Contributing

This modular design makes it easy to:
- Add new models in the `models/` directory
- Extend metrics in `utils/metrics.py`
- Add visualization in `visualization/`
- Modify configuration in `config.py`

## License

MIT License - see LICENSE file for details.
