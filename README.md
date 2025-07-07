NFL Field Goal Kicker Analysis

A comprehensive Python package for analyzing NFL field goal kicker performance using advanced statistical methods and machine learning.

Streamlit app: https://nfl-field-goal-kicker-leaderboard.streamlit.app/

## Features

* Data Loading & Preprocessing: Robust data pipeline for NFL kicker datasets
* EPA-FG+ Rating System: Expected Points Added metrics for kicker evaluation
* Age-Aware Modeling: Non-linear age curves with cubic splines and career progression tracking
* Multiple Model Support: Traditional ML models and advanced Bayesian approaches
* Comprehensive Testing: Full unit test coverage
* Modular Design: Clean, maintainable code architecture

## Package Structure

```
src/nfl_kicker_analysis/
├── __init__.py              # Main package imports
├── config.py                # Configuration and constants
├── pipeline.py              # Main analysis pipeline
├── data/
│   ├── __init__.py
│   ├── loader.py            # Data loading utilities
│   └── preprocessor.py      # Data preprocessing and feature engineering
├── models/
│   ├── __init__.py
│   └── traditional.py       # Traditional ML models
├── utils/
│   ├── __init__.py
│   └── metrics.py           # EPA calculations and evaluation metrics
└── visualization/
    └── __init__.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from nfl_kicker_analysis import KickerAnalysisPipeline

# Run complete analysis
pipeline = KickerAnalysisPipeline()
results = pipeline.run_complete_analysis()

# Get top kickers
elite_kickers = pipeline.get_elite_kickers()
print(elite_kickers[['player_name', 'epa_fg_plus', 'rank']])
```

## Top 5 Leaderboards

Below are the top 5 performers by model, ranked by EPA per attempt or posterior mean for Bayesian logistic regression.

### CatBoost

| Rank | Player Name     | Attempts | Success Rate | EPA per Attempt |
| ---- | --------------- | -------- | ------------ | --------------- |
| 1    | BRETT MAHER     | 16       | 0.9375       | 0.04218         |
| 2    | JUSTIN TUCKER   | 240      | 0.9250       | 0.01941         |
| 3    | TRAVIS COONS    | 36       | 0.9722       | 0.01630         |
| 4    | HARRISON BUTKER | 54       | 0.9074       | 0.01458         |
| 5    | PATRICK MURRAY  | 47       | 0.8511       | 0.01389         |

### Random Forest

| Rank | Player Name     | Attempts | Success Rate | EPA per Attempt |
| ---- | --------------- | -------- | ------------ | --------------- |
| 1    | BRETT MAHER     | 16       | 0.9375       | 0.27984         |
| 2    | JUSTIN TUCKER   | 240      | 0.9250       | 0.13818         |
| 3    | HARRISON BUTKER | 54       | 0.9074       | 0.11629         |
| 4    | TRAVIS COONS    | 36       | 0.9722       | 0.11402         |
| 5    | NICK ROSE       | 12       | 0.9167       | 0.07345         |

### Ridge Logistic Regression

| Rank | Player Name     | Attempts | Success Rate | EPA per Attempt |
| ---- | --------------- | -------- | ------------ | --------------- |
| 1    | BRETT MAHER     | 16       | 0.9375       | 0.82428         |
| 2    | TRAVIS COONS    | 36       | 0.9722       | 0.79173         |
| 3    | NICK ROSE       | 12       | 0.9167       | 0.70855         |
| 4    | JUSTIN TUCKER   | 240      | 0.9250       | 0.69114         |
| 5    | STEVEN HAUSCHKA | 255      | 0.9255       | 0.68151         |

### Simple Logistic Regression

| Rank | Player Name     | Attempts | Success Rate | EPA per Attempt |
| ---- | --------------- | -------- | ------------ | --------------- |
| 1    | BRETT MAHER     | 16       | 0.9375       | 0.22824         |
| 2    | JUSTIN TUCKER   | 240      | 0.9250       | 0.16690         |
| 3    | TRAVIS COONS    | 36       | 0.9722       | 0.14796         |
| 4    | NICK ROSE       | 12       | 0.9167       | 0.08633         |
| 5    | STEVEN HAUSCHKA | 255      | 0.9255       | 0.07666         |

### XGBoost

| Rank | Player Name     | Attempts | Success Rate | EPA per Attempt |
| ---- | --------------- | -------- | ------------ | --------------- |
| 1    | ALDRICK ROSAS   | 37       | 0.8378       | 0.06867         |
| 2    | BRETT MAHER     | 16       | 0.9375       | 0.06233         |
| 3    | HARRISON BUTKER | 54       | 0.9074       | 0.05055         |
| 4    | ANDREW FRANKS   | 35       | 0.8571       | 0.02564         |
| 5    | BRANDON MCMANUS | 131      | 0.8550       | 0.01582         |

### Bayesian Logistic Regression

| Rank | Player Name        | EPA-FG+ Mean |
| ---- | ------------------ | ------------ |
| 1    | JUSTIN TUCKER      | 0.19870      |
| 2    | MATT BRYANT        | 0.08744      |
| 3    | STEVEN HAUSCHKA    | 0.07821      |
| 4    | STEPHEN GOSTKOWSKI | 0.07381      |
| 5    | ADAM VINATIERI     | 0.05899      |

## Individual Module Usage

Each module can be used independently:

```python
# Data loading
from nfl_kicker_analysis.data import DataLoader
loader = DataLoader()
data = loader.load_complete_dataset()

# EPA calculations
from nfl_kicker_analysis.utils import EPACalculator
epa_calc = EPACalculator()
ratings = epa_calc.calculate_all_kicker_ratings(data)

# Traditional models
from nfl_kicker_analysis.models import TraditionalModelSuite
models = TraditionalModelSuite()
results = models.fit_all_models(data)
```

## Testing

Run all tests:

```bash
python -m pytest tests/
```

Run individual module tests:

```bash
python src/nfl_kicker_analysis/config.py
python src/nfl_kicker_analysis/data/loader.py
python src/nfl_kicker_analysis/utils/metrics.py
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

* **EPA-F+**: Expected Points Added per field goal attempt, adjusted for distance
* **Success Rate**: Raw percentage of successful field goals
* **Difficulty-Adjusted Rating**: Accounts for attempt distance distribution
* **Age Curves & Career Stage**: Non-linear age effects with 3-knot cubic splines on age and explicit career progression modeling. Empirically, league-wide make probability peaks around age 30 and declines by \~0.3 percentage points per season thereafter.

## Model Performance

The package includes multiple modeling approaches:

* Simple Logistic Regression (distance-only baseline)
* Ridge Logistic Regression (with kicker effects)
* Random Forest (non-linear relationships)
* Hierarchical Bayesian Models (with age splines and career effects)

## Age-Aware Enhancement

Recent upgrades include sophisticated age modeling:

* Cubic spline basis functions (age\_spline\_1/2/3) for non-linear age curves
* Career progression tracking (seasons\_of\_experience, career\_year)
* Automatic feature preservation in selection pipelines
* Enhanced model performance with typical AUC improvements of 0.005-0.010

## 6. Comparison to Tree Models

| Model                 | AUC-ROC | AUC-PR | Log Loss | Brier  | ECE    | Accuracy | Precision | Recall | F1     |
| --------------------- | ------- | ------ | -------- | ------ | ------ | -------- | --------- | ------ | ------ |
| Bayesian hierarchical | 0.7975  | 0.9606 | 0.3151   | 0.0946 | 0.0250 | 0.8755   | 0.8750    | 1.0000 | 0.9333 |
| Simple Logistic       | 0.8049  | 0.9637 | 0.3240   | 0.1005 | 0.0379 | 0.8692   | 0.8688    | 1.0000 | 0.9298 |
| Ridge Logistic        | 0.7881  | 0.9562 | 0.4630   | 0.1408 | 0.1908 | 0.8692   | 0.8688    | 1.0000 | 0.9298 |
| Random Forest         | 0.7878  | 0.9584 | 0.3410   | 0.1030 | 0.0250 | 0.8663   | 0.8663    | 1.0000 | 0.9283 |
| XGBoost               | 0.8056  | 0.9621 | 0.3238   | 0.0990 | 0.0455 | 0.8634   | 0.8659    | 0.9966 | 0.9267 |
| CatBoost              | 0.7423  | 0.9486 | 0.3911   | 0.1119 | 0.0631 | 0.8576   | 0.8673    | 0.9866 | 0.9231 |

## Feature Columns and Reasoning

| Column            | Role                                | Reasoning                                                                                |
| ----------------- | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| attempt\_yards    | Distance (standardized)             | Primary driver of success; we z-score to aid sampler convergence and interpretability.   |
| distance\_squared | Distance²                           | Captures non-linear drop-off in make probability; minimal co-linearity after centering.  |
| is\_long\_attempt | (>50 yds) flag                      | Encodes extreme attempts where logistic slope may differ.                                |
| age\_c            | Centered age                        | Allows us to detect age trends around the league-average age (30 yrs).                   |
| age\_c2           | Age² (quadratic)                    | Models potential curvature in age effect.                                                |
| exp\_100          | Scaled experience (career attempts) | Captures learning curve and veteran steadiness; standardized to zero mean/unit variance. |
| season\_progress  | Fraction through season (0–1)       | Reflects potential clutch or fatigue effects as season advances.                         |
| rolling\_success  | Recent performance (exp. decay)     | Adjusts for momentum or slumps; exponential decay emphasizes the last few attempts.      |

## Contributing

This modular design makes it easy to:

* Add new models in the models/ directory
* Extend metrics in utils/metrics.py
* Add visualization in visualization/
* Modify configuration in config.py

## License

MIT License - see LICENSE file for details.
