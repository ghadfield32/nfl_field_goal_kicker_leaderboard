"""
Traditional ML models for NFL kicker analysis.
Includes simple logistic regression, ridge logistic regression, and random forest.
Each model can be optionally tuned using Bayesian optimization with Optuna.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Optional, Union, Any
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier
from numpy.typing import NDArray

from src.nfl_kicker_analysis.utils.metrics import ModelEvaluator
from src.nfl_kicker_analysis.utils.metrics import EPACalculator
from src.nfl_kicker_analysis.config import config, FEATURE_LISTS
from src.nfl_kicker_analysis.utils.model_utils import save_model, load_model
from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
import scipy.special  # new import

def predict_proba_for_ridge(model, X):
    """
    Fallback for RidgeClassifier: use decision_function + sigmoid to approximate probabilities.
    """
    scores = model.decision_function(X)
    probs  = scipy.special.expit(scores)
    return np.vstack([1 - probs, probs]).T


class TreeBasedModelSuite:
    def __init__(self, *, feature_lists: dict[str, list[str]] | None = None):
        """Initialise evaluator, preprocessor & CV splitter."""
        self.fitted_models: dict[str, Any] = {}
        self.evaluator      = ModelEvaluator()
        self._tss           = TimeSeriesSplit(n_splits=3)

        # One shared preprocessor per suite
        self.preprocessor = DataPreprocessor()
        if feature_lists is not None:
            # Keep config MIN/MAX distance etc. but overwrite column roles
            self.preprocessor.update_feature_lists(**feature_lists)
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_], OneHotEncoder]:
        """
        Prepare feature matrices for modeling.
        
        Args:
            data: DataFrame with attempt_yards and kicker_id
            
        Returns:
            Tuple of:
            - Distance-only features
            - Combined features (distance + one-hot kicker)
            - Kicker IDs
            - OneHotEncoder for kickers
        """
        # Distance features
        X_distance = data['attempt_yards'].values.astype(np.float_).reshape(-1, 1)
        
        # Kicker IDs for tree models (fall back to player_id)
        ids_col = 'kicker_id' if 'kicker_id' in data.columns else 'player_id'
        kicker_ids = data[ids_col].values.astype(np.int_).reshape(-1, 1)
        
        # One-hot encode kickers for linear models
        encoder = OneHotEncoder(sparse_output=True)
        kicker_onehot = encoder.fit_transform(kicker_ids)
        X_combined = np.hstack([X_distance, kicker_onehot.toarray()])
        
        return X_distance, X_combined, kicker_ids, encoder
    
    def create_time_split(self, data: pd.DataFrame) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Create train/test split by time.
        
        Args:
            data: DataFrame with game_date
            
        Returns:
            Train and test indices
        """
        train_mask = data['season'] <= 2017
        test_mask = data['season'] == 2018
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        print(f"Train: {len(train_idx):,} attempts ({train_mask.mean():.1%})")
        print(f"Test: {len(test_idx):,} attempts ({test_mask.mean():.1%})")
        
        return train_idx, test_idx

    def _tune_simple_logistic_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int | None = None,
    ) -> LogisticRegression:
        """Bayesian-optimize a simple LogisticRegression."""
        n_trials = n_trials or config.OPTUNA_TRIALS["simple_logistic"]
        
        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 1e-5, 100, log=True)
            model = LogisticRegression(C=C, random_state=42)
            model.fit(X, y)
            return model.score(X, y)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best params and refit
        best_C = study.best_params["C"]
        best_model = LogisticRegression(C=best_C, random_state=42)
        best_model.fit(X, y)
        return best_model
        
    def _tune_ridge_logistic_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int | None = None,
    ) -> RidgeClassifier:
        """Bayesian-optimize a RidgeClassifier."""
        n_trials = n_trials or config.OPTUNA_TRIALS["ridge_logistic"]
        
        def objective(trial: optuna.Trial) -> float:
            alpha = trial.suggest_float("alpha", 1e-5, 100, log=True)
            model = RidgeClassifier(alpha=alpha, random_state=42)
            model.fit(X, y)
            return model.score(X, y)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best params and refit
        best_alpha = study.best_params["alpha"]
        best_model = RidgeClassifier(alpha=best_alpha, random_state=42)
        best_model.fit(X, y)
        return best_model
        
    def _tune_random_forest_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int | None = None,
    ) -> RandomForestClassifier:
        """Bayesian-optimize a RandomForestClassifier."""
        n_trials = n_trials or config.OPTUNA_TRIALS["random_forest"]
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
            }
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X, y)
            return model.score(X, y)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best params and refit
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X, y)
        return best_model
        
    def _tune_xgboost_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int | None = None,
    ) -> xgb.XGBClassifier:
        """Bayesian-optimize an XGBoost classifier."""
        n_trials = n_trials or config.OPTUNA_TRIALS["xgboost"]
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X, y)
            return model.score(X, y)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best params and refit
        best_params = study.best_params
        best_model = xgb.XGBClassifier(**best_params, random_state=42)
        best_model.fit(X, y)
        return best_model
        
    def _tune_catboost_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int | None = None,
    ) -> CatBoostClassifier:
        """Bayesian-optimize a CatBoost classifier on numeric array."""
        n_trials = n_trials or config.OPTUNA_TRIALS["catboost"]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth":      trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True)
            }
            model = CatBoostClassifier(**params, random_state=42, verbose=False)
            model.fit(X, y)
            return model.score(X, y)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_model = CatBoostClassifier(**best_params, random_state=42, verbose=False)
        best_model.fit(X, y)
        return best_model


    def fit_all_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Fit all traditional & boosted models and return their metrics.
        Also saves pipelines, leaderboards, and metrics to disk.
        """
        print("🔍 [fit_all_models] columns:", list(data.columns))

        # Ensure 'success' exists
        if "success" not in data.columns:
            if "field_goal_result" in data.columns:
                print("🔄 Deriving 'success' from 'field_goal_result'")
                data = data.copy()
                data["success"] = (data["field_goal_result"] == "Made").astype(int)
            else:
                raise KeyError("fit_all_models expects a 'success' column or 'field_goal_result'")

        # Preprocess and transform
        processed = self.preprocessor.preprocess_complete(data)
        X_full, y = self.preprocessor.fit_transform_features()
        train_idx, test_idx = self.create_time_split(processed)

        # Train each model
        print("\nTraining models on transformed features...")
        self.fitted_models['simple_logistic'] = self._tune_simple_logistic_optuna(
            X_full[train_idx], y[train_idx]
        )
        self.fitted_models['ridge_logistic'] = self._tune_ridge_logistic_optuna(
            X_full[train_idx], y[train_idx]
        )
        self.fitted_models['random_forest'] = self._tune_random_forest_optuna(
            X_full[train_idx], y[train_idx]
        )
        self.fitted_models['xgboost'] = self._tune_xgboost_optuna(
            X_full[train_idx], y[train_idx]
        )
        self.fitted_models['catboost'] = self._tune_catboost_optuna(
            X_full[train_idx], y[train_idx]
        )

        # Evaluate on hold-out
        print("\nEvaluating models on hold-out set…")
        metrics: Dict[str, Dict[str, float]] = {}
        for name, model in self.fitted_models.items():
            df_slice = processed.iloc[test_idx].reset_index(drop=True)
            X_test = self.preprocessor.transform_features(df_slice)
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = predict_proba_for_ridge(model, X_test)[:, 1]
            y_true = data["success"].values[test_idx]
            metrics[name] = self.evaluator.calculate_classification_metrics(y_true, y_pred)

        # Persist pipelines
        from sklearn.pipeline import Pipeline
        for name, model in self.fitted_models.items():
            pipeline = Pipeline([
                ("transformer", self.preprocessor.column_transformer_),
                ("model", model),
            ])
            save_model(pipeline, name, metrics=metrics[name])

        # Save leaderboards for each model
        for name in self.fitted_models.keys():
            lb = self.get_epa_leaderboard(data, model_name=name, top_n=10)
            lb_path = config.OUTPUT_DIR / f"{name}_leaderboard.csv"
            lb.to_csv(lb_path, index=False)
            print(f"✅ Saved {name} leaderboard to {lb_path}")

        # Save metrics summary
        metrics_df = pd.DataFrame(metrics).T
        metrics_path = config.OUTPUT_DIR / "model_metrics.csv"
        metrics_df.to_csv(metrics_path)
        print(f"✅ Saved model metrics to {metrics_path}")

        return metrics


    def predict(self, model_name: str, data: pd.DataFrame) -> NDArray[np.float_]:
        """
        Predict probabilities with any fitted model in the suite,
        applying the same preprocessing pipeline as used at training.
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} not fitted")

        model = self.fitted_models[model_name]

        # -- Use the preprocessor's transform_features for all models --
        X = self.preprocessor.transform_features(data)

        # -- Obtain probabilities whether predict_proba exists or not --
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            # Fallback for RidgeClassifier using logistic sigmoid
            from scipy.special import expit
            scores = model.decision_function(X)
            p = expit(scores)  # logistic(sigmoid) transform
            probs = p  # already the positive-class probability

        return probs.astype(np.float_)


    def get_feature_importance(self, model_name: str) -> Optional[NDArray[np.float_]]:
        """
        Get feature importance for tree-based models.
        """
        if model_name not in self.fitted_models:
            return None

        model = self.fitted_models[model_name]

        if model_name in {"catboost"}:
            return model.get_feature_importance().astype(np.float_)
        elif hasattr(model, "feature_importances_"):
            return model.feature_importances_.astype(np.float_)
        else:
            return None

    def get_epa_leaderboard(
        self,
        data: pd.DataFrame,
        model_name: str = "random_forest",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Generate EPA leaderboard using predictions from specified model.
        
        Parameters
        ----------
        data : DataFrame with required columns
        model_name : which model to use for predictions
        top_n : how many kickers to include (default 10)
        
        Returns
        -------
        DataFrame with columns:
            rank, player_name, attempts, success_rate, epa_per_attempt
        """
        # Get predictions
        y_pred = self.predict(model_name, data)
        
        # Compute EPA
        data = data.copy()
        data["pred_prob"] = y_pred
        data["epa"] = 3 * (data["success"] - data["pred_prob"])
        
        # Build leaderboard
        lb = (data.groupby(["player_id", "player_name"])
                  .agg(
                      attempts=("success", "size"),
                      success_rate=("success", "mean"),
                      epa_per_attempt=("epa", "mean")
                  )
                  .reset_index()
                  .sort_values("epa_per_attempt", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))
        lb["rank"] = lb.index + 1
        
        return lb[["rank", "player_name", "attempts", "success_rate", "epa_per_attempt"]]

if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor

    # 1️⃣ Ensure directories
    config.ensure_directories()

    # 2️⃣ Load & preprocess
    loader = DataLoader()
    raw = loader.merge_datasets()
    pre = DataPreprocessor()

    processed = pre.preprocess_slice(raw)

    # 3️⃣ Train & save
    suite = TreeBasedModelSuite()
    metrics = suite.fit_all_models(processed)
    print("\n📊 Final metrics:")
    for name, m in metrics.items():
        print(f"  • {name}: {m}")

    # 4️⃣ Generate, save & print RF leaderboard
    try:
        rf_lb = suite.get_epa_leaderboard(processed, model_name="random_forest", top_n=10)
        out_path = config.OUTPUT_DIR / "rf_leaderboard.csv"
        rf_lb.to_csv(out_path, index=False)
        print(f"\n✅ Saved RF leaderboard to {out_path}\n")
        print("🏆 Top-10 RF EPA Leaderboard:")
        print(rf_lb.to_string(index=False))
    except Exception as e:
        print(f"⚠️ Could not build RF leaderboard: {e}")

    print("\n🔄 Loading saved 'random_forest' pipeline for inference…")
    rf_pipe = load_model("random_forest")
    sample = processed.iloc[:5]   
    preds  = rf_pipe.predict_proba(sample)[:, 1]
    print("Sample RF preds:", preds)
