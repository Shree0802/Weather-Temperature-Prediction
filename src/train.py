"""
Machine Learning Model Training, Comparison, and Evaluation pipeline
for Weather Temperature Prediction.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing import WeatherDataPreprocessor
from src.utils import MODELS_DIR, ensure_directories
from src.visualization import generate_all_eda_plots

# Check if XGBoost is available
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class ModelTrainer:
    """
    Automated Machine Learning trainer and evaluator.
    Trains multiple regression models, compares evaluation metrics, selects the best model,
    and serializes model artifacts to disk.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(
                max_depth=12, min_samples_split=5, random_state=random_state
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=4, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=random_state
            ),
        }

        if XGB_AVAILABLE:
            self.models["XGBoost Regressor"] = XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=random_state, n_jobs=-1
            )
        else:
            self.models["Extra Trees Regressor"] = ExtraTreesRegressor(
                n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1
            )

        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> pd.DataFrame:
        """
        Train all models, perform 5-fold cross-validation, calculate evaluation metrics,
        and build a comparison table.
        """
        ensure_directories()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        comparison_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            # 5-fold cross validation score
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
            cv_mean = np.mean(cv_scores)

            # Fit model on training set
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            # Extract feature importance if available
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_.tolist()
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).tolist()
            else:
                importances = [1.0 / X.shape[1]] * X.shape[1]

            self.results[name] = {
                "model": model,
                "R2_Score": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "CV_Score": cv_mean,
                "predictions": preds,
                "y_test": y_test,
                "feature_importances": importances,
            }

            comparison_list.append({
                "Model": name,
                "R² Score": round(r2, 4),
                "MAE (°C)": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE (°C)": round(rmse, 4),
                "CV Score (R²)": round(cv_mean, 4),
            })

        comparison_df = pd.DataFrame(comparison_list).sort_values(by="R² Score", ascending=False).reset_index(drop=True)

        # Select best model based on R² score
        self.best_model_name = comparison_df.iloc[0]["Model"]
        self.best_model = self.results[self.best_model_name]["model"]

        return comparison_df

    def save_best_model(self, X_cols: list) -> Path:
        """Save the best model and metadata to models/ Directory."""
        best_path = MODELS_DIR / "best_model.pkl"
        meta_path = MODELS_DIR / "metadata.json"

        joblib.dump(self.best_model, best_path)

        best_res = self.results[self.best_model_name]
        metadata = {
            "best_model_name": self.best_model_name,
            "R2_Score": best_res["R2_Score"],
            "MAE": best_res["MAE"],
            "MSE": best_res["MSE"],
            "RMSE": best_res["RMSE"],
            "CV_Score": best_res["CV_Score"],
            "feature_names": X_cols,
            "feature_importances": best_res["feature_importances"],
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return best_path


def run_full_training_pipeline() -> dict:
    """
    Run complete end-to-end training execution pipeline:
    1. Preprocess data
    2. Generate EDA static figures
    3. Train & compare all ML models
    4. Save best model
    """
    preprocessor = WeatherDataPreprocessor()
    df, X, y = preprocessor.fit_transform_pipeline()

    # Generate & save EDA figures
    eda_images = generate_all_eda_plots(df)

    # Train and evaluate models
    trainer = ModelTrainer()
    comparison_df = trainer.train_and_evaluate(X, y)
    best_model_path = trainer.save_best_model(list(X.columns))

    return {
        "processed_df": df,
        "X": X,
        "y": y,
        "comparison_df": comparison_df,
        "best_model_name": trainer.best_model_name,
        "best_model_path": best_model_path,
        "results": trainer.results,
        "eda_images": eda_images,
    }


if __name__ == "__main__":
    print("Starting full ML pipeline execution...")
    res = run_full_training_pipeline()
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(res["comparison_df"].to_string(index=False))
    print(f"\nBest Model Selected: {res['best_model_name']}")
    print(f"Artifact Saved to: {res['best_model_path']}")
