"""
Inference Module for Weather Temperature Prediction.
Provides real-time single-sample inference and multi-day time series future trend forecasting.
"""

from pathlib import Path
import sys

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import json
import time
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import WeatherDataPreprocessor
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR, ensure_directories


class TemperaturePredictor:
    """
    Real-Time Temperature Inference Engine & Time-Series Multi-Step Forecaster.
    """

    def __init__(self):
        self.preprocessor = WeatherDataPreprocessor()
        self.model = None
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained best model and metadata from models/ directory."""
        best_model_path = MODELS_DIR / "best_model.pkl"
        metadata_path = MODELS_DIR / "metadata.json"

        if not best_model_path.exists() or not metadata_path.exists():
            from src.train import run_full_training_pipeline
            run_full_training_pipeline()

        self.model = joblib.load(best_model_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def predict(
        self,
        humidity: float,
        pressure: float,
        wind_speed: float,
        rainfall: float,
        month: int,
        day: int,
        weather_condition: str,
        year: int = 2026,
    ) -> dict:
        """
        Execute single-sample temperature inference.
        """
        start_time = time.time()

        input_payload = {
            "Humidity": float(humidity),
            "Pressure": float(pressure),
            "Wind Speed": float(wind_speed),
            "Rainfall": float(rainfall),
            "Month": int(month),
            "Day": int(day),
            "Year": int(year),
            "Weather Condition": str(weather_condition),
        }

        X_scaled = self.preprocessor.transform_single_input(input_payload)

        pred_array = self.model.predict(X_scaled)
        predicted_temp = float(np.round(pred_array[0], 2))

        rmse = self.metadata.get("RMSE", 0.5)
        conf_min = float(np.round(predicted_temp - 1.96 * max(rmse, 0.2), 2))
        conf_max = float(np.round(predicted_temp + 1.96 * max(rmse, 0.2), 2))

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "predicted_temperature": predicted_temp,
            "confidence_lower": conf_min,
            "confidence_upper": conf_max,
            "model_used": self.metadata.get("best_model_name", "Best ML Model"),
            "prediction_time_ms": elapsed_ms,
            "r2_score": self.metadata.get("R2_Score", 1.0),
            "rmse": rmse,
        }

    def predict_future_trend(self, n_days: int = 14) -> pd.DataFrame:
        """
        Generate autoregressive multi-step time series temperature forecast
        for the next n_days into the future.
        """
        processed_path = PROCESSED_DATA_DIR / "weather_data_processed.csv"
        if processed_path.exists():
            hist_df = pd.read_csv(processed_path)
            hist_df["Date"] = pd.to_datetime(hist_df["Date"])
            last_date = hist_df["Date"].max()
            last_humidity = hist_df["Humidity"].iloc[-1]
            last_pressure = hist_df["Pressure"].iloc[-1]
            last_wind = hist_df["Wind Speed"].iloc[-1]
            last_temp = hist_df["Temperature"].iloc[-1]
        else:
            last_date = pd.Timestamp.now()
            last_humidity, last_pressure, last_wind, last_temp = 65.0, 1013.25, 12.0, 22.0

        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]
        predictions = []

        curr_temp = last_temp
        rmse = self.metadata.get("RMSE", 0.5)

        for f_date in future_dates:
            # Simulate natural minor weather fluctuations
            sim_humidity = np.clip(last_humidity + np.random.normal(0, 2.0), 30, 95)
            sim_pressure = np.clip(last_pressure + np.random.normal(0, 1.5), 990, 1030)
            sim_wind = np.clip(last_wind + np.random.normal(0, 1.0), 3, 40)
            sim_rain = 0.0 if np.random.rand() > 0.2 else np.round(np.random.exponential(5.0), 1)

            payload = {
                "Date": f_date.strftime("%Y-%m-%d"),
                "Humidity": sim_humidity,
                "Pressure": sim_pressure,
                "Wind Speed": sim_wind,
                "Rainfall": sim_rain,
                "Month": f_date.month,
                "Day": f_date.day,
                "Year": f_date.year,
                "Weather Condition": "Rainy" if sim_rain > 5.0 else ("Sunny" if f_date.month in [6, 7, 8] else "Clear"),
                "Temp_Lag1": curr_temp,
                "Temp_Lag7": curr_temp,
                "Temp_Rolling7_Mean": curr_temp,
                "Temp_Rolling30_Mean": curr_temp,
            }

            X_scaled = self.preprocessor.transform_single_input(payload)
            pred_temp = float(np.round(self.model.predict(X_scaled)[0], 2))

            predictions.append({
                "Date": f_date,
                "Forecasted_Temperature": pred_temp,
                "Confidence_Lower": round(pred_temp - 1.96 * max(rmse, 0.3), 2),
                "Confidence_Upper": round(pred_temp + 1.96 * max(rmse, 0.3), 2),
                "Simulated_Humidity": round(sim_humidity, 1),
                "Simulated_Pressure": round(sim_pressure, 1),
            })
            curr_temp = pred_temp

        return pd.DataFrame(predictions)


if __name__ == "__main__":
    predictor = TemperaturePredictor()
    print("Single Inference:", predictor.predict(60, 1013, 10, 0, 7, 15, "Clear"))
    print("\nFuture 7-Day Trend Forecast:")
    print(predictor.predict_future_trend(7))
