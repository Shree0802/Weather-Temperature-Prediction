"""
Inference Module for Weather Temperature Prediction.
Provides real-time prediction using the serialized best ML model.
"""

import sys
from pathlib import Path

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
from src.utils import MODELS_DIR, ensure_directories


class TemperaturePredictor:
    """
    Real-time Temperature Inference Engine.
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
        
        Returns:
            dict: Containing predicted_temperature, confidence bounds, model name, and execution time.
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

        # Preprocess input payload into scaled feature DataFrame
        X_scaled = self.preprocessor.transform_single_input(input_payload)

        # Make prediction
        pred_array = self.model.predict(X_scaled)
        predicted_temp = float(np.round(pred_array[0], 2))

        rmse = self.metadata.get("RMSE", 1.5)
        conf_min = float(np.round(predicted_temp - 1.96 * rmse, 2))
        conf_max = float(np.round(predicted_temp + 1.96 * rmse, 2))

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "predicted_temperature": predicted_temp,
            "confidence_lower": conf_min,
            "confidence_upper": conf_max,
            "model_used": self.metadata.get("best_model_name", "Best ML Model"),
            "prediction_time_ms": elapsed_ms,
            "r2_score": self.metadata.get("R2_Score", 0.0),
            "rmse": rmse,
        }


if __name__ == "__main__":
    predictor = TemperaturePredictor()
    result = predictor.predict(
        humidity=65.0,
        pressure=1013.25,
        wind_speed=12.5,
        rainfall=0.0,
        month=7,
        day=15,
        weather_condition="Sunny"
    )
    print("Sample Inference Result:")
    print(json.dumps(result, indent=4))
