"""
Unit tests for Machine Learning Trainer and Predictor modules.
"""

from pathlib import Path
import pytest
from src.predict import TemperaturePredictor
from src.train import ModelTrainer, run_full_training_pipeline
from src.utils import MODELS_DIR


def test_model_training_pipeline():
    res = run_full_training_pipeline()
    assert "comparison_df" in res
    assert res["comparison_df"].shape[0] >= 4
    assert res["best_model_path"].exists()


def test_temperature_predictor():
    predictor = TemperaturePredictor()
    res = predictor.predict(
        humidity=60.0,
        pressure=1013.0,
        wind_speed=15.0,
        rainfall=0.0,
        month=6,
        day=10,
        weather_condition="Clear"
    )

    assert "predicted_temperature" in res
    assert isinstance(res["predicted_temperature"], float)
    assert res["confidence_lower"] <= res["predicted_temperature"] <= res["confidence_upper"]
    assert res["prediction_time_ms"] > 0
