"""
Unit tests for Data Preprocessor module.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import WeatherDataPreprocessor
from src.utils import generate_synthetic_weather_data


@pytest.fixture
def sample_df(tmp_path):
    csv_path = tmp_path / "test_raw.csv"
    df = generate_synthetic_weather_data(output_path=csv_path, n_years=1)
    return df


def test_missing_values(sample_df):
    preprocessor = WeatherDataPreprocessor()
    cleaned = preprocessor.handle_missing_values(sample_df)
    assert cleaned["Humidity"].isnull().sum() == 0
    assert cleaned["Pressure"].isnull().sum() == 0


def test_duplicate_removal(sample_df):
    preprocessor = WeatherDataPreprocessor()
    dup_df = pd.concat([sample_df, sample_df.iloc[:3]], ignore_index=True)
    dedup = preprocessor.remove_duplicates(dup_df)
    assert len(dedup) == len(sample_df.drop_duplicates())


def test_date_feature_extraction(sample_df):
    preprocessor = WeatherDataPreprocessor()
    df_date = preprocessor.extract_date_features(sample_df)
    for col in ["Year", "Month", "Day", "Week", "DayOfWeek", "IsWeekend"]:
        assert col in df_date.columns


def test_fit_transform_pipeline(tmp_path):
    csv_path = tmp_path / "raw.csv"
    generate_synthetic_weather_data(csv_path, n_years=1)
    preprocessor = WeatherDataPreprocessor()
    df, X, y = preprocessor.fit_transform_pipeline(file_path=csv_path, save_processed=False)

    assert not X.isnull().values.any()
    assert len(X) == len(y)
    assert "Humidity" in X.columns
