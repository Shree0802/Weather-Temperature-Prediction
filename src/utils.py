"""
Utility functions for Weather Temperature Prediction system.
Includes directory setup, dataset generation, and path management.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
TESTS_DIR = BASE_DIR / "tests"


def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        IMAGES_DIR,
        SCREENSHOTS_DIR,
        TESTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def generate_synthetic_weather_data(output_path: Path = None, n_years: int = 5) -> pd.DataFrame:
    """
    Generate a realistic, multi-year historical weather dataset with authentic seasonal
    correlations, trends, and weather conditions.
    
    Parameters:
        output_path (Path): Destination CSV path.
        n_years (int): Number of years of historical daily data to synthesize.
        
    Returns:
        pd.DataFrame: Raw synthetic weather dataset.
    """
    ensure_directories()
    if output_path is None:
        output_path = RAW_DATA_DIR / "weather_data_raw.csv"

    np.random.seed(42)
    start_date = pd.Timestamp("2020-01-01")
    days = n_years * 365 + (n_years // 4)
    date_range = pd.date_range(start=start_date, periods=days, freq="D")

    # Day of year sine/cosine wave for seasonal temperature dynamics
    day_of_year = date_range.dayofyear.values
    
    # Base temperature sinusoidal cycle (min in Jan, max in July)
    # Average around 22°C with variation amplitude of 14°C
    base_temp = 22 + 14 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    
    # Add random daily noise to temperature
    temp_noise = np.random.normal(0, 3.2, size=days)
    temperature = np.round(base_temp + temp_noise, 2)

    # Humidity: Inverse relationship with temp + random noise (range 20% to 98%)
    humidity = np.clip(
        np.round(75 - 0.7 * (temperature - 20) + np.random.normal(0, 10, size=days), 2),
        20.0,
        98.0
    )

    # Pressure: Fluctuation around 1013.25 hPa with slight inverse temp correlation
    pressure = np.clip(
        np.round(1014.0 - 0.15 * (temperature - 20) + np.random.normal(0, 6.0, size=days), 2),
        985.0,
        1035.0
    )

    # Wind Speed: 2 to 45 km/h with log-normal distribution
    wind_speed = np.round(np.random.lognormal(mean=2.2, sigma=0.45, size=days), 2)
    wind_speed = np.clip(wind_speed, 2.0, 50.0)

    # Rainfall: Higher when humidity is high (>70%) and pressure is dropping (<1012 hPa)
    rain_prob = np.where((humidity > 70) & (pressure < 1013), 0.65, 0.12)
    has_rain = np.random.rand(days) < rain_prob
    rainfall = np.where(
        has_rain,
        np.round(np.random.exponential(scale=12.0, size=days) + 0.5, 2),
        0.0
    )

    # Determine Weather Condition categorical column based on physics rules
    conditions = []
    for h, p, r, t in zip(humidity, pressure, rainfall, temperature):
        if r > 25.0:
            conditions.append("Thunderstorm")
        elif r > 0.0:
            conditions.append("Rainy")
        elif h > 88.0 and t < 15.0:
            conditions.append("Foggy")
        elif h > 75.0 or p < 1008.0:
            conditions.append("Cloudy")
        elif t > 30.0:
            conditions.append("Sunny")
        else:
            conditions.append("Clear")

    df = pd.DataFrame({
        "Date": date_range.strftime("%Y-%m-%d"),
        "Temperature": temperature,
        "Humidity": humidity,
        "Pressure": pressure,
        "Wind Speed": wind_speed,
        "Rainfall": rainfall,
        "Weather Condition": conditions
    })

    # Introduce a small controlled percentage of missing values and duplicates for realistic cleaning tests
    # (0.5% missing in Humidity, 0.5% in Pressure)
    missing_idx_h = np.random.choice(days, size=int(days * 0.005), replace=False)
    missing_idx_p = np.random.choice(days, size=int(days * 0.005), replace=False)
    df.loc[missing_idx_h, "Humidity"] = np.nan
    df.loc[missing_idx_p, "Pressure"] = np.nan

    # Add a few exact duplicate rows
    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv(output_path, index=False)
    return df
