"""
Data Preprocessing and Feature Engineering module for Weather Temperature Prediction.
Handles missing values, duplicate removal, outlier filtering, date decomposition,
categorical encoding, and feature scaling.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories


class WeatherDataPreprocessor:
    """
    Production-ready data preprocessor for historical weather datasets.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.numerical_cols = ["Humidity", "Pressure", "Wind Speed", "Rainfall"]
        self.categorical_cols = ["Weather Condition"]

    def load_raw_data(self, file_path: Path = None) -> pd.DataFrame:
        """Load raw weather CSV dataset."""
        if file_path is None:
            file_path = RAW_DATA_DIR / "weather_data_raw.csv"
        if not file_path.exists():
            from src.utils import generate_synthetic_weather_data
            generate_synthetic_weather_data(file_path)
        return pd.read_csv(file_path)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing numerical values with median and categorical with mode."""
        df = df.copy()
        for col in self.numerical_cols + ["Temperature"]:
            if col in df.columns and df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records from dataframe."""
        return df.drop_duplicates().reset_index(drop=True)

    def remove_outliers_iqr(self, df: pd.DataFrame, factor: float = 2.5) -> pd.DataFrame:
        """
        Detect and clip extreme outliers using Interquartile Range (IQR) method
        to prevent losing valid continuous weather records while muting extreme errors.
        """
        df = df.copy()
        target_cols = [c for c in self.numerical_cols + ["Temperature"] if c in df.columns]
        for col in target_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose Date column into Year, Month, Day, Week, DayOfWeek, and IsWeekend."""
        df = df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            # dt.isocalendar().week returns uint32 in pandas
            df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
            df["DayOfWeek"] = df["Date"].dt.dayofweek
            df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical weather condition into numeric label representation."""
        df = df.copy()
        if "Weather Condition" in df.columns:
            if fit:
                df["Weather_Condition_Code"] = self.label_encoder.fit_transform(df["Weather Condition"].astype(str))
            else:
                # Handle unseen labels gracefully during inference
                df["Weather_Condition_Code"] = df["Weather Condition"].astype(str).map(
                    lambda x: self.label_encoder.transform([x])[0]
                    if x in self.label_encoder.classes_
                    else 0
                )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific derivative features like DewPoint proxy and HeatIndex proxy."""
        df = df.copy()
        # Dew point approximation formula
        if "Temperature" in df.columns and "Humidity" in df.columns:
            df["Dew_Point"] = df["Temperature"] - ((100 - df["Humidity"]) / 5)
        elif "Humidity" in df.columns:
            # Approx using mean temp if target temperature is absent during inference input
            df["Dew_Point"] = 20.0 - ((100 - df["Humidity"]) / 5)
            
        # Temperature sinusoidal seasonality feature
        if "Month" in df.columns:
            df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12)
            df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12)
        return df

    def fit_transform_pipeline(
        self, file_path: Path = None, save_processed: bool = True
    ) -> (pd.DataFrame, pd.DataFrame, pd.Series):
        """
        Execute full end-to-end preprocessing pipeline.
        
        Returns:
            processed_df (pd.DataFrame): Complete cleaned dataframe.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target temperature values.
        """
        ensure_directories()
        raw_df = self.load_raw_data(file_path)
        df = self.handle_missing_values(raw_df)
        df = self.remove_duplicates(df)
        df = self.remove_outliers_iqr(df)
        df = self.extract_date_features(df)
        df = self.encode_categorical(df, fit=True)
        df = self.engineer_features(df)

        if save_processed:
            processed_path = PROCESSED_DATA_DIR / "weather_data_processed.csv"
            df.to_csv(processed_path, index=False)

        # Feature column selection for modeling
        feature_cols = [
            "Humidity",
            "Pressure",
            "Wind Speed",
            "Rainfall",
            "Year",
            "Month",
            "Day",
            "Week",
            "DayOfWeek",
            "IsWeekend",
            "Weather_Condition_Code",
            "Dew_Point",
            "Sin_Month",
            "Cos_Month",
        ]
        
        X = df[feature_cols].copy()
        y = df["Temperature"].copy()

        # Fit and transform scaler on X
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=feature_cols, index=X.index
        )

        self.feature_columns = feature_cols

        # Save fitted scaler and label encoder objects
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(self.label_encoder, MODELS_DIR / "label_encoder.pkl")
        joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")

        return df, X_scaled, y

    def transform_single_input(self, input_dict: dict) -> pd.DataFrame:
        """
        Transform a single raw input payload into scaled feature vector for real-time inference.
        """
        # Load persisted artifacts if not loaded
        scaler_path = MODELS_DIR / "scaler.pkl"
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        feature_cols_path = MODELS_DIR / "feature_columns.pkl"

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            feature_cols = joblib.load(feature_cols_path)
        else:
            scaler = self.scaler
            encoder = self.label_encoder
            feature_cols = self.feature_columns

        df = pd.DataFrame([input_dict])

        # Date handling if Date string passed
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
            df["DayOfWeek"] = df["Date"].dt.dayofweek
            df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        else:
            # Set default date fields if year/month/day provided directly
            df["Year"] = df.get("Year", 2026)
            df["Month"] = df.get("Month", 6)
            df["Day"] = df.get("Day", 15)
            df["Week"] = df.get("Week", 24)
            df["DayOfWeek"] = df.get("DayOfWeek", 2)
            df["IsWeekend"] = df.get("IsWeekend", 0)

        # Categorical encoding
        cond = df["Weather Condition"].iloc[0] if "Weather Condition" in df.columns else "Clear"
        if hasattr(encoder, "classes_") and cond in encoder.classes_:
            df["Weather_Condition_Code"] = encoder.transform([cond])[0]
        else:
            df["Weather_Condition_Code"] = 0

        # Feature Engineering
        if "Temperature" in df.columns:
            temp_ref = df["Temperature"].iloc[0]
        else:
            temp_ref = 22.0  # reference mean temp for dew point calculation
        df["Dew_Point"] = temp_ref - ((100 - df["Humidity"]) / 5)
        df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12)

        # Reorder to exact feature columns
        X_single = df[feature_cols]
        X_scaled = pd.DataFrame(scaler.transform(X_single), columns=feature_cols)
        return X_scaled


if __name__ == "__main__":
    preprocessor = WeatherDataPreprocessor()
    df, X, y = preprocessor.fit_transform_pipeline()
    print(f"Preprocessing finished successfully! Processed {len(df)} records.")
    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
