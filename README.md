# Weather Data Analysis & Temperature Prediction ML System 🌤️🌡️

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://weather-temperature-prediction-3jgs3a5gmtyunyqdyjdwxw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-150458?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end production-grade Machine Learning system and interactive Streamlit web application that analyzes multi-year historical weather datasets and predicts daily temperature trends with high precision.

---

## 🔗 Live Application Link

🚀 **Deployed Live Web App**: [https://weather-temperature-prediction-3jgs3a5gmtyunyqdyjdwxw.streamlit.app/](https://weather-temperature-prediction-3jgs3a5gmtyunyqdyjdwxw.streamlit.app/)

---

## 📌 Project Overview

This project was built from scratch following modern ML engineering best practices. It implements a complete data pipeline (cleaning, imputation, IQR outlier treatment, feature engineering), an automated model training & evaluation benchmark comparing 5 regression algorithms, and an interactive glassmorphic web dashboard built with Streamlit.

---

## ✨ Features

- **Automated Data Pipeline**: Seamless handling of missing values, duplicates, outliers (IQR clipping), date feature decomposition (`Year`, `Month`, `Day`, `Week`, `DayOfWeek`), and categorical label encoding.
- **Time Series Feature Engineering**: Autoregressive temperature lags (`Temp_Lag1`, `Temp_Lag7`), 7-day/30-day rolling moving averages, and seasonal sine/cosine terms (`Sin_DayOfYear`).
- **Multi-Step Future Forecasting**: Autoregressively project temperature trends 7 to 30 days into the future with 95% confidence interval bounds.
- **Exploratory Data Analysis (EDA)**: Automatic generation of 10+ professional static figures (`assets/images/`) and dynamic Plotly charts.
- **Machine Learning Benchmark**: Evaluates Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost/ExtraTrees.
- **Automated Model Selection**: Compares test metrics ($R^2$, MAE, MSE, RMSE, 5-Fold Cross Validation) and automatically serializes the best performing model (`models/best_model.pkl`).
- **Interactive Web App**: 6-page Streamlit dashboard featuring:
  - 🏠 **Home**: Landing page, technology stack badges, project objectives.
  - 📊 **Dashboard**: KPI metric cards (Total Records, Avg/Max/Min Temp, Avg Humidity, Avg Pressure) and key charts.
  - 🔍 **EDA & Time Series**: Moving average smoothing, dynamic column plotter, interactive filters, and CSV export.
  - 🔮 **Prediction & Forecasting**: Single-day prediction & 14-day future trend forecast.
  - 📈 **Model Performance**: Metric comparison leaderboard, feature importance, residual analysis, and Actual vs Predicted alignment plots.
  - ℹ️ **About**: Architectural workflow diagram, pipeline explanation, future scope, and author info.

---

## 📊 Dataset Structure

The dataset contains historical daily weather records with the following features:

| Feature Column | Type | Description |
|---|---|---|
| `Date` | Datetime | Date of observation (YYYY-MM-DD) |
| `Temperature` | Float | **Target variable** - Daily mean temperature (°C) |
| `Humidity` | Float | Relative humidity percentage (%) |
| `Pressure` | Float | Atmospheric sea-level pressure (hPa) |
| `Wind Speed` | Float | Mean daily wind velocity (km/h) |
| `Rainfall` | Float | Daily precipitation volume (mm) |
| `Weather Condition` | Categorical | Condition (Sunny, Clear, Cloudy, Rainy, Thunderstorm, Foggy) |

---

## 📁 Folder Structure

```
Weather-Temperature-Prediction/
│
├── app.py                     # Main Streamlit Web Application
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── DEPLOYMENT.md              # Deployment guide
├── LICENSE                    # MIT License
├── .gitignore                 # Files to exclude from version control
├── packages.txt               # System packages for deployment
├── runtime.txt                # Python runtime configuration
│
├── data/
│   ├── raw/                   # Unprocessed raw weather datasets
│   └── processed/             # Cleaned & scaled dataset output
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb   # Interactive EDA Notebook
│
├── models/
│   ├── best_model.pkl         # Serialized best regression model
│   ├── scaler.pkl             # Fitted StandardScaler object
│   ├── label_encoder.pkl      # Fitted LabelEncoder object
│   └── metadata.json          # Evaluation metrics & feature names
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning & time-series lag feature engineering
│   ├── train.py               # Model training, CV & benchmark
│   ├── predict.py             # Multi-step autoregressive future forecaster
│   ├── utils.py               # Helper paths & synthetic generator
│   └── visualization.py       # Matplotlib & Plotly chart generators
│
├── assets/
│   └── images/                # Saved static EDA plots
│
├── screenshots/               # Application UI screenshots
│
└── tests/                     # Pytest unit testing suite
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_app.py
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shree0802/Weather-Temperature-Prediction.git
   cd Weather-Temperature-Prediction
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Run Data Preprocessing & Model Training:
```bash
python src/train.py
```

### 2. Run Pytest Test Suite:
```bash
pytest tests/
```

### 3. Launch Streamlit Web Dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Model Performance & Benchmark

All models were evaluated using 5-Fold Cross Validation and Test Set split (80/20):

| Model | $R^2$ Score | MAE (°C) | RMSE (°C) | CV Score ($R^2$) |
|---|---|---|---|---|
| **Linear Regression** | **1.0000** | **0.0000** | **0.0000** | **1.0000** |
| Gradient Boosting Regressor | 0.9991 | 0.2177 | 0.3183 | 0.9987 |
| XGBoost Regressor | 0.9989 | 0.2289 | 0.3438 | 0.9984 |
| Random Forest Regressor | 0.9984 | 0.2648 | 0.4164 | 0.9978 |
| Decision Tree Regressor | 0.9942 | 0.5860 | 0.7906 | 0.9936 |

> **Best Model**: **Linear Regression** (Serialized to `models/best_model.pkl`).

---

## 🔮 Future Improvements

- [ ] Incorporate time-series Deep Learning models (LSTM, GRU, Temporal Fusion Transformer).
- [ ] Connect live weather API feeds (OpenWeatherMap / NOAA) for real-time inference.
- [ ] Implement automated ML model retraining & drift monitoring.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Shree0802**  
*Built with Python, Scikit-Learn & Streamlit* 🚀
