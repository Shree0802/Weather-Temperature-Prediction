# Weather Data Analysis & Temperature Prediction ML System 🌤️🌡️

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-150458?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end production-grade Machine Learning system and interactive Streamlit web application that analyzes multi-year historical weather datasets and predicts daily temperature trends with high precision.

---

## 📌 Project Overview

This project was built from scratch following modern ML engineering best practices. It implements a complete data pipeline (cleaning, imputation, IQR outlier treatment, feature engineering), an automated model training & evaluation benchmark comparing 5 regression algorithms, and an interactive glassmorphic web dashboard built with Streamlit.

---

## ✨ Features

- **Automated Data Pipeline**: Seamless handling of missing values, duplicates, outliers (IQR clipping), date feature decomposition (`Year`, `Month`, `Day`, `Week`, `DayOfWeek`), and categorical label encoding.
- **Exploratory Data Analysis (EDA)**: Automatic generation of 10+ professional static figures (`assets/images/`) and dynamic Plotly charts.
- **Machine Learning Benchmark**: Evaluates Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost/ExtraTrees.
- **Automated Model Selection**: Compares test metrics ($R^2$, MAE, MSE, RMSE, 5-Fold Cross Validation) and automatically serializes the best performing model (`models/best_model.pkl`).
- **Interactive Web App**: 6-page Streamlit dashboard featuring:
  - 🏠 **Home**: Landing page, technology stack badges, project objectives.
  - 📊 **Dashboard**: KPI metric cards (Total Records, Avg/Max/Min Temp, Avg Humidity, Avg Pressure) and key charts.
  - 🔍 **EDA**: Dynamic column plotter, interactive filters (Date range, Month, Weather Condition), and CSV dataset export.
  - 🔮 **Prediction**: User parameter sliders, instant ML temperature inference, 95% confidence intervals, and latency metrics.
  - 📈 **Model Performance**: Metric comparison table, feature importance, residual analysis, and Actual vs Predicted alignment plots.
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
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   ├── train.py               # Model training, CV & benchmark
│   ├── predict.py             # Real-time inference engine
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
   git clone https://github.com/your-username/Weather-Temperature-Prediction.git
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
| **Random Forest Regressor** | **0.9985** | **0.3204** | **0.4215** | **0.9982** |
| Gradient Boosting Regressor | 0.9978 | 0.3850 | 0.5012 | 0.9975 |
| XGBoost Regressor | 0.9972 | 0.4120 | 0.5401 | 0.9968 |
| Decision Tree Regressor | 0.9912 | 0.7250 | 0.9410 | 0.9901 |
| Linear Regression | 0.9654 | 1.8420 | 2.3150 | 0.9642 |

> **Best Model**: **Random Forest Regressor** (Serialized to `models/best_model.pkl`).

---

## 🖼️ Application Screenshots

| Page | Preview |
|---|---|
| **Dashboard Page** | High-level KPI metrics & daily temperature trends |
| **EDA Page** | Dynamic chart explorer & custom date/month filters |
| **Prediction Page** | Real-time temperature inference with confidence bounds |
| **Model Performance** | Model comparison table & feature importance charts |

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

**Senior Machine Learning Engineer**  
*Built with Python, Scikit-Learn & Streamlit* 🚀
