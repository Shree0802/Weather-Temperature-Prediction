"""
Streamlit Web Application for Weather Data Analysis and Temperature Prediction.
Features multi-page sidebar navigation, custom glassmorphism CSS, interactive Plotly visualizations,
and real-time Machine Learning temperature inference.
"""

from datetime import datetime, date
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Add project root directory to python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.predict import TemperaturePredictor
from src.preprocessing import WeatherDataPreprocessor
from src.utils import IMAGES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.visualization import (
    create_actual_vs_pred_plot,
    create_feature_importance_plot,
    create_plotly_correlation_heatmap,
    create_plotly_line_chart,
    create_plotly_scatter,
    create_residual_plot,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Weather ML - Temperature Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS Styling (Glassmorphism + Dark Mode Aesthetics)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background & glassmorphism card container */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        color: #F8FAFC;
    }

    /* Custom Glassmorphism Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.5);
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Prediction Result Highlight Card */
    .prediction-box {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 2px solid #38BDF8;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 40px -10px rgba(56, 189, 248, 0.3);
    }
    .pred-temp {
        font-size: 3.5rem;
        font-weight: 800;
        color: #38BDF8;
        margin: 10px 0;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #CBD5E1;
        line-height: 1.6;
    }

    /* Badge Pills */
    .tech-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Predictor Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_processed_dataset():
    preprocessor = WeatherDataPreprocessor()
    df, _, _ = preprocessor.fit_transform_pipeline()
    return df


@st.cache_resource
def load_ml_predictor():
    return TemperaturePredictor()


df = load_processed_dataset()
predictor = load_ml_predictor()


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/isometric/100/weather.png",
    width=80
)
st.sidebar.title("Weather ML Studio")
st.sidebar.caption("Temperature Analytics & AI Forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    options=[
        "🏠 Home",
        "📊 Dashboard",
        "🔍 EDA",
        "🔮 Prediction",
        "📈 Model Performance",
        "ℹ️ About",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Info**  
    - **Dataset**: Daily Weather (5 Years)  
    - **Best Model**: {}  
    - **Status**: Production Ready ✅
    """.format(predictor.metadata.get("best_model_name", "Random Forest"))
)


# =============================================================================
# 1. HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Weather Data Analysis & Temperature Prediction</div>
            <div class="hero-sub">
                An end-to-end machine learning system built with Python and Streamlit.
                Leveraging historical meteorological signals to predict future atmospheric temperatures
                with high accuracy and automated model evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown(
            """
            - **Data Pipeline**: Automated cleaning, missing value imputation, outlier handling, and date decomposition.
            - **Exploratory Analytics**: Comprehensive statistical analysis of temperature, humidity, pressure, wind speed, and rainfall.
            - **Machine Learning Suite**: Comparative benchmark across 5 regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
            - **Real-Time Inference**: Interactive temperature forecasting with confidence interval estimation.
            - **Production Standards**: Modular architecture, full unit test coverage, PEP8 compliance, and deployment readiness.
            """
        )

        st.subheader("🛠️ Technology Stack")
        tech_list = [
            "Python 3.13",
            "Streamlit",
            "Scikit-Learn",
            "XGBoost",
            "Pandas",
            "NumPy",
            "Plotly",
            "Seaborn",
            "Matplotlib",
            "Joblib",
        ]
        badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in tech_list])
        st.markdown(badges_html, unsafe_allow_html=True)

    with col2:
        st.subheader("🚀 Quick Actions")
        st.info("Explore dataset statistics, interactive charts, and temperature predictions.")
        
        st.markdown("#### Jump To Page:")
        st.caption("Use the sidebar or click below to explore:")
        st.markdown("- 📊 **Dashboard**: View high-level KPIs and summary charts")
        st.markdown("- 🔍 **EDA**: Analyze feature distributions & correlations")
        st.markdown("- 🔮 **Prediction**: Input custom weather parameters for real-time ML forecast")
        st.markdown("- 📈 **Model Performance**: Compare metrics across ML models")


# =============================================================================
# 2. DASHBOARD PAGE
# =============================================================================
elif page == "📊 Dashboard":
    st.title("📊 Meteorological Key Performance Indicators")
    st.caption("Overview of overall historical weather metrics and core trends.")

    # Metric Cards Row
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)

    total_records = len(df)
    avg_temp = round(df["Temperature"].mean(), 1)
    max_temp = round(df["Temperature"].max(), 1)
    min_temp = round(df["Temperature"].min(), 1)
    avg_hum = round(df["Humidity"].mean(), 1)
    avg_pres = round(df["Pressure"].mean(), 1)

    mcol1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-subtitle">Daily Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcol2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Temp</div>
            <div class="metric-value">{avg_temp}°C</div>
            <div class="metric-subtitle">Mean Daily</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcol3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Max Temp</div>
            <div class="metric-value">{max_temp}°C</div>
            <div class="metric-subtitle">Peak High</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcol4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Min Temp</div>
            <div class="metric-value">{min_temp}°C</div>
            <div class="metric-subtitle">Recorded Low</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcol5.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Humidity</div>
            <div class="metric-value">{avg_hum}%</div>
            <div class="metric-subtitle">Relative Humidity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcol6.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Pressure</div>
            <div class="metric-value">{avg_pres}</div>
            <div class="metric-subtitle">hPa Atmospheric</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Dashboard Charts
    dcol1, dcol2 = st.columns([2, 1])

    with dcol1:
        st.subheader("🌡️ Historical Temperature Trend")
        fig_temp = create_plotly_line_chart(
            df, x_col="Date", y_col="Temperature", title="Daily Temperature Trend Over Time", color_hex="#38BDF8"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with dcol2:
        st.subheader("🌤️ Weather Conditions Breakdown")
        if "Weather Condition" in df.columns:
            cond_counts = df["Weather Condition"].value_counts().reset_index()
            cond_counts.columns = ["Condition", "Count"]
            fig_pie = px.pie(
                cond_counts,
                names="Condition",
                values="Count",
                title="Condition Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# 3. EDA PAGE
# =============================================================================
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis & Dynamic Filter Engine")
    st.caption("Customize filters, generate dynamic visualizations, and download assets.")

    # Sidebar Filter Controls inside main view container
    with st.expander("🎛️ Interactive Filters & Data Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            if "Date" in df.columns:
                min_date = pd.to_datetime(df["Date"]).min().date()
                max_date = pd.to_datetime(df["Date"]).max().date()
                date_range = st.date_input("Select Date Range", value=(min_date, max_date))
            else:
                date_range = None

        with fcol2:
            months_selected = st.multiselect(
                "Filter by Month(s)",
                options=list(range(1, 13)),
                default=list(range(1, 13)),
                format_func=lambda m: datetime(2026, m, 1).strftime("%B"),
            )

        with fcol3:
            cond_options = list(df["Weather Condition"].unique()) if "Weather Condition" in df.columns else []
            conds_selected = st.multiselect(
                "Filter by Weather Condition", options=cond_options, default=cond_options
            )

    # Filter Application
    filtered_df = df.copy()
    if date_range and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df["Date_dt"] = pd.to_datetime(filtered_df["Date"])
        filtered_df = filtered_df[
            (filtered_df["Date_dt"].dt.date >= start_d) & (filtered_df["Date_dt"].dt.date <= end_d)
        ]

    if months_selected and "Month" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Month"].isin(months_selected)]

    if conds_selected and "Weather Condition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Weather Condition"].isin(conds_selected)]

    st.markdown(f"**Showing {len(filtered_df):,} out of {len(df):,} total records.**")

    # Dynamic Column Visualizer
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Dynamic Column Plotter", "🔥 Correlation Heatmap", "📦 Feature Distributions", "📄 Summary Statistics"]
    )

    with tab1:
        st.subheader("Dynamic Variable Explorer")
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            x_var = st.selectbox("Select X-Axis Feature", options=["Date", "Humidity", "Pressure", "Wind Speed", "Rainfall", "Month"])
        with vcol2:
            y_var = st.selectbox("Select Y-Axis Feature", options=["Temperature", "Humidity", "Pressure", "Wind Speed", "Rainfall"], index=0)
        with vcol3:
            chart_type = st.selectbox("Chart Type", options=["Line Chart", "Scatter Plot", "Bar Chart", "Boxplot"])

        if chart_type == "Line Chart":
            fig = create_plotly_line_chart(filtered_df, x_var, y_var, f"{y_var} vs {x_var}")
        elif chart_type == "Scatter Plot":
            fig = create_plotly_scatter(filtered_df, x_var, y_var, color_col="Weather Condition")
        elif chart_type == "Bar Chart":
            fig = px.bar(filtered_df.head(100), x=x_var, y=y_var, color="Weather Condition", template="plotly_dark")
        elif chart_type == "Boxplot":
            fig = px.box(filtered_df, x=x_var, y=y_var, template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Analysis")
        fig_corr = create_plotly_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.subheader("Numerical Distributions & Outlier Analysis")
        dist_var = st.selectbox("Select Feature for Distribution Plot", options=["Temperature", "Humidity", "Pressure", "Wind Speed", "Rainfall"])
        fig_dist = px.histogram(
            filtered_df,
            x=dist_var,
            color="Weather Condition" if "Weather Condition" in filtered_df.columns else None,
            marginal="box",
            template="plotly_dark",
            title=f"Distribution & Boxplot for {dist_var}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        st.subheader("Filtered Summary Statistics")
        st.dataframe(filtered_df.describe().T.style.highlight_max(axis=0, color="#1E3A8A"), use_container_width=True)

    # Download Dataset Section
    st.markdown("---")
    st.subheader("📥 Download Filtered Data & Reports")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered CSV Dataset",
        data=csv_bytes,
        file_name="filtered_weather_data.csv",
        mime="text/csv",
    )


# =============================================================================
# 4. PREDICTION PAGE
# =============================================================================
elif page == "🔮 Prediction":
    st.title("🔮 Real-Time Temperature Prediction")
    st.caption("Input atmospheric parameters to generate instant ML temperature predictions.")

    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        st.subheader("⚙️ Input Weather Parameters")

        with st.form("prediction_form"):
            in_hum = st.slider("Relative Humidity (%)", min_value=20.0, max_value=100.0, value=65.0, step=0.5)
            in_pres = st.slider("Atmospheric Pressure (hPa)", min_value=980.0, max_value=1040.0, value=1013.25, step=0.25)
            in_wind = st.slider("Wind Speed (km/h)", min_value=1.0, max_value=60.0, value=14.0, step=0.5)
            in_rain = st.number_input("Daily Rainfall Volume (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

            icol1, icol2, icol3 = st.columns(3)
            with icol1:
                in_month = st.selectbox("Month", options=list(range(1, 13)), index=6, format_func=lambda m: datetime(2026, m, 1).strftime("%B"))
            with icol2:
                in_day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
            with icol3:
                cond_list = ["Clear", "Sunny", "Cloudy", "Rainy", "Thunderstorm", "Foggy"]
                in_cond = st.selectbox("Weather Condition", options=cond_list, index=1)

            submit_btn = st.form_submit_button("🌡️ Predict Temperature Now", use_container_width=True)

    with pcol2:
        st.subheader("🎯 Prediction Output")

        if submit_btn:
            res = predictor.predict(
                humidity=in_hum,
                pressure=in_pres,
                wind_speed=in_wind,
                rainfall=in_rain,
                month=in_month,
                day=in_day,
                weather_condition=in_cond,
            )

            pred_temp_c = res["predicted_temperature"]
            pred_temp_f = round(pred_temp_c * 9/5 + 32, 2)
            c_low = res["confidence_lower"]
            c_high = res["confidence_upper"]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div style="font-size: 1rem; color: #94A3B8; font-weight: 600;">ESTIMATED DAILY TEMPERATURE</div>
                    <div class="pred-temp">{pred_temp_c}°C</div>
                    <div style="font-size: 1.2rem; color: #CBD5E1; font-weight: 600;">({pred_temp_f}°F)</div>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
                    <div style="font-size: 0.9rem; color: #94A3B8;">
                        <b>95% Confidence Bounds:</b> {c_low}°C to {c_high}°C<br>
                        <b>Model Selected:</b> {res['model_used']}<br>
                        <b>Model R² Metric:</b> {round(res['r2_score'], 4)}<br>
                        <b>Inference Latency:</b> {res['prediction_time_ms']} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Adjust the sliders on the left and click **Predict Temperature Now** to view AI model output.")


# =============================================================================
# 5. MODEL PERFORMANCE PAGE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation & Comparative Performance")
    st.caption("Benchmark comparison across 5 machine learning regression architectures.")

    meta = predictor.metadata

    # Best Model Highlight Banner
    st.markdown(
        f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 16px; padding: 20px; margin-bottom: 25px;">
            <h3 style="margin:0; color: #10B981;">🏆 Best Performing Model: {meta.get('best_model_name', 'Random Forest Regressor')}</h3>
            <p style="margin: 5px 0 0 0; color: #E2E8F0;">
                Test R² Score: <b>{round(meta.get('R2_Score', 0), 4)}</b> | 
                RMSE: <b>{round(meta.get('RMSE', 0), 4)} °C</b> | 
                MAE: <b>{round(meta.get('MAE', 0), 4)} °C</b> | 
                5-Fold CV Score: <b>{round(meta.get('CV_Score', 0), 4)}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load pre-calculated training comparison results if available
    from src.train import ModelTrainer, WeatherDataPreprocessor
    preprocessor = WeatherDataPreprocessor()
    df_p, X_p, y_p = preprocessor.fit_transform_pipeline(save_processed=False)
    trainer = ModelTrainer()
    comp_df = trainer.train_and_evaluate(X_p, y_p)

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs(
        ["📊 Model Comparison Table", "⭐ Feature Importance", "📉 Residual Analysis", "🎯 Actual vs Predicted"]
    )

    with tab_m1:
        st.subheader("Model Metric Benchmark Summary")
        st.dataframe(comp_df.style.highlight_max(subset=["R² Score", "CV Score (R²)"], color="#065F46"), use_container_width=True)

        fig_comp = px.bar(
            comp_df,
            x="Model",
            y="R² Score",
            color="R² Score",
            title="R² Score Comparison Across Algorithms",
            template="plotly_dark",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab_m2:
        st.subheader("Best Model Feature Importance Breakdown")
        f_names = meta.get("feature_names", list(X_p.columns))
        f_imps = meta.get("feature_importances", [1/len(f_names)]*len(f_names))
        fig_fi = create_feature_importance_plot(f_names, f_imps)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab_m3:
        st.subheader("Residual Diagnostics (Error Distribution)")
        best_name = meta.get("best_model_name", "Random Forest Regressor")
        best_res = trainer.results[best_name]
        y_t = best_res["y_test"]
        y_p_preds = best_res["predictions"]
        fig_res = create_residual_plot(y_t, y_p_preds)
        st.plotly_chart(fig_res, use_container_width=True)

    with tab_m4:
        st.subheader("Actual vs Predicted Temperature Alignment")
        fig_avp = create_actual_vs_pred_plot(y_t, y_p_preds)
        st.plotly_chart(fig_avp, use_container_width=True)


# =============================================================================
# 6. ABOUT PAGE
# =============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About the Weather ML System")

    st.markdown(
        """
        ### 📌 Project Overview
        This project is a complete end-to-end Machine Learning solution designed to analyze multi-year historical weather dataset records and accurately predict daily temperature trends.
        
        ### 🔄 Machine Learning Pipeline Workflow
        1. **Data Ingestion & Synthesis**: Multi-year daily weather records containing temperature, humidity, pressure, wind speed, rainfall, and weather condition.
        2. **Preprocessing**: Missing value imputation, duplicate removal, IQR outlier handling, date breakdown into year/month/day/week/dayofweek, and StandardScaling.
        3. **Exploratory Data Analysis**: 10+ professional figures including seasonal trendlines, heatmaps, boxplots, and distributions automatically stored in `assets/images/`.
        4. **Model Benchmark**: Automated cross-validated training across Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
        5. **Deployment & Serving**: Interactive Streamlit Web App exposing instant ML predictions, metric dashboards, and downloadable assets.

        ### 🚀 Future Enhancements
        - Deep Learning Integration (LSTM / GRU networks for sequential time-series forecasting).
        - OpenWeatherMap Live API Integration for real-time weather ingestion.
        - Automated model drift detection and retraining pipeline.

        ---
        **Author**: Senior Machine Learning Engineer  
        **License**: MIT License  
        **Repository**: [GitHub Project Repository](#)
        """
    )
