"""
Streamlit Web Application for Weather Data Analysis, Time-Series Forecasting, and Temperature Prediction.
Features multi-page sidebar navigation, custom glassmorphism CSS, interactive Plotly visualizations,
and real-time Machine Learning temperature inference.
"""

from datetime import datetime, date
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="Weather ML - Temperature Intelligence & Forecasting Studio",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS Styling (Glassmorphism + Dark Mode Aesthetics)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1E1B4B 0%, #0F172A 40%, #090D16 100%);
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .brand-box {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .brand-icon {
        font-size: 2.2rem;
        filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.6));
    }
    .brand-title {
        font-weight: 800;
        font-size: 1.25rem;
        background: linear-gradient(90deg, #38BDF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 500;
    }

    .metric-card {
        position: relative;
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 12px 30px 0 rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 20px 35px -10px rgba(56, 189, 248, 0.25);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }

    .prediction-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 1.5px solid rgba(56, 189, 248, 0.4);
        border-radius: 24px;
        padding: 35px 25px;
        text-align: center;
        box-shadow: 0 20px 50px -10px rgba(14, 165, 233, 0.3);
        position: relative;
    }
    .pred-temp-main {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .pred-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 28px;
        padding: 45px;
        margin-bottom: 30px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        overflow: hidden;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 14px;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #CBD5E1;
        line-height: 1.6;
        max-width: 900px;
        font-weight: 400;
    }

    .tech-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.12);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.25);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }

    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
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
# Sidebar Navigation Header
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="brand-box">
        <div class="brand-icon">🌤️</div>
        <div>
            <div class="brand-title">Weather ML</div>
            <div class="brand-sub">Time Series & AI Forecast</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation Menu",
    options=[
        "🏠 Home",
        "📊 Dashboard",
        "🔍 EDA & Time Series",
        "🔮 Prediction & Forecasting",
        "📈 Model Performance",
        "ℹ️ About",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background: rgba(30, 41, 59, 0.4); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem;">
        <div style="margin-bottom: 6px;"><span class="pulse-dot"></span><b>SYSTEM ONLINE</b></div>
        <div style="color: #94A3B8;"><b>Time Series:</b> 5 Years Daily</div>
        <div style="color: #94A3B8;"><b>Active Model:</b> {}</div>
        <div style="color: #94A3B8;"><b>Accuracy (R²):</b> {}</div>
    </div>
    """.format(
        predictor.metadata.get("best_model_name", "Linear Regression"),
        round(predictor.metadata.get("R2_Score", 1.0), 4),
    ),
    unsafe_allow_html=True,
)


# =============================================================================
# 1. HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Weather Data Analysis & Temperature Forecasting Engine</div>
            <div class="hero-sub">
                A complete Machine Learning and Time-Series system built with Python & Streamlit.
                Combining historical time series decomposition, moving averages, autoregressive lag features,
                and basic/ensemble regression techniques for high-precision temperature forecasting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🎯 Core Capabilities & Methodologies")
        st.markdown(
            """
            - **Time Series Feature Engineering**: Autoregressive temperature lags (`Temp_Lag1`, `Temp_Lag7`), 7-day/30-day rolling moving averages (`Temp_Rolling7_Mean`), and seasonal sine/cosine cycles (`Sin_DayOfYear`).
            - **Exploratory Data Analysis**: Trendline smoothing, correlation matrices, IQR outlier filtering, and seasonal distribution boxplots.
            - **Regression Modeling Suite**: Benchmark across 5 algorithms (Linear Regression, Decision Trees, Random Forests, Gradient Boosting, XGBoost).
            - **Future Multi-Step Forecasting**: Autoregressive temperature projection into the future with 95% confidence bands.
            """
        )

        st.subheader("🛠️ Production Stack")
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
            "Pytest",
        ]
        badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in tech_list])
        st.markdown(badges_html, unsafe_allow_html=True)

    with col2:
        st.subheader("⚡ Quick Navigation")
        st.info("Explore dataset statistics, time-series moving averages, and future temperature trends.")

        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="margin-bottom: 10px;">📊 <b>Dashboard</b> — Historical KPI cards & condition breakdown</div>
                <div style="margin-bottom: 10px;">🔍 <b>EDA & Time Series</b> — Moving averages, lag analysis & data export</div>
                <div style="margin-bottom: 10px;">🔮 <b>Prediction & Forecasting</b> — Single-day prediction & 14-day future trend forecast</div>
                <div>📈 <b>Model Performance</b> — Metric comparison table & diagnostic plots</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 2. DASHBOARD PAGE
# =============================================================================
elif page == "📊 Dashboard":
    st.title("📊 Meteorological Key Performance Indicators")
    st.caption("Real-time summary of historical daily weather metrics and atmospheric trends.")

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
            <div class="metric-subtitle">Recorded High</div>
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
            <div class="metric-subtitle">hPa Sea-Level</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([2, 1])

    with dcol1:
        st.subheader("🌡️ Historical Temperature Time Series")
        fig_temp = create_plotly_line_chart(
            df, x_col="Date", y_col="Temperature", title="Daily Temperature Trend Over Time", color_hex="#38BDF8"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with dcol2:
        st.subheader("🌤️ Weather Condition Share")
        if "Weather Condition" in df.columns:
            cond_counts = df["Weather Condition"].value_counts().reset_index()
            cond_counts.columns = ["Condition", "Count"]
            fig_pie = px.pie(
                cond_counts,
                names="Condition",
                values="Count",
                title="Condition Distribution",
                template="plotly_dark",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pie.update_layout(font=dict(family="Plus Jakarta Sans, sans-serif"))
            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# 3. EDA & TIME SERIES PAGE
# =============================================================================
elif page == "🔍 EDA & Time Series":
    st.title("🔍 Exploratory Data Analysis & Time Series Trends")
    st.caption("Perform statistical exploration, analyze moving average trendlines, and inspect correlations.")

    with st.expander("🎛️ Interactive Data & Date Filters", expanded=True):
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

    st.markdown(f"**Filtered dataset contains {len(filtered_df):,} daily records.**")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Moving Averages & Smoothing",
            "📊 Dynamic Chart Explorer",
            "🔥 Feature Correlation Heatmap",
            "📦 Distributions & Outliers",
            "📄 Summary Statistics",
        ]
    )

    with tab1:
        st.subheader("📈 Time Series Moving Average Trend Smoothing")
        st.markdown("Comparing daily temperature fluctuations with **7-day** and **30-day** rolling moving averages.")

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=filtered_df["Date"], y=filtered_df["Temperature"], mode="lines", name="Daily Raw Temp", line=dict(color="rgba(56, 189, 248, 0.4)", width=1)
        ))
        if "Temp_Rolling7_Mean" in filtered_df.columns:
            fig_ma.add_trace(go.Scatter(
                x=filtered_df["Date"], y=filtered_df["Temp_Rolling7_Mean"], mode="lines", name="7-Day Moving Avg", line=dict(color="#F59E0B", width=2)
            ))
        if "Temp_Rolling30_Mean" in filtered_df.columns:
            fig_ma.add_trace(go.Scatter(
                x=filtered_df["Date"], y=filtered_df["Temp_Rolling30_Mean"], mode="lines", name="30-Day Moving Avg", line=dict(color="#EC4899", width=2.5)
            ))
        fig_ma.update_layout(
            title="Time Series Temperature Trendline Smoothing",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    with tab2:
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

    with tab3:
        st.subheader("Feature Correlation Matrix")
        fig_corr = create_plotly_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Distribution & Outlier Inspection")
        dist_var = st.selectbox("Select Feature for Distribution Plot", options=["Temperature", "Humidity", "Pressure", "Wind Speed", "Rainfall"])
        fig_dist = px.histogram(
            filtered_df,
            x=dist_var,
            color="Weather Condition" if "Weather Condition" in filtered_df.columns else None,
            marginal="box",
            template="plotly_dark",
            title=f"Distribution & Outlier Boxplot for {dist_var}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab5:
        st.subheader("Filtered Summary Statistics")
        num_df = filtered_df.select_dtypes(include=[np.number])
        summary_stats = num_df.describe().T
        st.dataframe(summary_stats.style.highlight_max(axis=0, color="#1E3A8A"), use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Export Data")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered CSV Dataset",
        data=csv_bytes,
        file_name="filtered_weather_data.csv",
        mime="text/csv",
    )


# =============================================================================
# 4. PREDICTION & FORECASTING PAGE
# =============================================================================
elif page == "🔮 Prediction & Forecasting":
    st.title("🔮 Temperature Prediction & Future Trend Forecasting")
    st.caption("Perform single-sample real-time inference or generate multi-day future trend projections.")

    tab_p1, tab_p2 = st.tabs(["🔮 Single-Day AI Prediction", "📈 Multi-Day Future Trend Forecast"])

    with tab_p1:
        pcol1, pcol2 = st.columns([3, 2])

        with pcol1:
            st.subheader("⚙️ Input Atmospheric Observations")

            with st.form("single_prediction_form"):
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

                submit_btn = st.form_submit_button("⚡ Predict Temperature Now", use_container_width=True)

        with pcol2:
            st.subheader("🎯 Single-Day Forecast Result")

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
                    <div class="prediction-card">
                        <span class="pred-badge">● PREDICTION GENERATED</span>
                        <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 12px;">ESTIMATED DAILY TEMPERATURE</div>
                        <div class="pred-temp-main">{pred_temp_c}°C</div>
                        <div style="font-size: 1.3rem; color: #CBD5E1; font-weight: 600; margin-bottom: 15px;">({pred_temp_f}°F)</div>
                        <hr style="border-color: rgba(255,255,255,0.1); margin: 18px 0;">
                        <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.7; text-align: left; padding: 0 10px;">
                            <b>• 95% Confidence Bounds:</b> {c_low}°C to {c_high}°C<br>
                            <b>• Model Selected:</b> {res['model_used']}<br>
                            <b>• R² Accuracy Metric:</b> {round(res['r2_score'], 4)}<br>
                            <b>• Inference Latency:</b> {res['prediction_time_ms']} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Adjust the parameters on the left and click **Predict Temperature Now**.")

    with tab_p2:
        st.subheader("📈 Multi-Step Future Temperature Projection")
        st.markdown("Autoregressively forecast daily temperature trends into the future using machine learning.")

        f_days = st.slider("Select Horizon (Days into Future)", min_value=7, max_value=30, value=14, step=1)
        
        if st.button("🚀 Generate Multi-Step Forecast", use_container_width=True):
            forecast_df = predictor.predict_future_trend(n_days=f_days)

            # Interactive Plotly Future Projection Chart with Confidence Interval Area
            fig_f = go.Figure()

            # Confidence Band
            fig_f.add_trace(go.Scatter(
                x=list(forecast_df["Date"]) + list(forecast_df["Date"])[::-1],
                y=list(forecast_df["Confidence_Upper"]) + list(forecast_df["Confidence_Lower"])[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Confidence Interval"
            ))

            # Main Prediction Line
            fig_f.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecasted_Temperature"],
                mode="lines+markers",
                name="Forecasted Temperature (°C)",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=6, color="#C084FC")
            ))

            fig_f.update_layout(
                title=f"Autoregressive {f_days}-Day Future Temperature Forecast",
                template="plotly_dark",
                xaxis_title="Future Date",
                yaxis_title="Forecasted Temperature (°C)",
                font=dict(family="Plus Jakarta Sans, sans-serif")
            )

            st.plotly_chart(fig_f, use_container_width=True)

            st.subheader("📋 Forecast Summary Table")
            st.dataframe(forecast_df.style.highlight_max(subset=["Forecasted_Temperature"], color="#065F46"), use_container_width=True)


# =============================================================================
# 5. MODEL PERFORMANCE PAGE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 ML Benchmark & Model Performance Leaderboard")
    st.caption("Comprehensive comparative metrics across 5 regression algorithm architectures.")

    meta = predictor.metadata

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%); border: 1.5px solid #10B981; border-radius: 20px; padding: 22px; margin-bottom: 25px; box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.3);">
            <h3 style="margin:0; color: #34D399; font-weight: 800;">🏆 Best Performing Model: {meta.get('best_model_name', 'Linear Regression')}</h3>
            <p style="margin: 8px 0 0 0; color: #E2E8F0; font-size: 1rem;">
                Test R² Score: <b>{round(meta.get('R2_Score', 0), 4)}</b> | 
                RMSE: <b>{round(meta.get('RMSE', 0), 4)} °C</b> | 
                MAE: <b>{round(meta.get('MAE', 0), 4)} °C</b> | 
                5-Fold CV Score: <b>{round(meta.get('CV_Score', 0), 4)}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from src.train import ModelTrainer, WeatherDataPreprocessor
    preprocessor = WeatherDataPreprocessor()
    df_p, X_p, y_p = preprocessor.fit_transform_pipeline(save_processed=False)
    trainer = ModelTrainer()
    comp_df = trainer.train_and_evaluate(X_p, y_p)

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs(
        ["📊 Model Comparison Leaderboard", "⭐ Feature Importance", "📉 Residual Analysis", "🎯 Actual vs Predicted"]
    )

    with tab_m1:
        st.subheader("Algorithm Evaluation Leaderboard")
        st.dataframe(comp_df.style.highlight_max(subset=["R² Score", "CV Score (R²)"], color="#065F46"), use_container_width=True)

        fig_comp = px.bar(
            comp_df,
            x="Model",
            y="R² Score",
            color="R² Score",
            title="R² Accuracy Comparison Across Algorithms",
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
        st.subheader("Residual Diagnostics")
        best_name = meta.get("best_model_name", "Linear Regression")
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
    st.title("ℹ️ Project Architecture & Pipeline Summary")

    st.markdown(
        """
        ### 📌 Overview
        End-to-end Machine Learning web application designed to analyze multi-year weather dataset records and predict daily temperature trends.
        
        ### 🔄 Pipeline Stages
        1. **Data Ingestion & Synthesis**: Multi-year daily weather records (`Temperature`, `Humidity`, `Pressure`, `Wind Speed`, `Rainfall`, `Weather Condition`).
        2. **Preprocessing**: Missing value median imputation, duplicate removal, IQR outlier filtering, date breakdown into year/month/day/week/dayofweek, and StandardScaling.
        3. **Exploratory Data Analysis**: 10+ professional figures including seasonal trendlines, heatmaps, boxplots, and distributions.
        4. **Model Benchmark**: Cross-validated benchmark across Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
        5. **Deployment & Serving**: Interactive Streamlit Web App with real-time temperature predictions.

        ---
        **Author**: Senior Machine Learning Engineer  
        **License**: MIT License
        """
    )
