"""
Visualization module for Weather Data Analysis and EDA.
Generates matplotlib/seaborn plots (saved automatically into assets/images/)
and interactive Plotly charts for the Streamlit web dashboard.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for background image generation
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as gg
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils import IMAGES_DIR, ensure_directories

# Set global aesthetic styling for matplotlib/seaborn
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("muted")


def generate_all_eda_plots(df: pd.DataFrame, output_dir: Path = None) -> dict:
    """
    Generate and save all 10+ standard EDA static figures to assets/images/.
    
    Returns:
        dict: Mapping of figure identifiers to image file paths.
    """
    ensure_directories()
    if output_dir is None:
        output_dir = IMAGES_DIR

    saved_images = {}
    df_plot = df.copy()
    if "Date" in df_plot.columns and not pd.api.types.is_datetime64_any_dtype(df_plot["Date"]):
        df_plot["Date"] = pd.to_datetime(df_plot["Date"])

    # 1. Temperature Trend
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.lineplot(data=df_plot, x="Date", y="Temperature", color="#FF5722", linewidth=1.2, ax=ax)
    ax.set_title("Historical Daily Temperature Trend (°C)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    plt.tight_layout()
    temp_path = output_dir / "temperature_trend.png"
    plt.savefig(temp_path)
    plt.close()
    saved_images["temperature_trend"] = temp_path

    # 2. Humidity Trend
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.lineplot(data=df_plot, x="Date", y="Humidity", color="#0288D1", linewidth=1.2, ax=ax)
    ax.set_title("Historical Relative Humidity Trend (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Humidity (%)", fontsize=11)
    plt.tight_layout()
    hum_path = output_dir / "humidity_trend.png"
    plt.savefig(hum_path)
    plt.close()
    saved_images["humidity_trend"] = hum_path

    # 3. Pressure Trend
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.lineplot(data=df_plot, x="Date", y="Pressure", color="#7B1FA2", linewidth=1.2, ax=ax)
    ax.set_title("Atmospheric Pressure Trend (hPa)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    plt.tight_layout()
    pres_path = output_dir / "pressure_trend.png"
    plt.savefig(pres_path)
    plt.close()
    saved_images["pressure_trend"] = pres_path

    # 4. Rainfall Trend
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.lineplot(data=df_plot, x="Date", y="Rainfall", color="#388E3C", linewidth=1.2, ax=ax)
    ax.set_title("Daily Rainfall Volume Trend (mm)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Rainfall (mm)", fontsize=11)
    plt.tight_layout()
    rain_path = output_dir / "rainfall_trend.png"
    plt.savefig(rain_path)
    plt.close()
    saved_images["rainfall_trend"] = rain_path

    # 5. Wind Speed Trend
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.lineplot(data=df_plot, x="Date", y="Wind Speed", color="#E64A19", linewidth=1.2, ax=ax)
    ax.set_title("Wind Speed Trend (km/h)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Wind Speed (km/h)", fontsize=11)
    plt.tight_layout()
    wind_path = output_dir / "wind_speed_trend.png"
    plt.savefig(wind_path)
    plt.close()
    saved_images["wind_speed_trend"] = wind_path

    # 6. Monthly Temperature Boxplot / Aggregation
    if "Month" in df_plot.columns:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        sns.boxplot(data=df_plot, x="Month", y="Temperature", hue="Month", palette="magma", legend=False, ax=ax)
        ax.set_title("Monthly Temperature Distribution (°C)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Month of Year", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        plt.tight_layout()
        month_path = output_dir / "monthly_temperature.png"
        plt.savefig(month_path)
        plt.close()
        saved_images["monthly_temperature"] = month_path

    # 7. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    num_cols = df_plot.select_dtypes(include=[np.number]).columns
    corr = df_plot[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = output_dir / "correlation_heatmap.png"
    plt.savefig(corr_path)
    plt.close()
    saved_images["correlation_heatmap"] = corr_path

    # 8. Distribution Plot for Temperature & Humidity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sns.histplot(df_plot["Temperature"], kde=True, color="#FF7043", ax=axes[0])
    axes[0].set_title("Temperature Distribution", fontsize=12, fontweight="bold")
    sns.histplot(df_plot["Humidity"], kde=True, color="#29B6F6", ax=axes[1])
    axes[1].set_title("Humidity Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    dist_path = output_dir / "distribution_plot.png"
    plt.savefig(dist_path)
    plt.close()
    saved_images["distribution_plot"] = dist_path

    # 9. Boxplots for Numerical Features
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    cols = ["Temperature", "Humidity", "Pressure", "Wind Speed"]
    for idx, col in enumerate(cols):
        r, c = idx // 2, idx % 2
        sns.boxplot(y=df_plot[col], color="#26A69A", ax=axes[r, c])
        axes[r, c].set_title(f"{col} Boxplot", fontsize=12, fontweight="bold")
    plt.tight_layout()
    box_path = output_dir / "boxplots.png"
    plt.savefig(box_path)
    plt.close()
    saved_images["boxplots"] = box_path

    # 10. Scatterplot: Humidity vs Temperature colored by Weather Condition
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    hue_col = "Weather Condition" if "Weather Condition" in df_plot.columns else None
    sns.scatterplot(
        data=df_plot,
        x="Humidity",
        y="Temperature",
        hue=hue_col,
        alpha=0.7,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Humidity vs Temperature by Weather Condition", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    scat_path = output_dir / "scatterplot.png"
    plt.savefig(scat_path)
    plt.close()
    saved_images["scatterplots"] = scat_path

    # 11. Pairplot
    try:
        pair_cols = ["Temperature", "Humidity", "Pressure", "Wind Speed"]
        g = sns.pairplot(df_plot[pair_cols], diag_kind="kde", corner=True)
        g.fig.suptitle("Multivariate Feature Pairplot", y=1.02, fontsize=14, fontweight="bold")
        pair_path = output_dir / "pairplots.png"
        g.savefig(pair_path)
        plt.close()
        saved_images["pairplots"] = pair_path
    except Exception as e:
        print(f"Pairplot warning: {e}")

    return saved_images


# Interactive Plotly chart builders for Streamlit dashboard
def create_plotly_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_hex: str = "#FF5722"):
    """Create a sleek interactive line chart with Plotly."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        template="plotly_dark",
        color_discrete_sequence=[color_hex]
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    return fig


def create_plotly_correlation_heatmap(df: pd.DataFrame):
    """Create interactive Plotly heatmap for numerical correlations."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Interactive Feature Correlation Matrix",
        template="plotly_dark"
    )
    fig.update_layout(font=dict(family="Inter, sans-serif", size=12))
    return fig


def create_plotly_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """Create interactive Plotly scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{x_col} vs {y_col}",
        template="plotly_dark",
        opacity=0.8
    )
    fig.update_layout(font=dict(family="Inter, sans-serif", size=12))
    return fig


def create_feature_importance_plot(feature_names: list, importances: list):
    """Create interactive feature importance bar plot."""
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Best Model Feature Importance Breakdown",
        template="plotly_dark",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(font=dict(family="Inter, sans-serif", size=12))
    return fig


def create_residual_plot(y_true, y_pred):
    """Create residual error distribution plot."""
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted Temperature (°C)", "y": "Residual Error (°C)"},
        title="Model Residual Analysis (Predicted vs Residuals)",
        template="plotly_dark",
        opacity=0.75
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#FF5252")
    fig.update_layout(font=dict(family="Inter, sans-serif", size=12))
    return fig


def create_actual_vs_pred_plot(y_true, y_pred):
    """Create Actual vs Predicted scatter plot with 45-degree reference identity line."""
    fig = gg.Figure()
    fig.add_trace(gg.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(color="#00E676", opacity=0.7)
    ))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(gg.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Fit (y=x)",
        line=dict(color="#FFD600", dash="dash", width=2)
    ))
    fig.update_layout(
        title="Actual vs Predicted Temperature Alignment",
        xaxis_title="Actual Temperature (°C)",
        yaxis_title="Predicted Temperature (°C)",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", size=12)
    )
    return fig
