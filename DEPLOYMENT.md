# Deployment Guide - Weather Data Analysis & Temperature Prediction ML System

This guide outlines step-by-step instructions for deploying your Streamlit Machine Learning web application to **Streamlit Community Cloud**, **Render**, **Hugging Face Spaces**, and **Docker / Cloud VMs**.

---

## Option 1: Deploy on Streamlit Community Cloud (Recommended - 100% Free & Easiest)

Streamlit Community Cloud directly connects to your GitHub repository and deploys your app in under 2 minutes.

### Step 1: Push Project to GitHub

1. Open your terminal in the project directory:
   ```bash
   cd C:\Users\Prathamesh\.gemini\antigravity\scratch\Weather-Temperature-Prediction
   ```

2. Create a new repository on [GitHub](https://github.com/new) named `Weather-Temperature-Prediction`.

3. Connect your local git repository to GitHub and push:
   ```bash
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/Weather-Temperature-Prediction.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository (`YOUR_GITHUB_USERNAME/Weather-Temperature-Prediction`).
4. Set **Branch** to `main`.
5. Set **Main file path** to `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your app with a public URL!

---

## Option 2: Deploy on Render (Free Hosting)

1. Sign up at [Render.com](https://render.com/).
2. Click **"New +"** ➔ **"Web Service"**.
3. Connect your GitHub repository `Weather-Temperature-Prediction`.
4. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python src/train.py`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## Option 3: Deploy on Hugging Face Spaces (Free)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Name your space `weather-temperature-prediction`.
3. Select **Streamlit** as the SDK.
4. Upload your project repository files (`app.py`, `requirements.txt`, `src/`, `models/`, `data/`, `assets/`).
5. Hugging Face will automatically build and host your app.

---

## Option 4: Deploy using Docker (Containerized)

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run model training to ensure artifacts exist
RUN python src/train.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run Docker Container locally:
```bash
docker build -t weather-ml-app .
docker run -p 8501:8501 weather-ml-app
```

---

## ⚙️ Deployment Checklist

- [x] `requirements.txt` contains all required libraries.
- [x] `.gitignore` ignores `venv`, `__pycache__`, and temporary caches.
- [x] `app.py` has no hardcoded local absolute Windows file paths.
- [x] `src/train.py` auto-generates missing models/data if running on a fresh cloud instance.
