# SSIF Observatory: Streamlit Cloud Deployment Guide
### Deploying to Streamlit Community Cloud (similar to dlsm-research.streamlit.app)

**Repository:** [https://github.com/HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)  
**Main App Path:** `app/main.py`  
**License:** Apache 2.0  

---

## 🚀 3-Minute Deployment to Streamlit Cloud

Follow these steps to deploy your live research observatory online:

### Step 1: Sign in to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Log in using your GitHub account (`HarshkumarG007`).

### Step 2: Create a New App
1. Click **"New app"** (or **"Create app"**).
2. Choose **"Deploy an existing app from GitHub"**.

### Step 3: Configure Deployment Parameters
Set the following fields in the deployment modal:
- **Repository:** `HarshkumarG007/SSIF`
- **Branch:** `main`
- **Main file path:** `app/main.py`
- **App URL (subdomain):** e.g., `ssif-research.streamlit.app` (or your preferred subdomain)

### Step 4: Advanced Settings (Optional)
- **Python version:** Select `3.11` (or `3.10+`).
- Streamlit Cloud will automatically detect `requirements.txt` and install all required scientific and ML dependencies (`pandas`, `scikit-learn`, `lifelines`, `plotly`, `shap`, etc.).
- Streamlit Cloud will automatically load theme styling from `.streamlit/config.toml`.

### Step 5: Click "Deploy!"
The cloud build process will initiate. Within 1–2 minutes, your research observatory will be live at:
> **`https://ssif-research.streamlit.app`**

---

## 🐳 Alternative: Deploying via Docker

If deploying to AWS ECS, Google Cloud Run, or Azure App Service:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and data
COPY . .

# Expose Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ssif-observatory .
docker run -p 8501:8501 ssif-observatory
```
