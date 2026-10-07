# 🚆 Indian Railways Predictive Maintenance System (IR-PMIS)
### Modern FastAPI + React Technical Architecture

A decoupled, production-ready AI telemetry and predictive maintenance platform for Indian Railways track infrastructure, wheelsets, and rolling stock components.

---

## 🏗️ Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        React + Vite Frontend                           │
│     (Vercel Deployable • Non-Technical 3-Second Verdict • No Sliders)   │
├──────────────────────┬────────────────────────┬────────────────────────┤
│ 1. Quick Check       │ 2. Batch Upload        │ 3. Model Performance   │
│ • 6 Curated Presets  │ • Drag & Drop Fleet CSV│ • Benchmark Metrics    │
│ • Subsystem Health   │ • Risk-Ranked Table    │ • Feature Importances  │
│ • Root Causes Box    │ • Urgency Filtering   │ • Confusion Matrix     │
│ • Advanced Override  │ • Export Enriched CSV  │ • ROC-AUC Visualizer   │
└──────────────────────┴───────────▲────────────┴────────────────────────┘
                                   │ REST API (JSON / FormData)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (ASGI)                          │
│        (Azure App Service Ready • Lifespan Memory Model Cache)          │
├────────────────────────────────────────────────────────────────────────┤
│ • POST /api/predict       : Single Asset Real-time Telemetry Inference │
│ • POST /api/predict-batch : Fleet CSV Automated Risk Ranking           │
│ • GET  /api/presets       : 6 Realistic Operational Preset Profiles    │
│ • GET  /api/model-info    : Precomputed Benchmarks & Gini Importances  │
│ • GET  /api/sample-csv    : Downloadable Fleet CSV Template            │
│ • GET  /health            : Liveness & Readiness Azure Health Probe    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
                 [ Preprocessor & Trained Random Forest ]
```

---

## 🌟 Key Features & Improvements over Monolithic Version

1. **Elimination of the "30-Slider Dump"**:
   - Instead of overwhelming users with 30 raw sliders on load, the **Quick Check** view features **6 realistic scenario presets** (*Vande Bharat Express*, *Monsoon Coastal Warning*, *Heavy Mineral Freight*, *Suburban EMU*, *Rajdhani Superfast*, *Yard Shunter*).
   - An optional **Advanced Diagnostics** toggle is tucked away for expert sensor overrides.

2. **3-Second Verdict Clarity**:
   - Single clear verdict card with large status indicators (`SAFE` vs `MAINTENANCE REQUIRED`).
   - Risk probability percentage with animated visual threshold bars.
   - 4 Subsystem Health Gauges: **Track Infrastructure**, **Mechanical Wheelsets**, **Pneumatic Brakes**, **Electrical & Signals**.
   - Plain-language **Root-Cause Analysis** listing top 3 contributing physical features.

3. **Multi-Asset Fleet Batch Upload**:
   - Drag-and-drop CSV upload for fleets of 10 to 10,000+ assets.
   - Automatic risk ranking (highest risk assets first).
   - Filter by Urgency (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), search by Asset ID, and export enriched predictions to CSV.

4. **Model Performance & Analytics**:
   - Zero live recomputation overhead on `GET /api/model-info`.
   - Side-by-side comparison between **Random Forest** (80.0% Accuracy, 51.5% Precision) and **Logistic Regression** (60.6% Accuracy, 40.4% Recall).
   - Gini Feature Importance rankings showing **Rail Wear (8.6%)**, **Wheel Wear (6.4%)**, **Track Vibration (5.5%)**, and **Ballast Condition (4.8%)** as top failure drivers.

---

## 🚀 How to Run Locally

### 1. Start the FastAPI Backend
```bash
# From workspace root
source venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
* **API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/health`

### 2. Start the React + Vite Frontend
```bash
# In a second terminal
cd frontend
npm run dev
```
* **Web UI Dashboard**: `http://127.0.0.1:5173`

---

## ☁️ Deployment Targets

### 1. Frontend $\rightarrow$ Vercel
* **Framework Preset**: Vite
* **Root Directory**: `frontend`
* **Build Command**: `npm run build`
* **Output Directory**: `dist`
* **Environment Variable**: `VITE_API_BASE_URL` = `https://<your-azure-app>.azurewebsites.net`

### 2. Backend $\rightarrow$ Azure App Service (Linux Python)
* **Startup Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app` or `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
* **App Settings / Environment Variables**:
  * `ALLOWED_ORIGINS` = `https://<your-vercel-app>.vercel.app`
  * `PORT` = `8000`

---

## 📁 Project Directory Structure
```text
AIML/
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI ASGI server & CORS middleware
│   ├── schemas.py                # Pydantic validation schemas
│   ├── inference.py              # ML inference & subsystem health engine
│   └── data_store.py             # Presets & precomputed evaluation metrics
├── frontend/
│   ├── .env                      # Local API base URL config
│   ├── .env.example              # Vercel deployment template
│   ├── package.json              # React 18/19, Lucide icons, Vite
│   ├── vite.config.js
│   ├── index.html                # HTML entry with Google fonts
│   └── src/
│       ├── config.js             # Centralized endpoint resolver
│       ├── App.jsx               # Main React container
│       ├── main.jsx
│       ├── styles/
│       │   └── index.css         # Industrial railway design system
│       └── components/
│           ├── Navbar.jsx        # Navigation & FastAPI status indicator
│           ├── QuickCheck/       # Presets, Verdict Card, Subsystems, Advanced Toggle
│           ├── BatchUpload/      # CSV Drag & Drop, KPI Cards, Ranked Table, Export
│           └── ModelPerformance/# Benchmark Table, Feature Importance, Confusion Matrix
├── saved_models/
│   ├── random_forest.joblib      # Production Model (80% Acc, 51.5% Prec)
│   ├── logistic_regression.joblib# Baseline Model
│   └── preprocessor.joblib       # ColumnTransformer (StandardScaler + OneHot)
├── Railway_Predictive_Maintenance_Phase1.ipynb # Jupyter Notebook
└── cleaned_railway_data.csv      # 100k Clean Dataset
```
