# 🔋 Smart Charge Locator

Live app: https://smart-charge-locator.streamlit.app

Smart Charge Locator predicts and visualizes ideal locations for EV charging stations using registration and geographic features. The app exposes an interactive map, top-city rankings, and on-demand charging-priority predictions driven by trained ML models (XGBoost is used by default).

## Key features

- Interactive Folium map with city markers and charging-priority popups
- Top-cities ranking table and per-city visual comparisons vs county averages
- On-demand predictions using pre-trained models (XGBoost by default)
- Simple, responsive Streamlit UI for quick exploration

## 🔋 Smart Charge Locator

Live demo: https://smart-charge-locator.streamlit.app

Smart Charge Locator predicts and visualizes high-priority locations for EV charging stations using vehicle registration and geographic features. The app provides an interactive map, top-city rankings, and on-demand charging-priority predictions powered by pre-trained ML models (XGBoost by default).

## Table of contents

- Overview
- Quick start — Users
- Local development — Developers
- Runtime assets & configuration
- Extended project structure
- Deployment
- Troubleshooting
- Contributing
- License & acknowledgements

## Overview

What this repo contains:
- A Streamlit web app for exploring EV data and predicting charging priority
- Processed data and trained model artifacts used at runtime
- Notebooks used for data cleaning, feature engineering and model training

Who this README is for:
- Users: want to try the live app or run a demo without installing anything
- Developers: want to run the app locally, reproduce results, or extend the codebase

Contract (small):
- Inputs: county/city selection, optional model choice, and the processed feature files
- Outputs: per-city charging score, visualizations, and exportable CSVs
- Success: app runs locally or on Streamlit Cloud and produces consistent predictions
- Error modes: missing data files, incompatible Python packages, or model artifact mismatch

## Quick start — Users

Try the live app (no install):
https://smart-charge-locator.streamlit.app

Usage highlights:
- Select a county from the left sidebar
- Explore the interactive map and click markers for city-level details
- View the Top Cities table and compare cities vs county averages
- Use the Predict button for a city's charging-priority score and visualizations

## Local development — Developers

This section explains how to set up a local dev environment on Windows (PowerShell). If you use macOS / Linux, the commands are similar but use your shell's activation steps.

1) Clone the repository and open a PowerShell prompt in the repo root:

```powershell
git clone <repository-url>
cd Smart-Charge-Locator
```

2) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

4) (Optional) Install dev/test dependencies:

```powershell
pip install -r requirements-dev.txt
```

5) Run the app locally:

```powershell
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

Developer notes:
- If the app crashes on missing files, set the `DATA_ROOT` environment variable to point to a folder containing required assets (see "Runtime assets"). Example (PowerShell):

```powershell
$env:DATA_ROOT = "E:\path\to\data"
```

- Use the `app/` package for app logic. `streamlit_app.py` is the entrypoint that wires Streamlit to the code in `app/`.

Edge cases to consider while developing:
- Missing or corrupted processed data files
- Model artifact versions that don't match feature columns/scaler
- Large file sizes (Streamlit Cloud limits) — provide smaller dev datasets

## Runtime assets & configuration

The app expects the following runtime files to be available under `data/processed/` (or under a path set with the `DATA_ROOT` env var):

- data/processed/city_features_engineered.csv
- data/processed/scaler.pkl
- data/processed/feature_columns.pkl
- data/processed/X_train.npy, X_test.npy, y_train.npy (optional for diagnostics)
- models/xgboost.pkl (default model)

If you don't want to store large artifacts in the repository, host them externally and either:
- set `DATA_ROOT` in your hosting environment (Streamlit Cloud: App settings → Advanced → Environment variables), or
- add a small bootstrap in `streamlit_app.py` that downloads assets on first run.

Configuration environment variables (optional):
- DATA_ROOT — folder with processed data and models (defaults to repo root)
- PORT or STREAMLIT_SERVER_PORT — to change the Streamlit port

## Extended project structure

The repository layout with key files explained:

Smart-Charge-Locator/
- .streamlit/                 · Streamlit configuration files
- app/                        · Application package (core app code)
   - __init__.py
   - app.py                    · high-level app wiring and UI components
   - utils.py                  · helper functions (data loading, IP/format helpers)
   - models.py                 · model loading & predict wrappers
- data/
   - raw/                      · original input datasets (committed for reference)
   - processed/                · prepared files used by the app (required at runtime)
- models/                     · trained model artifacts (.pkl)
- notebooks/                  · analysis and training notebooks
   - 01_Data_Loading_and_Cleaning.ipynb
   - 02_Exploratory_Data_Analysis.ipynb
   - 03_Feature_Engineering.ipynb
   - model_training/           · per-model training notebooks
- requirements.txt            · minimal runtime dependencies
- requirements-dev.txt        · extra packages for notebooks & testing
- runtime.txt                 · pinned Python runtime for hosting (e.g. python-3.11)
- streamlit_app.py            · Streamlit entrypoint (uses `app/` package)
- README.md                   · this document

Files you may inspect when developing:
- `app/app.py` — main UI and callback wiring
- `data/processed/feature_columns.pkl` — ensures model input ordering

## Deployment

Streamlit Community Cloud (Share) is the recommended simple host.

Steps:
1. Push your branch to GitHub
2. Create an app at https://share.streamlit.io and point it to this repo
3. Set the main file to `streamlit_app.py`
4. Add `DATA_ROOT` in app settings if assets are hosted externally

Notes & tips:
- Use `runtime.txt` to pin Python to `python-3.11` to avoid building native wheels on some hosted builders
- If artifacts are large, prefer hosting them on cloud storage (S3, GCS) and download them at startup

## Troubleshooting

- Error loading data: confirm `DATA_ROOT` or that `data/processed/` contains required files
- Port already in use: run `streamlit run streamlit_app.py --server.port 8503`
- Model mismatch errors: ensure `feature_columns.pkl` and `scaler.pkl` match the model you load
- Streamlit Cloud build errors: clear cache in app settings then redeploy

## Contributing

Small checklist for contributors:
1. Fork the repository
2. Create a branch named `feature/desc` or `fix/desc`
3. Add code + tests for new behavior when practical
4. Run `pip install -r requirements-dev.txt` and validate locally
5. Open a PR and use the PR template

If you'd like help adding CI (lint + tests) or badges, open an issue and I can add a GitHub Actions workflow for a basic check (flake8 / isort / pytest).

## License & acknowledgements

This project is provided under the terms in `LICENSE` (MIT)

Acknowledgements:
- Dataset: Electric Vehicle Population Data (see `data/raw/`)
- Built with: Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Plotly, Folium

---
