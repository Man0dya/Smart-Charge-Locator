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
## 🔋 Smart Charge Locator

Live demo: https://smart-charge-locator.streamlit.app

Smart Charge Locator helps planners and researchers identify high-priority locations for EV charging stations. It uses vehicle registration and geographic features to produce city-level charging priority scores, an interactive map, and top-city rankings. The Streamlit app loads pre-trained models (XGBoost by default) and processed datasets to generate insights quickly.

## Table of contents

- Overview
- For users (quick start)
- For developers (local setup)
- Configuration and runtime assets
- Project structure
- Training and notebooks
- Deployment
- Troubleshooting
- Contributing
- License and governance

## Overview

This repository contains:
- A Streamlit web app for exploring EV data and predicting charging priority
- Preprocessed datasets and trained model artifacts used at runtime
- Jupyter notebooks for data preparation, feature engineering, and model training

Audience:
- Users: try the live demo or run the app with minimal setup
- Developers: extend the app logic, retrain models, or adapt inputs/outputs

Success criteria:
- App runs locally or on Streamlit Cloud and produces reproducible predictions
- Users can explore maps, rankings, and make per-city predictions without errors

## For users (quick start)

No installation required — use the hosted app:
https://smart-charge-locator.streamlit.app

How to use:
- Select a county in the left sidebar
- Explore the interactive map and click markers for city details
- Review the Top Cities table and compare cities against county averages
- Use Predict for a city’s charging-priority score and visualizations

## For developers (local setup)

The following steps assume Windows PowerShell. On macOS/Linux, adapt the venv activation path.

1) Clone and enter the project:

```powershell
git clone <repository-url>
cd Smart-Charge-Locator
```

2) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) (Optional) Install additional dev/notebook tools:

```powershell
pip install -r requirements-dev.txt
```

5) Run the app:

```powershell
streamlit run streamlit_app.py
```

Open http://localhost:8501.

Notes for development:
- The Streamlit entrypoint is `streamlit_app.py`, which calls `app/app.py:main()`.
- If data artifacts are not in the repository path, set `DATA_ROOT` to a folder that contains them (see Configuration).

## Configuration and runtime assets

Environment variables (optional):
- DATA_ROOT — folder that contains required data and models (default: repository root)
- STREAMLIT_SERVER_PORT or run with `--server.port` — change local port if needed

Minimal required runtime files (relative to repo root or DATA_ROOT):
- data/processed/city_features_engineered.csv
- data/processed/scaler.pkl
- data/processed/feature_columns.pkl
- models/xgboost.pkl (default model at runtime)

Optional but useful (diagnostics/metrics):
- data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy
- data/processed/*_performance_metrics.json (per-model metrics)

Tip: To avoid committing large artifacts, host them in cloud storage and download them at startup, or set DATA_ROOT in your hosting environment.

## Project structure

Top-level layout with key files:

- .github/                         GitHub templates and automation (if any)
- .streamlit/                      Streamlit configuration
- app/
   - app.py                         Main app logic and UI
- data/
   - raw/                           Original datasets (reference)
   - processed/                     Prepared datasets and artifacts used by the app
- models/                          Trained model artifacts (.pkl)
- notebooks/
   - 01_Data_Loading_and_Cleaning.ipynb
   - 02_Exploratory_Data_Analysis.ipynb
   - 03_Feature_Engineering.ipynb
   - model_training/
      - 4.1_Linear_Regression.ipynb
      - 4.2_Ridge_Regression.ipynb
      - 4.3_Random_Forest.ipynb
      - 4.4_XGBoost.ipynb
- Model Accuracy Chart.ipynb       Summary/visual notebook at repo root
- requirements.txt                 Runtime dependencies
- requirements-dev.txt             Dev/notebook dependencies
- runtime.txt                      Pinned Python for hosting (e.g., python-3.11)
- streamlit_app.py                 Streamlit entrypoint
- CODE_OF_CONDUCT.md               Community guidelines
- CONTRIBUTING.md                  Contribution guide
- SECURITY.md                      Security policy
- LICENSE                          MIT license

## Training and notebooks

The `notebooks/` folder documents data preparation and model training. The `model_training/` subfolder contains per-model notebooks for linear regression, ridge, random forest, and XGBoost. Trained artifacts should be exported to `models/` and their corresponding preprocessors (e.g., `scaler.pkl`, `feature_columns.pkl`) to `data/processed/` for use by the app.

## Deployment

Streamlit Community Cloud (Share) is the simplest deployment target.

1. Push your branch to GitHub
2. Create an app at https://share.streamlit.io and select this repository/branch
3. Set the main file to `streamlit_app.py`
4. If artifacts are hosted externally, set `DATA_ROOT` in App settings → Advanced → Environment variables

Tips:
- Keep `requirements.txt` minimal to speed builds; pin `runtime.txt` to `python-3.11`
- If build issues occur, clear the app cache and redeploy

## Troubleshooting

- Missing data/artifacts: ensure required files exist under `data/processed/` and `models/`, or set `DATA_ROOT`
- Port in use: `streamlit run streamlit_app.py --server.port 8503`
- Model/feature mismatch: ensure `feature_columns.pkl` and `scaler.pkl` match the trained model
- Streamlit Cloud build errors: clear cache and verify dependency versions

## Contributing

We welcome improvements and extensions.
1. Fork the repository
2. Create a branch: `feature/<name>` or `fix/<name>`
3. Implement changes and add tests where practical
4. Install dev dependencies and validate locally
5. Open a PR and follow `CONTRIBUTING.md`

## License and governance

This project is licensed under the MIT License — see `LICENSE`.

Community and security:
- Code of Conduct — `CODE_OF_CONDUCT.md`
- Contributing Guide — `CONTRIBUTING.md`
- Security Policy — `SECURITY.md`

---

Questions or ideas? Open an issue — feedback helps improve data coverage, model quality, and UX.
