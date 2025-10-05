# Smart Charge Locator

[![Open in Streamlit](https://img.shields.io/badge/Live%20App-Open%20Smart%20Charge%20Locator-ff4b4b?logo=streamlit&logoColor=white)](https://smart-charge-locator.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Identify high‑priority locations for EV charging stations across Washington with an interactive Streamlit app powered by ML models and geospatial visuals.


## Table of contents

- Overview
- What this app does
- Quick start (for users)
- Screens and features
- Configuration
- Data and models
- Project structure
- Troubleshooting
- For developers
- Reproducing training/artifacts
- FAQ
- Roadmap
- Deploying on Streamlit Community Cloud
- Acknowledgements


## Overview

Smart Charge Locator helps planners, utilities, and station operators prioritize where to deploy EV charging. It aggregates city‑level indicators (EV adoption, range, MSRP, and more) into a Charging Score, then visualizes and ranks locations with county filters and intuitive comparisons.


## What this app does

- Interactive map of cities with EV adoption and Charging Score overlays
- County filter + top-10 city ranking table
- Per-city "Charging Score" prediction using the trained XGBoost model
- Side-by-side visual comparisons: selected city vs. county averages
- Distribution and tier breakdowns to aid siting decisions


## Quick start (for users)

Prerequisites
- Python 3.11
- pip

Install and run

```powershell
# From the repository root
pip install -r requirements.txt

# Start the Streamlit app
streamlit run streamlit_app.py
# Then open the link in the terminal (typically http://localhost:8501)
```

Required runtime files
- data/processed/city_features_engineered.csv
- data/processed/scaler.pkl
- data/processed/feature_columns.pkl
- models/xgboost.pkl

If these aren’t present, the app will show a helpful error. You can also point the app to a different data location via DATA_ROOT (see “Configuration”).


## Features at a glance

- Interactive Folium map of Washington with city markers sized by EV count and colored by charging priority
- County filter and top‑10 city ranking table
- One‑click “Charging Score” prediction for the selected city (using XGBoost)
- Plotly visuals: city vs. county average comparison, score distribution, priority tiers
- Robust file discovery via DATA_ROOT or repo defaults, with in‑app diagnostics


## Screens and features

- Map: Folium-based interactive map, markers sized by EV_Count and colored by Charging_Score. A county selector filters the view.
- Ranking: A sortable table of the top cities by Charging_Score.
- Prediction: Select a county and city to get a model-based Charging Score, with a friendly priority interpretation.
- Visuals: Compact comparisons of key features for the selected city versus the county average, plus score distributions and tier pie chart.


## Configuration

- Override data/models location with an environment variable:

```powershell
# Windows PowerShell
$env:DATA_ROOT = "E:\\FDM\\PROJECT\\Newest\\Smart-Charge-Locator"
streamlit run streamlit_app.py
```

The app searches these locations for files (in order):
1) DATA_ROOT (if set)
2) Current working directory
3) Repository root


## Data and models

- Raw source: data/raw/Electric_Vehicle_Population_Data.csv
- Processed features and artifacts: data/processed/*
- Trained models: models/*.pkl

Training notebooks (optional)
- notebooks/model_training/4.1_Linear_Regression.ipynb
- notebooks/model_training/4.2_Ridge_Regression.ipynb
- notebooks/model_training/4.3_Random_Forest.ipynb
- notebooks/model_training/4.4_XGBoost.ipynb

Model performance (from provided metrics files)
- RandomForest: test R² ≈ 0.974, test MAE ≈ 20.59
- XGBoost: test R² ≈ 0.963, test MAE ≈ 27.84

Note: The app currently defaults to XGBoost. You can retrain/update models via the notebooks and save them to models/xgboost.pkl.

Data dictionary (selected columns)
- City, County: geographic identifiers
- Latitude_mean, Longitude_mean: city centroid coordinates used for mapping
- EV_Count: number of EVs in the city
- Avg_Range: average electric range (miles)
- Avg_MSRP: average MSRP ($)
- Charging_Score: engineered target/priority score used for ranking and visualization
- Plus additional engineered features (total ≈ 11) used during model training


## Project structure

```text
Smart-Charge-Locator/
├─ streamlit_app.py             # Streamlit entrypoint (Cloud/local)
├─ app/
│  └─ app.py                    # Main UI + data/model loading and visuals
├─ data/
│  ├─ raw/                      # Original dataset(s)
│  └─ processed/                # Features, artifacts, metrics
├─ models/                      # Trained model pickles
├─ notebooks/                   # Data prep, EDA, feature engineering, training
├─ requirements.txt             # Runtime deps (app only)
├─ requirements-dev.txt         # Dev/Notebook deps (optional)
├─ runtime.txt                  # Python version hint for Streamlit Cloud
├─ CONTRIBUTING.md | SECURITY.md | CODE_OF_CONDUCT.md | LICENSE
```


## Troubleshooting

- FileNotFoundError / missing artifacts
  - Ensure the required files listed above exist. If you keep data elsewhere, set DATA_ROOT.
- “XGBoost model not available”
  - Run the XGBoost training notebook and export models/xgboost.pkl, or copy it from a previous run.
- Map rendering error mentioning JSON serialization
  - The app automatically falls back to HTML rendering for Folium if needed. Ensure streamlit-folium is installed (it is in requirements.txt).
- Version conflicts
  - Use requirements.txt for the app; requirements-dev.txt is for notebooks and may pin different versions for scientific stacks.


## For developers

Local dev setup
```powershell
# Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime deps
pip install -r requirements.txt

# Optional: install extra tools for notebooks, EDA, and training
pip install -r requirements-dev.txt
```

Recommended workflow
- Use a feature branch for changes
- Keep runtime requirements minimal; heavier notebook tooling should stay in requirements-dev.txt
- If you change data processing or training logic, re-generate processed artifacts in data/processed and update models/*.pkl accordingly
- Prefer small, focused PRs and include a brief note on data/model changes

Coding conventions
- Python 3.11, PEP 8 style
- Keep Streamlit UI snappy and user-friendly; prefer simple, readable visuals

Where things happen
- UI and interaction: app/app.py
- Entry point for Streamlit Cloud: streamlit_app.py
- Data and model files: data/processed, models/
- Notebooks: notebooks/*

Contributing
- See CONTRIBUTING.md for our PR workflow and tips
- Be kind and follow the CODE_OF_CONDUCT.md

Security
- See SECURITY.md to report vulnerabilities privately

License
- MIT (see LICENSE)


## Reproducing training/artifacts

1) Data prep and feature engineering
   - Run notebooks: 01_Data_Loading_and_Cleaning → 02_Exploratory_Data_Analysis → 03_Feature_Engineering
   - Export artifacts to data/processed (e.g., city_features_engineered.csv, scaler.pkl, feature_columns.pkl)
2) Model training
   - Use notebooks in notebooks/model_training to train Linear, Ridge, RF, and XGBoost
   - Save the chosen model to models/xgboost.pkl (the runtime default)
3) Metrics
   - Optionally persist metrics JSONs in data/processed for the sidebar performance panel


## FAQ

- Can I run with my own dataset?
  - Yes. Prepare a city‑level CSV with similar columns and regenerate artifacts (scaler, columns list, model). Point DATA_ROOT to your folder.
- Does the app support other states?
  - The UI and code are state‑agnostic. Update the datasets to your geography and center the map as needed.
- What if I only have the CSV but not the model?
  - You can still explore the map and rankings based on Charging_Score in the CSV. For predictions, retrain via the notebooks.
- Why XGBoost by default when RF tests higher here?
  - XGBoost is a solid baseline and widely portable. You can switch to RF by saving models/random_forest.pkl and adjusting the app logic if desired.


## Roadmap

- Multi‑state/national data support and automatic map centering
- Scenario analysis (e.g., simulate added chargers and re‑score)
- Caching for faster startup in Streamlit Cloud
- Optional Dockerfile and devcontainer for reproducible environments
- Basic CI (linting) and data validation checks

## Deploying on Streamlit Community Cloud

- Repo: link this GitHub repository
- Main file: streamlit_app.py
- Python version: 3.11 (runtime.txt already included)
- Python packages: requirements.txt
- Data availability: ensure the necessary files are in the repo or accessible via DATA_ROOT or external storage


## Acknowledgements

- EV population data file provided in data/raw. Processed features and metrics in data/processed were generated via the included notebooks.

—
Questions or ideas? Open an issue or start a discussion. Happy charging! ⚡