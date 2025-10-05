# 🔋 Smart Charge Locator

Live app: https://smart-charge-locator.streamlit.app

Smart Charge Locator predicts and visualizes ideal locations for EV charging stations using registration and geographic features. The app exposes an interactive map, top-city rankings, and on-demand charging-priority predictions driven by trained ML models (XGBoost is used by default).

## Key features

- Interactive Folium map with city markers and charging-priority popups
- Top-cities ranking table and per-city visual comparisons vs county averages
- On-demand predictions using pre-trained models (XGBoost by default)
- Simple, responsive Streamlit UI for quick exploration

## Repository layout

```
Smart-Charge-Locator/
├── .streamlit/                  # Streamlit config
├── app/                         # Streamlit app code (entry: streamlit_app.py)
├── data/                        # Data (raw & processed)
│   └── processed/               # Files required at runtime
├── models/                      # Trained model artifacts (.pkl)
├── notebooks/                   # Notebooks used to generate data and train models
├── requirements.txt             # Runtime Python dependencies
├── requirements-dev.txt         # Dev / notebook dependencies
└── README.md
```

## Quick start (users)

Open the live app (no install required):

https://smart-charge-locator.streamlit.app

Basic flow:

- Choose a county from the left sidebar
- Inspect the interactive map and marker popups
- View the Top Cities table
- Select a city and click Predict to show the Charging Score and visualizations

## Install and run locally (developers)

1. Clone the repo and switch to the project folder:

```powershell
git clone <repository-url>
cd Smart-Charge-Locator
```

2. Create and activate a venv (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the app locally:

```powershell
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Required runtime assets

The app expects these files to be present (relative to repo root or pointed to by the `DATA_ROOT` env var):

- data/processed/city_features_engineered.csv
- data/processed/scaler.pkl
- data/processed/feature_columns.pkl
- models/xgboost.pkl

If you don't want to commit these large files, host them externally and set `DATA_ROOT` in Streamlit Cloud (App settings → Advanced → Environment variables) or add a small download bootstrap in the app.

## Deployment (Streamlit Community Cloud)

1. Push your branch to GitHub
2. On https://share.streamlit.io create a new app and select this repository
3. Set the main file to `streamlit_app.py`
4. If you changed `runtime.txt` or package versions, go to Advanced → Clear cache and Redeploy

Notes:
- To avoid native builds, use the `requirements.txt` provided (it targets versions with prebuilt wheels for recent Python versions). If the builder compiles pandas/numpy, try clearing cache or pin `runtime.txt` to `python-3.11`.

## Development workflow

- Branching: `git checkout -b feature/short-description`
- Run the app locally and validate flows
- Open a pull request and use the PR template in `.github/pull_request_template.md`
- Follow the contributing guide in `CONTRIBUTING.md`

## Troubleshooting

- Error loading data: Ensure the files under `data/processed/` are present and not ignored by Git. You can set `DATA_ROOT` to an alternative path.
- Port already in use: `streamlit run streamlit_app.py --server.port 8503`
- Streamlit Cloud build errors: Clear the app cache (Advanced → Clear cache) and redeploy; ensure `requirements.txt` uses compatible package versions/wheels.

## Community & governance

- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [Security Policy](./SECURITY.md)
- [License (MIT)](./LICENSE)

## Acknowledgements

- Data source: Electric Vehicle Population Data (see `data/raw/`)
- Built with: Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Plotly, Folium

---

If you'd like, I can add CI (lint + a simple import/test), badges to the top of this README, or a small architecture diagram—tell me which you'd prefer next.

## 📈 Model Performance

The project includes multiple machine learning models with different strengths:

- **Linear Regression**: Fast, interpretable, good baseline
- **Ridge Regression**: Regularized, prevents overfitting
- **Random Forest**: Handles non-linear relationships, feature importance
- **XGBoost**: High performance, gradient boosting

## 🔧 Key Features of the Web App

- **Interactive Maps**: Folium-based maps with EV distribution
- **Real-time Predictions**: Get charging scores for any city
- **Model Comparison**: Switch between different ML models
- **County Filtering**: Focus on specific geographic areas
- **Performance Metrics**: View model accuracy and performance
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Dataset Information

The project uses the **Electric Vehicle Population Data** which includes:
- Vehicle information (VIN, make, model, year)
- Geographic data (county, city, state, coordinates)
- EV specifications (electric range, MSRP, vehicle type)
- Registration details and utility information

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Folium**: Interactive maps
- **Jupyter**: Notebook environment

## 📝 Usage Examples

### For Infrastructure Planners
1. Select your target county
2. View the interactive map to see current EV distribution
3. Check the top cities ranking for charging station priority
4. Use the prediction tool to evaluate specific locations
5. Make data-driven decisions for charging station placement

### For Researchers
1. Run the analysis notebooks to understand EV adoption patterns
2. Experiment with different feature engineering approaches
3. Compare model performance across different algorithms
4. Analyze feature importance to understand key factors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of the FDM module mini project.

## 🙏 Acknowledgments

- Electric Vehicle Population Data source
- FDM module instructors
- Open source libraries and frameworks used

## 📞 Support

For questions or issues:
1. Check the notebook documentation
2. Review the error messages in the web app
3. Ensure all dependencies are installed correctly
4. Verify that all notebooks have been executed in order

---

**Smart Charge Locator** - Making EV infrastructure planning smarter with data science! 🔋⚡

---

## 🧭 Community & Governance

- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [Security Policy](./SECURITY.md)
- [License (MIT)](./LICENSE)
- Issue templates and PR template available under [./.github](./.github)

## ☁️ Deploying to Streamlit Community Cloud

The repo is ready to deploy on Streamlit Cloud.

Required files committed:
- `streamlit_app.py` (entrypoint)
- `app/app.py` (main app code)
- `requirements.txt` (runtime deps)
- `models/*.pkl` and `data/processed/*` used by the app

Steps:
1. Push your latest changes to GitHub.
2. Go to https://share.streamlit.io, sign in, and click New app.
3. Select this repo and branch, set Main file to `streamlit_app.py`.
4. Click Deploy.

Notes:
- Keep `requirements.txt` minimal to speed up builds. Use `requirements-dev.txt` locally for notebooks.
- Ensure the `models/` and `data/processed/` folders are tracked in Git and under 1 GB total. If large, consider storing smaller subsets or hosting assets externally.
- If Folium map fails to render due to serialization issues, the app automatically falls back to HTML rendering.
- Pin Python to 3.11 for Streamlit Cloud to avoid building native deps on 3.13:
   - `runtime.txt` should contain `python-3.11` (preferred by Streamlit Cloud)
   - `.python-version` should contain `3.11` (helps some builders pick the right version)
   - After pushing these files, open the app settings in Streamlit Cloud → Advanced → Clear cache, then Redeploy so the new Python is used.
