# 🔋 Smart Charge Locator

Live app: https://smart-charge-locator.streamlit.app

A machine learning project that predicts optimal locations for electric vehicle (EV) charging stations based on EV population data and geographic factors.

## 📋 Project Overview

This project analyzes Electric Vehicle Population Data to identify the most suitable cities for installing new charging stations. Users can input a county name to get predictions for cities within that county, helping infrastructure planners make data-driven decisions.

## 🎯 Key Features

- **Data Analysis**: Comprehensive analysis of EV distribution patterns
- **Machine Learning Models**: Multiple ML models (Linear Regression, Ridge, Random Forest, XGBoost)
- **Interactive Web App**: Streamlit-based interface for easy interaction
- **Geographic Visualization**: Interactive maps showing EV distribution and charging station suitability
- **County-based Predictions**: Input county name to get city-level predictions

## 🏗️ Project Structure

```
SMART_CHARGE_LOCATOR/
├── .venv/                          # Virtual environment
├── app/                            # Streamlit web application
│   └── app.py                     # Main application file
├── data/                          # Data directory
│   ├── processed/                 # Processed and cleaned data
│   ├── raw/                       # Original dataset
│   └── Electric_Vehicle_Population_Data.csv
├── models/                        # Trained ML models
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── notebooks/                     # Jupyter notebooks
│   ├── model_training/           # Model training notebooks
│   │   ├── 4.1_Linear_Regression.ipynb
│   │   ├── 4.2_Ridge_Regression.ipynb
│   │   ├── 4.3_Random_Forest.ipynb
│   │   └── 4.4_XGBoost.ipynb
│   ├── 01_Data_Loading_and_Cleaning.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   └── 03_Feature_Engineering.ipynb
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## 🚀 Getting Started

### For users (no install)

- Open the live app: https://smart-charge-locator.streamlit.app
- Choose a county from the left sidebar.
- Explore the interactive map and top cities table.
- Pick a city on the right and click Predict to see its Charging Score and visualizations.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd SMART_CHARGE_LOCATOR
   
   # Or simply navigate to the project directory
   cd SMART_CHARGE_LOCATOR
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Execute the data processing notebooks** (in order):
   ```bash
   # Start Jupyter Notebook
   jupyter notebook
   
   # Run notebooks in this order:
   # 1. 01_Data_Loading_and_Cleaning.ipynb
   # 2. 02_Exploratory_Data_Analysis.ipynb
   # 3. 03_Feature_Engineering.ipynb
   # 4. notebooks/model_training/4.1_Linear_Regression.ipynb
   # 5. notebooks/model_training/4.2_Ridge_Regression.ipynb
   # 6. notebooks/model_training/4.3_Random_Forest.ipynb
   # 7. notebooks/model_training/4.4_XGBoost.ipynb
   ```

2. **Launch the Streamlit web application (local)**
   ```bash
   # Windows PowerShell
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

### For developers

- Local setup
   - Create and activate a virtual environment
   - Install runtime dependencies: `pip install -r requirements.txt`
   - Ensure the following runtime assets exist (relative to repo root):
      - `data/processed/city_features_engineered.csv`
      - `data/processed/scaler.pkl`
      - `data/processed/feature_columns.pkl`
      - `models/xgboost.pkl`
      - (optional) other model files and metrics JSONs in `data/processed/`
   - Run locally: `streamlit run streamlit_app.py`

- Notebooks and full pipeline
   - Optional: `pip install -r requirements-dev.txt` for Jupyter and geo/plot libraries
   - Use the notebooks in `notebooks/` to regenerate features and train models

- Environment variables
   - `DATA_ROOT` (optional): If set, the app will look for data and models under this folder before falling back to current working directory and project root.

## 🧩 Troubleshooting

- File not found (e.g., city_features_engineered.csv)
   - Ensure the files listed above are present in the repository (they’re allowed by `.gitignore`).
   - On Streamlit Cloud, push these files to GitHub so they’re deployed with the app.
   - You can also set `DATA_ROOT` in the Streamlit app’s Settings → Advanced → Environment variables to point to the base folder containing `data/` and `models/`.

- Port already in use locally
   - Run on another port: `streamlit run streamlit_app.py --server.port 8503`

- Build errors on Streamlit Cloud involving pandas/numpy compilation
   - The current `requirements.txt` uses versions with prebuilt wheels compatible with Python 3.13. If the builder still tries to compile, clear cache and redeploy.
   - You can also keep `runtime.txt` = `python-3.11` and clear cache to force Python 3.11.

## 📊 Data Processing Pipeline

### 1. Data Loading and Cleaning (`01_Data_Loading_and_Cleaning.ipynb`)
- Loads the Electric Vehicle Population Data
- Handles missing values and data inconsistencies
- Creates additional features like vehicle age
- Extracts geographic coordinates
- Saves cleaned data for further processing

### 2. Exploratory Data Analysis (`02_Exploratory_Data_Analysis.ipynb`)
- Analyzes EV distribution by county and city
- Creates charging station suitability scores
- Generates interactive maps
- Identifies patterns in EV adoption
- Saves analysis results

### 3. Feature Engineering (`03_Feature_Engineering.ipynb`)
- Creates city-level aggregated features
- Engineers new features for machine learning
- Handles categorical variables
- Prepares training and test datasets
- Saves processed data for model training

### 4. Model Training (`notebooks/model_training/`)
- **Linear Regression**: Baseline model with interpretable coefficients
- **Ridge Regression**: Regularized linear model to prevent overfitting
- **Random Forest**: Ensemble method capturing non-linear relationships
- **XGBoost**: Gradient boosting for high performance predictions

## 🎮 Using the Web Application

1. **Select a Model**: Choose from available trained models
2. **Choose a County**: Select a county to analyze
3. **View the Map**: Interactive map showing EV distribution and charging scores
4. **Check Rankings**: See top cities by charging station suitability
5. **Make Predictions**: Select a specific city to get charging score predictions
6. **View Statistics**: County-level statistics and model performance metrics

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
