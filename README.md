
# EV Charging Location Optimizer

This project helps electric vehicle (EV) charging station manufacturers identify optimal locations for new charging stations, using real-world EV population data and machine learning. It provides a user-friendly frontend for exploring recommendations and planning deployments.

## Features

- **Data Preprocessing & Analysis:** Cleans, aggregates, and profiles EV registration data.
- **Demand Modeling:** Predicts required chargers and identifies high-demand areas.
- **Optimal Placement:** Uses clustering and optimization to suggest station sites.
- **Streamlit App:** Interactive dashboard for manufacturers to view recommendations and download results.
- **Reusable Codebase:** Modular Python scripts for data, modeling, and utility functions.
- **CLI Tool:** Command-line interface for running key pipeline steps.

## Project Structure

```
.
├── app/                # Streamlit frontend
│   ├── app.py          # Main app code
│   └── utils.py        # App utilities
├── artifacts/          # Exported CSVs, models, and maps for the app
│   ├── stations.csv
│   ├── aggregated_demand.csv
│   └── ... (imputers, encoders, maps)
├── data/
│   ├── processed/
│   │   └── raw_snapshot.parquet
│   └── interim/        # (empty, for intermediate files)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Main workflow notebook
│   ├── artifacts/      # Notebook-generated artifacts
│   ├── models/         # Trained model files
│   └── data/           # Notebook-generated data
├── src/
│   ├── cli.py          # Command-line pipeline runner
│   ├── features/       # Feature engineering
│   ├── models/         # Model training/inference
│   └── utils/          # I/O and helpers
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── tasks.json          # Project milestone tracker
```

## Key Data & Artifacts

- **Electric_Vehicle_Population_Data.csv:** Raw input data (not included, place in project root).
- **data/processed/raw_snapshot.parquet:** Fast, columnar snapshot of raw data for analysis.
- **notebooks/data/processed/agg_city_county.parquet:** Aggregated demand by city/county.
- **notebooks/data/processed/agg_targets.parquet:** Demand targets (required chargers, high demand flags).
- **artifacts/stations.csv:** Recommended station locations for manufacturers.
- **artifacts/aggregated_demand.csv:** Aggregated demand for app and download.
- **artifacts/stations_map.html:** Interactive map of suggested sites.

## How to Run

1. **Setup Environment**
	 - Create a Python 3.10+ virtual environment:
		 ```
		 python -m venv .venv
		 .\.venv\Scripts\activate
		 ```
	 - Install dependencies:
		 ```
		 pip install -r requirements.txt
		 ```

2. **Prepare Data**
	 - Place `Electric_Vehicle_Population_Data.csv` in the project root.
	 - Run the notebook `notebooks/01_data_preprocessing.ipynb` step by step to generate processed data and artifacts.

3. **Launch the App**
	 - After running the notebook, start the Streamlit app:
		 ```
		 streamlit run app/app.py
		 ```
	 - The app will show recommended cities and station sites, with options to filter, view maps, and download results.

4. **Command-Line Usage**
	 - Run pipeline steps via CLI:
		 ```
		 python src/cli.py --step prep|train|place|export
		 ```

## For Developers

- All code is modular and reusable. See `src/` for feature engineering, modeling, and utility functions.
- Add unit tests in the `tests/` folder as needed.
- Extend the notebook for new data sources or modeling approaches.

## For Manufacturers

- Use the Streamlit app to view and download recommended locations for new EV charging stations.
- All technical terms are simplified for clarity.
- Download CSVs for planning and deployment.


