Smart-Charge-Locator (Remade)
=============================

Goal
----
Identify the top 10 Washington State cities with the highest density of EV owners and support placement of new charging stations. Deliver a notebook-driven workflow that preprocesses data, clusters cities, trains predictive models, and reports results.

Workflow
--------
1) Preprocessing (01_Preprocessing.ipynb)
   - Ingest `data/raw/Electric_Vehicle_Population_Data.xlsx`
   - Build city-year dataset and save:
     - `data/processed/city_ev_agg.csv`
     - `data/processed/model_ready_city_ev.csv`
2) Clustering (02_Clustering.ipynb)
   - Cluster cities by EV density proxy using KMeans/DBSCAN
   - Save `data/processed/city_clustering.csv`, report silhouette score
3) Model Training (03_Model_Training.ipynb)
   - Train Linear/Ridge/RandomForest regressors on city-year features
   - Save `models/best_model.pkl` and `data/processed/model_eval.csv`
4) Prediction & Report (04_Prediction_Report.ipynb)
   - Load best model, predict for a target year
   - Show accuracy metrics and top-10 city rankings with plots

Setup (Windows PowerShell)
--------------------------
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Data
----
- Place the raw Excel at `data/raw/Electric_Vehicle_Population_Data.xlsx`.
- Outputs are written to `data/processed/` and `models/`.

Notes
-----
- EV density proxy is computed as EV registrations per unique ZIP within a city (fallback when population/area data are unavailable). Replace with census population or city area when available to compute true density.
- Clustering quality is validated with silhouette score; the workflow chooses the best among tried configurations.

License
-------
For internal project use.

