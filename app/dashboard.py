import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Smart Charge Locator", layout="wide")
st.title("⚡ Smart Charge Locator – City Insights")

METRICS = Path('data/processed/model_eval.csv')
AGG = Path('data/processed/city_ev_agg.csv')
CLUSTERS = Path('data/processed/city_clustering.csv')
MODEL = Path('models/best_model.pkl')
FEATS = Path('data/processed/model_ready_city_ev.csv')

@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame()
    agg = pd.read_csv(AGG) if AGG.exists() else pd.DataFrame()
    clusters = pd.read_csv(CLUSTERS) if CLUSTERS.exists() else pd.DataFrame()
    feats = pd.read_csv(FEATS) if FEATS.exists() else pd.DataFrame()
    return metrics, agg, clusters, feats

metrics, agg, clusters, feats = load_data()

if feats.empty or agg.empty or metrics.empty or not MODEL.exists():
    st.error("Required artifacts missing. Please run the scripts: preprocess, cluster, train, report.")
    st.stop()

# Determine best model from metrics
best_row = metrics.sort_values(['r2', 'mae'], ascending=[False, True]).iloc[0]
best_model_name = str(best_row['model'])

# Build model selection options based on available artifacts
MODEL_DIR = Path('models')
candidate_paths = {
    f"Best ({best_model_name})": MODEL,
    'LinearRegression': MODEL_DIR / 'linear_regression.pkl',
    'Ridge': MODEL_DIR / 'ridge_regression.pkl',
    'RandomForest': MODEL_DIR / 'random_forest.pkl',
    'XGBoost': MODEL_DIR / 'xgboost.pkl',
}
options = [name for name, p in candidate_paths.items() if p.exists()]
default_index = 0 if options else -1

st.subheader("Model selection")
selected_label = st.selectbox("Choose a model to use for predictions", options, index=default_index)
selected_path = candidate_paths[selected_label]
selected_name_for_metrics = best_model_name if selected_label.startswith('Best') else selected_label

@st.cache_resource
def load_model(path_str: str):
    return joblib.load(path_str)

model = load_model(str(selected_path))

# Prominent model accuracy section (selected model)
sel_row = metrics[metrics['model'] == selected_name_for_metrics]
if sel_row.empty:
    sel_r2, sel_mae = None, None
else:
    sel_r2 = float(sel_row.iloc[0]['r2'])
    sel_mae = float(sel_row.iloc[0]['mae'])

st.subheader("Model Performance (selected)")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.metric(label="Model", value=selected_name_for_metrics)
with mc2:
    st.metric(label="R² (higher is better)", value=(f"{sel_r2:.3f}" if sel_r2 is not None else "N/A"))
with mc3:
    st.metric(label="MAE (lower is better)", value=(f"{sel_mae:,.0f}" if sel_mae is not None else "N/A"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Evaluation")
    st.dataframe(metrics.sort_values('r2', ascending=False).reset_index(drop=True))
with col2:
    st.subheader("Clustering Summary")
    if not clusters.empty:
        st.caption(clusters[['Method','Silhouette']].drop_duplicates().to_string(index=False))

st.divider()
st.subheader("Top Cities by EV Density Proxy")
top10 = agg.sort_values('EV_Density_Proxy', ascending=False).head(10)
st.dataframe(
    top10[['City','EV_Density_Proxy','EV_Count_Total']]
    .rename(columns={
        'EV_Density_Proxy': 'EV_Density_Proxy (EVs per unique ZIP)',
        'EV_Count_Total': 'EV_Count_Total (total EVs)'
    })
    .set_index('City')
)

st.divider()
st.subheader("Predict for Target Year")
latest_year = int(feats['Model Year'].max())
target_year = st.slider("Target Year", min_value=int(feats['Model Year'].min()), max_value=latest_year, value=latest_year)

base = feats[feats['Model Year'] == target_year].copy()
if base.empty:
    st.info("No data for selected year.")
else:
    X = base[['Model Year','Prev_Year_EV_Count','Year_Delta']]
    base['Pred_EV_Count'] = model.predict(X)
    city_preds = base.groupby('City', as_index=False)['Pred_EV_Count'].sum()
    out = city_preds.merge(agg[['City','EV_Density_Proxy']], on='City', how='left')
    top10_pred = out.sort_values('EV_Density_Proxy', ascending=False).head(10)
    st.dataframe(
        top10_pred.rename(columns={
            'EV_Density_Proxy': 'EV_Density_Proxy (EVs per unique ZIP)'
        }).set_index('City')
    )


