import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk
import os
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")

# --- MAPBOX TOKEN SETUP ---
# Prefer environment variable; fallback to Streamlit secrets if available.
MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY")
try:
    if not MAPBOX_TOKEN and hasattr(st, "secrets") and "MAPBOX_API_KEY" in st.secrets:
        MAPBOX_TOKEN = st.secrets["MAPBOX_API_KEY"]
except Exception:
    # st.secrets may raise in some local contexts; ignore and continue
    pass

if MAPBOX_TOKEN:
    try:
        pdk.settings.mapbox_api_key = MAPBOX_TOKEN
    except Exception:
        pass
else:
    st.warning("Mapbox token not found. Set MAPBOX_API_KEY in your environment or Streamlit secrets for proper map rendering.")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    processed_df = pd.read_csv('data/processed/model_ready_ev_data.csv')
    coords_df = pd.read_csv('data/Washington_State_Coordinates.csv')
    full_data = pd.merge(processed_df, coords_df, on='Postal Code', how='left').dropna(subset=['lat', 'lon'])
    return full_data

def load_model(model_filename):
    path = os.path.join('models', model_filename)
    with open(path, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

data = load_data()

# --- MODEL SELECTION (AUTO: BEST ONLY) ---
st.sidebar.header("Controls")
model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
if not model_files:
    st.error("No model files found in '/models'. Please run training notebooks first.")
    st.stop()

def evaluate_model(_model, _data: pd.DataFrame):
    # Use the latest historical year as a hold-out test set
    latest_year = int(_data['Model Year'].max())
    test_df = _data[_data['Model Year'] == latest_year]
    if test_df.empty:
        # Fallback to entire dataset if latest year slice is empty
        test_df = _data
    X_test = test_df[['Postal Code', 'Model Year', 'Prev_Year_EV_Count', 'Year_Delta']]
    y_test = test_df['EV_Count']
    try:
        y_pred = _model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        return {"mae": mae, "r2": r2}
    except Exception as e:
        return {"mae": float('inf'), "r2": float('-inf'), "error": str(e)}

evaluations = []
for mf in model_files:
    try:
        m = load_model(mf)
        metrics = evaluate_model(m, data)
        evaluations.append({"file": mf, "metrics": metrics, "model": m})
    except Exception as e:
        evaluations.append({"file": mf, "metrics": {"mae": float('inf'), "r2": float('-inf'), "error": str(e)}, "model": None})

# Pick best by highest R2, tie-break by lowest MAE
evaluations = [e for e in evaluations if e["model"] is not None]
if not evaluations:
    st.error("All models failed to load or evaluate. Please retrain models.")
    st.stop()

best = sorted(evaluations, key=lambda x: (-x["metrics"]["r2"], x["metrics"]["mae"]))[0]
selected_model_file = best["file"]
model = best["model"]
latest_year_in_data = data['Model Year'].max()
prediction_year = st.sidebar.slider("Select Year for Prediction", int(latest_year_in_data), 2030, 2025)
st.sidebar.success(f"Using best model: {selected_model_file}")
st.sidebar.caption(f"R²: {best['metrics']['r2']:.3f} | MAE: {best['metrics']['mae']:.1f} (evaluated on year {int(latest_year_in_data)})")

# --- PREDICTION FUNCTION ---
def predict_future(start_year, end_year):
    coords = data[['Postal Code', 'lat', 'lon']].drop_duplicates()
    prediction_df = data[data['Model Year'] == start_year].copy()
    all_predictions = [prediction_df]
    for year in range(start_year + 1, end_year + 1):
        last_year_df = all_predictions[-1]
        features_to_predict = last_year_df[['Postal Code']].copy()
        features_to_predict['Model Year'] = year
        features_to_predict['Prev_Year_EV_Count'] = last_year_df['EV_Count']
        features_to_predict['Year_Delta'] = year - data['Model Year'].min()
        predicted_counts = model.predict(features_to_predict)
        current_year_predictions = features_to_predict.copy()
        current_year_predictions['EV_Count'] = np.round(predicted_counts).astype(int)
        current_year_predictions['EV_Count'] = np.maximum(current_year_predictions['EV_Count'], current_year_predictions['Prev_Year_EV_Count'])
        current_year_predictions = pd.merge(current_year_predictions, coords, on='Postal Code', how='left')
        all_predictions.append(current_year_predictions)
    return pd.concat(all_predictions)

# --- MAIN PAGE LAYOUT ---
st.title("⚡️ Electric Vehicle Hotspot Predictor")
model_name_display = selected_model_file.replace('_', ' ').replace('.pkl', '').title()
st.markdown(f"Using the automatically selected best model: **{model_name_display}**.")

predicted_data = predict_future(latest_year_in_data, prediction_year)
display_data = predicted_data[predicted_data['Model Year'] == prediction_year].dropna(subset=['lat', 'lon'])

st.header(f"Predicted EV Hotspots for {prediction_year}")

# --- MAP & TABLE ---
layer = pdk.Layer("HexagonLayer", data=display_data[['lon', 'lat', 'EV_Count']], get_position=['lon', 'lat'], get_weight='EV_Count', auto_highlight=True, elevation_scale=50, pickable=True, elevation_range=[0, 3000], extruded=True, coverage=1)
view_state = pdk.ViewState(longitude=-120.74, latitude=47.75, zoom=6, pitch=45)
st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[layer], tooltip={"text": "{elevationValue} EVs"}))
st.subheader("Top 10 Predicted Hotspots by Postal Code")
st.dataframe(display_data[['Postal Code', 'EV_Count']].nlargest(10, 'EV_Count').set_index('Postal Code'))