import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="EV Charger Planner", layout="wide")

st.title("EV Charger Planner")


@st.cache_data
def load_data():
    agg_path = Path('data/processed/agg_city_county.parquet')
    if agg_path.exists():
        agg = pd.read_parquet(agg_path)
    else:
        agg_csv = Path('artifacts/aggregated_demand.csv')
        agg = pd.read_csv(agg_csv) if agg_csv.exists() else pd.DataFrame()

    stations_csv = Path('artifacts/stations.csv')
    stations = pd.read_csv(stations_csv) if stations_csv.exists() else pd.DataFrame()
    return agg, stations

    # ...existing code...



agg, stations = load_data()

st.subheader("Station Placement Planner")
st.write("Below are recommended locations for new EV charging stations, based on demand analysis. All technical terms have been simplified for clarity.")

# Section: Top cities to place new chargers
st.markdown("### Recommended Cities for New Charging Stations")
if agg.empty:
    st.warning("No aggregated demand found. Please run the notebook to export artifacts/aggregated_demand.csv or data/processed/agg_city_county.parquet.")
else:
    # Determine available columns
    lat_col = next((c for c in ['centroid_lat', 'latitude', 'lat'] if c in agg.columns), None)
    lon_col = next((c for c in ['centroid_lon', 'longitude', 'lon'] if c in agg.columns), None)

    with st.container(border=True):
        # Controls row 1: region filter (county only) and demand toggle
        c1, c2 = st.columns([2,1])
        with c1:
            counties = sorted(agg['county'].dropna().unique().tolist()) if 'county' in agg.columns else []
            county_sel = st.multiselect("Filter by county", counties, default=[])
        with c2:
            only_hd = st.checkbox("High demand only", value=('high_demand' in agg.columns))

        # Controls row 2: ranking and count
        c3, c4 = st.columns([2,2])
        with c3:
            demand_metric = st.selectbox(
                "Demand metric",
                options=[opt for opt in ['required_chargers', 'ev_count'] if opt in agg.columns],
                index=0 if 'required_chargers' in agg.columns else 1,
                help="Metric representing demand to prioritize."
            )
        with c4:
            max_n = int(min(200, len(agg))) if len(agg) > 0 else 1
            n_top = st.slider("How many stations to plan?", 1, max_n, min(10, max_n))

        # Build candidate set with filters
        candidates = agg.copy()
        if only_hd and 'high_demand' in candidates.columns:
            candidates = candidates[candidates['high_demand'] == 1]
        if county_sel and 'county' in candidates.columns:
            candidates = candidates[candidates['county'].isin(county_sel)]

        # Require coords for mapping; ranking is demand-only
        if lat_col and lon_col:
            candidates = candidates.dropna(subset=[lat_col, lon_col]).copy()

        # Rank and select top N (by demand metric only)
        if demand_metric in candidates.columns:
            top_df = candidates.sort_values(demand_metric, ascending=False).head(n_top).copy()
        else:
            top_df = candidates.head(n_top).copy()

        # Map and details
        if not top_df.empty and lat_col and lon_col:
            map_df = top_df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
            st.map(map_df[['lat', 'lon']])
        else:
            st.info("No mappable city centroids found in the selection.")

        # Simplified columns and friendly names
        col_map = {
            'city': 'City',
            'county': 'County',
            'ev_count': 'Number of EVs',
            'required_chargers': 'Suggested Chargers',
            'high_demand': 'High Demand Area',
        }
        show_cols = [c for c in ['city','county','ev_count','required_chargers','high_demand'] if c in top_df.columns]
        if lat_col and lon_col:
            show_cols += [lat_col, lon_col]
            col_map[lat_col] = 'Latitude'
            col_map[lon_col] = 'Longitude'
        # Rename columns for display
        display_df = top_df[show_cols].rename(columns=col_map).reset_index(drop=True)
        st.dataframe(display_df)

        st.download_button(
            "Download recommended cities (CSV)",
            data=display_df.to_csv(index=False),
            file_name="recommended_cities_new_chargers.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("### Suggested Locations for New Charging Stations")
st.write("These locations are recommended based on demand and geographic analysis. The selection method shows how each site was chosen.")
if not stations.empty:
    # Simplify columns and rename for manufacturers
    col_map = {
        'station_id': 'Suggested Site ID',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'method': 'Selection Method',
    }
    show_cols = [c for c in ['station_id','latitude','longitude','method'] if c in stations.columns]
    display_stations = stations[show_cols].rename(columns=col_map)
    st.map(display_stations.rename(columns={'Latitude':'lat','Longitude':'lon'}))
    st.dataframe(display_stations)
    st.download_button(
        "Download suggested station sites (CSV)",
        data=display_stations.to_csv(index=False),
        file_name="suggested_station_sites.csv",
        mime="text/csv"
    )
else:
    st.info("No suggested station sites found. Please run the analysis to generate artifacts/stations.csv.")
