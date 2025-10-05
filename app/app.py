import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Smart Charge Locator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed data and models with robust path resolution"""
    def resolve_path(rel_path: str) -> Path | None:
        # Candidates: explicit env override, CWD, repo root (two levels up from this file)
        env_root = os.environ.get("DATA_ROOT")
        if env_root:
            p = Path(env_root) / rel_path
            if p.exists():
                return p
        candidates = [
            Path.cwd() / rel_path,
            Path(__file__).resolve().parent.parent / rel_path,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def must_read_csv(rel_path: str) -> pd.DataFrame:
        p = resolve_path(rel_path)
        if p is None:
            bases = [str(Path.cwd()), str(Path(__file__).resolve().parent.parent)]
            raise FileNotFoundError(f"Missing {rel_path}. Looked under: {bases}")
        return pd.read_csv(p)

    def must_load_joblib(rel_path: str):
        p = resolve_path(rel_path)
        if p is None:
            bases = [str(Path.cwd()), str(Path(__file__).resolve().parent.parent)]
            raise FileNotFoundError(f"Missing {rel_path}. Looked under: {bases}")
        return joblib.load(p)

    try:
        # Load city features
        city_features = must_read_csv('data/processed/city_features_engineered.csv')

        # Load models
        models = {}
        model_names = ['linear_regression', 'ridge_regression', 'random_forest', 'xgboost']

        for model_name in model_names:
            model_path = f'models/{model_name}.pkl'
            try:
                models[model_name] = must_load_joblib(model_path)
            except FileNotFoundError:
                st.warning(f"Model {model_name} not found at {model_path}. If you don't need it, you can ignore this warning; the app defaults to XGBoost.")

        # Load scaler and feature columns
        scaler = must_load_joblib('data/processed/scaler.pkl')
        feature_columns = must_load_joblib('data/processed/feature_columns.pkl')

        # Load performance metrics
        performance_metrics = {}
        for model_name in model_names:
            metrics_rel = f'data/processed/{model_name}_performance_metrics.json'
            p = resolve_path(metrics_rel)
            if p and p.exists():
                try:
                    with open(p, 'r') as f:
                        performance_metrics[model_name] = json.load(f)
                except Exception:
                    pass

        return city_features, models, scaler, feature_columns, performance_metrics
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Provide quick diagnostics panel
        with st.expander("Diagnostics: file locations"):
            st.write({
                "cwd": str(Path.cwd()),
                "repo_root": str(Path(__file__).resolve().parent.parent),
                "exists_city_features": (Path.cwd() / 'data/processed/city_features_engineered.csv').exists() or (Path(__file__).resolve().parent.parent / 'data/processed/city_features_engineered.csv').exists(),
            })
        return None, None, None, None, None

def predict_charging_score(city_data, model, scaler, feature_columns):
    """Predict charging score for a city"""
    try:
        # Prepare features
        features = city_data[feature_columns].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_city_map(city_features, selected_county=None):
    """Create an interactive map of cities"""
    # Filter by county if selected
    if selected_county and selected_county != "All Counties":
        filtered_data = city_features[city_features['County'] == selected_county]
    else:
        filtered_data = city_features
    
    # Create map centered on Washington state
    m = folium.Map(
        location=[47.7511, -120.7401],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add markers for cities
    for idx, row in filtered_data.iterrows():
        if pd.notna(row['Latitude_mean']) and pd.notna(row['Longitude_mean']):
            # Color based on charging score
            score = row['Charging_Score']
            if score > 100:
                color = 'red'
            elif score > 50:
                color = 'orange'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=[row['Latitude_mean'], row['Longitude_mean']],
                radius=min(row['EV_Count']/20, 15),
                popup=f"""
                <b>{row['City']}, {row['County']}</b><br>
                EVs: {int(row['EV_Count'])}<br>
                Charging Score: {row['Charging_Score']:.1f}<br>
                Avg Range: {row['Avg_Range']:.0f} miles
                """,
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header"> Smart Charge Locator</h1>', unsafe_allow_html=True)
    st.markdown("### Optimal locations for EV charging stations in washington")
    
    # Load data
    city_features, models, scaler, feature_columns, performance_metrics = load_data()
    
    if city_features is None:
        st.error("Failed to load data. Please ensure all data files are available.")
        return
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Use XGBoost model only (remove model selector)
    selected_model = 'xgboost'
    if models.get('xgboost') is None:
        st.error("XGBoost model not available. Please run the model training notebooks first.")
        return
    
    # County selection
    counties = ["All Counties"] + sorted(city_features['County'].unique().tolist())
    selected_county = st.sidebar.selectbox("Select County", counties)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📍 Interactive Map")
        
        # Create and display map
        city_map = create_city_map(city_features, selected_county)
        # Basic sanity check
        if not isinstance(city_map, folium.Map):
            st.error(f"Expected folium.Map but got {type(city_map)}")
        else:
            # Inspect children for callables (common source of JSON serialization issues)
            problematic = []
            try:
                for k, v in city_map._children.items():
                    if callable(v):
                        problematic.append((k, type(v)))
            except Exception:
                # If introspection fails, continue to attempting to render
                problematic = []

            if problematic:
                st.warning("Found non-serializable objects inside the map (callables). Falling back to HTML render and logging details.")
                st.write(problematic)
                try:
                    components.html(city_map._repr_html_(), height=500)
                except Exception as e:
                    st.error(f": {e}")
            else:
                # Try the normal st_folium render and fall back to HTML on error with diagnostics
                try:
                    st_folium(city_map, width=700, height=500)
                except Exception as e:
                    # Suppress the known folium->streamlit JSON serialization error message
                    msg = str(e)
                    if ("Object of type function is not JSON serializable" in msg
                            or "Could not convert component args to JSON" in msg):
                        # do not display the large TypeError to the user; silently fallback
                        pass
                    else:
                        st.error(f"Map rendering error: {e}")
                        # fallback to HTML without showing an additional info message

                    try:
                        components.html(city_map._repr_html_(), height=500)
                    except Exception as e2:
                        st.error(f"HTML fallback failed: {e2}")
        
        # City rankings
        st.header("Top Cities for charging stations")
        
        # Filter data
        if selected_county and selected_county != "All Counties":
            filtered_data = city_features[city_features['County'] == selected_county]
        else:
            filtered_data = city_features
        
        # Display top cities
        top_cities = filtered_data.nlargest(10, 'Charging_Score')[
            ['City', 'County', 'EV_Count', 'Avg_Range', 'Avg_MSRP', 'Charging_Score']
        ].round(2)
        
        st.dataframe(
            top_cities,
            use_container_width=True,
            column_config={
                "City": "City",
                "County": "County", 
                "EV_Count": "EV Count",
                "Avg_Range": "Avg Range (miles)",
                "Avg_MSRP": "Avg MSRP ($)",
                "Charging_Score": "Charging Score"
            }
        )

        # Collapsible visualization section for station owners (hidden by default)
        with st.expander("Visualizations for selected city:", expanded=True):
            try:
                viz_city = st.session_state.get('selected_city')
                viz_county = selected_county
                if not viz_city:
                    st.info('Select a city from the right panel to view visualizations.')
                else:
                    # Fetch city row
                    city_row = city_features[
                        (city_features['City'] == viz_city) & 
                        (city_features['County'] == viz_county if viz_county != "All Counties" else True)
                    ].iloc[0]

                    # Compute prediction on-demand
                    pred = predict_charging_score(city_row, models[selected_model], scaler, feature_columns)

                    # Select candidate features (limit to top 4 for clarity for non-technical users)
                    candidate_feats = []
                    if isinstance(feature_columns, (list, tuple)) and len(feature_columns) > 0:
                        candidate_feats = [f for f in feature_columns if f in city_features.columns]
                    defaults = ['EV_Count', 'Avg_Range', 'Avg_MSRP']
                    for d in defaults:
                        if d not in candidate_feats and d in city_features.columns:
                            candidate_feats.append(d)
                    feats = candidate_feats[:4]  # keep chart simple and readable

                    county_df = city_features[city_features['County'] == viz_county] if viz_county != 'All Counties' else city_features

                    if len(feats) > 0:
                        county_means = county_df[feats].mean()
                        city_vals = city_row[feats]
                        comp_df = pd.DataFrame({
                            'Feature': feats,
                            'City': [city_vals[f] if pd.notna(city_vals[f]) else 0 for f in feats],
                            'County Avg': [county_means[f] if pd.notna(county_means[f]) else 0 for f in feats]
                        })

                        # Create a simple horizontal grouped bar chart with numeric labels
                        melt_df = comp_df.melt(id_vars='Feature', value_vars=['City', 'County Avg'], var_name='Series', value_name='Value')
                        fig_comp = px.bar(melt_df, x='Value', y='Feature', color='Series', orientation='h', barmode='group',
                                          title='Top features: City vs County average',
                                          color_discrete_map={'City':'#ff9aa2','County Avg':'#b5ead7'},
                                          text='Value')
                        fig_comp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig_comp.update_layout(yaxis={'categoryorder':'total ascending'},
                                               title_x=0.02,
                                               legend_title_text='',
                                               font=dict(size=12))
                        st.plotly_chart(fig_comp, use_container_width=True)
                        st.caption("Comparison of a few key metrics for the selected city against the county average.")

                    if 'Charging_Score' in county_df.columns:
                        # Simpler pastel histogram with a clear marker for the selected city
                        fig_hist = px.histogram(county_df, x='Charging_Score', nbins=20, title='Charging Score Distribution (County)',
                                                color_discrete_sequence=['#b5ead7'])
                        fig_hist.add_vline(x=pred, line_dash='dash', line_color='#0b3d91', annotation_text='Selected city', annotation_position='top')
                        fig_hist.update_layout(xaxis_title='Charging Score', yaxis_title='Number of Cities', font=dict(size=12))
                        st.plotly_chart(fig_hist, use_container_width=True)
                        st.caption(' how the selected city compares to other cities in the county. The dashed line is the city score.')

                        def tier_label(s):
                            if s > 100:
                                return 'High'
                            elif s > 50:
                                return 'Medium'
                            else:
                                return 'Low'

                        tiers = county_df['Charging_Score'].dropna().apply(tier_label).value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
                        pie_df = pd.DataFrame({'Tier': tiers.index, 'Count': tiers.values})
                        # soft pastel color palette for better aesthetics
                        fig_pie = px.pie(pie_df, names='Tier', values='Count', title='County Priority Breakdown', color='Tier',
                                         color_discrete_map={'High':'#ff9aa2','Medium':'#ffd3b6','Low':'#b5ead7'})
                        st.plotly_chart(fig_pie, use_container_width=True)

            except Exception as viz_e:
                print('Visualization error:', viz_e)
    
    with col2:
        st.header("")
        
        if selected_model in performance_metrics:
            metrics = performance_metrics[selected_model]
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{metrics['model_name']}</h4>
                <p><strong>Test R²:</strong> {metrics['test_r2']:.3f}</p>
                <p><strong>Test MSE:</strong> {metrics['test_mse']:.3f}</p>
                <p><strong>Test MAE:</strong> {metrics['test_mae']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction section
        st.header("Check Location Priority")
        
        # Get unique cities for the selected county
        if selected_county and selected_county != "All Counties":
            available_cities = city_features[city_features['County'] == selected_county]['City'].unique()
        else:
            available_cities = city_features['City'].unique()

        selected_city = st.selectbox("Select City", sorted(available_cities), key='selected_city')

        if st.button("Predict", type="primary"):
            # Get city data
            city_data = city_features[
                (city_features['City'] == selected_city) & 
                (city_features['County'] == selected_county if selected_county != "All Counties" else True)
            ].iloc[0]
            
            # Make prediction
            prediction = predict_charging_score(
                city_data, models[selected_model], scaler, feature_columns
            )
            
            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Charging Score</h3>
                    <h2 style="color: #0b3d91; font-weight: 700;">{prediction:.2f}</h2>
                    <p style="color: #0b2a5a; font-weight:600;">for {selected_city}, {selected_county}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                if prediction > 100:
                    st.success("🚀 High Priority: Excellent location for charging station!")
                elif prediction > 50:
                    st.info("⚡ Medium Priority: Good location for charging station")
                else:
                    st.warning("🔋 Low Priority: Consider other locations first")
        
        # Statistics
        st.header("County Statistics")
        
        if selected_county and selected_county != "All Counties":
            county_data = city_features[city_features['County'] == selected_county]
        else:
            county_data = city_features
        
        total_evs = county_data['EV_Count'].sum()
       # avg_score = county_data['Charging_Score'].mean()
        num_cities = len(county_data)
        
        st.metric("Total EVs", f"{total_evs:,}")
        #st.metric("Average Score", f"{avg_score:.1f}")
        st.metric("Number of Cities", num_cities)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Smart Charge Locator - Predicting optimal EV charging station locations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
