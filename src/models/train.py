from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from src.features.preprocess import build_preprocess_pipeline

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "processed")
os.makedirs(MODEL_DIR, exist_ok=True)


def select_features(df: pd.DataFrame):
    # Assumptions based on common EV datasets; adjust in notebook after EDA
    candidate_numeric = [
        col for col in df.columns if col.lower() in (
            "model_year", "electric_range", "base_msrp")
    ]
    # Location and type information for clustering
    candidate_categorical = [
        col for col in df.columns if col.lower() in (
            "city", "county", "state", "vehicle_location", "clean_alternative_fuel_vehicle_eligibility",
            "electric_vehicle_type", "electric_utility", "census_tract"
        )
    ]
    return candidate_numeric, candidate_categorical


def train_location_clusters(df: pd.DataFrame, k: int = 12, random_state: int = 42):
    num_cols, cat_cols = select_features(df)

    pre = build_preprocess_pipeline(num_cols, cat_cols)

    # KMeans on engineered features to capture demand hotspots
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)

    pipe = Pipeline([
        ("pre", pre),
        ("kmeans", model)
    ])

    X = df[num_cols + cat_cols].copy()
    pipe.fit(X)

    # Evaluate clustering compactness
    try:
        # Use transformed features for silhouette
        X_t = pipe.named_steps["pre"].transform(X)
        score = silhouette_score(X_t, pipe.named_steps["kmeans"].labels_, sample_size=10_000, random_state=42)
    except Exception:
        score = np.nan

    joblib.dump(pipe, os.path.join(MODEL_DIR, "location_cluster_model.joblib"))
    return pipe, score


if __name__ == "__main__":
    # This is an optional runner; real flow is in notebooks
    raise SystemExit("Use notebooks to prepare data, then import and call train_location_clusters().")
