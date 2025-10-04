from __future__ import annotations
import os
import joblib
import pandas as pd
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "processed", "location_cluster_model.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict_cluster(df: pd.DataFrame) -> List[int]:
    model = load_model()
    X = df[[c for c in df.columns if c in model.named_steps['pre'].get_feature_names_out()]]
    return model.predict(X)
