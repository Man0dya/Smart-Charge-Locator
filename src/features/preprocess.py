from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols: list[str]):
        self.text_cols = text_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.text_cols:
            if c in X.columns:
                X[c] = (
                    X[c]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"\s+", " ", regex=True)
                )
        return X


class ZipCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, zip_col: str):
        self.zip_col = zip_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.zip_col in X.columns:
            X[self.zip_col] = (
                X[self.zip_col]
                .astype(str)
                .str.extract(r"(\d{5})", expand=False)
            )
        return X


def build_preprocess_pipeline(numeric_features: list[str], categorical_features: list[str], ordinal_features: list[str] | None = None):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers)
    return preprocessor
