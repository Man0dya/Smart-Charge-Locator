import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

IN_PATH = Path('data/processed/model_ready_city_ev.csv')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_OUT = Path('data/processed/model_eval.csv')

# Try optional XGBoost candidate without hard dependency
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    X = df[['Model Year','Prev_Year_EV_Count','Year_Delta']].copy()
    y = df['EV_Count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }
    if HAS_XGB:
        # Reasonable default; tree_method='hist' for speed, n_estimators moderate
        candidates['XGBoost'] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

    # Where to save each individual model artifact
    out_paths = {
        'LinearRegression': MODEL_DIR / 'linear_regression.pkl',
        'Ridge': MODEL_DIR / 'ridge_regression.pkl',
        'RandomForest': MODEL_DIR / 'random_forest.pkl',
    }
    if 'XGBoost' in candidates:
        out_paths['XGBoost'] = MODEL_DIR / 'xgboost.pkl'

    records = []
    best_name, best_model, best_score, best_mae = None, None, -np.inf, np.inf
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        records.append({'model': name, 'r2': r2, 'mae': mae})

        # Save each trained model artifact
        try:
            joblib.dump(model, out_paths[name])
        except Exception:
            # Non-fatal: continue even if a particular dump fails
            pass

        if r2 > best_score or (r2 == best_score and mae < best_mae):
            best_name, best_model, best_score, best_mae = name, model, r2, mae

    pd.DataFrame(records).to_csv(EVAL_OUT, index=False)
    joblib.dump(best_model, MODEL_DIR / 'best_model.pkl')

if __name__ == '__main__':
    main()


