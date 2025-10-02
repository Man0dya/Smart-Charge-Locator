import pandas as pd
import numpy as np
from pathlib import Path
import joblib

MODEL = Path('models/best_model.pkl')
DF_FEATS = Path('data/processed/model_ready_city_ev.csv')
DF_AGG = Path('data/processed/city_ev_agg.csv')
EVAL = Path('data/processed/model_eval.csv')

def main() -> None:
    if not (MODEL.exists() and DF_FEATS.exists() and DF_AGG.exists() and EVAL.exists()):
        raise FileNotFoundError('Missing one or more inputs for report')

    best = joblib.load(MODEL)
    feats = pd.read_csv(DF_FEATS)
    agg = pd.read_csv(DF_AGG)
    metrics = pd.read_csv(EVAL)

    latest_year = int(feats['Model Year'].max())
    base = feats[feats['Model Year'] == latest_year].copy()
    X = base[['Model Year','Prev_Year_EV_Count','Year_Delta']]
    base['Pred_EV_Count'] = best.predict(X)

    out = base[['City','Pred_EV_Count']].groupby('City', as_index=False).sum()
    out = out.merge(agg[['City','EV_Density_Proxy','EV_Count_Total']], on='City', how='left')

    top10 = out.sort_values('EV_Density_Proxy', ascending=False).head(10)
    print('Model evaluation (held-out):')
    print(metrics.sort_values('r2', ascending=False).head(3))
    print('\nTop 10 cities by density proxy and predicted EV count:')
    print(top10[['City','EV_Density_Proxy','Pred_EV_Count']])

if __name__ == '__main__':
    main()


