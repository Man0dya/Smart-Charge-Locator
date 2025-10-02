import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

IN_PATH = Path('data/processed/city_ev_agg.csv')
OUT_PATH = Path('data/processed/city_clustering.csv')

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")
    city = pd.read_csv(IN_PATH)
    city = city.dropna(subset=['EV_Density_Proxy'])
    X = np.log1p(city[['EV_Density_Proxy']].values)

    best = {'method': None, 'labels': None, 'score': -1.0}
    for k in range(3, 7):
        try:
            km = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            if score > best['score']:
                best = {'method': f'KMeans(k={k})', 'labels': labels, 'score': score}
        except Exception:
            pass

    for eps in [0.1, 0.2, 0.3]:
        try:
            db = DBSCAN(eps=eps, min_samples=3)
            labels = db.fit_predict(X)
            if len(np.unique(labels)) > 1 and np.max(labels) >= 1:
                score = silhouette_score(X, labels)
                if score > best['score']:
                    best = {'method': f'DBSCAN(eps={eps})', 'labels': labels, 'score': score}
        except Exception:
            pass

    city['Cluster'] = best['labels'] if best['labels'] is not None else 0
    city['Silhouette'] = best['score']
    city['Method'] = best['method']
    city.to_csv(OUT_PATH, index=False)

if __name__ == '__main__':
    main()


