from __future__ import annotations
import numpy as np
from sklearn.neighbors import BallTree

# Build and query a BallTree with haversine metric

def build_tree(latitudes, longitudes):
    radians = np.deg2rad(np.c_[latitudes, longitudes])
    return BallTree(radians, metric='haversine')


def query_tree(tree, q_lat, q_lon, k=3):
    q = np.deg2rad(np.array([[q_lat, q_lon]]))
    dist, idx = tree.query(q, k=k)
    # Convert radians distance to km (Earth radius ~6371km)
    return dist[0] * 6371.0, idx[0]
