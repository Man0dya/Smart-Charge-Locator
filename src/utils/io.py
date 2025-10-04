from __future__ import annotations
import os
import pandas as pd
from typing import Iterator, Optional

RAW_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Electric_Vehicle_Population_Data.csv')
INTERIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'interim')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def read_ev_csv_chunks(chunksize: int = 100_000, usecols: Optional[list[str]] = None) -> Iterator[pd.DataFrame]:
    """Stream the large EV CSV safely in chunks.

    Parameters
    - chunksize: number of rows per chunk
    - usecols: subset of columns to read (for speed)
    """
    return pd.read_csv(RAW_CSV, chunksize=chunksize, low_memory=False, usecols=usecols)


def write_parquet(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
    return pd.read_parquet(path)
