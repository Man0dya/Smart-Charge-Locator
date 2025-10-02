import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path('data/raw/Electric_Vehicle_Population_Data.xlsx')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_excel(RAW_PATH)
    df.columns = [c.strip().title() for c in df.columns]

    expected = ['City', 'County', 'Postal Code', 'Model Year', 'Electric Range']
    keep = [c for c in expected if c in df.columns]
    extra = [c for c in ['Vehicle Location','Latitude','Longitude'] if c in df.columns]
    df = df[keep + extra]

    if 'Model Year' in df:
        df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
    if 'Postal Code' in df:
        df['Postal Code'] = pd.to_numeric(df['Postal Code'], errors='coerce')
    if 'City' in df:
        df = df.dropna(subset=['City'])
        df['City'] = df['City'].astype(str).str.strip().str.title()

    city_year = df.groupby(['City','Model Year'], as_index=False).size().rename(columns={'size':'EV_Count'})
    city_year = city_year.sort_values(['City','Model Year'])
    city_year['Prev_Year_EV_Count'] = city_year.groupby('City')['EV_Count'].shift(1).fillna(0)
    city_year['Year_Delta'] = city_year['Model Year'] - city_year['Model Year'].min()

    city_totals = city_year.groupby('City', as_index=False)['EV_Count'].sum().rename(columns={'EV_Count':'EV_Count_Total'})

    if 'Postal Code' in df:
        zips_per_city = df.groupby('City', as_index=False)['Postal Code'].nunique().rename(columns={'Postal Code':'Unique_Zips'})
        city_totals = city_totals.merge(zips_per_city, on='City', how='left')
        city_totals['Unique_Zips'] = city_totals['Unique_Zips'].replace(0, np.nan)
        city_totals['EV_Density_Proxy'] = city_totals['EV_Count_Total'] / city_totals['Unique_Zips']
    else:
        city_totals['EV_Density_Proxy'] = city_totals['EV_Count_Total']

    city_totals.to_csv(OUT_DIR / 'city_ev_agg.csv', index=False)
    city_year.to_csv(OUT_DIR / 'model_ready_city_ev.csv', index=False)

if __name__ == '__main__':
    main()


